"""
train_model.py

Trains a learned pitch-quality model on historical Statcast data, in place
of the fixed 20-80 heuristic scale in data/quality.py.

WHAT THIS DOES DIFFERENTLY FROM THE HEURISTIC SCORER
------------------------------------------------------
Instead of comparing velocity/spin/location to a fixed league-average
formula, this trains a gradient-boosted model to predict each pitch's
*run value* (how good/bad it actually was for the pitcher, based on the
real outcome) using:
  - the pitch's physical characteristics (velocity, spin, movement, location)
  - the count (balls/strikes)
  - the batter's and pitcher's handedness
  - the batter's and pitcher's IDENTITY (as learned categorical features)

Because batter_id and pitcher_id are included as features, the trees can
learn matchup-specific patterns (e.g. "this pitcher's slider is unusually
effective against left-handed hitters with a high chase rate") without any
of that being hand-coded. Where there isn't enough data for a specific
matchup, gradient boosting naturally shrinks toward league-average
behavior, so a rookie facing an unfamiliar hitter doesn't get a wild
overconfident grade.

RUNNING THIS
------------
Needs internet access (pulls from Baseball Savant via `pybaseball`) and
these packages: pybaseball, lightgbm, pandas, scikit-learn, joblib.
This sandbox has no network access, so this script is meant to be run in
your own environment:

    pip install pybaseball lightgbm scikit-learn joblib pandas
    python ml/train_model.py --start 2024-04-01 --end 2024-09-30

It saves a trained model + calibration stats to ml/models/pitch_quality_model.pkl,
which ml/model.py then loads for live inference in the dashboard.

RETRAINING CADENCE
-------------------
Don't try to update this model pitch-by-pitch during a live game -- a
single pitch's outcome is far too noisy to learn from in isolation. The
intended workflow is to re-run this script periodically (nightly, or after
each game day) on all data collected so far. Recency weighting (see
`RECENCY_HALF_LIFE_DAYS` below) makes the model track a hitter's current
form without violently overreacting to any one plate appearance.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# How many days it takes for a pitch's training weight to decay by half.
# Lower = model adapts faster to recent form; higher = more stable, slower
# to react. 45 days is a reasonable starting point for an in-season model.
RECENCY_HALF_LIFE_DAYS = 45

FEATURE_COLUMNS = [
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
    "plate_x", "plate_z", "sz_top", "sz_bot",
    "balls", "strikes",
]
CATEGORICAL_COLUMNS = ["pitch_type", "stand", "p_throws", "pitcher", "batter"]


def pitch_outcome_value(row: pd.Series) -> float | None:
    """Approximate per-pitch run value from Statcast's description/events.

    Higher = better for the PITCHER (that's what we're grading: pitch
    quality from the pitcher's perspective). Values are rough linear-weight
    style approximations, not exact run expectancy -- good enough to rank
    pitches relative to each other, which is all a quality grade needs.
    """
    desc = str(row.get("description", "")).lower()
    events = str(row.get("events", "")).lower()

    if "called_strike" in desc:
        return 0.08
    if "swinging_strike" in desc:  # includes swinging_strike_blocked
        return 0.13
    if desc == "foul" or desc.startswith("foul"):
        return 0.02
    if "ball" in desc and "in_play" not in desc:
        return -0.06
    if "hit_by_pitch" in desc:
        return -0.12

    if "in_play" in desc or events:
        # Ball put in play -- use contact quality if available.
        xwoba = row.get("estimated_woba_using_speedangle")
        if pd.notna(xwoba):
            # League-average xwOBA on contact is roughly 0.35; scale so
            # weak contact (~0.15) is good for the pitcher, hard contact
            # (~0.55+) is bad.
            return float((0.35 - xwoba) * 0.6)
        if events in ("strikeout",):
            return 0.13
        if events in ("field_out", "force_out", "grounded_into_double_play",
                      "double_play", "sac_fly", "sac_bunt"):
            return 0.05
        if events in ("single", "double", "triple", "home_run"):
            return -0.15
        if events == "walk":
            return -0.06

    return None  # unrecognized event -- drop from training


def build_training_frame(start: str, end: str) -> pd.DataFrame:
    """Pull Statcast pitch-level data and engineer features + target."""
    try:
        from pybaseball import statcast
    except ImportError:
        print("pybaseball is required: pip install pybaseball", file=sys.stderr)
        raise

    print(f"Pulling Statcast data {start} -> {end} (this can take a while)...")
    df = statcast(start_dt=start, end_dt=end)
    print(f"Pulled {len(df)} raw rows.")

    df["target"] = df.apply(pitch_outcome_value, axis=1)
    df = df.dropna(subset=["target"] + FEATURE_COLUMNS)
    print(f"{len(df)} rows remain after dropping unusable/incomplete rows.")

    # Recency weight: more recent pitches count more in training.
    df["game_date"] = pd.to_datetime(df["game_date"])
    most_recent = df["game_date"].max()
    age_days = (most_recent - df["game_date"]).dt.days
    df["sample_weight"] = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    return df


def train(df: pd.DataFrame):
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    feature_cols = FEATURE_COLUMNS + CATEGORICAL_COLUMNS
    X = df[feature_cols]
    y = df["target"]
    w = df["sample_weight"]

    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, w, test_size=0.15, random_state=42
    )

    train_set = lgb.Dataset(
        X_train, label=y_train, weight=w_train,
        categorical_feature=CATEGORICAL_COLUMNS, free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_val, label=y_val, weight=w_val,
        categorical_feature=CATEGORICAL_COLUMNS, reference=train_set,
        free_raw_data=False,
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params, train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    preds = model.predict(X, num_iteration=model.best_iteration)
    calibration = {"mean": float(np.mean(preds)), "std": float(np.std(preds))}

    return model, calibration, feature_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--out", default="ml/models/pitch_quality_model.pkl",
        help="Where to save the trained model bundle",
    )
    args = parser.parse_args()

    df = build_training_frame(args.start, args.end)
    model, calibration, feature_cols = train(df)

    bundle = {
        "model": model,
        "calibration": calibration,
        "feature_cols": feature_cols,
        "categorical_cols": CATEGORICAL_COLUMNS,
        "trained_at": datetime.utcnow().isoformat(),
        "trained_range": [args.start, args.end],
    }
    joblib.dump(bundle, args.out)
    print(f"Saved model bundle to {args.out}")
    print(f"Calibration: mean={calibration['mean']:.4f} std={calibration['std']:.4f}")


if __name__ == "__main__":
    main()
