"""
auto_train.py

Zero-argument, autonomous version of train_model.py, meant to be triggered
by the dashboard itself (or a scheduler) with no human picking dates or
typing commands.

WHAT IT DOES AUTOMATICALLY
---------------------------
- Figures out its own training window: from the current MLB season's
  Opening Day through yesterday. (If it's currently the offseason, it uses
  last season's full range instead.)
- Uses a lock file so two overlapping runs (e.g. two dashboard sessions
  both noticing a stale model) can't stomp on each other.
- Writes the trained model to a temp file and atomically renames it into
  place, so the dashboard never reads a half-written model file.
- Records how long it took and how many pitches it trained on so the
  dashboard can show a meaningful status message.

This still needs internet access to pull Statcast data (via pybaseball) --
that part can't be worked around. It's meant to run somewhere that has it
(your own machine, a server, a scheduled job), not inside this sandbox.

USAGE
-----
Run directly:
    python -m ml.auto_train

Or import and call from the dashboard (non-blocking, in a background
thread/subprocess -- see app/dashboard.py):
    from ml.auto_train import run_auto_training
    run_auto_training()

To keep the model fresh with zero manual steps long-term, schedule this
to run nightly (cron / Task Scheduler) in addition to (or instead of) the
dashboard's own background trigger:
    0 6 * * * cd /path/to/pitch-dashboard && python -m ml.auto_train
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "pitch_quality_model.pkl"
LOCK_PATH = MODEL_DIR / ".training.lock"
LOG_PATH = MODEL_DIR / "auto_train.log"

# Roughly when MLB's regular season starts each year. Good enough for
# picking a training window automatically -- doesn't need to be exact.
SEASON_START_MONTH_DAY = (3, 25)


def _log(msg: str):
    line = f"[{datetime.utcnow().isoformat()}] {msg}"
    print(line)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def determine_training_window() -> tuple[str, str]:
    """Pick a start/end date automatically -- no human input required."""
    today = date.today()
    yesterday = today - timedelta(days=1)

    this_season_start = date(today.year, *SEASON_START_MONTH_DAY)
    if today >= this_season_start:
        start = this_season_start
        end = yesterday
    else:
        # Offseason (or very early in the year) -- train on last full season.
        start = date(today.year - 1, *SEASON_START_MONTH_DAY)
        end = date(today.year - 1, 10, 31)

    return start.isoformat(), end.isoformat()


def is_locked() -> bool:
    return LOCK_PATH.exists()


def needs_retrain(max_age_hours: float = 24.0) -> bool:
    """True if there's no model yet, or the existing one is older than
    max_age_hours."""
    if not MODEL_PATH.exists():
        return True
    age_hours = (time.time() - MODEL_PATH.stat().st_mtime) / 3600
    return age_hours >= max_age_hours


def run_auto_training(force: bool = False) -> bool:
    """Run a full training pass and atomically install the result.

    Returns True if training ran, False if it was skipped (already locked,
    or not needed). Safe to call speculatively -- it no-ops rather than
    erroring if another process already has the lock or the model is fresh.
    """
    if is_locked():
        _log("Skipped: another training run is already in progress.")
        return False
    if not force and not needs_retrain():
        _log("Skipped: existing model is still fresh.")
        return False

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(str(os.getpid()))
    try:
        # Imported lazily so the dashboard can import this module even in
        # environments where lightgbm/pybaseball aren't installed yet --
        # it'll only fail here, inside the actual training attempt.
        from ml.train_model import build_training_frame, train
        import joblib

        start, end = determine_training_window()
        _log(f"Starting auto-training for window {start} -> {end}")
        t0 = time.time()

        df = build_training_frame(start, end)
        model, calibration, feature_cols = train(df)

        bundle = {
            "model": model,
            "calibration": calibration,
            "feature_cols": feature_cols,
            "categorical_cols": ["pitch_type", "stand", "p_throws", "pitcher", "batter"],
            "trained_at": datetime.utcnow().isoformat(),
            "trained_range": [start, end],
            "pitch_count": len(df),
        }

        tmp_path = MODEL_PATH.with_suffix(".pkl.tmp")
        joblib.dump(bundle, tmp_path)
        os.replace(tmp_path, MODEL_PATH)  # atomic on POSIX and Windows

        elapsed = time.time() - t0
        _log(f"Finished in {elapsed:.0f}s. Trained on {len(df)} pitches. Model saved.")
        return True
    except Exception as e:  # noqa: BLE001 - log and move on, never crash the caller
        _log(f"Auto-training failed: {e}")
        return False
    finally:
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    force = "--force" in sys.argv
    ran = run_auto_training(force=force)
    sys.exit(0 if ran else 1)
