"""
model.py

Loads a trained pitch-quality model (see train_model.py) and scores live
pitches on a 20-80 scale, personalized to the specific batter/pitcher
matchup where the model has learned enough to say something meaningful.

Falls back gracefully (returns None) if no trained model file is found,
so the dashboard can drop back to the heuristic scorer in data/quality.py
without crashing -- useful before you've run training the first time, or
if the model file isn't shipped with the repo (it shouldn't be committed;
it's a multi-MB binary you generate yourself).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "pitch_quality_model.pkl"


class PitchQualityModel:
    """Wraps a trained LightGBM model for live 20-80 scoring."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._bundle = None
        self._load_error: Optional[str] = None
        self._loaded_mtime: Optional[float] = None
        self._try_load()

    def _try_load(self):
        if not self.model_path.exists():
            self._load_error = (
                f"No trained model found at {self.model_path}. "
                "Run `python ml/train_model.py --start ... --end ...` first."
            )
            return
        try:
            import joblib
            self._bundle = joblib.load(self.model_path)
            self._loaded_mtime = self.model_path.stat().st_mtime
        except Exception as e:  # noqa: BLE001 - want to degrade gracefully
            self._load_error = f"Failed to load model: {e}"

    def refresh_if_changed(self) -> bool:
        """Reload from disk if the model file has changed since we last
        loaded it (e.g. a background auto-training run just finished).
        Returns True if a reload happened."""
        if not self.model_path.exists():
            return False
        current_mtime = self.model_path.stat().st_mtime
        if self._loaded_mtime is None or current_mtime != self._loaded_mtime:
            self._try_load()
            return True
        return False

    @property
    def is_available(self) -> bool:
        return self._bundle is not None

    @property
    def status_message(self) -> str:
        if self.is_available:
            trained_at = self._bundle.get("trained_at", "unknown")
            trained_range = self._bundle.get("trained_range", ["?", "?"])
            pitch_count = self._bundle.get("pitch_count")
            count_bit = f", {pitch_count:,} pitches" if pitch_count else ""
            return (
                f"Learned model active — trained {trained_at[:10]} on data "
                f"{trained_range[0]} to {trained_range[1]}{count_bit}."
            )
        return self._load_error or "Model unavailable."

    def score(
        self,
        *,
        pitch_type: str | None,
        velocity: float | None,
        spin_rate: float | None,
        pfx_x: float | None,
        pfx_z: float | None,
        plate_x: float | None,
        plate_z: float | None,
        sz_top: float | None,
        sz_bot: float | None,
        balls: int | None,
        strikes: int | None,
        batter_stand: str | None,
        pitcher_throws: str | None,
        pitcher_id: int | None,
        batter_id: int | None,
    ) -> Optional[float]:
        """Return a 20-80 score, or None if the model can't score this pitch
        (missing model, or missing required fields)."""
        if not self.is_available:
            return None

        required = [velocity, spin_rate, plate_x, plate_z, sz_top, sz_bot]
        if any(v is None for v in required):
            return None

        import pandas as pd

        row = {
            "release_speed": velocity,
            "release_spin_rate": spin_rate,
            "pfx_x": pfx_x if pfx_x is not None else 0.0,
            "pfx_z": pfx_z if pfx_z is not None else 0.0,
            "plate_x": plate_x,
            "plate_z": plate_z,
            "sz_top": sz_top,
            "sz_bot": sz_bot,
            "balls": balls if balls is not None else 0,
            "strikes": strikes if strikes is not None else 0,
            "pitch_type": pitch_type or "UN",
            "stand": batter_stand or "R",
            "p_throws": pitcher_throws or "R",
            # Unseen IDs (a batter/pitcher not in training data) fall back
            # to a generic bucket rather than erroring -- the model treats
            # this like "league average player" via LightGBM's handling of
            # unseen categories.
            "pitcher": str(pitcher_id) if pitcher_id is not None else "unknown",
            "batter": str(batter_id) if batter_id is not None else "unknown",
        }

        feature_cols = self._bundle["feature_cols"]
        categorical_cols = self._bundle["categorical_cols"]
        X = pd.DataFrame([row])[feature_cols]
        for col in categorical_cols:
            X[col] = X[col].astype("category")

        model = self._bundle["model"]
        raw_pred = float(model.predict(X, num_iteration=model.best_iteration)[0])

        calib = self._bundle["calibration"]
        z = (raw_pred - calib["mean"]) / calib["std"] if calib["std"] else 0.0
        score = 50 + (z * 10)
        return max(20.0, min(80.0, score))
