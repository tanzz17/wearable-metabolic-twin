"""Energy expenditure proxy modeling."""

from __future__ import annotations

from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Simple MET lookup by activity_id (approximate)
MET_LOOKUP = {
    1: 1.0,   # lying
    2: 1.3,   # sitting
    3: 1.6,   # standing
    4: 3.5,   # walking
    5: 7.0,   # running
    6: 6.8,   # cycling
    7: 6.0,   # nordic walking
    9: 1.0,   # watching TV
    10: 1.5,  # computer work
    11: 2.0,  # car driving
    12: 4.0,  # ascending stairs
    13: 3.5,  # descending stairs
    16: 3.5,  # vacuum cleaning
    17: 2.3,  # ironing
    18: 2.0,  # folding laundry
    19: 3.0,  # house cleaning
    20: 7.0,  # playing soccer
    24: 8.8,  # rope jumping
}


def build_met_target(activity_ids: np.ndarray) -> np.ndarray:
    """Return a MET target using a simple lookup table."""
    return np.array([MET_LOOKUP.get(int(a), 3.0) for a in activity_ids], dtype=float)


def train_energy_model(X, y, n_estimators: int = 300, random_state: int = 42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def evaluate_energy_model(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
