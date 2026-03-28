"""Train/evaluate activity classifier."""

from __future__ import annotations

from typing import Dict

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def train_activity_model(
    X,
    y,
    n_estimators: int = 300,
    random_state: int = 42,
    model_type: str = "rf",
    max_depth: int = -1,
    learning_rate: float = 0.1,
    num_leaves: int = 63,
):
    if model_type.lower() == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("LightGBM not installed. Install lightgbm or set model_type=rf.") from exc
        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
        )
    model.fit(X, y)
    return model


def evaluate_activity_model(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1_macro": float(f1_score(y, preds, average="macro")),
    }


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
