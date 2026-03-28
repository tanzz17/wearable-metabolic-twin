"""Windowing and feature extraction."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _feature_stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "energy": float(np.sum(x ** 2) / max(len(x), 1)),
    }


def build_feature_matrix(window: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for col in feature_cols:
        stats = _feature_stats(window[col].values)
        for k, v in stats.items():
            feats[f"{col}_{k}"] = v
    return feats


def make_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    window_size: int,
    step_size: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create sliding windows and return features + labels.
    window_size, step_size are in samples (not seconds).
    """
    rows = []
    labels = []

    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start : start + window_size]
        feats = build_feature_matrix(window, feature_cols)
        rows.append(feats)

        # Majority label in the window
        labels.append(int(window[label_col].mode()[0]))

    X = pd.DataFrame(rows)
    y = np.array(labels)
    return X, y
