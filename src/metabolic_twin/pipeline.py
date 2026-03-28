"""High-level pipeline helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .data import load_pamap2_dat
from .preprocessing import add_magnitudes, clean_dataframe, resample_by_rate
from .features import make_windows


def load_and_preprocess(
    paths: List[str | Path],
    sampling_rate_hz: int | None = None,
    target_rate_hz: int | None = None,
) -> pd.DataFrame:
    frames = [load_pamap2_dat(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = clean_dataframe(df)
    if sampling_rate_hz is not None:
        df = resample_by_rate(df, sampling_rate_hz, target_rate_hz)
    df = add_magnitudes(df)
    return df


def build_activity_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int,
    step_size: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_windows(df, feature_cols, "activity_id", window_size, step_size)
    return X, y
