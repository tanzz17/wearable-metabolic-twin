"""Preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Remove transient activity label (0)
    - Forward-fill heart rate, then drop remaining NaNs
    """
    df = df.copy()
    df = df[df["activity_id"] != 0]

    if "heart_rate" in df.columns:
        df["heart_rate"] = df["heart_rate"].replace(-1, np.nan)
        df["heart_rate"] = df["heart_rate"].ffill()

    df = df.dropna(axis=0)
    return df


def resample_by_rate(df: pd.DataFrame, sampling_rate_hz: int, target_rate_hz: int | None) -> pd.DataFrame:
    """
    Lightweight resampling by decimation (keep every k-th row).
    This assumes roughly uniform sampling.
    """
    if target_rate_hz is None:
        return df
    if target_rate_hz >= sampling_rate_hz:
        return df
    step = max(int(round(sampling_rate_hz / target_rate_hz)), 1)
    return df.iloc[::step].reset_index(drop=True)


def add_magnitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Add accel and gyro magnitude features for each sensor."""
    df = df.copy()

    for prefix in ["hand", "chest", "ankle"]:
        ax = f"{prefix}_accel16_x"
        ay = f"{prefix}_accel16_y"
        az = f"{prefix}_accel16_z"
        gx = f"{prefix}_gyro_x"
        gy = f"{prefix}_gyro_y"
        gz = f"{prefix}_gyro_z"

        df[f"{prefix}_accel_mag"] = np.sqrt(df[ax] ** 2 + df[ay] ** 2 + df[az] ** 2)
        df[f"{prefix}_gyro_mag"] = np.sqrt(df[gx] ** 2 + df[gy] ** 2 + df[gz] ** 2)

    return df
