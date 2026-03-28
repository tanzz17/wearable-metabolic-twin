"""PAMAP2 data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

# PAMAP2 column definitions (54 columns)
PAMAP2_COLUMNS: List[str] = [
    "timestamp",
    "activity_id",
    "heart_rate",
    # IMU hand
    "hand_temp",
    "hand_accel16_x",
    "hand_accel16_y",
    "hand_accel16_z",
    "hand_accel6_x",
    "hand_accel6_y",
    "hand_accel6_z",
    "hand_gyro_x",
    "hand_gyro_y",
    "hand_gyro_z",
    "hand_mag_x",
    "hand_mag_y",
    "hand_mag_z",
    "hand_orient_w",
    "hand_orient_x",
    "hand_orient_y",
    "hand_orient_z",
    # IMU chest
    "chest_temp",
    "chest_accel16_x",
    "chest_accel16_y",
    "chest_accel16_z",
    "chest_accel6_x",
    "chest_accel6_y",
    "chest_accel6_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "chest_mag_x",
    "chest_mag_y",
    "chest_mag_z",
    "chest_orient_w",
    "chest_orient_x",
    "chest_orient_y",
    "chest_orient_z",
    # IMU ankle
    "ankle_temp",
    "ankle_accel16_x",
    "ankle_accel16_y",
    "ankle_accel16_z",
    "ankle_accel6_x",
    "ankle_accel6_y",
    "ankle_accel6_z",
    "ankle_gyro_x",
    "ankle_gyro_y",
    "ankle_gyro_z",
    "ankle_mag_x",
    "ankle_mag_y",
    "ankle_mag_z",
    "ankle_orient_w",
    "ankle_orient_x",
    "ankle_orient_y",
    "ankle_orient_z",
]

PAMAP2_ACTIVITY_LABELS = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic_walking",
    9: "watching_TV",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}


def load_pamap2_dat(path: str | Path) -> pd.DataFrame:
    """
    Load a single PAMAP2 .dat file into a DataFrame.
    The file is whitespace-separated and has 54 columns.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=PAMAP2_COLUMNS,
        engine="python",
    )
    return df


def load_many(paths: List[str | Path]) -> pd.DataFrame:
    frames = [load_pamap2_dat(p) for p in paths]
    return pd.concat(frames, ignore_index=True)
