"""Metabolic zone classification from heart rate."""

from __future__ import annotations

import numpy as np


def heart_rate_zones(hr: np.ndarray, hr_max: float) -> np.ndarray:
    """
    Zones based on %HRmax:
    - Rest: <50%
    - Fat Burn: 50-70%
    - Cardio: 70-85%
    - Peak: >85%
    """
    hr = np.asarray(hr, dtype=float)
    pct = hr / hr_max
    zones = np.empty_like(pct, dtype=int)

    zones[pct < 0.50] = 0
    zones[(pct >= 0.50) & (pct < 0.70)] = 1
    zones[(pct >= 0.70) & (pct < 0.85)] = 2
    zones[pct >= 0.85] = 3

    return zones


def zone_labels():
    return {
        0: "rest",
        1: "fat_burn",
        2: "cardio",
        3: "peak",
    }
