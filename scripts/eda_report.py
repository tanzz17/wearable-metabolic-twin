import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from metabolic_twin.data import load_pamap2_dat, PAMAP2_ACTIVITY_LABELS
from metabolic_twin.preprocessing import clean_dataframe, add_magnitudes, resample_by_rate
from metabolic_twin.models.zone import heart_rate_zones, zone_labels

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = ROOT / "data" / "raw" / "PAMAP2_Dataset" / "PAMAP2_Dataset" / "Protocol"
files = sorted(RAW_DIR.glob("*.dat"))
if not files:
    raise SystemExit(f"No .dat files found in {RAW_DIR}")

# Load a single subject for fast EDA
f = files[0]
print("EDA file:", f.name)

df = load_pamap2_dat(f)
df = clean_dataframe(df)
# Assume 100 Hz, target 20 Hz for lighter plots
sr = 100
tr = 20
df = resample_by_rate(df, sr, tr)
df = add_magnitudes(df)

# Activity distribution
plt.figure(figsize=(6, 4))
activity_counts = df["activity_id"].value_counts().sort_index()
labels = [PAMAP2_ACTIVITY_LABELS.get(int(a), str(a)) for a in activity_counts.index]
plt.bar(labels, activity_counts.values)
plt.xticks(rotation=90)
plt.title("Activity Distribution (subject sample)")
plt.tight_layout()
plt.savefig(OUT_DIR / "activity_distribution.png", dpi=150)
plt.close()

# Heart rate distribution
plt.figure(figsize=(5, 4))
plt.hist(df["heart_rate"], bins=30, color="#1f77b4", alpha=0.8)
plt.title("Heart Rate Distribution")
plt.xlabel("BPM")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "heart_rate_distribution.png", dpi=150)
plt.close()

# IMU magnitude trends (short window)
plt.figure(figsize=(6, 4))
for col in ["hand_accel_mag", "chest_accel_mag", "ankle_accel_mag"]:
    plt.plot(df[col].values[:2000], label=col)
plt.title("Accel Magnitude (short window)")
plt.xlabel("Sample")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "imu_magnitude_trends.png", dpi=150)
plt.close()

# Correlation heatmap (selected features)
feat_cols = [
    "hand_accel_mag",
    "chest_accel_mag",
    "ankle_accel_mag",
    "hand_gyro_mag",
    "chest_gyro_mag",
    "ankle_gyro_mag",
    "heart_rate",
]
plt.figure(figsize=(6, 4))
sns.heatmap(df[feat_cols].corr(), annot=False, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Feature Correlation (sample)")
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_correlation.png", dpi=150)
plt.close()

# Zone distribution
zones = heart_rate_zones(df["heart_rate"].values, hr_max=220 - 30)
zone_map = zone_labels()
zone_counts = pd.Series([zone_map[int(z)] for z in zones]).value_counts()
plt.figure(figsize=(5, 4))
plt.bar(zone_counts.index, zone_counts.values)
plt.title("Zone Distribution (sample)")
plt.tight_layout()
plt.savefig(OUT_DIR / "zone_distribution.png", dpi=150)
plt.close()

print("Saved EDA plots to", OUT_DIR)
