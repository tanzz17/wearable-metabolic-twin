"""Train energy (MET proxy) model."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from metabolic_twin.pipeline import load_and_preprocess
from metabolic_twin.features import make_windows
from metabolic_twin.models.energy import build_met_target, train_energy_model, evaluate_energy_model, save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    model_dir = Path(cfg["paths"]["models_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.dat"))
    if not files:
        raise SystemExit(f"No .dat files found in {raw_dir}")

    df = load_and_preprocess(files)

    sr = cfg["windowing"]["sampling_rate_hz"]
    window_size = int(cfg["windowing"]["window_seconds"] * sr)
    step_size = int(cfg["windowing"]["step_seconds"] * sr)

    feature_cols = cfg["features"]["columns"]
    X, y_act = make_windows(df, feature_cols, "activity_id", window_size, step_size)
    y = build_met_target(y_act)

    model = train_energy_model(X, y, n_estimators=cfg["energy_model"]["n_estimators"])
    metrics = evaluate_energy_model(model, X, y)

    save_path = model_dir / "energy_rf.joblib"
    save_model(model, str(save_path))

    print("Energy model metrics:", metrics)
    print("Saved:", save_path)


if __name__ == "__main__":
    main()
