"""End-to-end pipeline: activity + energy + zone demo."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from metabolic_twin.pipeline import load_and_preprocess
from metabolic_twin.features import make_windows
from metabolic_twin.models.activity import train_activity_model, evaluate_activity_model, save_model as save_activity
from metabolic_twin.models.energy import build_met_target, train_energy_model, evaluate_energy_model, save_model as save_energy
from metabolic_twin.models.zone import heart_rate_zones, zone_labels
from metabolic_twin.data import PAMAP2_ACTIVITY_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--age", type=int, default=30)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a faster demo on fewer files/windows.",
    )
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=None,
        help="Limit number of subject files loaded (overrides --quick default).",
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Evaluate with leave-one-subject-out (LOSO) split.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    model_dir = Path(cfg["paths"]["models_dir"])
    out_dir = Path(cfg["paths"]["outputs_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.dat"))
    if not files:
        raise SystemExit(f"No .dat files found in {raw_dir}")

    # Quick mode defaults
    if args.quick and args.max_subjects is None:
        args.max_subjects = 1

    if args.max_subjects is not None:
        files = files[: args.max_subjects]

    print(f"Loading {len(files)} file(s) from {raw_dir} ...")
    start_time = time.time()
    sr = cfg["windowing"]["sampling_rate_hz"]
    target_rate = cfg["windowing"].get("target_rate_hz")
    df = load_and_preprocess(files, sampling_rate_hz=sr, target_rate_hz=target_rate)
    print(f"Loaded rows after cleaning: {len(df):,}")

    window_size = int(cfg["windowing"]["window_seconds"] * sr)
    step_size = int(cfg["windowing"]["step_seconds"] * sr)
    if args.quick:
        window_size = max(200, window_size // 2)
        step_size = max(100, step_size // 2)

    feature_cols = cfg["features"]["columns"]
    print("Building windows/features ...")
    X, y_act = make_windows(df, feature_cols, "activity_id", window_size, step_size)
    print(f"Feature matrix: {X.shape}, labels: {y_act.shape}")

    loso_metrics = None
    if args.loso:
        print("Running LOSO evaluation ...")
        # Build windows per subject file to keep subject IDs
        X_parts = []
        y_parts = []
        subj_parts = []
        for fpath in files:
            subject_id = fpath.stem
            sdf = load_and_preprocess([fpath], sampling_rate_hz=sr, target_rate_hz=target_rate)
            Xs, ys = make_windows(sdf, feature_cols, "activity_id", window_size, step_size)
            X_parts.append(Xs)
            y_parts.append(ys)
            subj_parts.append(np.full(len(ys), subject_id))

        X_all = np.vstack([x.values for x in X_parts])
        y_all = np.concatenate(y_parts)
        subj_all = np.concatenate(subj_parts)

        unique_subjects = sorted(set(subj_all))
        loso_scores = []
        for sid in unique_subjects:
            train_mask = subj_all != sid
            test_mask = subj_all == sid

            X_train = X_all[train_mask]
            y_train = y_all[train_mask]
            X_test = X_all[test_mask]
            y_test = y_all[test_mask]

            model = train_activity_model(
                X_train,
                y_train,
                n_estimators=cfg["activity_model"]["n_estimators"],
                model_type=cfg["activity_model"]["model_type"],
                max_depth=cfg["activity_model"]["max_depth"],
                learning_rate=cfg["activity_model"]["learning_rate"],
                num_leaves=cfg["activity_model"]["num_leaves"],
            )
            metrics = evaluate_activity_model(model, X_test, y_test)
            loso_scores.append(metrics)

        loso_metrics = {
            "accuracy_mean": float(np.mean([m["accuracy"] for m in loso_scores])),
            "f1_macro_mean": float(np.mean([m["f1_macro"] for m in loso_scores])),
            "n_subjects": len(unique_subjects),
        }
        print("LOSO metrics:", loso_metrics)

    # Activity model
    print("Training activity model ...")
    activity_model = train_activity_model(
        X,
        y_act,
        n_estimators=cfg["activity_model"]["n_estimators"],
        model_type=cfg["activity_model"]["model_type"],
        max_depth=cfg["activity_model"]["max_depth"],
        learning_rate=cfg["activity_model"]["learning_rate"],
        num_leaves=cfg["activity_model"]["num_leaves"],
    )
    activity_metrics = evaluate_activity_model(activity_model, X, y_act)
    save_activity(activity_model, str(model_dir / "activity_rf.joblib"))
    print("Activity metrics:", activity_metrics)

    # Energy model
    print("Training energy model ...")
    y_met = build_met_target(y_act)
    energy_model = train_energy_model(X, y_met, n_estimators=cfg["energy_model"]["n_estimators"])
    energy_metrics = evaluate_energy_model(energy_model, X, y_met)
    save_energy(energy_model, str(model_dir / "energy_rf.joblib"))
    print("Energy metrics:", energy_metrics)

    # Zone demo on heart rate (window median)
    hr_window = X[[c for c in X.columns if c.endswith("heart_rate_median")]].values.flatten()
    if len(hr_window) == 0:
        hr_window = X[[c for c in X.columns if c.endswith("heart_rate_mean")]].values.flatten()

    hr_max = 220 - args.age
    zones = heart_rate_zones(hr_window, hr_max)
    labels = zone_labels()
    zone_counts = {labels[int(k)]: int(v) for k, v in zip(*np.unique(zones, return_counts=True))}

    report_path = out_dir / "pipeline_report.txt"
    report_json_path = out_dir / "pipeline_report.json"
    eda_dir = out_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - start_time
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Activity metrics: {activity_metrics}\n")
        f.write(f"Energy metrics: {energy_metrics}\n")
        if loso_metrics:
            f.write(f"LOSO metrics: {loso_metrics}\n")
        f.write(f"Zone counts: {zone_counts}\n")

    print("Zone counts:", zone_counts)
    print("Saved report:", report_path)
    report_payload = {
        "activity_metrics": activity_metrics,
        "energy_metrics": energy_metrics,
        "loso_metrics": loso_metrics,
        "zone_counts": zone_counts,
        "activity_labels": PAMAP2_ACTIVITY_LABELS,
        "n_samples": int(len(y_act)),
        "n_features": int(X.shape[1]),
        "n_subject_files": int(len(files)),
        "elapsed_seconds": float(round(elapsed, 2)),
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print("Saved report JSON:", report_json_path)

    # Save a small prediction sample for the dashboard
    try:
        sample_n = min(200, len(X))
        X_sample = X.iloc[:sample_n]
        y_sample = y_act[:sample_n]
        pred_act = activity_model.predict(X_sample)
        pred_met = energy_model.predict(X_sample)

        # Use HR mean for zones if available
        hr_col = [c for c in X_sample.columns if c.endswith("heart_rate_mean")]
        if hr_col:
            hr_vals = X_sample[hr_col[0]].values
        else:
            hr_vals = np.full(sample_n, 120.0)
        z = heart_rate_zones(hr_vals, hr_max=220 - args.age)
        z_labels = zone_labels()

        sample_df = X_sample.copy()
        sample_df["activity_true"] = y_sample
        sample_df["activity_pred"] = pred_act
        sample_df["met_pred"] = pred_met
        sample_df["zone_pred"] = [z_labels[int(i)] for i in z]
        sample_path = out_dir / "sample_predictions.csv"
        sample_df.to_csv(sample_path, index=False)
    except Exception:
        pass

    # Save confusion matrix plot (train data)
    try:
        preds = activity_model.predict(X)
        disp = ConfusionMatrixDisplay.from_predictions(y_act, preds, xticks_rotation="vertical")
        plt.title("Activity Confusion Matrix (Train)")
        plt.tight_layout()
        cm_path = eda_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
    except Exception:
        pass

    # Energy scatter plot (train data)
    try:
        y_pred = energy_model.predict(X)
        plt.figure(figsize=(5, 4))
        plt.scatter(y_met, y_pred, s=6, alpha=0.4)
        plt.xlabel("MET (proxy)")
        plt.ylabel("Predicted MET")
        plt.title("Energy Model: Pred vs Actual (Train)")
        plt.tight_layout()
        scatter_path = eda_dir / "energy_scatter.png"
        plt.savefig(scatter_path, dpi=150)
        plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
