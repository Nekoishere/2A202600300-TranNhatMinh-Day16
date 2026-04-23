#!/usr/bin/env python3
"""CPU benchmark for the Kaggle credit card fraud dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM fraud model and write benchmark_result.json."
    )
    parser.add_argument(
        "--data",
        default="creditcard.csv",
        help="Path to the Kaggle credit card fraud CSV file.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_result.json",
        help="Path to the JSON file where benchmark metrics will be saved.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible train/validation/test splits.",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, float]:
    start = time.perf_counter()
    df = pd.read_csv(csv_path)
    load_seconds = time.perf_counter() - start

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    features = df.drop(columns=["Class"])
    labels = df["Class"].astype(int)
    return features, labels, load_seconds


def split_dataset(
    features: pd.DataFrame, labels: pd.Series, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=random_state,
    )
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def build_model(y_train: pd.Series) -> lgb.LGBMClassifier:
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / max(positive_count, 1)

    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=os.cpu_count() or 1,
        verbose=-1,
    )


def benchmark_inference(model: lgb.LGBMClassifier, x_test: pd.DataFrame) -> tuple[float, float]:
    sample_row = x_test.iloc[[0]]
    single_row_repeats = 200
    batch = x_test.iloc[: min(len(x_test), 1000)]
    batch_repeats = 20

    start = time.perf_counter()
    for _ in range(single_row_repeats):
        model.predict_proba(sample_row)
    latency_seconds = (time.perf_counter() - start) / single_row_repeats

    start = time.perf_counter()
    for _ in range(batch_repeats):
        model.predict_proba(batch)
    throughput_rows_per_second = (len(batch) * batch_repeats) / (time.perf_counter() - start)

    return latency_seconds * 1000.0, throughput_rows_per_second


def main() -> None:
    args = parse_args()
    csv_path = Path(args.data).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download and unzip creditcard.csv first."
        )

    features, labels, load_seconds = load_dataset(csv_path)
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_dataset(
        features, labels, args.random_state
    )

    model = build_model(y_train)

    train_start = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    training_seconds = time.perf_counter() - train_start

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    inference_latency_ms, inference_throughput = benchmark_inference(model, x_test)

    best_iteration = model.best_iteration_
    if best_iteration is None or best_iteration <= 0:
        best_iteration = model.n_estimators_

    results = {
        "dataset_path": str(csv_path),
        "dataset_rows": int(len(features)),
        "dataset_columns": int(features.shape[1]),
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_valid)),
        "test_rows": int(len(x_test)),
        "positive_rate": float(labels.mean()),
        "load_data_seconds": round(load_seconds, 4),
        "training_seconds": round(training_seconds, 4),
        "best_iteration": int(best_iteration),
        "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 6),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 6),
        "inference_latency_ms_1_row": round(float(inference_latency_ms), 6),
        "inference_throughput_rows_per_sec_1000_rows": round(float(inference_throughput), 2),
    }

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("LightGBM CPU benchmark completed")
    print(f"Dataset: {csv_path}")
    print(f"Rows: {results['dataset_rows']}, Features: {results['dataset_columns']}")
    print(f"Load data time: {results['load_data_seconds']:.4f} s")
    print(f"Training time: {results['training_seconds']:.4f} s")
    print(f"Best iteration: {results['best_iteration']}")
    print(f"AUC-ROC: {results['auc_roc']:.6f}")
    print(f"Accuracy: {results['accuracy']:.6f}")
    print(f"F1-Score: {results['f1_score']:.6f}")
    print(f"Precision: {results['precision']:.6f}")
    print(f"Recall: {results['recall']:.6f}")
    print(f"Inference latency (1 row): {results['inference_latency_ms_1_row']:.6f} ms")
    print(
        "Inference throughput (1000 rows): "
        f"{results['inference_throughput_rows_per_sec_1000_rows']:.2f} rows/sec"
    )
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
