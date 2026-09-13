"""Multiclass severity metrics without sklearn."""

import csv
import json
from pathlib import Path
from typing import Any


def multiclass_metrics(targets: list[int], predictions: list[int], labels: list[str]) -> dict[str, Any]:
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for actual, predicted in zip(targets, predictions):
        matrix[actual][predicted] += 1
    total = sum(sum(row) for row in matrix)
    accuracy = sum(matrix[index][index] for index in range(size)) / total if total else 0.0
    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(size) if row != index)
        fn = sum(matrix[index][column] for column in range(size) if column != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(matrix[index])}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {"labels": labels, "accuracy": accuracy, "precision_macro": sum(precisions) / size if size else 0.0, "recall_macro": sum(recalls) / size if size else 0.0, "f1_macro": sum(f1s) / size if size else 0.0, "per_class": per_class, "confusion_matrix": matrix, "support": {label: sum(matrix[index]) for index, label in enumerate(labels)}}


def write_metrics(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "severity_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    matrix = metrics["confusion_matrix"]
    labels = metrics["labels"]
    with (output_dir / "severity_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])
