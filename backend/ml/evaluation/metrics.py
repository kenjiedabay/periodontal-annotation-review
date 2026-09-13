"""Binary classification metrics without an sklearn dependency."""

import csv
import json
from pathlib import Path
from typing import Any

import torch


def classification_metrics(targets: list[int], predictions: list[int]) -> dict[str, Any]:
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(targets, predictions))
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in zip(targets, predictions))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(targets, predictions))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(targets, predictions))
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "support": {"absent": tn + fp, "present": tp + fn},
    }


def write_evaluation(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    matrix = metrics["confusion_matrix"]
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", "absent", "present"])
        writer.writerow(["absent", matrix[0][0], matrix[0][1]])
        writer.writerow(["present", matrix[1][0], matrix[1][1]])
