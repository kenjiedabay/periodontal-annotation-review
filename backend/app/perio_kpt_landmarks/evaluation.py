"""Landmark-only evaluation A and qualitative visualization."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .geometry import calculate_rbl
from .schema import LANDMARK_NAMES, rbl_indices


def decode_heatmaps(probabilities: torch.Tensor,
                    valid_content: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, _, width = probabilities.shape
    if valid_content is not None:
        mask = valid_content[:, None].bool()
        probabilities = probabilities.masked_fill(~mask, float("-inf"))
    flattened = probabilities.reshape(batch, channels, -1)
    confidence, index = flattened.max(dim=-1)
    points = torch.stack(((index % width).float(), (index // width).float()), dim=-1)
    if valid_content is not None:
        for item in range(batch):
            assert bool(valid_content[item, points[item, :, 1].long(), points[item, :, 0].long()].all())
    return points, confidence


def summarize_predictions(rows: list[dict], output_size: int) -> dict:
    errors: dict[int, list[float]] = defaultdict(list)
    rbl_errors: dict[str, list[float]] = defaultdict(list)
    failures = defaultdict(int)
    diagonal = float(np.sqrt(2.0) * output_size)
    for row in rows:
        for index, available in enumerate(row["availability"]):
            if not available:
                continue
            error = float(np.linalg.norm(np.asarray(row["prediction"][index]) - np.asarray(row["target"][index])))
            if np.isfinite(error):
                errors[index].append(error)
            else:
                failures[index] += 1
        target_points = [tuple(point) if available else None
                         for point, available in zip(row["target"], row["availability"])]
        predicted_points = [tuple(point) for point in row["prediction"]]
        for surface in ("mesial", "distal"):
            indices = rbl_indices(row["class_id"], surface)
            truth = calculate_rbl(target_points, surface, indices=indices)
            prediction = calculate_rbl(predicted_points, surface, indices=indices)
            if truth["status"] == "assessable" and prediction["status"] == "assessable":
                rbl_errors[surface].append(prediction["rbl_percent"] - truth["rbl_percent"])
            elif truth["status"] == "assessable":
                failures[f"rbl_{surface}"] += 1
    landmark_metrics = {}
    all_errors = []
    for index, name in enumerate(LANDMARK_NAMES):
        values = np.asarray(errors[index], dtype=np.float64)
        all_errors.extend(values.tolist())
        landmark_metrics[name] = {
            "count": int(values.size), "failed": int(failures[index]),
            "mean_pixel_error": float(values.mean()) if values.size else None,
            "normalized_mean_error": float((values / diagonal).mean()) if values.size else None,
            "pck_0.05": float((values / diagonal <= 0.05).mean()) if values.size else None,
            "pck_0.10": float((values / diagonal <= 0.10).mean()) if values.size else None,
        }
    rbl_metrics = {}
    for surface in ("mesial", "distal"):
        values = np.asarray(rbl_errors[surface], dtype=np.float64)
        rbl_metrics[surface] = {
            "paired_assessable_count": int(values.size),
            "prediction_invalid_count": int(failures[f"rbl_{surface}"]),
            "bias_percentage_points": float(values.mean()) if values.size else None,
            "mae_percentage_points": float(np.abs(values).mean()) if values.size else None,
            "rmse_percentage_points": float(np.sqrt(np.square(values).mean())) if values.size else None,
        }
    all_values = np.asarray(all_errors)
    return {
        "normalization": "256x256 crop diagonal",
        "overall_mean_pixel_error": float(all_values.mean()) if all_values.size else None,
        "overall_normalized_mean_error": float((all_values / diagonal).mean()) if all_values.size else None,
        "overall_pck_0.05": float((all_values / diagonal <= 0.05).mean()) if all_values.size else None,
        "overall_pck_0.10": float((all_values / diagonal <= 0.10).mean()) if all_values.size else None,
        "per_landmark": landmark_metrics, "rbl": rbl_metrics,
        "status": "preliminary_one_epoch_smoke_test",
    }


def save_overlay(image: torch.Tensor, row: dict, destination: Path) -> None:
    pixels = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    for index, (prediction, target, available) in enumerate(zip(row["prediction"], row["target"], row["availability"])):
        if not available:
            continue
        px, py = (int(round(value)) for value in prediction)
        tx, ty = (int(round(value)) for value in target)
        cv2.circle(overlay, (tx, ty), 4, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.drawMarker(overlay, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        cv2.putText(overlay, LANDMARK_NAMES[index], (min(px + 4, 215), max(py - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 255), 1, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), overlay)
