"""Compare a DenPAR Validation image's tooth masks with current model output."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The system Python has the working CPU torch build; the backend environment supplies OpenCV.
sys.path.append(str(ROOT / "backend" / ".venv" / "Lib" / "site-packages"))
import cv2
import numpy as np

from app.tooth_segmentation import service


VALIDATION = ROOT / "DenPAR Radiographs Dataset" / "Dataset" / "Validation"
COLORS = [(50, 220, 255), (230, 190, 40), (240, 75, 180), (75, 235, 110), (170, 110, 250)]


def box(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def label(canvas: np.ndarray, mask: np.ndarray, name: str, color: tuple[int, int, int]) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, 3)
    x1, y1, _, _ = box(mask)
    cv2.putText(canvas, name, (x1 + 3, max(24, y1 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_id", help="Numeric DenPAR Validation image ID")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.image_id.isdecimal():
        parser.error("image_id must be numeric")
    image_path = VALIDATION / "Images" / f"{args.image_id}.jpg"
    mask_dir = VALIDATION / "Masks (Tooth-wise)" / args.image_id
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        parser.error(f"image not found: {image_path}")
    mask_paths = sorted(mask_dir.glob("mask*.png"), key=lambda path: int(path.stem[4:]))
    truth = [(path.name, cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0) for path in mask_paths]
    if not truth or any(mask.shape != original.shape[:2] for _, mask in truth):
        parser.error("missing or mis-sized DenPAR masks")
    output = args.output or ROOT / "artifacts" / "tooth-segmentation" / f"case-{args.image_id}"
    output.mkdir(parents=True, exist_ok=True)

    result = service.predict(args.image_id, original)
    if result["model_status"] != "available" or result["ground_truth"] is None:
        raise RuntimeError(f"Model or exact-pixel DenPAR comparison unavailable: {result.get('model_error')}")
    predicted = []
    for item in result["instances"]:
        encoded = item["mask_url"].split(",", 1)[1]
        rgba = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_UNCHANGED)
        predicted.append((item, rgba[:, :, 3] > 0))

    matrix = []
    truth_rows = []
    for name, mask in truth:
        _, _, component_stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        substantial_components = [int(stats[cv2.CC_STAT_AREA]) for stats in component_stats[1:] if stats[cv2.CC_STAT_AREA] > 100]
        overlaps = []
        for item, prediction in predicted:
            intersection = np.logical_and(mask, prediction).sum()
            union = np.logical_or(mask, prediction).sum()
            overlaps.append(float(intersection / union) if union else 0.0)
        matrix.append(overlaps)
        best = int(np.argmax(overlaps)) if overlaps else None
        truth_rows.append({"mask": name, "bbox_xyxy": box(mask), "pixels": int(mask.sum()),
                           "components_over_100_pixels": substantial_components,
                           "best_prediction_instance": predicted[best][0]["instance_id"] if best is not None and overlaps[best] > 0 else None,
                           "best_iou": round(overlaps[best], 4) if best is not None else 0.0,
                           "matched_at_iou_0_50": best is not None and overlaps[best] >= 0.5})

    source_panel = original.copy()
    prediction_panel = original.copy()
    combined_panel = original.copy()
    for index, (name, mask) in enumerate(truth):
        color = COLORS[index % len(COLORS)]
        label(source_panel, mask, name.removesuffix(".png"), color)
        label(combined_panel, mask, name.removesuffix(".png"), color)
    for item, mask in predicted:
        name = f"P{item['instance_id']} {item['confidence']:.2f}"
        label(prediction_panel, mask, name, (50, 60, 255))
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(combined_panel, contours, -1, (50, 60, 255), 2)
    titled = []
    for panel, title in ((source_panel, "DENPAR: source masks"), (prediction_panel, "MODEL: confidence >= 0.50"), (combined_panel, "COMPARISON: source colors / prediction red")):
        panel = cv2.copyMakeBorder(panel, 45, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 30, 30))
        cv2.putText(panel, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        titled.append(panel)
    cv2.imwrite(str(output / "comparison.png"), np.concatenate(titled, axis=1))
    report = {"image_id": args.image_id, "source_image": str(image_path.relative_to(ROOT)),
              "exact_source_pixel_match": True, "model_version": result["model_version"],
              "confidence_threshold": result["confidence_threshold"], "denpar_instance_count": len(truth),
              "predicted_instance_count": len(predicted), "denpar_masks": truth_rows,
              "predictions": [{"instance_id": item["instance_id"], "confidence": round(item["confidence"], 4),
                               "bbox_xyxy": [round(value, 1) for value in item["bbox"]]} for item, _ in predicted],
              "iou_matrix_denpar_rows_prediction_columns": [[round(value, 4) for value in row] for row in matrix],
              "matching_rule": "Per-mask best IoU; 0.50 threshold; model output at confidence >= 0.50",
              "note": "Structural tooth segmentation comparison only; no disease or tooth-identity claim."}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(output), "denpar": len(truth), "predicted": len(predicted), "mask_best_iou": [(row["mask"], row["best_iou"]) for row in truth_rows]}, indent=2))


if __name__ == "__main__":
    main()
