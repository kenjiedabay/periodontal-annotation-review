"""Compare final DenPAR pseudo-label experiment checkpoints on Validation misses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import ToothInstanceDataset, validate_official_split
from .model import create_model


def analyze_prediction(prediction, target):
    truth = target["masks"].numpy().astype(bool)
    masks = (prediction["masks"][:, 0].cpu().numpy() >= 0.5)
    scores = prediction["scores"].cpu().numpy()
    ious = np.zeros((len(truth), len(masks)), dtype=np.float32)
    for i, reference in enumerate(truth):
        for j, proposed in enumerate(masks):
            intersection = np.logical_and(reference, proposed).sum()
            union = np.logical_or(reference, proposed).sum()
            ious[i, j] = intersection / union if union else 0
    assigned = {}
    used = set()
    # Match the existing experiment evaluator: model score order, greedy IoU >= 0.50.
    for j in range(len(masks)):
        candidates = [(ious[i, j], i) for i in range(len(truth)) if i not in used]
        if candidates:
            value, i = max(candidates)
            if value >= 0.5:
                assigned[i] = j
                used.add(i)
    rows = []
    for i, reference in enumerate(truth):
        ys, xs = np.where(reference)
        best = int(np.argmax(ious[i])) if len(masks) else None
        overlap = float(ious[i, best]) if best is not None else 0.0
        components, _ = cv2.connectedComponents(reference.astype(np.uint8))
        rows.append({
            "mask": f"mask{i + 1}",
            "matched": i in assigned,
            "matched_score": float(scores[assigned[i]]) if i in assigned else None,
            "best_iou": overlap,
            "best_score": float(scores[best]) if best is not None else None,
            "best_prediction_index": best,
            "area_pixels": int(reference.sum()),
            "bbox_xyxy_processed": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            "components": components - 1,
        })
    return rows, masks, scores


def load_model(checkpoint, device):
    model, _ = create_model(False, 1024)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
    return model.to(device).eval()


def overlay(image, truth, mask_sets, path):
    base = (image[0].numpy() * 255).astype(np.uint8)
    panels = []
    for title, masks, scores in mask_sets:
        canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        for i, reference in enumerate(truth):
            contours, _ = cv2.findContours(reference.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (40, 210, 40), 2)
            ys, xs = np.where(reference)
            cv2.putText(canvas, f"T{i+1}", (int(xs.min()), max(16, int(ys.min()))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 210, 40), 1)
        for j, mask in enumerate(masks):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (20, 30, 240), 2)
            ys, xs = np.where(mask)
            if len(xs):
                cv2.putText(canvas, f"P{j+1} {scores[j]:.2f}", (int(xs.min()), min(1000, int(ys.max()) + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 30, 240), 1)
        canvas = cv2.copyMakeBorder(canvas, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 15))
        cv2.putText(canvas, title + " | green=reference red=prediction", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        panels.append(canvas)
    cv2.imwrite(str(path), np.hstack(panels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--training-images-dir", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, _ = validate_official_split(Path(args.dataset_root), "validation", Path(args.training_images_dir))
    dataset = ToothInstanceDataset(records, 1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = {}
    for name, checkpoint in (("control", args.control_checkpoint), ("student", args.student_checkpoint)):
        model = load_model(checkpoint, device)
        per_image = {}
        with torch.inference_mode():
            for index in range(len(dataset)):
                image, target = dataset[index]
                result = model([image.to(device)])[0]
                rows, _, _ = analyze_prediction(result, target)
                per_image[target["metadata"]["image_id"]] = rows
                if (index + 1) % 25 == 0:
                    print(f"{name}: {index + 1}/{len(dataset)}", flush=True)
        summaries[name] = per_image
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    differences = []
    for image_id in sorted(summaries["control"], key=int):
        control = summaries["control"][image_id]
        student = summaries["student"][image_id]
        for left, right in zip(control, student):
            if not left["matched"] or not right["matched"]:
                differences.append({"image_id": image_id, "mask": left["mask"], "case": "both_miss" if not left["matched"] and not right["matched"] else "student_only_miss" if left["matched"] else "control_only_miss", "control": left, "student": right})
    counts = {label: sum(item["case"] == label for item in differences) for label in ("both_miss", "student_only_miss", "control_only_miss")}
    chosen = []
    for label, limit in (("student_only_miss", 3), ("both_miss", 3), ("control_only_miss", 2)):
        chosen.extend([item["image_id"] for item in differences if item["case"] == label][:limit])
    chosen = list(dict.fromkeys(chosen))
    if "584" in summaries["control"] and "584" not in chosen:
        chosen.append("584")
    for image_id in chosen:
        index = next(i for i, record in enumerate(records) if record.image_id == image_id)
        image, target = dataset[index]
        rendered = []
        for name, checkpoint in (("control", args.control_checkpoint), ("student", args.student_checkpoint)):
            model = load_model(checkpoint, device)
            with torch.inference_mode():
                result = model([image.to(device)])[0]
            _, masks, scores = analyze_prediction(result, target)
            rendered.append((name, masks, scores))
            del model
        overlay(image, target["masks"].numpy().astype(bool), rendered, out / f"case_{image_id}.png")
    report = {"split": "official_validation", "matching_iou": 0.5, "mask_threshold": 0.5, "counts": counts, "difference_cases": differences, "overlay_images": chosen}
    (out / "miss_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"counts": counts, "overlays": chosen}, indent=2))


if __name__ == "__main__":
    main()
