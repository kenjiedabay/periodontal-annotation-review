"""Post-generation diagnostic audit against hidden DenPAR Training masks."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_masks(folder):
    masks = []
    for path in sorted(folder.glob("mask*.png")):
        with Image.open(path) as image:
            masks.append(np.asarray(image.convert("L")) > 0)
    return masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--pseudo-labels-dir", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    root = Path(args.split_manifest).resolve().parents[2]
    pseudo_root = Path(args.pseudo_labels_dir)
    true_root = root / manifest["source_mask_dir"]
    totals = {"images": 0, "true_instances": 0, "pseudo_instances": 0, "matched_instances_iou50": 0}
    dice = []
    for image_id in manifest["pseudo_label_candidate_ids"]:
        truth = read_masks(true_root / image_id)
        predicted = read_masks(pseudo_root / image_id)
        totals["images"] += 1
        totals["true_instances"] += len(truth)
        totals["pseudo_instances"] += len(predicted)
        used = set()
        for mask in predicted:
            candidates = [(np.logical_and(mask, target).sum() / max(1, np.logical_or(mask, target).sum()), i) for i, target in enumerate(truth) if i not in used]
            if candidates:
                score, i = max(candidates)
                if score >= 0.5:
                    used.add(i)
                    totals["matched_instances_iou50"] += 1
        predicted_union = np.logical_or.reduce(predicted) if predicted else np.zeros_like(truth[0])
        true_union = np.logical_or.reduce(truth)
        dice.append(2 * np.logical_and(predicted_union, true_union).sum() / max(1, predicted_union.sum() + true_union.sum()))
    totals["instance_recall_iou50"] = totals["matched_instances_iou50"] / totals["true_instances"]
    totals["mean_union_dice"] = float(np.mean(dice))
    output = pseudo_root / "hidden_reference_audit.json"
    output.write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
