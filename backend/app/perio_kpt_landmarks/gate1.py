"""Gate 1: CPU-only Perio-KPT audit and deterministic preparation evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random

import cv2
import numpy as np

from .data import (annotation_points_original, find_image, load_image,
                   parse_label, prepare_annotation)
from .geometry import calculate_rbl, gaussian_heatmaps
from .schema import (LANDMARK_NAMES, OBJECT_CLASS_NAMES, SCHEMA_VERSION,
                     TOOTH_CLASSES, rbl_indices)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT / "datasets/sources/perio-kpt/extracted/Periodontal_Keypoint_Dataset"
DEFAULT_OUTPUT = ROOT / "artifacts/perio-kpt-landmarks/gate1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def split_paths(dataset: Path, fold: str, split: str) -> tuple[Path, Path]:
    base = dataset / "1_Experiment/standard_box" / fold / split
    return base / "images", base / "labels"


def canonical_sources(dataset: Path) -> tuple[dict[str, tuple[Path, Path]], set[str]]:
    result: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "val"):
        image_dir, label_dir = split_paths(dataset, "f0", split)
        for image in sorted(image_dir.glob("*.png")):
            result[image.stem] = (image, label_dir / f"{image.stem}.txt")
    holdout = dataset / "1_Experiment/holdout_test_standard_box"
    holdout_ids = set()
    for image in sorted((holdout / "images").glob("*.png")):
        holdout_ids.add(image.stem)
        result[image.stem] = (image, holdout / "labels" / f"{image.stem}.txt")
    return result, holdout_ids


def build_folds(dataset: Path, sources: dict[str, tuple[Path, Path]], holdout_ids: set[str]) -> dict:
    folds = []
    for number in range(5):
        fold = f"f{number}"
        train_dir, _ = split_paths(dataset, fold, "train")
        val_dir, _ = split_paths(dataset, fold, "val")
        train = sorted(path.stem for path in train_dir.glob("*.png"))
        validation = sorted(path.stem for path in val_dir.glob("*.png"))
        if set(train) & set(validation):
            raise ValueError(f"radiograph overlap inside {fold}")
        if (set(train) | set(validation)) & holdout_ids:
            raise ValueError(f"holdout leakage inside {fold}")
        folds.append({"fold": number, "train_image_ids": train, "validation_image_ids": validation,
                      "train_count": len(train), "validation_count": len(validation)})
    development_ids = sorted(set().union(*(set(item["train_image_ids"]) | set(item["validation_image_ids"]) for item in folds)))
    if set(development_ids) | holdout_ids != set(sources):
        raise ValueError("fold identities do not cover the canonical source inventory")
    source_records = []
    for image_id, (image, label) in sorted(sources.items()):
        source_records.append({"image_id": image_id, "partition": "holdout" if image_id in holdout_ids else "development",
                               "image_path": relative(image), "image_sha256": sha256(image),
                               "label_path": relative(label), "label_sha256": sha256(label)})
    manifest = {"manifest_version": "perio-kpt-five-fold-v1", "schema_version": SCHEMA_VERSION,
                "grouping_unit": "radiograph", "patient_grouping_available": False,
                "development_image_count": len(development_ids), "holdout_image_count": len(holdout_ids),
                "development_image_ids": development_ids, "holdout_image_ids": sorted(holdout_ids),
                "folds": folds, "sources": source_records}
    frozen_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(frozen_bytes).hexdigest()
    return manifest


def _render_example(image: np.ndarray, annotation, output_dir: Path, index: int) -> dict:
    height, width = image.shape
    transform, crop_points = prepare_annotation(annotation, width, height)
    x1, y1, _, _ = transform.crop_xyxy
    affine = np.asarray([[transform.scale, 0, transform.offset_x - transform.scale * x1],
                         [0, transform.scale, transform.offset_y - transform.scale * y1]], dtype=np.float32)
    crop = cv2.warpAffine(image, affine, (transform.output_size, transform.output_size),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    overlay = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    colors = ((35, 230, 180), (40, 180, 255), (230, 80, 230), (35, 230, 180),
              (40, 180, 255), (230, 80, 230), (200, 110, 255), (255, 210, 70),
              (80, 200, 255), (80, 200, 255), (50, 80, 255))
    for keypoint_index, point in enumerate(crop_points):
        if point is None:
            continue
        px, py = int(round(point[0])), int(round(point[1]))
        cv2.circle(overlay, (px, py), 4, colors[keypoint_index], -1, lineType=cv2.LINE_AA)
        cv2.putText(overlay, LANDMARK_NAMES[keypoint_index], (min(px + 5, 210), max(py - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, colors[keypoint_index], 1, cv2.LINE_AA)
    heatmaps, availability = gaussian_heatmaps(crop_points, transform.output_size)
    combined = np.clip(heatmaps.max(axis=0) * 255, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(combined, cv2.COLORMAP_TURBO)
    heatmap_overlay = cv2.addWeighted(overlay, 0.70, heatmap_color, 0.30, 0)
    stem = f"{index:02d}_{annotation.image_id}_object{annotation.object_index}_{OBJECT_CLASS_NAMES[annotation.class_id]}"
    overlay_path = output_dir / f"{stem}_landmarks.png"
    heatmap_path = output_dir / f"{stem}_heatmaps.png"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(heatmap_path), heatmap_overlay)
    return {"record_id": annotation.record_id, "class_id": annotation.class_id,
            "class_name": OBJECT_CLASS_NAMES[annotation.class_id],
            "expanded_crop": transform.to_dict(), "available_landmarks": int(availability.sum()),
            "landmark_overlay": relative(overlay_path), "heatmap_overlay": relative(heatmap_path)}


def run(dataset: Path, output: Path) -> dict:
    dataset, output = dataset.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples_dir = output / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    sources, holdout_ids = canonical_sources(dataset)
    manifest = build_folds(dataset, sources, holdout_ids)

    annotations_by_image, quarantined = {}, []
    class_counts: Counter[int] = Counter()
    landmark_counts: Counter[str] = Counter()
    visibility_counts: Counter[int] = Counter()
    image_dimensions: Counter[str] = Counter()
    valid_rows = 0
    rbl_results = []
    heatmap_ready_records = 0
    for image_id, (image_path, label_path) in sorted(sources.items()):
        if not label_path.is_file():
            quarantined.append({"record_id": image_id, "image_id": image_id, "reason": "missing_label",
                                "source_preserved": True})
            annotations_by_image[image_id] = []
            continue
        image = load_image(image_path)
        height, width = image.shape
        image_dimensions[f"{width}x{height}"] += 1
        annotations, rejected = parse_label(label_path)
        annotations_by_image[image_id] = annotations
        quarantined.extend(rejected)
        valid_rows += len(annotations)
        for annotation in annotations:
            class_counts[annotation.class_id] += 1
            available_count = 0
            for name, landmark in zip(LANDMARK_NAMES, annotation.landmarks):
                visibility_counts[landmark.visibility] += 1
                if landmark.available:
                    landmark_counts[name] += 1
                    available_count += 1
            if available_count:
                heatmap_ready_records += 1
            if annotation.class_id in TOOTH_CLASSES:
                points = annotation_points_original(annotation, width, height)
                for surface in ("mesial", "distal"):
                    rbl_results.append({"record_id": annotation.record_id, "class_id": annotation.class_id,
                                        **calculate_rbl(points, surface,
                                                        indices=rbl_indices(annotation.class_id, surface))})

    # Ensure repeated fold copies are byte-identical to their canonical label and image.
    inconsistent_copies = []
    for fold in range(5):
        for split in ("train", "val"):
            image_dir, label_dir = split_paths(dataset, f"f{fold}", split)
            for image in image_dir.glob("*.png"):
                canonical_image, canonical_label = sources[image.stem]
                label = label_dir / f"{image.stem}.txt"
                if sha256(image) != sha256(canonical_image) or sha256(label) != sha256(canonical_label):
                    inconsistent_copies.append({"fold": fold, "split": split, "image_id": image.stem})

    candidates = [annotation for image_id in manifest["development_image_ids"]
                  for annotation in annotations_by_image[image_id]
                  if annotation.class_id in TOOTH_CLASSES and sum(point.available for point in annotation.landmarks) >= 4]
    by_class = {}
    for annotation in candidates:
        by_class.setdefault(annotation.class_id, []).append(annotation)
    randomizer = random.Random(42)
    chosen = []
    for class_id in sorted(by_class):
        chosen.extend(randomizer.sample(by_class[class_id], min(2, len(by_class[class_id]))))
    examples = []
    for index, annotation in enumerate(chosen, 1):
        image_path, _ = sources[annotation.image_id]
        examples.append(_render_example(load_image(image_path), annotation, examples_dir, index))

    assessable = [item for item in rbl_results if item["status"] == "assessable"]
    not_assessable = [item for item in rbl_results if item["status"] != "assessable"]
    rbl_values = [item["rbl_percent"] for item in assessable]
    rbl_summary = {
        "candidate_surface_count": len(rbl_results),
        "assessable_surface_count": len(assessable),
        "not_assessable_surface_count": len(not_assessable),
        "not_assessable_reasons": dict(Counter(item["reason"] for item in not_assessable)),
        "rbl_percent": ({"minimum": min(rbl_values), "maximum": max(rbl_values),
                         "mean": float(np.mean(rbl_values)), "median": float(np.median(rbl_values))}
                        if rbl_values else None),
        "formula": "100 * euclidean_distance(CEJ, BL) / euclidean_distance(CEJ, root_landmark)",
        "interpretation": "ground-truth radiographic geometry only; not CAL, probing depth, diagnosis, or treatment",
    }
    report = {
        "gate": 1,
        "training_performed": False,
        "dataset_root": relative(dataset),
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "processor": platform.processor(), "device": "CPU-only preparation"},
        "inventory": {"unique_radiographs": len(sources), "development_radiographs": len(sources) - len(holdout_ids),
                      "holdout_radiographs": len(holdout_ids), "valid_annotation_rows": valid_rows,
                      "quarantined_rows": len(quarantined), "heatmap_ready_records": heatmap_ready_records,
                      "class_counts": {OBJECT_CLASS_NAMES[key]: value for key, value in sorted(class_counts.items())},
                      "landmark_counts": {name: landmark_counts[name] for name in LANDMARK_NAMES},
                      "visibility_counts": {str(key): value for key, value in sorted(visibility_counts.items())},
                      "image_dimensions": dict(image_dimensions)},
        "validation": {"expected_fold_count": 5, "folds_valid": len(manifest["folds"]) == 5,
                       "radiograph_level_split": True, "patient_ids_available": False,
                       "repeated_fold_copy_mismatches": inconsistent_copies},
        "quarantined": quarantined,
        "rbl": rbl_summary,
        "examples": examples,
        "source_modified": False,
        "limitations": ["Patient identifiers are not supplied, so patient-disjoint splitting cannot be verified.",
                        "ARR regions cannot be proposed by the tooth-only Mask R-CNN stage.",
                        "PLS rows with no available keypoint provide no landmark heatmap supervision."],
    }
    (output / "five_fold_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "quarantine.json").write_text(json.dumps({"records": quarantined}, indent=2), encoding="utf-8")
    (output / "ground_truth_rbl.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rbl_results), encoding="utf-8")
    (output / "examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    (output / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.dataset, args.output)
    print(json.dumps({"inventory": report["inventory"], "validation": report["validation"],
                      "quarantined": report["quarantined"], "rbl": report["rbl"],
                      "examples": report["examples"]}, indent=2))


if __name__ == "__main__":
    main()
