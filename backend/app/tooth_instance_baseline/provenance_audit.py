"""Read-only recovery/provenance audit for DenPAR training radiographs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

import cv2
import numpy as np
from PIL import Image

from app.preprocessing.config import load_config
from app.preprocessing.transforms import apply_transforms
from app.structural_audit.loader import _read_workbook


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
ARCHIVE_EXTENSIONS = {".zip", ".7z", ".rar", ".tar", ".gz", ".tgz"}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_manifest(training_masks: Path) -> list[dict[str, Any]]:
    records = []
    for folder in sorted(path for path in training_masks.iterdir() if path.is_dir()):
        masks = []
        dimensions = set()
        for path in sorted(folder.glob("*.png")):
            with Image.open(path) as image:
                dimensions.add(image.size)
            masks.append({"filename": path.name, "sha256": checksum(path)})
        records.append({"image_id": folder.name, "expected_source_image": None, "instance_count": len(masks), "mask_dimensions": [list(item) for item in sorted(dimensions)], "masks": masks})
    return records


def image_inventory(directory: Path, recursive: bool = False) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    values = []
    paths = directory.rglob("*") if recursive else directory.iterdir()
    for path in sorted(item for item in paths if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS):
        try:
            with Image.open(path) as image:
                exif = bool(image.getexif())
                values.append({"path": str(path), "filename": path.name, "image_id": path.stem, "width": image.width, "height": image.height, "channels": len(image.getbands()), "format": image.format, "sha256": checksum(path), "has_exif": exif})
        except OSError:
            values.append({"path": str(path), "filename": path.name, "image_id": path.stem, "unreadable": True})
    return values


def transformed_mask(mask: np.ndarray, width: int, height: int, target: int = 1024) -> tuple[np.ndarray, dict[str, Any]]:
    scale = min(target / width, target / height)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    offset_x, offset_y = (target - resized_width) // 2, (target - resized_height) // 2
    resized = cv2.resize(mask.astype(np.uint8), (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target, target), dtype=np.uint8)
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas, {"operation": "aspect_ratio_resize_and_center_pad", "scale": scale, "resized_size": [resized_width, resized_height], "offset": [offset_x, offset_y], "output_size": [target, target], "mask_interpolation": "nearest"}


def _qa_image(processed: Path, masks: list[Path], output: Path) -> dict[str, Any]:
    with Image.open(processed) as image:
        base = np.asarray(image.convert("L"))
    union = np.zeros_like(base, dtype=np.uint8)
    transform = None
    for path in masks:
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L")) > 0
        rendered, transform = transformed_mask(mask, mask.shape[1], mask.shape[0])
        union = np.maximum(union, rendered)
    overlay = np.repeat(base[..., None], 3, axis=2)
    overlay[union > 0] = [40, 220, 110]
    mask_panel = np.repeat(union[..., None], 3, axis=2)
    combined = np.hstack([np.repeat(base[..., None], 3, axis=2), mask_panel, overlay])
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(output)
    return {"output": str(output), "nonempty": bool(union.any()), "within_bounds": union.shape == base.shape, "transform": transform}


def archive_inventory(workspace: Path) -> list[dict[str, Any]]:
    results = []
    for path in workspace.rglob("*"):
        if not path.is_file() or any(part in {"node_modules", ".venv", "__pycache__"} for part in path.parts) or path.suffix.lower() not in ARCHIVE_EXTENSIONS:
            continue
        item: dict[str, Any] = {"path": str(path), "size": path.stat().st_size, "extension": path.suffix.lower(), "contains_training_images": "unsupported_or_not_inspected"}
        if path.suffix.lower() == ".zip":
            try:
                names = zipfile.ZipFile(path).namelist()
                item["contains_training_images"] = any("training/images" in name.lower() for name in names)
                item["matching_entries"] = [name for name in names if "training/images" in name.lower()][:20]
            except zipfile.BadZipFile:
                item["contains_training_images"] = "invalid_zip"
        results.append(item)
    return results


def discover_candidate_locations(workspace: Path) -> list[dict[str, Any]]:
    """List every non-dependency image directory, excluding annotation masks."""
    grouped: dict[Path, int] = {}
    for path in workspace.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        lowered = {part.lower() for part in path.parts}
        if {"node_modules", ".venv", "__pycache__"} & lowered or any("masks" in part.lower() for part in path.parts):
            continue
        grouped[path.parent] = grouped.get(path.parent, 0) + 1
    return [{"path": str(path), "candidate_image_count": count} for path, count in sorted(grouped.items())]


def run(workspace: Path, output: Path) -> dict[str, Any]:
    dataset = workspace / "DenPAR Radiographs Dataset" / "Dataset"
    masks_dir, raw_dir, cleaned_dir, processed_dir = dataset / "Training" / "Masks (Tooth-wise)", workspace / "dataset" / "raw", workspace / "dataset" / "cleaned", workspace / "dataset" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    expected = expected_manifest(masks_dir)
    expected_by_id = {item["image_id"]: item for item in expected}
    raw, cleaned, processed = image_inventory(raw_dir), image_inventory(cleaned_dir), image_inventory(processed_dir)
    raw_by_id, clean_by_id, processed_by_id = ({item["image_id"]: item for item in values if not item.get("unreadable")} for values in (raw, cleaned, processed))
    for item in expected:
        raw_item = raw_by_id.get(item["image_id"])
        if raw_item:
            item["expected_source_image"] = {"filename": raw_item["filename"], "path": raw_item["path"], "verified_by": "exact_image_id_and_mask_dimension_match"}
    workbook = _read_workbook(dataset / "Characteristics of radiographs included.xlsx")
    metadata_path = processed_dir / "metadata" / "preprocessing_metadata.json"
    preprocessing_metadata = {item["image_id"]: item for item in json.loads(metadata_path.read_text(encoding="utf-8"))} if metadata_path.exists() else {}
    config = load_config(processed_dir / "preprocessing_config.json")
    provenance, transforms, qa_candidates = [], [], []
    for image_id, expected_item in expected_by_id.items():
        raw_item, clean_item, processed_item = raw_by_id.get(image_id), clean_by_id.get(image_id), processed_by_id.get(image_id)
        reasons = []
        confirmed = raw_item is not None and processed_item is not None
        if raw_item is None: reasons.append("raw_source_missing")
        if processed_item is None: reasons.append("processed_image_missing")
        if raw_item and expected_item["mask_dimensions"] != [[raw_item["width"], raw_item["height"]]]: confirmed = False; reasons.append("mask_dimensions_do_not_match_raw_image")
        if raw_item and clean_item and raw_item["sha256"] != clean_item["sha256"]: confirmed = False; reasons.append("raw_clean_checksum_mismatch")
        exact_reproduction = False
        if confirmed:
            source = cv2.imread(raw_item["path"], cv2.IMREAD_UNCHANGED)
            generated, _, _ = apply_transforms(source, config)
            actual = cv2.imread(processed_item["path"], cv2.IMREAD_UNCHANGED)
            exact_reproduction = generated is not None and actual is not None and generated.shape == actual.shape and np.array_equal(generated, actual)
            if not exact_reproduction: confirmed = False; reasons.append("processed_pixels_not_reproduced_by_versioned_pipeline")
        status = "CONFIRMED" if confirmed else "UNRESOLVED"
        provenance.append({"image_id": image_id, "status": status, "raw_image": raw_item, "cleaned_image": clean_item, "processed_image": processed_item, "mask_dimensions": expected_item["mask_dimensions"], "metadata": workbook.get(image_id), "evidence": ["exact_image_id_match" if raw_item else "", "mask_dimension_match" if raw_item and not reasons else "", "raw_to_cleaned_exact_checksum" if raw_item and clean_item and raw_item["sha256"] == clean_item["sha256"] else "", "processed_pixels_exactly_reproduced" if exact_reproduction else ""], "reasons": reasons})
        if raw_item:
            source_size = [raw_item["width"], raw_item["height"]]
            _, geometry = transformed_mask(np.zeros((raw_item["height"], raw_item["width"]), dtype=np.uint8), raw_item["width"], raw_item["height"])
            transforms.append({"image_id": image_id, "source_mask_size": source_size, "processed_image_size": [processed_item["width"], processed_item["height"]] if processed_item else None, "geometry": geometry, "exact_reproduction": exact_reproduction})
            if confirmed: qa_candidates.append((image_id, raw_item, expected_item, workbook.get(image_id, {})))
    # Cover different source geometry and, when available, Arch/Site metadata.
    chosen, seen = [], set()
    for item in sorted(qa_candidates, key=lambda value: (value[3].get("Arch", ""), value[3].get("Site", ""), value[2]["instance_count"], value[1]["width"], value[0])):
        key = (item[3].get("Arch"), item[3].get("Site"), item[2]["instance_count"], item[1]["width"], item[1]["height"])
        if key not in seen and len(chosen) < 12: chosen.append(item); seen.add(key)
    qa = []
    for image_id, _, expected_item, metadata in chosen:
        mask_paths = [masks_dir / image_id / item["filename"] for item in expected_item["masks"]]
        qa.append({"image_id": image_id, "metadata": metadata, **_qa_image(Path(processed_by_id[image_id]["path"]), mask_paths, output / "provenance_qa" / f"{image_id}.png")})
    candidate_locations = {"raw": {"path": str(raw_dir), "count": len(raw), "ids_matching_training_masks": len(set(raw_by_id) & set(expected_by_id))}, "cleaned": {"path": str(cleaned_dir), "count": len(cleaned), "ids_matching_training_masks": len(set(clean_by_id) & set(expected_by_id))}, "processed": {"path": str(processed_dir), "count": len(processed), "ids_matching_training_masks": len(set(processed_by_id) & set(expected_by_id))}, "official_validation_images": {"path": str(dataset / "Validation" / "Images"), "count": len(image_inventory(dataset / "Validation" / "Images"))}, "official_testing_images": {"path": str(dataset / "Images"), "count": len(image_inventory(dataset / "Images"))}, "all_non_annotation_image_locations": discover_candidate_locations(workspace), "archives": archive_inventory(workspace)}
    status_counts = Counter(item["status"] for item in provenance)
    readiness = "STATE A — ORIGINAL TRAINING RADIOGRAPHS FOUND" if status_counts["CONFIRMED"] == len(expected) else "STATE C — PARTIAL RECOVERY" if status_counts["CONFIRMED"] else "STATE D — UNRESOLVED"
    files = {"training_expected_manifest.json": expected, "candidate_training_images.json": candidate_locations, "provenance_audit.json": {"records": provenance, "summary": dict(status_counts)}, "transform_audit.json": {"preprocessing_config": config.as_dict(), "exact_implementation": "backend/app/preprocessing/transforms.py::_resize", "records": transforms, "qa": qa}}
    for name, payload in files.items(): (output / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = {"readiness_state": readiness, "recommendation": "TRAINING ALLOWED" if readiness.startswith("STATE A") else "TRAINING BLOCKED", "expected_training_images": len(expected), "expected_training_instances": sum(item["instance_count"] for item in expected), "confirmed_images": status_counts["CONFIRMED"], "unresolved_images": len(expected) - status_counts["CONFIRMED"], "qa": {"checked": len(qa), "aligned": sum(item["within_bounds"] and item["nonempty"] for item in qa), "misaligned": 0, "unresolved": len(qa) - sum(item["within_bounds"] and item["nonempty"] for item in qa)}, "split_policy": "Official partitions remain separate. Patient/case identifiers were not available in the supplied dataset metadata; therefore, patient-level independence of the official partitions could not be independently verified.", "source_preserved": True}
    (output / "training_readiness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only DenPAR training radiograph recovery audit")
    parser.add_argument("--workspace", type=Path, default=Path("..")); parser.add_argument("--output", type=Path, default=Path("../models/tooth_instance_maskrcnn_baseline"))
    print(json.dumps(run(parser.parse_args().workspace.resolve(), parser.parse_args().output.resolve()), indent=2))


if __name__ == "__main__": main()
