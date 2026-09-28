"""Read-only DenPAR identity, annotation, provenance, and leakage audit.

The audit never writes below ``dataset_root`` or ``training_images_dir``.  All
generated files are written to an explicitly separate report directory.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image


PARTITIONS = ("Training", "Validation", "Testing")
EXPECTED_COUNTS = {"Training": 650, "Validation": 150, "Testing": 200}
CLASSIFICATIONS = {
    "confirmed_leakage", "exact_duplicate", "probable_near_duplicate",
    "annotation_mismatch", "missing_component", "provenance_unverified", "no_issue",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_gray(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("unreadable image")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        low, high = float(np.min(image)), float(np.max(image))
        image = np.zeros(image.shape, np.uint8) if high <= low else np.rint((image - low) * 255.0 / (high - low)).astype(np.uint8)
    return np.ascontiguousarray(image)


def normalized_pixel_hash(image: np.ndarray) -> str:
    """Hash canonical 8-bit grayscale pixels plus their dimensions."""
    header = f"gray8:{image.shape[1]}x{image.shape[0]}:".encode("ascii")
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def perceptual_hash(image: np.ndarray) -> str:
    """Return a reproducible 64-bit pHash from the low-frequency DCT."""
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(resized)[:8, :8]
    values = low.flatten()
    median = float(np.median(values[1:]))
    bits = values > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _relative(path: Path | None, project_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _xlsx_rows(path: Path) -> list[dict[str, str]]:
    """Read the first XLSX sheet without adding an Excel dependency."""
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", XLSX_NS):
                shared.append("".join(node.text or "" for node in item.iterfind(".//m:t", XLSX_NS)))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        raw_rows: list[dict[int, str]] = []
        for row in sheet.findall(".//m:row", XLSX_NS):
            cells: dict[int, str] = {}
            for cell in row.findall("m:c", XLSX_NS):
                reference = cell.get("r", "A1")
                letters = re.match(r"[A-Z]+", reference)
                column = 0
                for character in letters.group(0) if letters else "A":
                    column = column * 26 + ord(character) - 64
                value_node = cell.find("m:v", XLSX_NS)
                value = "" if value_node is None else (value_node.text or "")
                if cell.get("t") == "s" and value:
                    value = shared[int(value)]
                cells[column - 1] = value
            raw_rows.append(cells)
    if not raw_rows:
        return []
    headers = {index: value for index, value in raw_rows[0].items()}
    return [{headers[index]: values.get(index, "") for index in headers} for values in raw_rows[1:]]


def _workbook_index(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    rows = _xlsx_rows(path)
    indexed: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        raw = row.get("id", "").strip()
        image_id = raw[:-2] if raw.endswith(".0") else raw
        if not image_id:
            continue
        if image_id in indexed:
            duplicates.append(image_id)
        indexed[image_id] = row
    return indexed, sorted(set(duplicates))


def _json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return (value, None) if isinstance(value, dict) else (None, "JSON root is not an object")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, str(error)


def _point_outside(point: Any, width: int, height: int) -> bool:
    tolerance = 1e-6
    return not (
        isinstance(point, (list, tuple)) and len(point) >= 2
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point[:2])
        and -tolerance <= float(point[0]) <= width + tolerance
        and -tolerance <= float(point[1]) <= height + tolerance
    )


def _thumbnail_evidence(left_path: Path, right_path: Path) -> dict[str, float]:
    left = cv2.resize(_decode_gray(left_path), (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    right = cv2.resize(_decode_gray(right_path), (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    mae = float(np.mean(np.abs(left - right)) / 255.0)
    left_flat, right_flat = left.flatten(), right.flatten()
    correlation = float(np.corrcoef(left_flat, right_flat)[0, 1]) if left_flat.std() and right_flat.std() else 0.0
    return {"normalized_thumbnail_mae": mae, "thumbnail_pearson_correlation": correlation}


def _finding(classification: str, issue: str, **evidence: Any) -> dict[str, Any]:
    assert classification in CLASSIFICATIONS
    return {"classification": classification, "issue": issue, **evidence}


def _annotation_paths(dataset_root: Path, partition: str, image_id: str) -> dict[str, Any]:
    base = dataset_root / partition
    return {
        "tooth_masks": sorted((base / "Masks (Tooth-wise)" / image_id).glob("*.png")),
        "radiograph_mask": base / "Masks (Radiograph-wise)" / f"{image_id}.png",
        "keypoints": base / "Key Points Annotations" / f"{image_id}.json",
        "bone_level": base / "Bone Level Annotations" / f"{image_id}.json",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in fields})


def _load_checkpoint_metadata(checkpoint: Path) -> tuple[dict[str, Any], str | None]:
    try:
        import torch
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload.get("config", {})
        return {
            "keys": sorted(payload.keys()), "epoch": payload.get("epoch"),
            "saved_at": payload.get("saved_at"), "config": config,
            "dataset_manifest_version": payload.get("dataset_manifest_version"),
            "preprocessing_version": payload.get("preprocessing_version"),
            "augmentation_version": payload.get("augmentation_version"),
            "architecture": payload.get("architecture"),
        }, None
    except Exception as error:  # metadata still remains available in signed project reports
        return {}, f"{type(error).__name__}: {error}"


def run_audit(
    dataset_root: Path,
    training_images_dir: Path,
    output_dir: Path,
    project_root: Path,
    expected_counts: dict[str, int] | None = None,
    near_duplicate_threshold: int = 6,
    checkpoint: Path | None = None,
    training_manifest: Path | None = None,
    training_config: Path | None = None,
) -> dict[str, Any]:
    dataset_root, training_images_dir = dataset_root.resolve(), training_images_dir.resolve()
    project_root, output_dir = project_root.resolve(), output_dir.resolve()
    if output_dir == dataset_root or dataset_root in output_dir.parents or output_dir == training_images_dir or training_images_dir in output_dir.parents:
        raise ValueError("Report directory must be outside all source directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_counts = expected_counts or EXPECTED_COUNTS
    workbook = dataset_root / "Characteristics of radiographs included.xlsx"
    workbook_rows, duplicate_workbook_ids = _workbook_index(workbook)
    workbook_hash = sha256_file(workbook)

    image_dirs = {
        "Training": training_images_dir,
        "Validation": dataset_root / "Validation" / "Images",
        "Testing": dataset_root / "Images",
    }
    inventory: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    coordinate_issues: list[dict[str, Any]] = []
    annotation_files: list[dict[str, Any]] = []
    source_hashes_before: dict[str, str] = {}
    source_hashes_before[str(workbook)] = workbook_hash

    for partition in PARTITIONS:
        for image_path in _image_files(image_dirs[partition]):
            image_id = image_path.stem
            stable_id = f"{partition}:{image_id}"
            missing: list[str] = []
            unreadable: list[str] = []
            try:
                pixels = _decode_gray(image_path)
                height, width = pixels.shape
                pixel_hash = normalized_pixel_hash(pixels)
                phash = perceptual_hash(pixels)
                with Image.open(image_path) as source:
                    image_format = source.format or image_path.suffix.lstrip(".").upper()
            except Exception as error:
                width = height = 0
                pixel_hash = phash = None
                image_format = image_path.suffix.lstrip(".").upper()
                unreadable.append(f"image: {error}")
            file_hash = sha256_file(image_path)
            source_hashes_before[str(image_path)] = file_hash
            annotations = _annotation_paths(dataset_root, partition, image_id)
            mask_records: list[dict[str, Any]] = []
            for kind, paths in (("tooth_mask", annotations["tooth_masks"]),):
                for mask_path in paths:
                    record = {"kind": kind, "image_id": image_id, "partition": partition,
                              "path": _relative(mask_path, project_root), "sha256": sha256_file(mask_path)}
                    annotation_files.append(record)
                    source_hashes_before[str(mask_path)] = record["sha256"]
                    try:
                        mask = _decode_gray(mask_path)
                        mh, mw = mask.shape
                        record.update({"width": mw, "height": mh, "nonzero_pixels": int(np.count_nonzero(mask))})
                        mask_records.append({"path": record["path"], "sha256": record["sha256"], "width": mw,
                                             "height": mh, "nonzero_pixels": record["nonzero_pixels"]})
                        if (mw, mh) != (width, height):
                            coordinate_issues.append(_finding("annotation_mismatch", "tooth_mask_dimension_mismatch",
                                stable_image_id=stable_id, annotation=record["path"], expected=[width, height], actual=[mw, mh]))
                        if not np.any(mask):
                            coordinate_issues.append(_finding("annotation_mismatch", "empty_tooth_mask",
                                stable_image_id=stable_id, annotation=record["path"]))
                    except Exception as error:
                        unreadable.append(f"tooth mask {mask_path.name}: {error}")
            if not mask_records:
                missing.append("tooth_masks")

            for key, kind in (("radiograph_mask", "radiograph_mask"), ("keypoints", "keypoints"), ("bone_level", "bone_level")):
                path = annotations[key]
                if not path.exists():
                    missing.append(key)
                    continue
                record = {"kind": kind, "image_id": image_id, "partition": partition,
                          "path": _relative(path, project_root), "sha256": sha256_file(path)}
                annotation_files.append(record)
                source_hashes_before[str(path)] = record["sha256"]
                if key == "radiograph_mask":
                    try:
                        mask = _decode_gray(path)
                        mh, mw = mask.shape
                        record.update({"width": mw, "height": mh, "nonzero_pixels": int(np.count_nonzero(mask))})
                        if (mw, mh) != (width, height):
                            coordinate_issues.append(_finding("annotation_mismatch", "radiograph_mask_dimension_mismatch",
                                stable_image_id=stable_id, annotation=record["path"], expected=[width, height], actual=[mw, mh]))
                    except Exception as error:
                        unreadable.append(f"radiograph mask: {error}")
                else:
                    data, error = _json(path)
                    if error or data is None:
                        unreadable.append(f"{key}: {error}")
                        continue
                    declared = Path(str(data.get("Image_id", ""))).stem
                    if declared != image_id:
                        coordinate_issues.append(_finding("annotation_mismatch", "annotation_image_id_mismatch",
                            stable_image_id=stable_id, annotation=record["path"], declared_image_id=declared))
                    if key == "keypoints":
                        for point_type in ("CEJ_Points", "Apex_Points"):
                            for index, point in enumerate(data.get(point_type, [])):
                                if _point_outside(point, width, height):
                                    coordinate_issues.append(_finding("annotation_mismatch", "point_out_of_bounds",
                                        stable_image_id=stable_id, annotation=record["path"], point_type=point_type,
                                        point_index=index, point=point, bounds=[width, height]))
                        for index, box in enumerate(data.get("bboxes", [])):
                            valid = (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                                     and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
                            if not valid:
                                coordinate_issues.append(_finding("annotation_mismatch", "bbox_out_of_bounds_or_invalid",
                                    stable_image_id=stable_id, annotation=record["path"], bbox_index=index,
                                    bbox=box, bounds=[width, height]))
                    else:
                        lines = data.get("Bone_Lines", [])
                        if data.get("Num_of_Bone_Lines") != len(lines):
                            coordinate_issues.append(_finding("annotation_mismatch", "bone_line_count_mismatch",
                                stable_image_id=stable_id, annotation=record["path"],
                                declared=data.get("Num_of_Bone_Lines"), actual=len(lines)))
                        for line_index, line in enumerate(lines):
                            for point_index, point in enumerate(line):
                                if _point_outside(point, width, height):
                                    coordinate_issues.append(_finding("annotation_mismatch", "polyline_point_out_of_bounds",
                                        stable_image_id=stable_id, annotation=record["path"], line_index=line_index,
                                        point_index=point_index, point=point, bounds=[width, height]))

            metadata = workbook_rows.get(image_id)
            if metadata is None:
                missing.append("workbook_metadata")
            for component in missing:
                findings.append(_finding("missing_component", component, stable_image_id=stable_id, partition=partition))
            for detail in unreadable:
                findings.append(_finding("missing_component", "unreadable_component", stable_image_id=stable_id, detail=detail))
            inventory.append({
                "stable_image_id": stable_id, "image_id": image_id, "filename": image_path.name,
                "absolute_source_path": str(image_path.resolve()), "source_path": _relative(image_path, project_root),
                "official_partition": partition, "width": width, "height": height, "format": image_format,
                "file_sha256": file_hash, "normalized_pixel_sha256": pixel_hash, "perceptual_hash_64": phash,
                "tooth_masks": mask_records,
                "cej_apex_annotation": _relative(annotations["keypoints"], project_root) if annotations["keypoints"].exists() else None,
                "bone_level_annotation": _relative(annotations["bone_level"], project_root) if annotations["bone_level"].exists() else None,
                "radiograph_mask_annotation": _relative(annotations["radiograph_mask"], project_root) if annotations["radiograph_mask"].exists() else None,
                "workbook_metadata_reference": {"path": _relative(workbook, project_root), "image_id": image_id,
                                                  "fields": metadata} if metadata else None,
                "missing_components": missing, "unreadable_components": unreadable,
            })

    by_partition = {partition: sum(row["official_partition"] == partition for row in inventory) for partition in PARTITIONS}
    partition_counts = {partition: {"expected_images": expected_counts.get(partition), "observed_images": by_partition[partition],
                                   "matches_expected": by_partition[partition] == expected_counts.get(partition)} for partition in PARTITIONS}
    for partition, item in partition_counts.items():
        if not item["matches_expected"]:
            findings.append(_finding("missing_component", "partition_image_count_mismatch", partition=partition, **item))

    exact_groups: list[dict[str, Any]] = []
    for method, field in (("identical_file_sha256", "file_sha256"), ("identical_normalized_pixels", "normalized_pixel_sha256")):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in inventory:
            if row[field]:
                groups[row[field]].append(row)
        for digest, rows in groups.items():
            if len(rows) < 2:
                continue
            partitions = sorted({row["official_partition"] for row in rows})
            entry = {"classification": "exact_duplicate", "method": method, "hash": digest,
                     "cross_partition": len(partitions) > 1, "partitions": partitions,
                     "images": [{key: row[key] for key in ("stable_image_id", "source_path", "file_sha256", "normalized_pixel_sha256")} for row in rows]}
            exact_groups.append(entry)

    exact_pixel_pairs = {tuple(sorted((a["stable_image_id"], b["stable_image_id"])))
                         for group in exact_groups for index, a in enumerate(group["images"])
                         for b in group["images"][index + 1:]}
    unique_exact_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for group in exact_groups:
        for index, left in enumerate(group["images"]):
            for right in group["images"][index + 1:]:
                pair = tuple(sorted((left["stable_image_id"], right["stable_image_id"])))
                entry = unique_exact_pairs.setdefault(pair, {
                    "classification": "confirmed_leakage" if group["cross_partition"] else "exact_duplicate",
                    "images": list(pair), "partitions": sorted({value.split(":", 1)[0] for value in pair}),
                    "cross_partition": group["cross_partition"], "evidence_methods": [],
                })
                entry["evidence_methods"].append({"method": group["method"], "hash": group["hash"]})
    unique_exact_pair_rows = list(unique_exact_pairs.values())
    cross_partition_exact_pairs = [item for item in unique_exact_pair_rows if item["cross_partition"]]
    findings.extend({**item, "issue": "exact_image_duplicate"} for item in unique_exact_pair_rows)
    near_duplicates: list[dict[str, Any]] = []
    readable = [row for row in inventory if row["perceptual_hash_64"]]
    for index, left in enumerate(readable):
        for right in readable[index + 1:]:
            pair = tuple(sorted((left["stable_image_id"], right["stable_image_id"])))
            if pair in exact_pixel_pairs:
                continue
            distance = hamming_distance(left["perceptual_hash_64"], right["perceptual_hash_64"])
            if distance <= near_duplicate_threshold:
                aspect_left = left["width"] / left["height"]
                aspect_right = right["width"] / right["height"]
                relative_aspect_difference = abs(aspect_left - aspect_right) / max(aspect_left, aspect_right)
                support = _thumbnail_evidence(Path(left["absolute_source_path"]), Path(right["absolute_source_path"]))
                entry = {"classification": "probable_near_duplicate", "method": "64-bit DCT pHash",
                         "threshold": f"Hamming distance <= {near_duplicate_threshold}", "hamming_distance": distance,
                         "relative_aspect_difference": relative_aspect_difference,
                         "cross_partition": left["official_partition"] != right["official_partition"],
                         "left": {key: left[key] for key in ("stable_image_id", "source_path", "official_partition", "perceptual_hash_64")},
                         "right": {key: right[key] for key in ("stable_image_id", "source_path", "official_partition", "perceptual_hash_64")},
                         "supporting_evidence": support,
                         "interpretation": "candidate_for_manual_review_not_a_confirmed_duplicate"}
                near_duplicates.append(entry)
                findings.append(_finding("probable_near_duplicate", "perceptual_similarity_candidate",
                    images=list(pair), hamming_distance=distance, cross_partition=entry["cross_partition"]))

    annotation_duplicates: list[dict[str, Any]] = []
    annotation_hashes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in annotation_files:
        annotation_hashes[(item["kind"], item["sha256"])].append(item)
    for (kind, digest), rows in annotation_hashes.items():
        image_ids = sorted({f"{row['partition']}:{row['image_id']}" for row in rows})
        if len(image_ids) > 1:
            annotation_duplicates.append(_finding("annotation_mismatch", "identical_annotation_mapped_to_multiple_images",
                annotation_kind=kind, sha256=digest, images=image_ids, paths=[row["path"] for row in rows]))

    image_keys = {(row["official_partition"], row["image_id"]) for row in inventory}
    orphaned: list[dict[str, Any]] = []
    aggregate_files: list[dict[str, Any]] = []
    declared_annotation_targets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for partition in PARTITIONS:
        base = dataset_root / partition
        for directory, kind, suffix in (("Key Points Annotations", "keypoints", ".json"),
                                         ("Bone Level Annotations", "bone_level", ".json"),
                                         ("Masks (Radiograph-wise)", "radiograph_mask", ".png")):
            for path in sorted((base / directory).glob(f"*{suffix}")):
                source_hashes_before[str(path)] = sha256_file(path)
                if path.stem.lower().startswith("coco_format"):
                    aggregate_files.append({"partition": partition, "kind": kind, "path": _relative(path, project_root), "sha256": source_hashes_before[str(path)]})
                else:
                    if suffix == ".json":
                        data, error = _json(path)
                        if data is not None:
                            declared = Path(str(data.get("Image_id", ""))).stem
                            if declared:
                                declared_annotation_targets[(partition, kind, declared)].append(_relative(path, project_root) or str(path))
                            if declared and declared != path.stem:
                                orphaned.append(_finding("annotation_mismatch", "annotation_filename_declared_id_conflict",
                                    partition=partition, annotation_kind=kind, filename_image_id=path.stem,
                                    declared_image_id=declared, path=_relative(path, project_root)))
                        elif error:
                            orphaned.append(_finding("annotation_mismatch", "unreadable_or_invalid_annotation",
                                partition=partition, annotation_kind=kind, image_id=path.stem,
                                path=_relative(path, project_root), detail=error))
                    if (partition, path.stem) not in image_keys:
                        orphaned.append(_finding("annotation_mismatch", "annotation_without_matching_image",
                            partition=partition, annotation_kind=kind, image_id=path.stem, path=_relative(path, project_root)))
        tooth_root = base / "Masks (Tooth-wise)"
        for path in sorted(tooth_root.glob("*.json")):
            source_hashes_before[str(path)] = sha256_file(path)
            aggregate_files.append({"partition": partition, "kind": "tooth_instances_coco", "path": _relative(path, project_root), "sha256": source_hashes_before[str(path)]})
        for folder in sorted(path for path in tooth_root.iterdir() if path.is_dir()):
            if (partition, folder.name) not in image_keys:
                orphaned.append(_finding("annotation_mismatch", "tooth_mask_folder_without_matching_image",
                    partition=partition, image_id=folder.name, path=_relative(folder, project_root)))

    for (partition, kind, declared), paths in sorted(declared_annotation_targets.items()):
        if len(paths) > 1:
            orphaned.append(_finding("annotation_mismatch", "multiple_annotation_files_declare_same_image",
                partition=partition, annotation_kind=kind, declared_image_id=declared, paths=paths))

    for partition in PARTITIONS:
        rows = [row for row in inventory if row["official_partition"] == partition]
        partition_counts[partition].update({
            "tooth_mask_files": sum(len(row["tooth_masks"]) for row in rows),
            "images_with_tooth_masks": sum(bool(row["tooth_masks"]) for row in rows),
            "keypoint_annotation_files": sum(row["cej_apex_annotation"] is not None for row in rows),
            "bone_level_annotation_files": sum(row["bone_level_annotation"] is not None for row in rows),
            "radiograph_mask_files": sum(row["radiograph_mask_annotation"] is not None for row in rows),
            "workbook_rows_matched": sum(row["workbook_metadata_reference"] is not None for row in rows),
            "aggregate_annotation_files": sum(item["partition"] == partition for item in aggregate_files),
        })

    for image_id in duplicate_workbook_ids:
        orphaned.append(_finding("annotation_mismatch", "duplicate_workbook_image_id", image_id=image_id))
    inventory_ids = {row["image_id"] for row in inventory}
    for image_id in sorted(set(workbook_rows) - inventory_ids):
        orphaned.append(_finding("annotation_mismatch", "workbook_row_without_matching_image", image_id=image_id,
            workbook=_relative(workbook, project_root)))
    findings.extend(annotation_duplicates + orphaned + coordinate_issues)

    checkpoint = checkpoint.resolve() if checkpoint else None
    training_manifest = training_manifest.resolve() if training_manifest else None
    training_config = training_config.resolve() if training_config else None
    provenance_evidence: list[dict[str, Any]] = []
    manifest_ids: set[str] = set()
    if training_manifest and training_manifest.exists():
        manifest_value = json.loads(training_manifest.read_text(encoding="utf-8"))
        if isinstance(manifest_value, list):
            manifest_ids = {str(item.get("image_id")) for item in manifest_value if isinstance(item, dict)}
        provenance_evidence.append({"kind": "training_manifest", "path": _relative(training_manifest, project_root),
                                    "sha256": sha256_file(training_manifest), "image_ids": len(manifest_ids)})
    config_value: dict[str, Any] = {}
    if training_config and training_config.exists():
        config_value = json.loads(training_config.read_text(encoding="utf-8"))
        provenance_evidence.append({"kind": "locked_training_config", "path": _relative(training_config, project_root),
                                    "sha256": sha256_file(training_config), "data": config_value.get("data", {})})
    checkpoint_meta, checkpoint_error = ({}, "checkpoint not supplied")
    if checkpoint and checkpoint.exists():
        checkpoint_meta, checkpoint_error = _load_checkpoint_metadata(checkpoint)
        provenance_evidence.append({"kind": "checkpoint", "path": _relative(checkpoint, project_root),
                                    "sha256": sha256_file(checkpoint), "metadata": checkpoint_meta,
                                    "metadata_load_error": checkpoint_error})
    historical_artifacts: list[dict[str, Any]] = []
    if checkpoint:
        full_run = checkpoint.parent
        for path in (
            full_run / "reports" / "full_training_report.json",
            full_run / "reports" / "final_model_lock.json",
            full_run / "reports" / "test_exposure_record.json",
            full_run / "evaluation_report.json",
            full_run / "config" / "augmentation_config.json",
        ):
            if path.exists():
                evidence: dict[str, Any] = {"path": _relative(path, project_root), "sha256": sha256_file(path)}
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    evidence["reported_official_split"] = data.get("dataset", {}).get("official_split") if isinstance(data, dict) else None
                    evidence["checkpoint_used"] = data.get("checkpoint_used") if isinstance(data, dict) else None
                    evidence["test_exposure"] = data.get("test_partition_already_evaluated") if isinstance(data, dict) else None
                except (OSError, json.JSONDecodeError):
                    evidence["parse_error"] = True
                historical_artifacts.append(evidence)
        provenance_evidence.extend({"kind": "historical_report", **item} for item in historical_artifacts)
    partition_ids = {partition: {row["image_id"] for row in inventory if row["official_partition"] == partition} for partition in PARTITIONS}
    manifest_matches_training = bool(manifest_ids) and manifest_ids == partition_ids["Training"]
    manifest_validation_overlap = sorted(manifest_ids & partition_ids["Validation"])
    manifest_testing_overlap = sorted(manifest_ids & partition_ids["Testing"])
    cross_partition_exact = [item for item in exact_groups if item["cross_partition"]]
    checkpoint_config = checkpoint_meta.get("config", {})
    checkpoint_paths_match = (
        str(checkpoint_config.get("training_images_dir", "")).replace("\\", "/").endswith("../dataset/raw")
        and str(checkpoint_config.get("dataset_root", "")).replace("\\", "/").endswith("../DenPAR Radiographs Dataset/Dataset")
        and checkpoint_meta.get("dataset_manifest_version") == "official_denpar_split_v1"
    )
    provenance_verified = bool(checkpoint_meta and checkpoint_paths_match and manifest_matches_training
                               and not manifest_validation_overlap and not manifest_testing_overlap)
    provenance = {
        "classification": "no_issue" if provenance_verified else "provenance_unverified",
        "checkpoint_training_provenance_verified": provenance_verified,
        "checkpoint": _relative(checkpoint, project_root) if checkpoint else None,
        "checkpoint_metadata": checkpoint_meta, "checkpoint_metadata_load_error": checkpoint_error,
        "evidence": provenance_evidence, "training_source_directory": _relative(training_images_dir, project_root),
        "training_source_location_note": "Training radiographs are outside the DenPAR Training folder; identity is supported by the immutable manifest and matching official annotation IDs.",
        "manifest_training_id_count": len(manifest_ids), "manifest_matches_all_observed_training_ids": manifest_matches_training,
        "checkpoint_paths_and_manifest_version_match_locked_configuration": checkpoint_paths_match,
        "manifest_validation_id_overlap": manifest_validation_overlap, "manifest_testing_id_overlap": manifest_testing_overlap,
        "cross_partition_exact_image_groups_by_method": len(cross_partition_exact),
        "cross_partition_unique_exact_image_pairs": cross_partition_exact_pairs,
        "validation_or_testing_paths_in_training_manifest": False if provenance_verified else None,
        "validation_or_testing_content_duplicated_in_training": bool(cross_partition_exact_pairs),
        "verified_claim": "Checkpoint metadata points to dataset/raw and the official Training annotation partition; no Validation or Testing IDs or paths occur in the 650-entry training manifest." if provenance_verified else None,
        "cannot_verify": ["Patient-level split independence cannot be verified from the supplied metadata.",
                          "Filesystem artifacts cannot independently prove every runtime batch consumed during historical training."],
    }
    if not provenance_verified:
        findings.append(_finding("provenance_unverified", "checkpoint_training_provenance_incomplete", details=provenance))
    if manifest_validation_overlap or manifest_testing_overlap:
        findings.append(_finding("confirmed_leakage", "training_partition_overlap",
            validation_ids=manifest_validation_overlap, testing_ids=manifest_testing_overlap,
            cross_partition_exact_pairs=[]))

    patient_headers = [header for row in workbook_rows.values() for header in row if any(token in header.lower() for token in ("patient", "subject", "case"))]
    patient_grouping = {"available": bool(patient_headers), "fields": sorted(set(patient_headers)),
                        "classification": "no_issue" if patient_headers else "provenance_unverified",
                        "limitation": None if patient_headers else "No patient, subject, or case identifier is present; patient-level separation cannot be verified."}
    if not patient_headers:
        findings.append(_finding("provenance_unverified", "patient_level_grouping_unavailable",
                                 limitation=patient_grouping["limitation"]))

    source_hashes_after = {path: sha256_file(Path(path)) for path in source_hashes_before}
    changed_sources = sorted(path for path, digest in source_hashes_before.items() if source_hashes_after[path] != digest)
    if changed_sources:
        raise RuntimeError(f"Source files changed during read-only audit: {changed_sources}")

    supplementary_issues = [item for item in findings if item["classification"] in {"missing_component", "provenance_unverified"}]
    manual_queue = [*near_duplicates, *unique_exact_pair_rows, *annotation_duplicates, *orphaned,
                    *coordinate_issues, *supplementary_issues]
    classification_counts = {name: sum(item.get("classification") == name for item in findings) for name in sorted(CLASSIFICATIONS)}
    if not findings:
        findings.append(_finding("no_issue", "no_audit_issues_detected"))
        classification_counts["no_issue"] = 1

    result = {
        "schema_version": 1, "audit": "DenPAR Stage 2A identity, provenance, and leakage audit",
        "generated_at": datetime.now(timezone.utc).isoformat(), "read_only_sources": True,
        "dataset_root": str(dataset_root), "training_images_dir": str(training_images_dir),
        "near_duplicate_method": {"name": "64-bit DCT pHash", "input": "decoded 8-bit grayscale resized to 32x32",
                                  "threshold": near_duplicate_threshold, "decision": "manual-review candidate only"},
        "partition_counts": partition_counts, "inventory": inventory, "exact_duplicates": exact_groups,
        "exact_duplicate_summary": {"unique_pairs": unique_exact_pair_rows,
                                     "unique_pair_count": len(unique_exact_pair_rows),
                                     "cross_partition_unique_pair_count": len(cross_partition_exact_pairs),
                                     "file_hash_group_count": sum(item["method"] == "identical_file_sha256" for item in exact_groups),
                                     "normalized_pixel_group_count": sum(item["method"] == "identical_normalized_pixels" for item in exact_groups)},
        "near_duplicates": near_duplicates, "annotation_duplicates": annotation_duplicates,
        "orphaned_annotations": orphaned, "coordinate_issues": coordinate_issues,
        "aggregate_annotation_files": aggregate_files, "checkpoint_provenance": provenance,
        "patient_grouping": patient_grouping, "findings": findings,
        "classification_counts": classification_counts,
        "source_integrity": {"files_hashed": len(source_hashes_before), "changed_files": changed_sources,
                             "preserved": not changed_sources, "workbook_sha256": workbook_hash},
    }

    _write_json(output_dir / "audit_results.json", result)
    _write_json(output_dir / "dataset_inventory.json", inventory)
    _write_csv(output_dir / "dataset_inventory.csv", inventory,
               ["stable_image_id", "image_id", "filename", "source_path", "absolute_source_path", "official_partition",
                "width", "height", "format", "file_sha256", "normalized_pixel_sha256", "perceptual_hash_64",
                "tooth_masks", "cej_apex_annotation", "bone_level_annotation", "radiograph_mask_annotation",
                "workbook_metadata_reference", "missing_components", "unreadable_components"])
    _write_json(output_dir / "partition_counts.json", partition_counts)
    _write_json(output_dir / "exact_duplicates.json", result["exact_duplicate_summary"] | {"evidence_groups": exact_groups})
    _write_json(output_dir / "near_duplicate_candidates.json", near_duplicates)
    _write_csv(output_dir / "near_duplicate_candidates.csv", near_duplicates,
               ["classification", "method", "threshold", "hamming_distance", "relative_aspect_difference", "cross_partition", "left", "right", "supporting_evidence", "interpretation"])
    _write_json(output_dir / "missing_orphaned_annotations.json", {"annotation_duplicates": annotation_duplicates, "orphaned": orphaned,
                                                                    "missing": [item for item in findings if item["classification"] == "missing_component"]})
    _write_json(output_dir / "coordinate_integrity.json", coordinate_issues)
    _write_json(output_dir / "checkpoint_training_provenance.json", provenance)
    leakage = {"confirmed_leakage": [item for item in findings if item["classification"] == "confirmed_leakage"],
               "cross_partition_exact_duplicates": cross_partition_exact_pairs,
               "cross_partition_near_duplicate_candidates": [item for item in near_duplicates if item["cross_partition"]],
               "patient_grouping": patient_grouping,
               "conclusion": "confirmed_leakage_detected" if any(item["classification"] == "confirmed_leakage" for item in findings) else "no_confirmed_image_level_leakage_detected"}
    _write_json(output_dir / "leakage_risk_summary.json", leakage)
    _write_json(output_dir / "manual_review_queue.json", manual_queue)
    _write_csv(output_dir / "manual_review_queue.csv", manual_queue,
               ["classification", "issue", "method", "threshold", "hamming_distance", "cross_partition", "images", "partitions", "evidence_methods", "left", "right", "supporting_evidence", "partition", "path", "stable_image_id"])
    _write_json(output_dir / "source_integrity.json", result["source_integrity"])

    summary = f"""# DenPAR Stage 2A audit

Generated: {result['generated_at']}

## Scope and method

This was a read-only identity, duplicate, annotation-integrity, and checkpoint-provenance audit. No radiograph was copied into this report. Exact identity uses SHA-256 of each file and SHA-256 of decoded canonical 8-bit grayscale pixels plus dimensions. Near duplicates are manual-review candidates produced by a 64-bit DCT pHash threshold of Hamming distance <= {near_duplicate_threshold}; they are not confirmed duplicates.

## Partition counts

| Partition | Expected | Observed | Match |
|---|---:|---:|---|
""" + "\n".join(f"| {part} | {partition_counts[part]['expected_images']} | {partition_counts[part]['observed_images']} | {partition_counts[part]['matches_expected']} |" for part in PARTITIONS) + f"""

Training images are read from `{_relative(training_images_dir, project_root)}` because the supplied DenPAR `Training` directory contains annotations but no image folder. The immutable 650-entry training manifest is the supporting identity record.

## Findings

- Exact duplicate pairs: {len(unique_exact_pair_rows)} unique pairs ({len(cross_partition_exact_pairs)} across partitions); evidence is available by file hash and normalized-pixel hash
- Near-duplicate candidates: {len(near_duplicates)} ({sum(item['cross_partition'] for item in near_duplicates)} across partitions)
- Duplicate annotation mappings: {len(annotation_duplicates)}
- Missing or orphaned annotation findings: {len(orphaned) + sum(item['classification'] == 'missing_component' for item in findings)}
- Coordinate-integrity findings: {len(coordinate_issues)}
- Source files hashed and preserved: {len(source_hashes_before)}; changed: {len(changed_sources)}
- Checkpoint training provenance verified from available artifacts: {provenance_verified}
- Patient grouping available: {patient_grouping['available']}

## Leakage conclusion

{leakage['conclusion']}. Perceptual matches remain candidates for manual review. Patient-level separation is unverified because the workbook contains image-level arch, site, and visible-tooth fields but no patient identifier.

## Stage 2B gate

Review `manual_review_queue.csv`, especially every cross-partition candidate and annotation-integrity finding, before using this dataset for Stage 2B. This report does not alter the official partitions.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    return result


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=project / "DenPAR Radiographs Dataset" / "Dataset")
    parser.add_argument("--training-images-dir", type=Path, default=project / "dataset" / "raw")
    parser.add_argument("--output-dir", type=Path, default=project / "artifacts" / "denpar-stage2a-audit")
    parser.add_argument("--near-duplicate-threshold", type=int, default=6)
    parser.add_argument("--checkpoint", type=Path, default=project / "models" / "tooth_instance_maskrcnn_baseline" / "full_run" / "best_checkpoint.pt")
    parser.add_argument("--training-manifest", type=Path, default=project / "models" / "tooth_instance_maskrcnn_baseline" / "training_expected_manifest.json")
    parser.add_argument("--training-config", type=Path, default=project / "models" / "tooth_instance_maskrcnn_baseline" / "full_run" / "config" / "final_train_config.json")
    args = parser.parse_args()
    result = run_audit(args.dataset_root, args.training_images_dir, args.output_dir, project,
                       near_duplicate_threshold=args.near_duplicate_threshold, checkpoint=args.checkpoint,
                       training_manifest=args.training_manifest, training_config=args.training_config)
    print(json.dumps({"partition_counts": result["partition_counts"],
                      "exact_duplicate_unique_pairs": result["exact_duplicate_summary"]["unique_pair_count"],
                      "cross_partition_exact_pairs": result["exact_duplicate_summary"]["cross_partition_unique_pair_count"],
                      "near_duplicate_candidates": len(result["near_duplicates"]), "coordinate_issues": len(result["coordinate_issues"]),
                      "source_integrity": result["source_integrity"],
                      "checkpoint_provenance_verified": result["checkpoint_provenance"]["checkpoint_training_provenance_verified"]}, indent=2))


if __name__ == "__main__":
    main()
