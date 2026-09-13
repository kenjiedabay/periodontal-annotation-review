"""Reproducible preparation for DenPAR structural tasks; never creates disease targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app.structural_audit.loader import audit_validation_dataset
from .validator import validate_manifest


TASKS = ("tooth_segmentation", "tooth_localization", "cej_apex_landmarks", "bone_line_structure")


@dataclass(frozen=True)
class PreparationConfig:
    validation_dir: Path
    output_dir: Path
    confirmed_correspondence: Path | None = None
    resize_width: int | None = None
    resize_height: int | None = None
    sample_count: int = 3

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return {key: str(item) if isinstance(item, Path) else item for key, item in value.items()}


def _resize_config(config: PreparationConfig, width: int, height: int) -> dict[str, Any]:
    if config.resize_width is None and config.resize_height is None:
        return {"operation": "identity", "source_size": [width, height], "output_size": [width, height], "scale_x": 1.0, "scale_y": 1.0}
    if not config.resize_width or not config.resize_height:
        raise ValueError("resize-width and resize-height must be supplied together")
    return {"operation": "resize", "mode": "explicit_stretch", "source_size": [width, height], "output_size": [config.resize_width, config.resize_height], "scale_x": config.resize_width / width, "scale_y": config.resize_height / height, "image_resample": "lanczos", "mask_resample": "nearest"}


def _load_confirmations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("confirmed correspondence must be an object with an images list")
    result = {}
    for item in payload["images"]:
        if isinstance(item, dict) and isinstance(item.get("image_id"), str):
            result[item["image_id"].removesuffix(".jpg")] = item
    return result


def _confirmed_landmarks(item: dict[str, Any], scale_x: float, scale_y: float) -> list[dict[str, Any]]:
    values = []
    for landmark in item.get("landmarks", []):
        if not isinstance(landmark, dict) or landmark.get("status") != "confirmed":
            continue
        cej, apex, instance_id = landmark.get("cej"), landmark.get("apex"), landmark.get("tooth_instance_id")
        if not (isinstance(instance_id, (str, int)) and isinstance(cej, list) and len(cej) == 2 and isinstance(apex, list) and len(apex) == 2):
            continue
        values.append({"tooth_instance_id": str(instance_id), "cej": [cej[0] * scale_x, cej[1] * scale_y], "apex": [apex[0] * scale_x, apex[1] * scale_y], "status": "confirmed"})
    return values


def _confirmed_bone_lines(item: dict[str, Any], scale_x: float, scale_y: float) -> list[dict[str, Any]]:
    values = []
    for line in item.get("bone_lines", []):
        if not isinstance(line, dict) or line.get("status") != "confirmed" or not line.get("anatomical_meaning"):
            continue
        points = line.get("points")
        if not (isinstance(line.get("tooth_instance_id"), (str, int)) and isinstance(points, list) and len(points) >= 2):
            continue
        if not all(isinstance(point, list) and len(point) == 2 for point in points):
            continue
        values.append({"tooth_instance_id": str(line["tooth_instance_id"]), "anatomical_meaning": line["anatomical_meaning"], "points": [[point[0] * scale_x, point[1] * scale_y] for point in points], "status": "confirmed"})
    return values


def _write_image(source: Path, destination: Path, transform: dict[str, Any], mask: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if transform["operation"] == "identity":
        shutil.copy2(source, destination)
        return
    with Image.open(source) as image:
        resample = Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS
        image.resize(tuple(transform["output_size"]), resample=resample).save(destination)


def _union_mask(mask_paths: list[Path], destination: Path, transform: dict[str, Any]) -> None:
    union: np.ndarray | None = None
    for path in mask_paths:
        with Image.open(path) as image:
            current = np.asarray(image.convert("L")) > 0
        union = current if union is None else np.logical_or(union, current)
    if union is None:
        raise ValueError("cannot create a union mask from no masks")
    output = Image.fromarray((union * 255).astype(np.uint8), mode="L")
    if transform["operation"] == "resize":
        output = output.resize(tuple(transform["output_size"]), resample=Image.Resampling.NEAREST)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination)


def _visualization(image_path: Path, output: Path, boxes: list[list[float]], transform: dict[str, Any]) -> None:
    with Image.open(image_path).convert("RGB") as image:
        rendered = image.copy()
        draw = ImageDraw.Draw(rendered)
        for x, y, width, height in boxes:
            draw.rectangle((x, y, x + width, y + height), outline=(30, 210, 150), width=max(1, image.width // 250))
        output.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(output)


def prepare_structural_dataset(config: PreparationConfig) -> dict[str, Any]:
    """Create derived assets/manifests without disease, severity, or FDI targets.

    The source does not expose patient/case identifiers. To prevent unprovable
    leakage, this pipeline leaves all records in an ``unsplit`` pool rather than
    fabricating image-level train/validation/test groups.
    """
    audit = audit_validation_dataset(config.validation_dir)
    confirmations = _load_confirmations(config.confirmed_correspondence)
    root = config.output_dir
    for directory in ("images/unsplit", "task_a_masks/unsplit", "task_a_instances/unsplit", "manifests", "reports", "samples"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    task_counts = {task: {"images": 0, "targets": 0, "excluded_unconfirmed": 0} for task in TASKS}
    for index, record in enumerate(audit.records):
        image_id, image = record.image.image_id, record.image
        transform = _resize_config(config, image.width, image.height)
        image_output = root / "images/unsplit" / image.filename
        _write_image(Path(image.path), image_output, transform)
        source_masks = [config.validation_dir / "Masks (Tooth-wise)" / image_id / item["filename"] for item in record.tooth_masks]
        union_output = root / "task_a_masks/unsplit" / f"{image_id}.png"
        _union_mask(source_masks, union_output, transform)
        instances = []
        for mask, source in zip(record.tooth_masks, source_masks):
            target = root / "task_a_instances/unsplit" / image_id / mask["filename"]
            _write_image(source, target, transform, mask=True)
            instances.append({"source_filename": mask["filename"], "path": str(target.relative_to(root)).replace("\\", "/"), "status": "unpaired_instance_mask"})
        sx, sy = transform["scale_x"], transform["scale_y"]
        boxes = [{"coco_annotation_id": annotation.get("id"), "bbox_xywh": [annotation["bbox"][0] * sx, annotation["bbox"][1] * sy, annotation["bbox"][2] * sx, annotation["bbox"][3] * sy]} for annotation in record.coco_annotations]
        confirmed = confirmations.get(image_id, {})
        landmarks = _confirmed_landmarks(confirmed, sx, sy)
        bone_lines = _confirmed_bone_lines(confirmed, sx, sy)
        task_counts["tooth_segmentation"]["images"] += 1
        task_counts["tooth_segmentation"]["targets"] += len(instances)
        task_counts["tooth_localization"]["images"] += 1
        task_counts["tooth_localization"]["targets"] += len(boxes)
        task_counts["cej_apex_landmarks"]["images"] += int(bool(landmarks))
        task_counts["cej_apex_landmarks"]["targets"] += len(landmarks)
        task_counts["cej_apex_landmarks"]["excluded_unconfirmed"] += len((record.keypoints or {}).get("CEJ_Points", [])) + len((record.keypoints or {}).get("Apex_Points", []))
        task_counts["bone_line_structure"]["images"] += int(bool(bone_lines))
        task_counts["bone_line_structure"]["targets"] += len(bone_lines)
        task_counts["bone_line_structure"]["excluded_unconfirmed"] += len((record.bone_lines or {}).get("Bone_Lines", []))
        records.append({"image_id": image_id, "split": "unsplit", "case_id": None, "image_path": str(image_output.relative_to(root)).replace("\\", "/"), "source_image_path": image.path, "source_dimensions": [image.width, image.height], "transform": transform, "tasks": {"tooth_segmentation": {"union_mask_path": str(union_output.relative_to(root)).replace("\\", "/"), "instance_masks": instances, "target_status": "available"}, "tooth_localization": {"boxes": boxes, "target_status": "available"}, "cej_apex_landmarks": {"landmarks": landmarks, "target_status": "available" if landmarks else "excluded_unconfirmed"}, "bone_line_structure": {"bone_lines": bone_lines, "target_status": "available" if bone_lines else "excluded_unconfirmed"}}})
        if index < config.sample_count:
            _visualization(image_output, root / "samples" / f"{image_id}_localization.png", [item["bbox_xywh"] for item in boxes], transform)
    manifest = {"schema_version": 1, "purpose": "non-diagnostic structural analysis only", "prohibited_targets": ["disease_status", "severity", "FDI_metadata", "Arch", "Site"], "split_policy": {"status": "not_generated", "reason": "No patient/case IDs were available; image-level splitting could leak cases.", "pool": "unsplit"}, "records": records}
    manifest_path = root / "manifests" / "structural_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (root / "manifests" / "structural_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "split", "case_id", "image_path", "source_dimensions", "transform", "segmentation_status", "localization_boxes", "landmark_status", "bone_line_status"])
        writer.writeheader()
        for item in records:
            tasks = item["tasks"]
            writer.writerow({"image_id": item["image_id"], "split": item["split"], "case_id": "", "image_path": item["image_path"], "source_dimensions": json.dumps(item["source_dimensions"]), "transform": json.dumps(item["transform"]), "segmentation_status": tasks["tooth_segmentation"]["target_status"], "localization_boxes": len(tasks["tooth_localization"]["boxes"]), "landmark_status": tasks["cej_apex_landmarks"]["target_status"], "bone_line_status": tasks["bone_line_structure"]["target_status"]})
    validation = validate_manifest(root, manifest)
    if not validation["valid"]:
        raise ValueError(f"derived structural manifest validation failed: {validation['errors']}")
    report = {"config": config.as_dict(), "source_preserved": True, "case_id_limitation": "Case/patient IDs unavailable in inspected DenPAR files; no train/validation/test split was generated.", "task_statistics": task_counts, "records": len(records), "manifest": str(manifest_path.relative_to(root)).replace("\\", "/"), "validation": {"images_with_dimension_preserving_identity_transform": sum(item["transform"]["operation"] == "identity" for item in records), "images_with_explicit_resize_transform": sum(item["transform"]["operation"] == "resize" for item in records), "disease_or_severity_targets_written": False, "manifest_validation": validation}}
    (root / "reports" / "structural_preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "manifests" / "preparation_config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    return report
