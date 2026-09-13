"""Validation for derived structural manifests, without interpreting anatomy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def validate_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    seen_images: set[str] = set()
    for record in manifest.get("records", []):
        image_id = record.get("image_id", "<unknown>")
        if image_id in seen_images:
            errors.append(f"duplicate image record: {image_id}")
        seen_images.add(image_id)
        if record.get("split") != "unsplit" or record.get("case_id") is not None:
            errors.append(f"unexpected split/case assignment: {image_id}")
        transform = record.get("transform", {})
        image_path = root / record.get("image_path", "")
        if not image_path.exists():
            errors.append(f"missing derived image: {image_id}")
            continue
        with Image.open(image_path) as image:
            if list(image.size) != transform.get("output_size"):
                errors.append(f"transform output dimensions differ from derived image: {image_id}")
        tasks = record.get("tasks", {})
        if tasks.get("cej_apex_landmarks", {}).get("target_status") == "available":
            if not all(item.get("status") == "confirmed" for item in tasks["cej_apex_landmarks"].get("landmarks", [])):
                errors.append(f"unconfirmed landmark target included: {image_id}")
        if tasks.get("bone_line_structure", {}).get("target_status") == "available":
            if not all(item.get("status") == "confirmed" and item.get("anatomical_meaning") for item in tasks["bone_line_structure"].get("bone_lines", [])):
                errors.append(f"unconfirmed or undefined bone-line target included: {image_id}")
    return {"valid": not errors, "record_count": len(manifest.get("records", [])), "errors": errors}
