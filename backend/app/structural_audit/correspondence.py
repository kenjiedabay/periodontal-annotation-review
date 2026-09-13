"""Conservative, read-only correspondence analysis for DenPAR annotations.

Spatial proximity is exposed for expert review, but it is never promoted to a
ground-truth pairing. A relationship is ``confirmed`` only when source files
explicitly provide an identifier-level relationship.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .loader import AuditResult, ImageAnnotations

CorrespondenceStatus = Literal["confirmed", "uncertain", "missing", "not_applicable"]


@dataclass(frozen=True)
class Relationship:
    name: str
    status: CorrespondenceStatus
    evidence: str
    unresolved_reason: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CorrespondenceRecord:
    image_id: str
    status: str
    relationships: tuple[Relationship, ...]
    missing_landmarks: tuple[str, ...]
    suspicious_mappings: tuple[str, ...]
    review_items: dict[str, list[dict[str, Any]]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "status": self.status,
            "relationships": [item.as_dict() for item in self.relationships],
            "missing_landmarks": list(self.missing_landmarks),
            "suspicious_mappings": list(self.suspicious_mappings),
            "review_items": self.review_items,
        }


def _relationship(name: str, status: CorrespondenceStatus, evidence: str, reason: str = "") -> Relationship:
    return Relationship(name, status, evidence, reason)


def _contains(bbox: list[float], point: list[float]) -> bool:
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def _coco_xyxy(annotation: dict[str, Any]) -> list[float] | None:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def analyze_record(record: ImageAnnotations) -> CorrespondenceRecord:
    """Describe source-established links and review-only spatial candidates."""
    image_id, keypoints = record.image.image_id, record.keypoints or {}
    masks, coco, bboxes = record.tooth_masks, record.coco_annotations, keypoints.get("bboxes", [])
    cej_points, apex_points = keypoints.get("CEJ_Points", []), keypoints.get("Apex_Points", [])
    fdi_values = []
    if record.characteristics:
        fdi_values = [value.strip() for value in record.characteristics.get("FDI notation of fully/partially visible teeth", "").split(",") if value.strip()]
    mask_count, coco_count, bbox_count = len(masks), len(coco), len(bboxes)
    relationships = [
        _relationship("image_to_tooth_masks", "confirmed" if mask_count else "missing", f"{mask_count} tooth-wise mask files are keyed by image ID." if mask_count else "No tooth-wise mask files found.", "" if mask_count else "tooth_masks_missing"),
        _relationship("image_to_coco_instances", "confirmed" if coco_count else "missing", f"{coco_count} COCO instances are keyed by image ID." if coco_count else "No COCO instances found.", "" if coco_count else "coco_instances_missing"),
        _relationship("image_to_keypoint_annotation", "confirmed" if record.keypoints else "missing", "Key-point JSON Image_id matches the image." if record.keypoints else "No key-point JSON found.", "" if record.keypoints else "keypoints_missing"),
        _relationship("image_to_fdi_metadata", "confirmed" if fdi_values else "missing", f"Workbook row supplies {len(fdi_values)} FDI candidate values for this image." if fdi_values else "No FDI metadata found.", "" if fdi_values else "fdi_metadata_missing"),
        _relationship("tooth_mask_to_coco_instance", "uncertain" if mask_count and coco_count else "missing", f"mask count={mask_count}; COCO instance count={coco_count}.", "No shared instance ID or declared ordering exists. Equal counts do not establish pairs." if mask_count and coco_count else "One or both assets are absent."),
        _relationship("keypoint_bbox_to_tooth_instance", "uncertain" if bbox_count else "missing", f"key-point bbox count={bbox_count}; mask count={mask_count}; COCO count={coco_count}.", "No stable instance ID links these boxes to masks or COCO annotations." if bbox_count else "No key-point bboxes found."),
        _relationship("cej_to_tooth_bbox", "uncertain" if cej_points else "missing", f"CEJ point count={len(cej_points)}; key-point bbox count={bbox_count}.", "No CEJ-to-box index or tooth identifier is encoded." if cej_points else "No CEJ points found."),
        _relationship("apex_to_tooth_bbox", "uncertain" if apex_points else "missing", f"Apex point count={len(apex_points)}; key-point bbox count={bbox_count}.", "No apex-to-box index or tooth identifier is encoded." if apex_points else "No apex points found."),
        _relationship("fdi_metadata_to_tooth_instance", "uncertain" if fdi_values else "missing", f"FDI candidate count={len(fdi_values)}; mask count={mask_count}; COCO count={coco_count}.", "FDI values have no source-defined link or ordering to maskN, COCO, boxes, or landmarks." if fdi_values else "No FDI metadata found."),
    ]
    coco_boxes = [_coco_xyxy(annotation) for annotation in coco]
    review_items: dict[str, list[dict[str, Any]]] = {
        "tooth_masks": [{"id": str(mask.get("filename", index)), "bbox_xyxy": mask.get("bbox_xyxy"), "status": "uncertain"} for index, mask in enumerate(masks)],
        "coco_instances": [{"id": str(annotation.get("id", index)), "bbox_xyxy": coco_boxes[index], "status": "uncertain"} for index, annotation in enumerate(coco)],
        "keypoint_bboxes": [{"id": str(index), "bbox_xyxy": bbox, "status": "uncertain"} for index, bbox in enumerate(bboxes)],
        "cej_points": [{"id": str(index), "point": point, "bbox_candidates": [str(box_index) for box_index, bbox in enumerate(bboxes) if _contains(bbox, point)], "status": "uncertain"} for index, point in enumerate(cej_points)],
        "apex_points": [{"id": str(index), "point": point, "bbox_candidates": [str(box_index) for box_index, bbox in enumerate(bboxes) if _contains(bbox, point)], "status": "uncertain"} for index, point in enumerate(apex_points)],
        "fdi_candidates": [{"id": value, "status": "uncertain"} for value in fdi_values],
    }
    missing = [label for label, values in (("CEJ", cej_points), ("apex", apex_points)) if not values]
    suspicious: list[str] = []
    for label, count in (("CEJ", len(cej_points)), ("apex", len(apex_points))):
        if count and count != bbox_count:
            suspicious.append(f"{label} count {count} differs from key-point bbox count {bbox_count}.")
    if fdi_values and len(fdi_values) != mask_count:
        suspicious.append(f"FDI candidate count {len(fdi_values)} differs from tooth mask count {mask_count}.")
    if mask_count and coco_count != mask_count:
        suspicious.append(f"Tooth mask count {mask_count} differs from COCO count {coco_count}.")
    for label, items in (("CEJ", review_items["cej_points"]), ("apex", review_items["apex_points"])):
        for item in items:
            if not item["bbox_candidates"]:
                suspicious.append(f"{label} point {item['id']} is not contained by any key-point bbox.")
    statuses = {item.status for item in relationships}
    overall = "unresolved" if "uncertain" in statuses else "partial" if "missing" in statuses else "complete"
    return CorrespondenceRecord(image_id, overall, tuple(relationships), tuple(missing), tuple(suspicious), review_items)


def analyze_correspondence(audit: AuditResult) -> dict[str, Any]:
    records = [analyze_record(record) for record in audit.records]
    counts = Counter(record.status for record in records)
    return {"records": [record.as_dict() for record in records], "summary": {
        "images_with_complete_correspondence": counts.get("complete", 0),
        "images_with_partial_correspondence": counts.get("partial", 0),
        "images_with_unresolved_correspondence": counts.get("unresolved", 0),
        "images_with_missing_landmarks": sum(bool(record.missing_landmarks) for record in records),
        "images_with_suspicious_mappings": sum(bool(record.suspicious_mappings) for record in records),
        "safe_for_landmark_supervision": False,
        "reason": "No deterministic CEJ/apex-to-tooth correspondence is encoded. Review candidates are not ground truth.",
    }}
