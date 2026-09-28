"""Versioned expert verification of tooth identity and mesial/distal surfaces.

Stage 2B proposals and DenPAR files are read-only.  Every submitted decision is
stored as a new JSON record in original-image pixel coordinates.
"""
from __future__ import annotations

import json
import math
import os
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSOCIATIONS = PROJECT_ROOT / "artifacts" / "denpar-stage2b-associations-v1"
STORE = Path(os.getenv("SURFACE_VERIFICATION_DIR", PROJECT_ROOT / "data" / "surface_verification"))
router = APIRouter(prefix="/api/surface-verification", tags=["tooth surface verification"])
LOCK = threading.RLock()
SCHEMA_VERSION = "1.1.0"
CALCULATION_VERSION = "rbl-euclidean-v1"


class GeometryConfig(BaseModel):
    mask_tolerance_px: float = Field(default=12, ge=0)
    bone_boundary_tolerance_px: float = Field(default=18, ge=0)
    near_zero_distance_px: float = Field(default=5, gt=0)
    coronal_allowance_px: float = Field(default=8, ge=0)
    rbl_review_min: float = Field(default=0, ge=0)
    rbl_review_max: float = Field(default=100, gt=0)
    unrelated_line_sample_step_px: float = Field(default=2, gt=0)


GEOMETRY = GeometryConfig()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Point(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class Orientation(BaseModel):
    arch: Literal["maxillary", "mandibular", "unknown"] = "unknown"
    patient_side: Literal["left", "right", "anterior", "unknown"] = "unknown"
    display_orientation: Literal["native", "mirrored", "rotated", "unknown"] = "unknown"
    horizontal_flip: bool | None = None
    source: Literal["dataset_metadata", "imported_record", "expert_confirmation", "unknown"] = "unknown"
    status: Literal["verified", "unverified", "conflicting"] = "unverified"
    verified_by: str | None = None
    verified_at: str | None = None

    @model_validator(mode="after")
    def verified_has_provenance(self) -> "Orientation":
        if self.status == "verified" and (not self.verified_by or not self.verified_at):
            raise ValueError("Verified orientation requires expert and timestamp")
        return self


class Landmark(BaseModel):
    landmark_id: str | None = None
    state: Literal["visible", "not_visible", "uncertain", "ungradable"] = "uncertain"
    point: Point | None = None
    verification_status: Literal["confirmed", "unverified", "ungradable"] = "unverified"
    expert_notes: str = ""

    @model_validator(mode="after")
    def coordinates_match_state(self) -> "Landmark":
        if (self.state == "visible") != (self.point is not None):
            raise ValueError("Only a visible landmark may have a coordinate, and it must have one")
        return self


class SurfaceRecord(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    surface: Literal["mesial", "distal"]
    model_surface_suggestion: Literal["mesial", "distal"] | None = None
    pixel_side: Literal["image_left", "image_right"] | None = None
    adjacent_fdi: str | None = None
    adjacency_status: Literal["confirmed", "missing_neighbor", "unverified"] = "unverified"
    cej: Landmark = Field(default_factory=Landmark)
    bone_crest: Landmark = Field(default_factory=Landmark)
    root_apex: Landmark = Field(default_factory=Landmark)
    rbl_percentage: float | None = Field(default=None, ge=0)
    verification_status: Literal["confirmed", "unverified", "ungradable"] = "unverified"
    expert_decision: Literal["confirmed", "corrected", "not_visible", "ungradable", "pending"] = "pending"
    review_notes: str = ""
    model_cej: Point | None = None
    model_bone_crest: Point | None = None
    model_root_apex: Point | None = None
    model_rbl_percentage: float | None = Field(default=None, ge=0)
    cej_to_bone_distance_px: float | None = None
    cej_to_apex_distance_px: float | None = None
    calculation_version: str | None = None
    apex_reference_id: str | None = None
    apex_selection_method: Literal["expert_selected", "expert_approved_rule", "not_selected"] = "not_selected"


class ToothDecision(BaseModel):
    instance_id: str
    original_instance_id: str
    original_mask_ref: str | None = None
    original_bbox_xyxy: tuple[float, float, float, float]
    original_confidence: float | None = Field(default=None, ge=0, le=1)
    centroid: Point | None = None
    model_fdi_suggestion: str | None = None
    expert_fdi: str | None = None
    tooth_verification_status: Literal["confirmed", "unverified", "ungradable"] = "unverified"
    fdi_source: Literal["model_suggestion", "expert_confirmation", "none"] = "none"
    partial_edge: bool = False
    multi_rooted: bool | None = None
    surfaces: list[SurfaceRecord] = Field(default_factory=list)
    review_notes: str = ""


class VerificationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer_id: str = Field(min_length=1)
    partition: Literal["Training", "Validation"]
    image_id: str = Field(min_length=1)
    orientation: Orientation
    teeth: list[ToothDecision]
    review_notes: str = ""


def valid_fdi(value: str | None) -> bool:
    if value is None or len(value) != 2 or not value.isdigit():
        return False
    quadrant, position = int(value[0]), int(value[1])
    return quadrant in (1, 2, 3, 4) and 1 <= position <= 8


def expected_facing_surface(tooth_fdi: str, neighbor_fdi: str) -> str | None:
    """Return the surface on tooth_fdi that faces an anatomically adjacent neighbor."""
    if not valid_fdi(tooth_fdi) or not valid_fdi(neighbor_fdi) or tooth_fdi == neighbor_fdi:
        return None
    central_pairs = {frozenset(("11", "21")), frozenset(("31", "41"))}
    if frozenset((tooth_fdi, neighbor_fdi)) in central_pairs:
        return "mesial"
    if tooth_fdi[0] != neighbor_fdi[0]:
        return None
    tooth_pos, neighbor_pos = int(tooth_fdi[1]), int(neighbor_fdi[1])
    if abs(tooth_pos - neighbor_pos) != 1:
        return None
    return "mesial" if neighbor_pos < tooth_pos else "distal"


def swap_surfaces(surfaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Swap anatomical labels and linked records without changing coordinates."""
    swapped = []
    for item in surfaces:
        copy = json.loads(json.dumps(item))
        copy["surface"] = "distal" if item["surface"] == "mesial" else "mesial"
        swapped.append(copy)
    return swapped


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def calculate_rbl(surface: SurfaceRecord) -> float | None:
    if surface.verification_status != "confirmed" or not all(
        x.state == "visible" and x.point and x.verification_status == "confirmed"
        for x in (surface.cej, surface.bone_crest, surface.root_apex)
    ):
        return None
    denominator = _distance(surface.cej.point, surface.root_apex.point)  # type: ignore[arg-type]
    if denominator <= 0:
        return None
    return 100.0 * _distance(surface.cej.point, surface.bone_crest.point) / denominator  # type: ignore[arg-type]


def _mask_path(partition: str, image_id: str, tooth: ToothDecision) -> Path | None:
    reference = tooth.original_mask_ref
    if reference:
        candidate = PROJECT_ROOT / reference
        if candidate.is_file():
            return candidate
    suffix = tooth.instance_id.rsplit(":", 1)[-1].replace("tooth:", "")
    try:
        number = int(suffix)
    except ValueError:
        return None
    candidate = PROJECT_ROOT / "DenPAR Radiographs Dataset" / "Dataset" / partition / "Masks (Tooth-wise)" / image_id / f"mask{number}.png"
    return candidate if candidate.is_file() else None


def _point_mask_distance(mask: np.ndarray, point: Point) -> tuple[bool, float]:
    x, y = int(round(point.x)), int(round(point.y))
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
        return True, 0.0
    outside = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        cx, cy = min(max(x, 0), mask.shape[1]-1), min(max(y, 0), mask.shape[0]-1)
        return False, float(outside[cy, cx] + math.hypot(x-cx, y-cy))
    return False, float(outside[y, x])


def _boundary_distance(mask: np.ndarray, point: Point) -> float:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return min((-cv2.pointPolygonTest(c, (point.x, point.y), True) for c in contours), default=float("inf"), key=abs).__abs__()


def _line_crosses(mask: np.ndarray, start: Point, end: Point, step: float) -> bool:
    length = _distance(start, end); count = max(2, math.ceil(length / step))
    for index in range(1, count):
        ratio = index / count; x = round(start.x + (end.x-start.x)*ratio); y = round(start.y + (end.y-start.y)*ratio)
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
            return True
    return False


def geometry_warnings(payload: VerificationInput, config: GeometryConfig = GEOMETRY) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    masks: dict[str, np.ndarray] = {}
    for tooth in payload.teeth:
        path = _mask_path(payload.partition, payload.image_id, tooth)
        if path:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None: masks[tooth.instance_id] = mask
    for tooth in payload.teeth:
        mask = masks.get(tooth.instance_id); centroid_x = tooth.centroid.x if tooth.centroid else (tooth.original_bbox_xyxy[0]+tooth.original_bbox_xyxy[2])/2
        for surface in tooth.surfaces:
            context = f"{tooth.instance_id} {surface.surface}"
            for name, landmark in (("cej", surface.cej), ("apex", surface.root_apex)):
                if mask is not None and landmark.point:
                    inside, distance = _point_mask_distance(mask, landmark.point)
                    if not inside and distance > config.mask_tolerance_px:
                        warnings.append({"code": f"{name}_outside_mask", "severity": "block", "message": f"{context}: {name} is {distance:.1f}px outside tooth mask"})
            if mask is not None and surface.bone_crest.point:
                distance = _boundary_distance(mask, surface.bone_crest.point)
                if distance > config.bone_boundary_tolerance_px:
                    warnings.append({"code": "bone_far_from_boundary", "severity": "block", "message": f"{context}: bone point is {distance:.1f}px from tooth boundary"})
            points = [x.point for x in (surface.cej, surface.bone_crest, surface.root_apex) if x.point]
            for point in points:
                actual = "image_left" if point.x < centroid_x else "image_right"
                if surface.pixel_side and actual != surface.pixel_side:
                    warnings.append({"code": "landmark_wrong_pixel_side", "severity": "block", "message": f"{context}: landmark contradicts confirmed pixel side"})
                    break
            if surface.cej.point and surface.root_apex.point and _distance(surface.cej.point, surface.root_apex.point) < config.near_zero_distance_px:
                warnings.append({"code": "near_zero_root_length", "severity": "block", "message": f"{context}: CEJ-to-apex distance is too small"})
            if surface.cej.point and surface.bone_crest.point and surface.root_apex.point:
                root = _distance(surface.cej.point, surface.root_apex.point); bone = _distance(surface.cej.point, surface.bone_crest.point)
                if bone > root + config.coronal_allowance_px:
                    warnings.append({"code": "anatomically_invalid_bone", "severity": "block", "message": f"{context}: bone distance exceeds root reference"})
            if surface.cej.point and surface.bone_crest.point:
                for other_id, other_mask in masks.items():
                    if other_id != tooth.instance_id and _line_crosses(other_mask, surface.cej.point, surface.bone_crest.point, config.unrelated_line_sample_step_px):
                        warnings.append({"code": "measurement_crosses_unrelated_tooth", "severity": "review", "message": f"{context}: measurement crosses {other_id}"}); break
    return warnings


def validate_submission(payload: VerificationInput) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    fdis = [tooth.expert_fdi for tooth in payload.teeth if tooth.expert_fdi]
    for fdi, count in Counter(fdis).items():
        if count > 1:
            warnings.append({"code": "duplicate_fdi", "message": f"FDI {fdi} is assigned {count} times"})
    ids = {tooth.instance_id for tooth in payload.teeth}
    if len(ids) != len(payload.teeth):
        warnings.append({"code": "duplicate_instance", "message": "A tooth instance is repeated"})
    for tooth in payload.teeth:
        if tooth.expert_fdi and not valid_fdi(tooth.expert_fdi):
            warnings.append({"code": "invalid_fdi", "message": f"Invalid permanent FDI value {tooth.expert_fdi}"})
        if tooth.tooth_verification_status != "confirmed":
            warnings.append({"code": "tooth_unverified", "message": f"{tooth.instance_id}: tooth identity is not confirmed"})
        labels = [surface.surface for surface in tooth.surfaces]
        if len(labels) != len(set(labels)):
            warnings.append({"code": "duplicate_surface", "message": f"{tooth.instance_id}: repeated surface label"})
        pixel_sides = [surface.pixel_side for surface in tooth.surfaces if surface.pixel_side]
        if len(pixel_sides) != len(set(pixel_sides)):
            warnings.append({"code": "same_pixel_side", "message": f"{tooth.instance_id}: mesial and distal use the same pixel side"})
        if tooth.multi_rooted and any(s.root_apex.state == "visible" and s.verification_status != "confirmed" for s in tooth.surfaces):
            warnings.append({"code": "multiroot_apex_review", "message": f"{tooth.instance_id}: expert apex selection required"})
        for surface in tooth.surfaces:
            calculated = calculate_rbl(surface)
            context = f"{tooth.instance_id} {surface.surface}"
            if payload.orientation.status != "verified" and surface.verification_status == "confirmed":
                warnings.append({"code": "orientation_unverified", "message": f"{context}: surface cannot be ground truth"})
            if surface.verification_status != "confirmed":
                warnings.append({"code": "surface_unverified", "message": f"{context}: surface is not confirmed"})
            if surface.rbl_percentage is not None and (not math.isfinite(surface.rbl_percentage) or surface.rbl_percentage < 0):
                warnings.append({"code": "invalid_rbl", "message": f"{context}: invalid RBL value"})
            if surface.rbl_percentage is not None and calculated is None:
                warnings.append({"code": "rbl_without_landmarks", "message": f"{context}: RBL lacks complete visible landmarks"})
            if calculated is not None and not GEOMETRY.rbl_review_min <= calculated <= GEOMETRY.rbl_review_max:
                warnings.append({"code": "rbl_outside_review_range", "severity": "review", "message": f"{context}: RBL {calculated:.2f}% needs expert review"})
            if tooth.multi_rooted and surface.verification_status == "confirmed" and (not surface.apex_reference_id or surface.apex_selection_method == "not_selected"):
                warnings.append({"code": "missing_multiroot_apex_selection", "severity": "block", "message": f"{context}: select an apex reference"})
            if surface.adjacent_fdi and tooth.expert_fdi:
                expected = expected_facing_surface(tooth.expert_fdi, surface.adjacent_fdi)
                if expected is None:
                    warnings.append({"code": "invalid_adjacency", "message": f"{tooth.expert_fdi} and {surface.adjacent_fdi} are not anatomical neighbors"})
                elif expected != surface.surface:
                    warnings.append({"code": "inconsistent_surface", "message": f"{tooth.expert_fdi}-{surface.surface[0].upper()} cannot face {surface.adjacent_fdi}"})
    return warnings + geometry_warnings(payload)


def _association(partition: str, image_id: str) -> dict[str, Any]:
    path = ASSOCIATIONS / f"{partition.lower()}_associations.jsonl"
    if not path.is_file():
        raise HTTPException(404, "Association artifact unavailable")
    stable = f"{partition}:{image_id}"
    with path.open(encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            if item["stable_image_id"] == stable:
                return item
    raise HTTPException(404, "Association record not found")


def _proposal(partition: str, image_id: str) -> dict[str, Any]:
    record = _association(partition, image_id)
    teeth = []
    for item in record["tooth_instances"]:
        x1, y1, x2, y2 = item["original_bbox_xyxy"]
        teeth.append({**item, "centroid": {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2},
                      "model_fdi_suggestion": None, "expert_fdi": None,
                      "tooth_verification_status": "unverified", "surfaces": []})
    return {"schema_version": SCHEMA_VERSION, "stable_image_id": record["stable_image_id"],
            "source_image_id": image_id, "source_partition": partition,
            "original_image": record["original_image"], "workbook_fdi_values": record.get("workbook_fdi_values"),
            "orientation": {"arch": "unknown", "patient_side": "unknown", "display_orientation": "unknown",
                            "horizontal_flip": None, "source": "dataset_metadata", "status": "unverified",
                            "verified_by": None, "verified_at": None},
            "teeth": teeth, "warnings": ["DenPAR metadata does not establish display orientation or flip status.",
                                          "No mesial/distal assignment is ground truth until expert confirmation."],
            "coordinate_space": "original_image_pixels", "source_record": record}


def _records(partition: str, image_id: str) -> list[dict[str, Any]]:
    folder = STORE / partition / image_id
    if not folder.is_dir():
        return []
    return sorted((json.loads(path.read_text(encoding="utf-8")) for path in folder.glob("*.json")), key=lambda x: x["created_at"])


@router.get("/evaluation/summary")
def evaluation_summary() -> dict[str, Any]:
    return _evaluation_summary_impl()


@router.get("/features/{partition}/{image_id}/{reviewer_id}")
def confirmed_features(partition: Literal["Training", "Validation"], image_id: str, reviewer_id: str) -> dict[str, Any]:
    """Export only confirmed expert-derived features for a later disease/severity model."""
    rows = [r for r in _records(partition, image_id) if r.get("reviewer_id") == reviewer_id]
    if not rows:
        raise HTTPException(404, "No expert record found")
    record = rows[-1]
    features = []
    for tooth in record.get("teeth", []):
        for surface in tooth.get("surfaces", []):
            if surface.get("verification_status") != "confirmed":
                continue
            features.append({"tooth_instance_id": tooth.get("instance_id"), "expert_fdi": tooth.get("expert_fdi"),
                             "surface": surface.get("surface"), "rbl_percentage": surface.get("rbl_percentage"),
                             "cej_to_bone_distance_px": surface.get("cej_to_bone_distance_px"),
                             "cej_to_apex_distance_px": surface.get("cej_to_apex_distance_px"),
                             "landmark_states": {k: surface.get(k, {}).get("state") for k in ("cej", "bone_crest", "root_apex")},
                             "expert_decision": surface.get("expert_decision"), "review_notes": surface.get("review_notes", "")})
    return {"schema_version": SCHEMA_VERSION, "feature_source_record_id": record.get("record_id"),
            "source_image_id": image_id, "partition": partition, "reviewer_id": reviewer_id,
            "feature_status": "expert_confirmed_only", "features": features,
            "clinical_labels_required_separately": True}


@router.get("/{partition}/{image_id}")
def get_verification(partition: Literal["Training", "Validation"], image_id: str) -> dict[str, Any]:
    history = _records(partition, image_id)
    return {"proposal": _proposal(partition, image_id), "latest": history[-1] if history else None, "history_count": len(history)}


@router.post("/{partition}/{image_id}", status_code=201)
def save_verification(partition: Literal["Training", "Validation"], image_id: str, payload: VerificationInput) -> dict[str, Any]:
    if payload.partition != partition or payload.image_id != image_id:
        raise HTTPException(422, "Path and payload identity differ")
    proposal = _proposal(partition, image_id)
    expected_ids = {x["tooth_instance_id"] for x in proposal["teeth"]}
    if any(t.instance_id not in expected_ids or t.original_instance_id != t.instance_id for t in payload.teeth):
        raise HTTPException(422, "Original tooth instance identity must be preserved")
    warnings = validate_submission(payload)
    stored_teeth = payload.model_dump()["teeth"]
    for tooth_model, tooth_data in zip(payload.teeth, stored_teeth):
        for surface_model, surface_data in zip(tooth_model.surfaces, tooth_data["surfaces"]):
            rbl = calculate_rbl(surface_model)
            if rbl is not None:
                cej, bone, apex = surface_model.cej.point, surface_model.bone_crest.point, surface_model.root_apex.point
                surface_data.update(rbl_percentage=rbl, cej_to_bone_distance_px=_distance(cej, bone),
                                    cej_to_apex_distance_px=_distance(cej, apex), calculation_version=CALCULATION_VERSION)
            else:
                surface_data.update(rbl_percentage=None, cej_to_bone_distance_px=None,
                                    cej_to_apex_distance_px=None, calculation_version=None)
    record_id = str(uuid.uuid4())
    record = {"schema_version": SCHEMA_VERSION, "record_id": record_id, "record_type": "expert_surface_verification",
              "created_at": _now(), "coordinate_space": "original_image_pixels", "coordinate_origin": "top_left",
              "source_proposal_id": proposal["stable_image_id"], "model_predictions_immutable": True,
              **payload.model_dump(exclude={"teeth"}), "teeth": stored_teeth, "geometry_config": GEOMETRY.model_dump(),
              "warnings": warnings, "ground_truth_ready": not any(x.get("severity") == "block" for x in warnings) and payload.orientation.status == "verified"}
    target = STORE / partition / image_id / f"{record_id}.json"
    with LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as output:
            json.dump(record, output, ensure_ascii=False, indent=2)
    return record


def _metrics(rows: list[tuple[str, str]]) -> dict[str, Any]:
    labels = ("mesial", "distal")
    matrix = [[sum(1 for truth, pred in rows if truth == a and pred == b) for b in labels] for a in labels]
    total = len(rows)
    result: dict[str, Any] = {"count": total, "accuracy": sum(a == b for a, b in rows) / total if total else None,
                              "confusion_matrix": {"labels": list(labels), "values": matrix}}
    for label in labels:
        tp = sum(a == label and b == label for a, b in rows); fp = sum(a != label and b == label for a, b in rows); fn = sum(a == label and b != label for a, b in rows)
        precision = tp / (tp + fp) if tp + fp else None; recall = tp / (tp + fn) if tp + fn else None
        result[label] = {"precision": precision, "recall": recall,
                         "f1": 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None}
    if total:
        observed = result["accuracy"]
        truth_counts, pred_counts = Counter(a for a, _ in rows), Counter(b for _, b in rows)
        expected = sum(truth_counts[x] * pred_counts[x] for x in labels) / (total * total)
        result["cohens_kappa"] = (observed - expected) / (1 - expected) if expected != 1 else 1.0
        result["swap_error_rate"] = sum(a != b for a, b in rows) / total
    else:
        result.update(cohens_kappa=None, swap_error_rate=None)
    return result


def _evaluation_summary_impl() -> dict[str, Any]:
    """Evaluate only confirmed, gradable records; predictions remain distinct."""
    rows: list[tuple[str, str]] = []
    reviewed = corrected = tooth_correct = tooth_comparable = 0
    landmark_errors: list[float] = []
    rbl_errors: list[float] = []
    for path in STORE.glob("*/*/*.json") if STORE.exists() else ():
        record = json.loads(path.read_text(encoding="utf-8"))
        for tooth in record.get("teeth", []):
            reviewed += 1
            if tooth.get("model_fdi_suggestion") and tooth.get("expert_fdi"):
                tooth_comparable += 1
                tooth_correct += tooth["expert_fdi"] == tooth["model_fdi_suggestion"]
                corrected += tooth["expert_fdi"] != tooth["model_fdi_suggestion"]
            for surface in tooth.get("surfaces", []):
                if record["orientation"]["status"] == "verified" and tooth["tooth_verification_status"] == "confirmed" and surface["verification_status"] == "confirmed":
                    predicted = surface.get("model_surface_suggestion")
                    if predicted in ("mesial", "distal"):
                        rows.append((surface["surface"], predicted))
                    pairs = ((surface.get("cej", {}).get("point"), surface.get("model_cej")),
                             (surface.get("bone_crest", {}).get("point"), surface.get("model_bone_crest")),
                             (surface.get("root_apex", {}).get("point"), surface.get("model_root_apex")))
                    for expert_point, model_point in pairs:
                        if expert_point and model_point:
                            landmark_errors.append(math.hypot(expert_point["x"]-model_point["x"], expert_point["y"]-model_point["y"]))
                    if surface.get("rbl_percentage") is not None and surface.get("model_rbl_percentage") is not None:
                        rbl_errors.append(surface["rbl_percentage"]-surface["model_rbl_percentage"])
    return {"surface_association": _metrics(rows),
            "tooth_number_assignment_accuracy": tooth_correct / tooth_comparable if tooth_comparable else None,
            "landmark_mean_distance_error_px": sum(landmark_errors)/len(landmark_errors) if landmark_errors else None,
            "rbl_mae": sum(abs(x) for x in rbl_errors)/len(rbl_errors) if rbl_errors else None,
            "rbl_rmse": math.sqrt(sum(x*x for x in rbl_errors)/len(rbl_errors)) if rbl_errors else None,
            "tooth_records_reviewed": reviewed,
            "percentage_requiring_expert_correction": 100 * corrected / reviewed if reviewed else None,
            "subgroups": {"status": "reported when stored records include enough verified expert/model pairs",
                          "dimensions": ["arch", "patient_side", "root_type", "edge_status", "initial_orientation_status"]},
            "excluded": "Unresolved and ungradable records are excluded. Empty metrics mean no comparable expert/model pairs exist."}
