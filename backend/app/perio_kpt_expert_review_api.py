"""Blinded, append-only API for the 12-case Perio-KPT landmark review."""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "artifacts/perio-kpt-landmarks/gate2/measurement_readiness/expert_review_package"
REVIEWER_BUNDLE = PACKAGE / "reviewer_bundle"
COMPLETED = PACKAGE / "completed_reviews"
CASE_PATTERN = re.compile(r"Case\d{3}\Z")
REVIEWER_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
Answer = Literal["yes", "no", "uncertain", "not_applicable"]
Reason = Literal["cej_misplaced", "bone_level_misplaced", "root_or_apex_inappropriate",
                 "mesial_distal_reversed", "geometry_implausible", "landmark_not_visible",
                 "overlapping_anatomy", "low_image_quality", "crop_context_insufficient",
                 "surface_not_assessable", "definition_ambiguous", "other"]
QUESTIONS = ("cej_correct", "bone_level_correct", "root_landmark_appropriate",
             "mesial_distal_association_correct", "geometry_clinically_plausible", "surface_assessable")


class SurfaceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    surface: Literal["mesial", "distal"]
    cej_correct: Answer
    bone_level_correct: Answer
    root_landmark_appropriate: Answer
    mesial_distal_association_correct: Answer
    geometry_clinically_plausible: Answer
    surface_assessable: Answer
    reason_codes: list[Reason] = Field(default_factory=list)
    comments: str = Field(default="", max_length=5000)

    @model_validator(mode="after")
    def require_reason_for_problem(self):
        if any(getattr(self, q) in {"no", "uncertain"} for q in QUESTIONS) and not self.reason_codes:
            raise ValueError("At least one reason code is required for a no or uncertain answer")
        return self


class CaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol_version: Literal["1.0"] = "1.0"
    case_id: str
    anonymous_reviewer_id: str
    surfaces: list[SurfaceReview]
    completed_at_utc: str

    @model_validator(mode="after")
    def validate_identifiers_and_surfaces(self):
        if not CASE_PATTERN.fullmatch(self.case_id):
            raise ValueError("Invalid blinded case ID")
        if not REVIEWER_PATTERN.fullmatch(self.anonymous_reviewer_id):
            raise ValueError("Invalid reviewer ID")
        if len(self.surfaces) != 2 or {item.surface for item in self.surfaces} != {"mesial", "distal"}:
            raise ValueError("Exactly one mesial and one distal review are required")
        return self


router = APIRouter(prefix="/api/perio-kpt-expert-review", tags=["Perio-KPT blinded expert review"])


def _public_cases() -> list[dict[str, str]]:
    form = REVIEWER_BUNDLE / "review_form.csv"
    if not form.is_file():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Perio-KPT review package unavailable")
    cases: dict[str, str] = {}
    with form.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            case_id = row.get("case_id", "")
            if not CASE_PATTERN.fullmatch(case_id):
                raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Invalid blinded review manifest")
            cases.setdefault(case_id, row.get("root_class", ""))
    return [{"case_id": case_id, "root_class": root_class} for case_id, root_class in cases.items()]


def _known_case(case_id: str) -> bool:
    return bool(CASE_PATTERN.fullmatch(case_id)) and any(x["case_id"] == case_id for x in _public_cases())


@router.get("/manifest")
def manifest() -> dict:
    return {"protocol_version": "1.0", "title": "Blinded Perio-KPT landmark review",
            "instructions": "Review ground-truth landmark placement only. Do not diagnose disease, assign severity, or alter source labels.",
            "allowed_answers": ["yes", "no", "uncertain", "not_applicable"],
            "reason_codes": list(Reason.__args__), "cases": _public_cases()}


@router.get("/cases/{case_id}/image")
def case_image(case_id: str) -> FileResponse:
    if not _known_case(case_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Blinded case not found")
    path = (REVIEWER_BUNDLE / "images" / f"{case_id}.png").resolve()
    allowed = (REVIEWER_BUNDLE / "images").resolve()
    if path.parent != allowed or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Blinded case image unavailable")
    return FileResponse(path, media_type="image/png")


@router.get("/status/{reviewer_id}")
def review_status(reviewer_id: str) -> dict:
    if not REVIEWER_PATTERN.fullmatch(reviewer_id):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid reviewer ID")
    expected = {x["case_id"] for x in _public_cases()}
    folder = COMPLETED / reviewer_id
    completed = ({p.stem for p in folder.glob("Case*.json")} if folder.is_dir() else set()) & expected
    return {"reviewer_id": reviewer_id, "completed": len(completed), "total": len(expected),
            "completed_case_ids": sorted(completed)}


@router.post("/reviews", status_code=status.HTTP_201_CREATED)
def submit_review(review: CaseReview) -> dict:
    if not _known_case(review.case_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Blinded case not found")
    folder = COMPLETED / review.anonymous_reviewer_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{review.case_id}.json"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise HTTPException(status.HTTP_409_CONFLICT, "Review already locked") from error
    try:
        os.write(descriptor, review.model_dump_json(indent=2).encode("utf-8"))
    finally:
        os.close(descriptor)
    return {"saved": True, "locked": True, "case_id": review.case_id}
