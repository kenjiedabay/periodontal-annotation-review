"""Pydantic-validated JSON storage for expert radiograph annotations."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

IMAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
ANNOTATIONS_DIR = Path(os.getenv("ANNOTATIONS_DIR", Path(__file__).resolve().parents[2] / "dataset" / "annotations"))
ANNOTATION_FIELDS = {"image_id", "disease_status", "affected_teeth", "severity", "findings", "regions", "expert_comment", "expert_validated", "validation_status", "validation_timestamp"}


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["bounding_box", "polygon", "freehand"]
    coordinates: list[list[float]] | list[float] = Field(default_factory=list)


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1, max_length=128)
    disease_status: Literal["present", "absent", "uncertain"]
    affected_teeth: list[str] = Field(default_factory=list, max_length=64)
    severity: Literal["mild", "moderate", "severe", "cannot_determine"]
    findings: list[str] = Field(default_factory=list, max_length=100)
    regions: list[Region] = Field(default_factory=list, max_length=100)
    expert_comment: str = Field(default="", max_length=10000)
    expert_validated: bool = False
    validation_status: Literal["pending", "validated", "needs_revision", "uncertain"] = "pending"
    validation_timestamp: str | None = None

    @field_validator("image_id")
    @classmethod
    def validate_image_id(cls, value: str) -> str:
        if not IMAGE_ID_PATTERN.fullmatch(value):
            raise ValueError("image_id may contain only letters, numbers, underscores, and hyphens")
        return value

    @field_validator("disease_status", "severity", mode="before")
    @classmethod
    def normalize_label(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return value.strip().lower().replace(" ", "_")

    @field_validator("affected_teeth", mode="before")
    @classmethod
    def normalize_teeth(cls, value: Any) -> Any:
        if value is None:
            return []
        return [str(tooth).strip() for tooth in value]

    @field_validator("findings", mode="before")
    @classmethod
    def normalize_findings(cls, value: Any) -> Any:
        if value is None:
            return []
        return [str(finding).strip() for finding in value if str(finding).strip()]


class AnnotationVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    saved_at: str
    annotation: Annotation


class StoredAnnotation(Annotation):
    version: int = 1
    created_at: str
    updated_at: str
    version_history: list[AnnotationVersion] = Field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(image_id: str) -> Path:
    if not IMAGE_ID_PATTERN.fullmatch(image_id):
        raise ValueError("Invalid image_id")
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return ANNOTATIONS_DIR / f"{image_id}.json"


def _read(path: Path) -> StoredAnnotation:
    return StoredAnnotation.model_validate_json(path.read_text(encoding="utf-8"))


def _write(record: StoredAnnotation) -> StoredAnnotation:
    path = _path(record.image_id)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return record


def create(annotation: Annotation) -> StoredAnnotation:
    path = _path(annotation.image_id)
    if path.exists():
        raise FileExistsError(annotation.image_id)
    timestamp = _now()
    return _write(StoredAnnotation(**annotation.model_dump(), created_at=timestamp, updated_at=timestamp))


def get(image_id: str) -> StoredAnnotation:
    path = _path(image_id)
    if not path.exists():
        raise FileNotFoundError(image_id)
    return _read(path)


def update(image_id: str, annotation: Annotation) -> StoredAnnotation:
    current = get(image_id)
    if annotation.image_id != image_id:
        raise ValueError("Path image_id must match body image_id")
    timestamp = _now()
    previous = Annotation.model_validate(current.model_dump(include=ANNOTATION_FIELDS))
    history = [*current.version_history, AnnotationVersion(version=current.version, saved_at=current.updated_at, annotation=previous)]
    updated = StoredAnnotation(
        **annotation.model_dump(),
        version=current.version + 1,
        created_at=current.created_at,
        updated_at=timestamp,
        version_history=history,
    )
    return _write(updated)


def delete(image_id: str) -> None:
    path = _path(image_id)
    if not path.exists():
        raise FileNotFoundError(image_id)
    path.unlink()


def mark_validated(image_id: str, expert_comment: str = "") -> StoredAnnotation:
    current = get(image_id)
    payload = Annotation.model_validate({**current.model_dump(include=ANNOTATION_FIELDS), "expert_validated": True, "expert_comment": expert_comment or current.expert_comment})
    return update(image_id, payload)
