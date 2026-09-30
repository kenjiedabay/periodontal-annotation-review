"""FastAPI adapter for the periodontal research PoC.

The in-memory store is deliberate for a thesis demonstration. Replace it with a
validated database and model service when moving beyond the prototype.
"""
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.annotation_storage import Annotation, StoredAnnotation, create, delete, get, mark_validated, update
from app.structural_audit.correspondence import analyze_record
from app.structural_audit.loader import find_validation_record
from app.tooth_segmentation import service as tooth_service
from app.anatomy_prediction import service as anatomy_service
from app.spatial_annotations import router as spatial_annotation_router
from app.review_foundation import router as review_foundation_router, assert_ai_access
from app.association_review import router as association_review_router, legacy_router as association_review_legacy_router
from app.surface_verification import router as surface_verification_router
from app.pilot_surface_review import router as pilot_surface_router
from app.ground_truth_pilot_api import router as ground_truth_pilot_router
from app.perio_kpt_expert_review_api import router as perio_kpt_expert_review_router
from app.perio_kpt_connected_inference import service as connected_perio_service

app = FastAPI(title="PerioLab Research API", version="0.1.0")
app.include_router(spatial_annotation_router)
app.include_router(review_foundation_router)
app.include_router(association_review_router)
app.include_router(association_review_legacy_router)
app.include_router(surface_verification_router)
app.include_router(pilot_surface_router)
app.include_router(ground_truth_pilot_router)
app.include_router(perio_kpt_expert_review_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post('/analysis/tooth-segmentation')
def tooth_segmentation(image_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    data = file.file.read(30 * 1024 * 1024 + 1)
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='Image exceeds 30 MB')
    assert_ai_access(image_id, data)
    original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if original is None:
        raise HTTPException(status_code=422, detail='Unsupported image')
    return tooth_service.predict(image_id, original)


@app.post('/analysis/anatomical-overlay')
def anatomical_overlay(image_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    data = file.file.read(30 * 1024 * 1024 + 1)
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='Image exceeds 30 MB')
    assert_ai_access(image_id, data)
    original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if original is None:
        raise HTTPException(status_code=422, detail='Unsupported image')
    return anatomy_service.predict(image_id, original)


@app.post('/analysis/perio-kpt-preview')
def perio_kpt_preview(image_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    data = file.file.read(30 * 1024 * 1024 + 1)
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='Image exceeds 30 MB')
    assert_ai_access(image_id, data)
    original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if original is None:
        raise HTTPException(status_code=422, detail='Unsupported image')
    return connected_perio_service.predict(image_id, original)


@app.post('/analysis/unified-model-review')
def unified_model_review(image_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """Run existing models without combining their datasets or checkpoints."""
    data = file.file.read(30 * 1024 * 1024 + 1)
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='Image exceeds 30 MB')
    assert_ai_access(image_id, data)
    original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if original is None:
        raise HTTPException(status_code=422, detail='Unsupported image')
    return {
        'image_id': image_id,
        'task': 'unified_research_annotation_review',
        'connected_perio_kpt': connected_perio_service.predict(image_id, original),
        'full_image_anatomy': anatomy_service.predict(image_id, original),
        'separation': {
            'datasets_merged': False,
            'checkpoints_modified': False,
            'expert_annotations_are_model_outputs': False,
        },
        'disclaimer': 'Experimental AI overlays for expert review only; not a diagnosis or treatment recommendation.',
    }

cases: dict[str, dict[str, Any]] = {}
DENPAR_VALIDATION_DIR = Path(os.getenv("DENPAR_VALIDATION_DIR", Path(__file__).resolve().parents[1] / "DenPAR Radiographs Dataset" / "Dataset" / "Validation"))


def _validation_image_path(image_id: str) -> Path:
    path = DENPAR_VALIDATION_DIR / "Images" / f"{image_id}.jpg"
    if not path.exists() or path.parent != (DENPAR_VALIDATION_DIR / "Images"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="DenPAR image not found")
    return path


@app.get("/structural-audit/{image_id}")
def structural_audit_record(image_id: str) -> dict[str, Any]:
    assert_ai_access(image_id)
    try:
        audit, record = find_validation_record(DENPAR_VALIDATION_DIR, image_id)
    except FileNotFoundError:
        # A filename alone does not establish that an uploaded image belongs to
        # the supplied Validation partition.
        return {
            "available": False,
            "image_id": image_id,
            "reason": "No DenPAR Validation structural-annotation record is available for this image ID.",
            "source_preserved": True,
        }
    return {
        "available": True,
        "image": {"image_id": record.image.image_id, "filename": record.image.filename, "width": record.image.width, "height": record.image.height},
        "tooth_masks": [{**mask, "url": f"/structural-audit/{image_id}/mask/tooth/{mask['filename']}"} for mask in record.tooth_masks],
        "radiograph_mask": {**record.radiograph_mask, "url": f"/structural-audit/{image_id}/mask/radiograph"} if record.radiograph_mask else None,
        "coco_annotations": record.coco_annotations,
        "keypoints": record.keypoints,
        "bone_lines": record.bone_lines,
        "characteristics": record.characteristics,
        "issues": [issue.as_dict() for issue in audit.issues if issue.image_id == image_id],
        "correspondence": analyze_record(record).as_dict(),
        "source_preserved": True,
    }


@app.get("/structural-audit/{image_id}/image")
def structural_audit_image(image_id: str) -> FileResponse:
    return FileResponse(_validation_image_path(image_id), media_type="image/jpeg")


@app.get("/structural-audit/{image_id}/bone-line-prediction")
def structural_audit_bone_prediction(image_id: str) -> FileResponse:
    assert_ai_access(image_id)
    if not image_id.isdecimal():
        raise HTTPException(status_code=400, detail="Invalid image ID")
    _validation_image_path(image_id)
    path = Path(__file__).resolve().parents[1] / "models" / "bone_line_unet_baseline" / "validation_predictions" / f"{image_id}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No prediction available for this Validation image")
    return FileResponse(path, media_type="image/png")


@app.get("/structural-audit/{image_id}/mask/radiograph")
def structural_audit_radiograph_mask(image_id: str) -> FileResponse:
    assert_ai_access(image_id)
    path = DENPAR_VALIDATION_DIR / "Masks (Radiograph-wise)" / f"{image_id}.png"
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Radiograph-wise mask not found")
    return FileResponse(path, media_type="image/png")


@app.get("/structural-audit/{image_id}/mask/tooth/{filename}")
def structural_audit_tooth_mask(image_id: str, filename: str) -> FileResponse:
    assert_ai_access(image_id)
    if Path(filename).name != filename or not filename.endswith(".png"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid mask filename")
    path = DENPAR_VALIDATION_DIR / "Masks (Tooth-wise)" / image_id / filename
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tooth-wise mask not found")
    return FileResponse(path, media_type="image/png")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "research-prototype"}


@app.post("/upload")
async def upload(image_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    original = await file.read()
    decoded = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return {"image_id": image_id, "accepted": False, "reason": "Unsupported image"}
    height, width = decoded.shape[:2]
    cases[image_id] = {"image_id": image_id, "filename": file.filename, "original_bytes": original, "width": width, "height": height, "created_at": now()}
    return {"image_id": image_id, "filename": file.filename, "width": width, "height": height, "original_preserved": True}


@app.post("/preprocess/{image_id}")
def preprocess(image_id: str, contrast: float = 1.0, brightness: int = 0, grayscale: bool = True) -> dict[str, Any]:
    case = cases[image_id]
    source = cv2.imdecode(np.frombuffer(case["original_bytes"], np.uint8), cv2.IMREAD_UNCHANGED)
    working = source.copy()
    if grayscale and len(working.shape) == 3:
        working = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    working = cv2.convertScaleAbs(working, alpha=contrast, beta=brightness)
    case["processed_shape"] = list(working.shape)
    return {"image_id": image_id, "operation": "opencv-display-copy", "processed_shape": list(working.shape), "original_preserved": True}


@app.post("/analyze/{image_id}")
def analyze(image_id: str) -> dict[str, Any]:
    if image_id not in cases:
        return {"image_id": image_id, "status": "not_found"}
    return {"image_id": image_id, "label": "PRELIMINARY SIMULATION", "possible_abnormality": "possible localized abnormality", "possible_region": "requires expert review", "clinically_validated": False, "model_backend": "replaceable-simulation-adapter"}


@app.get("/analysis/{image_id}")
def model_analysis(image_id: str) -> dict[str, Any]:
    """Expose model availability without fabricating predictions.

    A checkpoint alone is not treated as a prediction service. Inference adapters
    must be implemented and validated before this status can become available.
    """
    checkpoint_root = Path(__file__).resolve().parent / "ml" / "checkpoints"
    disease_checkpoint = checkpoint_root / "best.pt"
    severity_checkpoint = checkpoint_root / "severity" / "severity_best.pt"
    return {
        "image_id": image_id,
        "model_status": "unavailable",
        "disease_prediction": {"label": None, "confidence": None},
        "region_prediction": {"status": None, "confidence": None},
        "severity_prediction": {"label": None, "confidence": None},
        "structural_findings": [],
        "progression": {"status": "unsupported", "reason": "No longitudinal data or validated progression model is available."},
        "research_only": True,
        "checkpoint_inventory": {"disease": disease_checkpoint.exists(), "severity": severity_checkpoint.exists()},
        "message": "Inference adapters are not configured; no model prediction was generated.",
    }


@app.post("/annotations", response_model=StoredAnnotation, status_code=status.HTTP_201_CREATED)
def create_annotation(payload: Annotation) -> StoredAnnotation:
    try:
        return create(payload)
    except FileExistsError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Annotation already exists") from error


@app.get("/annotations/{image_id}", response_model=StoredAnnotation)
def get_annotation(image_id: str) -> StoredAnnotation:
    try:
        return get(image_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotation not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error


@app.get("/structural-analysis/{image_id}")
def structural_analysis(image_id: str) -> dict[str, Any]:
    """Return expert ground truth separately from unavailable model outputs."""
    try:
        annotation = get(image_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotation not found") from error
    ground_truth = None
    if annotation.expert_validated and annotation.validation_status == "validated":
        ground_truth = {
            "expert_validated": True,
            "findings": annotation.findings,
            "regions": [region.model_dump() for region in annotation.regions],
        }
    return {
        "image_id": image_id,
        "ground_truth": ground_truth,
        "preliminary_result": None,
        "model_prediction": None,
        "analysis_status": "ground_truth_only" if ground_truth else "model_unavailable",
    }


@app.put("/annotations/{image_id}", response_model=StoredAnnotation)
def update_annotation(image_id: str, payload: Annotation) -> StoredAnnotation:
    try:
        return update(image_id, payload)
    except FileNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotation not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error


@app.delete("/annotations/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_annotation(image_id: str) -> None:
    try:
        delete(image_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotation not found") from error
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error


@app.post("/validate/{image_id}")
def validate(image_id: str, expert_comments: str = "") -> dict[str, Any]:
    try:
        record = mark_validated(image_id, expert_comments)
    except FileNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotation not found") from error
    return {"image_id": image_id, "validation_status": "expert_validated", "ground_truth_source": "independent dental expert review", "record": record}


@app.get("/progression/{image_id}")
def progression(image_id: str) -> dict[str, Any]:
    return {"image_id": image_id, "label": "ESTIMATED POSSIBLE PROGRESSION", "status": "placeholder", "untreated": "possible structural deterioration", "controlled": "possible stabilization", "definitive_prediction": False, "guaranteed_outcome": False}


@app.get("/results/{image_id}")
def results(image_id: str) -> dict[str, Any]:
    try:
        annotation = get(image_id)
    except FileNotFoundError:
        annotation = None
    return {"image_id": image_id, "case": cases.get(image_id), "annotation": annotation, "research_only": True}
