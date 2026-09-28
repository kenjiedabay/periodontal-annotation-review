"""Append-only provenance and review records for periapical research images.

Existing annotation files and endpoints are deliberately independent of this store.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.preprocessing.config import load_config
from app.tooth_segmentation import CHECKPOINT, ROOT as PROJECT_ROOT, service as tooth_service

ROOT = Path(os.getenv('REVIEW_FOUNDATION_DIR', PROJECT_ROOT / 'data' / 'review_foundation'))
router = APIRouter(prefix='/api/review', tags=['versioned radiographic review'])
LOCK = threading.RLock()
IMAGE_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$')
AI_STATUS = 'AI-generated—awaiting expert review'
SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: str) -> str:
    if not IMAGE_ID.fullmatch(value):
        raise HTTPException(422, 'Invalid image ID')
    return value


def _folder(image_id: str) -> Path:
    return ROOT / 'images' / _safe(image_id)


def _records(image_id: str) -> list[dict[str, Any]]:
    folder = _folder(image_id) / 'records'
    if not folder.is_dir():
        return []
    records = [json.loads(path.read_text(encoding='utf-8')) for path in folder.glob('*.json')]
    return sorted(records, key=lambda item: (item['created_at'], item['record_id']))


def _record(image_id: str, record_id: str) -> dict[str, Any]:
    try:
        uuid.UUID(record_id)
    except ValueError as error:
        raise HTTPException(422, 'Invalid record ID') from error
    path = _folder(image_id) / 'records' / f'{record_id}.json'
    if not path.is_file():
        raise HTTPException(404, 'Record not found')
    return json.loads(path.read_text(encoding='utf-8'))


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    """Mutable review state only; immutable records use _append_json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary.open('x', encoding='utf-8') as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _append_json(path: Path, data: dict[str, Any]) -> None:
    """Publish a complete record without replacing an existing version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary.open('x', encoding='utf-8') as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())
    try:
        os.link(temporary, path)  # fails if the destination already exists
    finally:
        temporary.unlink(missing_ok=True)


def _source(image_id: str) -> dict[str, Any]:
    index = _folder(image_id) / 'source.json'
    if index.is_file():
        return json.loads(index.read_text(encoding='utf-8'))
    sources = [item for item in _records(image_id) if item['record_type'] == 'source_record']
    if not sources:
        raise HTTPException(404, 'Source record not found')
    return sources[0]


def _state(image_id: str) -> dict[str, Any]:
    source = _source(image_id)
    path = _folder(image_id) / 'review_state.json'
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    independent = [r for r in _records(image_id) if r['record_type'] == 'independent_expert_annotation']
    blind = source['review_mode'] == 'independent_evaluation'
    return {
        'image_id': image_id, 'mode': source['review_mode'],
        'independent_saved': bool(independent),
        'revealed': bool(state.get('revealed_at')) if blind else True,
        'can_reveal': bool(independent) if blind else True,
        'revealed_at': state.get('revealed_at'),
        'revealed_by': state.get('revealed_by'),
    }


def _can_view_ai(image_id: str) -> bool:
    state = _state(image_id)
    return state['mode'] == 'ai_assisted' or state['revealed']


def assert_ai_access(image_id: str, image_bytes: bytes | None = None) -> None:
    """Protect registered evaluation images on the legacy inference route too."""
    with LOCK:
        source = next((r for r in _records(image_id) if r['record_type'] == 'source_record'), None)
        if source and not _can_view_ai(image_id):
            raise HTTPException(403, 'Save an independent annotation before revealing AI output')
        if image_bytes is not None and ROOT.exists():
            digest = hashlib.sha256(image_bytes).hexdigest()
            for image_dir in (ROOT / 'images').glob('*') if (ROOT / 'images').exists() else ():
                if image_dir.name == image_id:
                    continue
                source_path = image_dir / 'source.json'
                if source_path.exists():
                    item = json.loads(source_path.read_text(encoding='utf-8'))
                    if item['image_hash'] == digest and not _can_view_ai(item['image_id']):
                        raise HTTPException(403, 'This evaluation image is awaiting an independent annotation')


class PixelPoint(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class Landmark(BaseModel):
    state: Literal['visible', 'not_visible', 'uncertain', 'not_applicable'] = 'uncertain'
    point: PixelPoint | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainty_reason: str | None = None

    @model_validator(mode='after')
    def check_point(self) -> 'Landmark':
        if (self.state == 'visible') != (self.point is not None):
            raise ValueError('Only a visible landmark may have a coordinate, and it must have one')
        return self


class RootApex(BaseModel):
    anatomical_label: str | None = None
    landmark: Landmark = Field(default_factory=Landmark)


class ToothAnatomy(BaseModel):
    tooth_id: str | None = None
    instance_id: int | None = None
    orientation_status: Literal['confirmed', 'provisional', 'uncertain'] = 'uncertain'
    tooth_mask: str | None = None  # immutable PNG data URL in a record snapshot
    tooth_bbox: tuple[float, float, float, float] | None = None
    crop_box: tuple[float, float, float, float] | None = None
    crop_margin_pixels: float | None = Field(default=None, ge=0)
    cej_mesial: Landmark = Field(default_factory=Landmark)
    cej_distal: Landmark = Field(default_factory=Landmark)
    bone_mesial: Landmark = Field(default_factory=Landmark)
    bone_distal: Landmark = Field(default_factory=Landmark)
    root_apices: list[RootApex] = Field(default_factory=list)
    mesial_apex_index: int | None = None
    distal_apex_index: int | None = None
    mesial_surface_status: Literal['provisional', 'uncertain', 'not_measurable'] = 'uncertain'
    distal_surface_status: Literal['provisional', 'uncertain', 'not_measurable'] = 'uncertain'
    confidence: float | None = Field(default=None, ge=0, le=1)
    uncertainty_reasons: list[str] = Field(default_factory=list)


class AnnotationItem(BaseModel):
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tooth_instance_id: int | None = None
    finding_type: Literal['cej', 'apex', 'bone_level', 'bone_loss_region', 'other']
    annotation_tool: Literal['point', 'polyline', 'polygon', 'brush']
    points: list[PixelPoint] = Field(default_factory=list)
    mask_data_url: str | None = None
    status: Literal['draft', 'confirmed', 'uncertain', 'rejected', 'cannot_determine'] = 'draft'
    expert_comment: str = ''


class AnnotationInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    created_by: str = Field(min_length=1)
    parent_record_id: str | None = None
    raw_prediction_record_id: str | None = None
    independent_annotation_record_id: str | None = None
    items: list[AnnotationItem] = Field(default_factory=list)
    teeth: list[ToothAnatomy] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    notes: str = ''
    uncertainty_reasons: list[str] = Field(default_factory=list)
    review_status: Literal['draft', 'uncertain', 'needs_revision'] = 'draft'


class ApprovalInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    created_by: str = Field(min_length=1)
    approved_record_id: str
    decision: Literal['approved', 'rejected', 'uncertain', 'needs_revision']
    notes: str = ''
    uncertainty_reasons: list[str] = Field(default_factory=list)


def _base(source: dict[str, Any], kind: str, created_by: str, parent: str | None,
          status: str, uncertainty: list[str] | None = None) -> dict[str, Any]:
    return {
        'schema_version': SCHEMA_VERSION, 'record_id': str(uuid.uuid4()),
        'record_type': kind, 'version': 1, 'image_id': source['image_id'],
        'image_hash': source['image_hash'], 'source_dataset': source['source_dataset'],
        'source_partition': source['source_partition'],
        'original_width': source['original_width'], 'original_height': source['original_height'],
        'coordinate_space': 'original_image_pixels', 'coordinate_origin': 'top_left',
        'created_at': _now(), 'created_by': created_by, 'parent_record_id': parent,
        'model_name': None, 'model_version': None, 'review_status': status,
        'uncertainty_reasons': uncertainty or [],
    }


def _save(record: dict[str, Any]) -> dict[str, Any]:
    _append_json(_folder(record['image_id']) / 'records' / f"{record['record_id']}.json", record)
    return record


def _denpar_path(image_id: str, partition: str) -> Path:
    root = PROJECT_ROOT / 'DenPAR Radiographs Dataset' / 'Dataset'
    canonical = ({'Training': root / 'Training' / 'Images',
                  'Validation': root / 'Validation' / 'Images',
                  'Testing': root / 'Testing' / 'Images'}[partition]) / f'{image_id}.jpg'
    if canonical.is_file():
        return canonical
    legacy = {'Training': PROJECT_ROOT / 'dataset' / 'raw',
              'Validation': root / 'Validation' / 'Images',
              'Testing': root / 'Images'}[partition]
    return legacy / f'{image_id}.jpg'


def _denpar_match(image_id: str, digest: str) -> tuple[str, Path] | None:
    for partition in ('Training', 'Validation', 'Testing'):
        path = _denpar_path(image_id, partition)
        if path.is_file() and _sha256_file(path) == digest:
            return partition, path
    return None


def _source_annotations(image_id: str, partition: str) -> list[dict[str, str]]:
    base = PROJECT_ROOT / 'DenPAR Radiographs Dataset' / 'Dataset' / partition
    dataset = PROJECT_ROOT / 'DenPAR Radiographs Dataset' / 'Dataset'
    suffix = {'Training': 'train', 'Validation': 'valid', 'Testing': 'test'}[partition]
    candidates = [
        ('keypoints', base / 'Key Points Annotations' / f'{image_id}.json'),
        ('bone_lines', base / 'Bone Level Annotations' / f'{image_id}.json'),
        ('radiograph_mask', base / 'Masks (Radiograph-wise)' / f'{image_id}.png'),
        ('tooth_instances_coco', base / 'Masks (Tooth-wise)' / f'coco_format_instances_{suffix}.json'),
        ('bone_lines_coco', base / 'Bone Level Annotations' / f'coco_format_bonelines_{suffix}.json'),
        ('radiograph_characteristics', dataset / 'Characteristics of radiographs included.xlsx'),
    ]
    candidates.extend(('tooth_mask', path) for path in sorted((base / 'Masks (Tooth-wise)' / image_id).glob('*.png')))
    return [{'kind': kind, 'path': str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'),
             'sha256': _sha256_file(path)}
            for kind, path in candidates if path.is_file()]


@router.post('/sources', status_code=201)
async def create_source(image_id: str = Form(...), review_mode: Literal['independent_evaluation', 'ai_assisted'] = Form(...),
                        created_by: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
    _safe(image_id)
    data = await file.read(30 * 1024 * 1024 + 1)
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(413, 'Image exceeds 30 MB')
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise HTTPException(422, 'Unsupported image')
    digest = hashlib.sha256(data).hexdigest()
    with LOCK:
        image_root = ROOT / 'images'
        for image_dir in image_root.glob('*') if image_root.exists() else ():
            if image_dir.name == image_id:
                continue
            index = image_dir / 'source.json'
            if index.is_file() and json.loads(index.read_text(encoding='utf-8'))['image_hash'] == digest:
                # Older Image Review builds prefixed library IDs (for example
                # ``case-Validation-496``). Permit re-registering that same
                # image under its canonical source ID after the ID fix; the
                # existing record remains untouched and locked assessments are
                # preserved.
                if not (image_id.isdigit() and image_dir.name.startswith('case-')):
                    raise HTTPException(409, 'This image is already registered under another image ID')
        current = [r for r in _records(image_id) if r['record_type'] == 'source_record']
        if current:
            if current[0]['image_hash'] != digest:
                raise HTTPException(409, 'Image ID already registered with a different image')
            # A source's original review mode is immutable. Reopening the same
            # bytes resumes that existing workflow instead of failing because
            # the intake selector currently has a different value.
            return current[0]
        denpar = _denpar_match(image_id, digest)
        if denpar:
            partition, path = denpar
            dataset, reference, stored = 'DenPAR', str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'), False
            annotations = _source_annotations(image_id, partition)
        else:
            partition, dataset, annotations, stored = 'Unassigned', 'uploaded', [], True
            suffix = '.png' if data.startswith(b'\x89PNG') else '.jpg'
            path = ROOT / 'source_images' / f'{digest}{suffix}'
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
                temporary.write_bytes(data)
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass
                finally:
                    temporary.unlink(missing_ok=True)
            reference = str(path.relative_to(ROOT)).replace('\\', '/')
        height, width = decoded.shape[:2]
        record = {
            'schema_version': SCHEMA_VERSION, 'record_id': str(uuid.uuid4()),
            'record_type': 'source_record', 'version': 1, 'image_id': image_id,
            'image_hash': digest, 'source_dataset': dataset, 'source_partition': partition,
            'original_width': width, 'original_height': height,
            'coordinate_space': 'original_image_pixels', 'coordinate_origin': 'top_left',
            'created_at': _now(), 'created_by': created_by, 'parent_record_id': None,
            'model_name': None, 'model_version': None, 'review_status': 'unreviewed',
            'uncertainty_reasons': [], 'review_mode': review_mode,
            'original_filename': file.filename, 'source_image_ref': reference,
            'stored_copy': stored, 'source_annotations': annotations,
        }
        _save(record)
        _atomic_json(_folder(image_id) / 'source.json', record)
        return record


@router.get('/sources/{image_id}')
def get_source(image_id: str) -> dict[str, Any]:
    return _source(image_id)


@router.get('/sources/{image_id}/image')
def get_source_image(image_id: str) -> FileResponse:
    source = _source(image_id)
    path = (ROOT if source['stored_copy'] else PROJECT_ROOT) / source['source_image_ref']
    return FileResponse(path)


@router.get('/sources/{image_id}/state')
def get_review_state(image_id: str) -> dict[str, Any]:
    return _state(image_id)


@router.post('/sources/{image_id}/reveal')
def reveal_ai(image_id: str, created_by: str = Form(...)) -> dict[str, Any]:
    with LOCK:
        state = _state(image_id)
        if not state['can_reveal']:
            raise HTTPException(403, 'Save an independent annotation before revealing AI output')
        if state['mode'] == 'independent_evaluation' and not state['revealed']:
            _atomic_json(_folder(image_id) / 'review_state.json',
                         {'revealed_at': _now(), 'revealed_by': created_by})
        return _state(image_id)


@router.post('/sources/{image_id}/predictions', status_code=201)
def create_prediction(image_id: str, created_by: str = Form('system')) -> dict[str, Any]:
    with LOCK:
        source = _source(image_id)
        path = (ROOT if source['stored_copy'] else PROJECT_ROOT) / source['source_image_ref']
        original = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if original is None:
            raise HTTPException(422, 'Source image unavailable')
        result = tooth_service.predict(image_id, original)
        if result['model_status'] != 'available':
            raise HTTPException(503, 'Mask R-CNN unavailable; no prediction saved')
        width, height = source['original_width'], source['original_height']
        scale = min(1024 / width, 1024 / height)
        rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
        offset_x, offset_y = (1024-rw)//2, (1024-rh)//2
        record = _base(source, 'raw_ai_prediction', created_by, source['record_id'], AI_STATUS)
        record.update(model_name='Mask R-CNN ResNet-50 FPN', model_version=result['model_version'],
                      checkpoint_identity={'path': str(CHECKPOINT.relative_to(PROJECT_ROOT) if CHECKPOINT.is_relative_to(PROJECT_ROOT) else CHECKPOINT.name).replace('\\', '/'),
                                           'sha256': _sha256_file(CHECKPOINT)},
                      prediction_timestamp=record['created_at'],
                      preprocessing=load_config().as_dict(),
                      coordinate_transform={'model_canvas': [1024, 1024], 'resized_image': [rw, rh],
                                            'pad_offset': [offset_x, offset_y],
                                            'stored_output_space': 'original_image_pixels'},
                      raw_model_output={**result, 'ground_truth': None},
                      teeth=[ToothAnatomy(instance_id=item['instance_id'], tooth_mask=item['mask_url'],
                                         tooth_bbox=tuple(item['bbox']), confidence=item['confidence'],
                                         uncertainty_reasons=['tooth_identity_and_orientation_unconfirmed']).model_dump()
                             for item in result['instances']],
                      uncertainty_reasons=['landmark_and_surface_relationships_not_predicted'])
        _save(record)
        return record if _can_view_ai(image_id) else {
            'record_id': record['record_id'], 'record_type': record['record_type'],
            'review_status': 'hidden_until_independent_annotation_and_reveal',
            'image_id': image_id,
        }


@router.get('/sources/{image_id}/predictions/{record_id}')
def get_prediction(image_id: str, record_id: str) -> dict[str, Any]:
    if not _can_view_ai(image_id):
        raise HTTPException(403, 'AI output hidden during independent review')
    record = _record(image_id, record_id)
    if record['record_type'] != 'raw_ai_prediction':
        raise HTTPException(422, 'Record is not a raw prediction')
    return record


def _check_items(payload: AnnotationInput, source: dict[str, Any]) -> None:
    if not payload.items and not payload.teeth and not payload.uncertainty_reasons:
        raise HTTPException(422, 'Add an annotation or an uncertainty reason before saving')
    width, height = source['original_width'], source['original_height']
    for item in payload.items:
        for point in item.points:
            if point.x > width or point.y > height:
                raise HTTPException(422, 'Point outside original image')
        if item.mask_data_url and not item.mask_data_url.startswith('data:image/png;base64,'):
            raise HTTPException(422, 'Mask must be a PNG data URL')
    for tooth in payload.teeth:
        landmarks = [tooth.cej_mesial, tooth.cej_distal, tooth.bone_mesial, tooth.bone_distal]
        landmarks += [root.landmark for root in tooth.root_apices]
        for landmark in landmarks:
            if landmark.point and (landmark.point.x > width or landmark.point.y > height):
                raise HTTPException(422, 'Landmark outside original image')
        if tooth.orientation_status == 'uncertain' and (tooth.mesial_surface_status == 'provisional' or tooth.distal_surface_status == 'provisional'):
            raise HTTPException(422, 'Surface relationship cannot be provisional with uncertain orientation')


def _annotation(image_id: str, payload: AnnotationInput, kind: str) -> dict[str, Any]:
    with LOCK:
        source = _source(image_id)
        _check_items(payload, source)
        state = _state(image_id)
        records = _records(image_id)
        prior = [r for r in records if r['record_type'] == kind]
        latest = max(prior, key=lambda item: item['version']) if prior else None
        if kind == 'independent_expert_annotation':
            if state['mode'] == 'independent_evaluation' and state['revealed']:
                raise HTTPException(409, 'Independent review is frozen after AI reveal')
            if payload.raw_prediction_record_id or payload.independent_annotation_record_id:
                raise HTTPException(422, 'Independent annotation cannot reference AI output')
            expected_parent = latest['record_id'] if latest else source['record_id']
        else:
            if not state['revealed']:
                raise HTTPException(403, 'Reveal AI after independent review before correcting')
            if not payload.raw_prediction_record_id:
                raise HTTPException(422, 'Correction requires a raw prediction reference')
            raw = _record(image_id, payload.raw_prediction_record_id)
            if raw['record_type'] != 'raw_ai_prediction':
                raise HTTPException(422, 'Invalid raw prediction reference')
            independents = [r for r in records if r['record_type'] == 'independent_expert_annotation']
            if independents:
                independent = _record(image_id, payload.independent_annotation_record_id or '')
                if independent['record_type'] != 'independent_expert_annotation':
                    raise HTTPException(422, 'Invalid independent annotation reference')
            elif payload.independent_annotation_record_id:
                raise HTTPException(422, 'Independent annotation reference not found')
            expected_parent = latest['record_id'] if latest else raw['record_id']
        if payload.parent_record_id not in (None, expected_parent):
            raise HTTPException(409, 'Parent must reference the latest version')
        record = _base(source, kind, payload.created_by, expected_parent,
                       payload.review_status, payload.uncertainty_reasons)
        record.update(version=(latest['version'] + 1 if latest else 1),
                      items=[item.model_dump() for item in payload.items],
                      teeth=[tooth.model_dump() for tooth in payload.teeth],
                      findings=payload.findings, notes=payload.notes)
        if kind == 'expert_correction':
            record.update(raw_prediction_record_id=payload.raw_prediction_record_id,
                          independent_annotation_record_id=payload.independent_annotation_record_id,
                          model_name=raw['model_name'], model_version=raw['model_version'])
        return _save(record)


@router.post('/sources/{image_id}/independent-annotations', status_code=201)
def create_independent_annotation(image_id: str, payload: AnnotationInput) -> dict[str, Any]:
    return _annotation(image_id, payload, 'independent_expert_annotation')


@router.post('/sources/{image_id}/corrections', status_code=201)
def create_correction(image_id: str, payload: AnnotationInput) -> dict[str, Any]:
    return _annotation(image_id, payload, 'expert_correction')


@router.get('/sources/{image_id}/annotations/{record_id}')
def get_annotation_record(image_id: str, record_id: str) -> dict[str, Any]:
    record = _record(image_id, record_id)
    if record['record_type'] not in ('independent_expert_annotation', 'expert_correction'):
        raise HTTPException(422, 'Record is not an annotation')
    if record['record_type'] == 'expert_correction' and not _can_view_ai(image_id):
        raise HTTPException(403, 'AI correction hidden during independent review')
    return record


@router.post('/sources/{image_id}/approvals', status_code=201)
def create_approval(image_id: str, payload: ApprovalInput) -> dict[str, Any]:
    with LOCK:
        source = _source(image_id)
        if source['review_mode'] == 'independent_evaluation' and not _can_view_ai(image_id):
            raise HTTPException(403, 'Reveal AI after independent review before final decision')
        approved = _record(image_id, payload.approved_record_id)
        if approved['record_type'] not in ('independent_expert_annotation', 'expert_correction'):
            raise HTTPException(422, 'Approval must reference an annotation or correction')
        if approved['record_type'] == 'expert_correction' and not _can_view_ai(image_id):
            raise HTTPException(403, 'Cannot approve hidden AI correction')
        previous = [r for r in _records(image_id) if r['record_type'] == 'final_approval']
        latest_approval = max(previous, key=lambda item: item['version']) if previous else None
        record = _base(source, 'final_approval', payload.created_by, approved['record_id'],
                       payload.decision, payload.uncertainty_reasons)
        record.update(version=len(previous) + 1, approved_record_id=approved['record_id'],
                      model_name=approved.get('model_name'), model_version=approved.get('model_version'),
                      supersedes_approval_record_id=latest_approval['record_id'] if latest_approval else None,
                      decision=payload.decision, notes=payload.notes)
        return _save(record)


@router.get('/sources/{image_id}/approvals/{record_id}')
def get_approval(image_id: str, record_id: str) -> dict[str, Any]:
    record = _record(image_id, record_id)
    if record['record_type'] != 'final_approval':
        raise HTTPException(422, 'Record is not a final approval')
    return record


@router.get('/sources/{image_id}/history')
def get_history(image_id: str) -> dict[str, Any]:
    state = _state(image_id)
    records = _records(image_id)
    if not _can_view_ai(image_id):
        records = [r for r in records if r['record_type'] in ('source_record', 'independent_expert_annotation')]
    return {'image_id': image_id, 'review_state': state, 'records': records}
