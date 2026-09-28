"""Independent expert spatial annotation records and lossless mask files."""
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(os.getenv('EXPERT_ANNOTATION_DIR', Path(__file__).resolve().parents[2] / 'data' / 'expert_annotations'))
router = APIRouter(prefix='/api/annotations', tags=['expert spatial annotations'])


class Point(BaseModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class Record(BaseModel):
    id: str | None = None
    image_id: str
    tooth_instance_id: int | None = None
    finding_type: Literal['cej', 'apex', 'bone_level', 'bone_loss_region', 'other']
    annotation_tool: Literal['point', 'polyline', 'polygon', 'brush']
    points: list[Point] = Field(default_factory=list)
    mask_path: str | None = None
    status: Literal['draft', 'confirmed', 'uncertain', 'rejected', 'cannot_determine'] = 'draft'
    expert_comment: str = ''
    created_at: str | None = None
    updated_at: str | None = None


def _safe(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value):
        raise HTTPException(422, 'Invalid identifier')
    return value


def _read() -> dict[str, dict]:
    path = ROOT / 'annotations.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def _write(records: dict[str, dict]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    temp = ROOT / 'annotations.tmp'
    temp.write_text(json.dumps(records, indent=2), encoding='utf-8')
    temp.replace(ROOT / 'annotations.json')


@router.get('/{image_id}')
def list_records(image_id: str) -> list[dict]:
    _safe(image_id)
    return [record for record in _read().values() if record['image_id'] == image_id]


@router.post('')
def create_record(record: Record) -> dict:
    _safe(record.image_id)
    data = record.model_dump()
    data['id'] = str(uuid.uuid4())
    data['created_at'] = data['updated_at'] = datetime.now(timezone.utc).isoformat()
    data['mask_path'] = None
    records = _read()
    records[data['id']] = data
    _write(records)
    return data


@router.put('/{annotation_id}')
def update_record(annotation_id: str, record: Record) -> dict:
    _safe(annotation_id)
    records = _read()
    if annotation_id not in records:
        raise HTTPException(404, 'Annotation not found')
    if record.image_id != records[annotation_id]['image_id']:
        raise HTTPException(422, 'Image cannot be changed')
    data = record.model_dump()
    data.update(id=annotation_id, created_at=records[annotation_id]['created_at'], mask_path=records[annotation_id]['mask_path'], updated_at=datetime.now(timezone.utc).isoformat())
    records[annotation_id] = data
    _write(records)
    return data


@router.delete('/{annotation_id}', status_code=204)
def delete_record(annotation_id: str) -> None:
    _safe(annotation_id)
    records = _read()
    if annotation_id not in records:
        raise HTTPException(404, 'Annotation not found')
    record = records.pop(annotation_id)
    if record.get('mask_path'):
        (ROOT / record['mask_path']).unlink(missing_ok=True)
    _write(records)


@router.post('/{annotation_id}/mask')
async def upload_mask(annotation_id: str, file: UploadFile = File(...)) -> dict:
    _safe(annotation_id)
    records = _read()
    if annotation_id not in records:
        raise HTTPException(404, 'Annotation not found')
    raw = await file.read(30 * 1024 * 1024 + 1)
    if len(raw) > 30 * 1024 * 1024:
        raise HTTPException(413, 'Mask too large')
    if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
        raise HTTPException(422, 'Expected lossless PNG mask')
    mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.isin(mask, [0, 255]).all():
        raise HTTPException(422, 'Expected binary PNG mask')
    record = records[annotation_id]
    path = Path('masks') / f"image_{record['image_id']}" / f'{annotation_id}.png'
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), mask)
    record['mask_path'] = path.as_posix()
    record['updated_at'] = datetime.now(timezone.utc).isoformat()
    _write(records)
    return {'mask_path': record['mask_path'], 'width': mask.shape[1], 'height': mask.shape[0]}


@router.get('/{annotation_id}/mask')
def get_mask(annotation_id: str) -> FileResponse:
    _safe(annotation_id)
    record = _read().get(annotation_id)
    if not record or not record.get('mask_path'):
        raise HTTPException(404, 'Mask not found')
    return FileResponse(ROOT / record['mask_path'], media_type='image/png')
