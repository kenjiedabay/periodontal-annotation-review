"""Focused persistence and mask checks for independent spatial annotations."""
import io
import numpy as np
import cv2
import asyncio
from starlette.datastructures import UploadFile

from app import spatial_annotations as storage


def test_multiple_findings_and_binary_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'ROOT', tmp_path)
    base = dict(image_id='1002', tooth_instance_id=4, points=[storage.Point(x=412.5, y=228.3)], status='draft', expert_comment='')
    first = storage.create_record(storage.Record(**base, finding_type='cej', annotation_tool='point'))
    second = storage.create_record(storage.Record(**{**base, 'points': []}, finding_type='bone_loss_region', annotation_tool='brush'))
    third = storage.create_record(storage.Record(**base, finding_type='cej', annotation_tool='point'))
    assert first['id'] != second['id']
    assert third['id'] != first['id']
    assert len(storage.list_records('1002')) == 3
    mask = np.zeros((19, 31), dtype=np.uint8)
    mask[3:8, 9:13] = 255
    encoded = cv2.imencode('.png', mask)[1].tobytes()
    uploaded = asyncio.run(storage.upload_mask(second['id'], UploadFile(filename='mask.png', file=io.BytesIO(encoded))))
    assert (uploaded['width'], uploaded['height']) == (31, 19)
    loaded = cv2.imread(str(tmp_path / uploaded['mask_path']), cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(loaded, mask)
    first['status'] = 'confirmed'
    assert storage.update_record(first['id'], storage.Record(**first))['status'] == 'confirmed'
    storage.delete_record(first['id'])
    assert len(storage.list_records('1002')) == 2
