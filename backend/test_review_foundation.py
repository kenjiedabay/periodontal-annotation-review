"""End-to-end checks for immutable review history and server-side blinding."""
import hashlib
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import annotation_storage, review_foundation
from main import app


def _image() -> bytes:
    pixels = np.full((24, 32), 150, dtype=np.uint8)
    return cv2.imencode('.png', pixels)[1].tobytes()


def _register(client, image_id='case_a', mode='independent_evaluation'):
    return client.post('/api/review/sources', data={
        'image_id': image_id, 'review_mode': mode, 'created_by': 'intake',
    }, files={'file': (f'{image_id}.png', _image(), 'image/png')})


def _item(x=8):
    return {'item_id': 'd4ebdb96-a44e-497d-9327-2f46f0bf6e42', 'finding_type': 'cej',
            'annotation_tool': 'point', 'points': [{'x': x, 'y': 10}], 'status': 'draft',
            'expert_comment': '', 'tooth_instance_id': 1}


def test_denpar_source_provenance_uses_official_partition_without_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(review_foundation, 'ROOT', tmp_path / 'review')
    image = review_foundation.PROJECT_ROOT / 'DenPAR Radiographs Dataset' / 'Dataset' / 'Validation' / 'Images' / '1002.jpg'
    original_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    with TestClient(app) as client:
        response = client.post('/api/review/sources', data={
            'image_id': '1002', 'review_mode': 'ai_assisted', 'created_by': 'intake',
        }, files={'file': ('1002.jpg', image.read_bytes(), 'image/jpeg')})
        assert response.status_code == 201
        source = response.json()
        assert source['source_dataset'] == 'DenPAR' and source['source_partition'] == 'Validation'
        assert source['stored_copy'] is False
        assert {'keypoints', 'bone_lines', 'radiograph_mask', 'tooth_mask',
                'tooth_instances_coco', 'radiograph_characteristics'} <= {
            item['kind'] for item in source['source_annotations']}
        assert not (review_foundation.ROOT / 'source_images').exists()
        assert hashlib.sha256(image.read_bytes()).hexdigest() == original_hash


@pytest.mark.parametrize('landmark_state', ['not_visible', 'uncertain', 'not_applicable'])
def test_nonvisible_landmarks_reject_coordinates(tmp_path, monkeypatch, landmark_state):
    monkeypatch.setattr(review_foundation, 'ROOT', tmp_path / 'review')
    with TestClient(app) as client:
        source = _register(client).json()
        payload = {'created_by': 'reviewer-1', 'parent_record_id': source['record_id'],
                   'items': [_item()], 'teeth': [{
                       'cej_mesial': {'state': landmark_state, 'point': {'x': 8, 'y': 10}}}]}
        assert client.post('/api/review/sources/case_a/independent-annotations', json=payload).status_code == 422
        assert client.get('/api/review/sources/case_a/state').json()['independent_saved'] is False


def test_routine_ai_assisted_review_is_available_without_independent_annotation(tmp_path, monkeypatch):
    monkeypatch.setattr(review_foundation, 'ROOT', tmp_path / 'review')
    fake_checkpoint = tmp_path / 'checkpoint.pt'
    fake_checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(review_foundation, 'CHECKPOINT', fake_checkpoint)
    monkeypatch.setattr(review_foundation.tooth_service, 'predict', lambda image_id, original: {
        'image_id': image_id, 'task': 'tooth_instance_segmentation', 'model_status': 'available',
        'model_version': 'maskrcnn_epoch8', 'instances': [], 'width': original.shape[1],
        'height': original.shape[0], 'ground_truth': None})
    with TestClient(app) as client:
        source = _register(client, mode='ai_assisted')
        assert source.status_code == 201
        state = client.get('/api/review/sources/case_a/state').json()
        assert state['mode'] == 'ai_assisted' and state['revealed'] is True
        assert state['independent_saved'] is False
        prediction = client.post('/api/review/sources/case_a/predictions')
        assert prediction.status_code == 201
        assert prediction.json()['raw_model_output']['model_status'] == 'available'
        record_id = prediction.json()['record_id']
        assert client.get(f'/api/review/sources/case_a/predictions/{record_id}').status_code == 200
        assert client.post('/analysis/tooth-segmentation?image_id=case_a',
                           files={'file': ('case_a.png', _image())}).status_code == 200


def test_reopening_source_resumes_original_review_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(review_foundation, 'ROOT', tmp_path / 'review')
    with TestClient(app) as client:
        original = _register(client, mode='independent_evaluation')
        reopened = _register(client, mode='ai_assisted')
        assert original.status_code == 201
        assert reopened.status_code == 201
        assert reopened.json()['record_id'] == original.json()['record_id']
        assert reopened.json()['review_mode'] == 'independent_evaluation'
        assert client.get('/api/review/sources/case_a/state').json()['mode'] == 'independent_evaluation'


def test_versioned_review_blinding_and_legacy_compatibility(tmp_path, monkeypatch):
    monkeypatch.setattr(review_foundation, 'ROOT', tmp_path / 'review')
    monkeypatch.setattr(annotation_storage, 'ANNOTATIONS_DIR', tmp_path / 'legacy')
    fake_checkpoint = tmp_path / 'checkpoint.pt'
    fake_checkpoint.write_bytes(b'existing checkpoint identity fixture')
    monkeypatch.setattr(review_foundation, 'CHECKPOINT', fake_checkpoint)

    def fake_predict(image_id, original):
        return {'image_id': image_id, 'task': 'tooth_instance_segmentation',
                'model_status': 'available', 'model_version': 'maskrcnn_epoch8',
                'confidence_threshold': .5, 'mask_threshold': .5,
                'instances': [{'instance_id': 1, 'confidence': .91,
                               'bbox': [3, 4, 20, 22], 'mask_url': 'data:image/png;base64,AA=='}],
                'width': original.shape[1], 'height': original.shape[0],
                'coordinate_space': 'original_image', 'bbox_format': 'xyxy',
                'ground_truth': {'split': 'Validation', 'mask_urls': ['must-not-copy']}}

    monkeypatch.setattr(review_foundation.tooth_service, 'predict', fake_predict)
    with TestClient(app) as client:
        source_response = _register(client)
        assert source_response.status_code == 201
        source = source_response.json()
        assert source['record_type'] == 'source_record'
        assert source['schema_version'] == 1
        assert source['image_hash'] == hashlib.sha256(_image()).hexdigest()
        assert (source['original_width'], source['original_height']) == (32, 24)
        assert (source['coordinate_space'], source['coordinate_origin']) == ('original_image_pixels', 'top_left')
        assert client.get('/api/review/sources/case_a').json()['record_id'] == source['record_id']
        assert _register(client, image_id='alias', mode='ai_assisted').status_code == 409

        raw_hidden = client.post('/api/review/sources/case_a/predictions', data={'created_by': 'service'})
        assert raw_hidden.status_code == 201
        raw_id = raw_hidden.json()['record_id']
        assert 'raw_model_output' not in raw_hidden.json()
        assert client.get(f'/api/review/sources/case_a/predictions/{raw_id}').status_code == 403
        assert client.post('/api/review/sources/case_a/reveal', data={'created_by': 'reviewer-1'}).status_code == 403
        assert client.post('/analysis/tooth-segmentation?image_id=case_a', files={'file': ('case_a.png', _image())}).status_code == 403
        assert client.post('/analysis/tooth-segmentation?image_id=alias', files={'file': ('alias.png', _image())}).status_code == 403
        assert client.get('/api/review/sources/case_a/history').json()['records'] == [source]

        invalid = {'created_by': 'reviewer-1', 'items': [_item()], 'teeth': [{
            'cej_mesial': {'state': 'not_visible', 'point': {'x': 8, 'y': 10}}}]}
        assert client.post('/api/review/sources/case_a/independent-annotations', json=invalid).status_code == 422
        invalid_parent = {'created_by': 'reviewer-1', 'parent_record_id': 'e35cb900-dbdd-4205-9ec8-07b1ff241c2b', 'items': [_item()]}
        assert client.post('/api/review/sources/case_a/independent-annotations', json=invalid_parent).status_code == 409

        independent_response = client.post('/api/review/sources/case_a/independent-annotations', json={
            'created_by': 'reviewer-1', 'parent_record_id': source['record_id'], 'items': [_item()]})
        assert independent_response.status_code == 201
        independent = independent_response.json()
        assert independent['record_type'] == 'independent_expert_annotation'
        assert independent['model_name'] is None and independent['parent_record_id'] == source['record_id']
        assert (independent['original_width'], independent['original_height']) == (32, 24)
        assert (independent['coordinate_space'], independent['coordinate_origin']) == ('original_image_pixels', 'top_left')
        assert independent['items'][0]['points'][0] == {'x': 8.0, 'y': 10.0}
        second_response = client.post('/api/review/sources/case_a/independent-annotations', json={
            'created_by': 'reviewer-1', 'parent_record_id': independent['record_id'], 'items': [_item(9)]})
        assert second_response.status_code == 201
        second = second_response.json()
        assert second['version'] == 2 and second['parent_record_id'] == independent['record_id']
        assert client.get(f"/api/review/sources/case_a/annotations/{independent['record_id']}").json()['items'][0]['points'][0]['x'] == 8

        revealed = client.post('/api/review/sources/case_a/reveal', data={'created_by': 'reviewer-1'})
        assert revealed.status_code == 200 and revealed.json()['revealed'] is True
        assert client.get('/api/review/sources/case_a/state').json()['revealed_by'] == 'reviewer-1'
        raw = client.get(f'/api/review/sources/case_a/predictions/{raw_id}').json()
        assert raw['review_status'] == review_foundation.AI_STATUS
        assert (raw['original_width'], raw['original_height']) == (32, 24)
        assert (raw['coordinate_space'], raw['coordinate_origin']) == ('original_image_pixels', 'top_left')
        assert raw['raw_model_output']['ground_truth'] is None
        assert raw['teeth'][0]['orientation_status'] == 'uncertain'
        raw_path = review_foundation.ROOT / 'images' / 'case_a' / 'records' / f'{raw_id}.json'
        raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        assert client.put(f'/api/review/sources/case_a/predictions/{raw_id}', json={}).status_code == 405
        assert client.delete(f'/api/review/sources/case_a/predictions/{raw_id}').status_code == 405
        assert client.post('/analysis/tooth-segmentation?image_id=case_a', files={'file': ('case_a.png', _image())}).status_code == 200

        bad_reference = {'created_by': 'reviewer-1', 'items': [_item(11)],
                         'raw_prediction_record_id': second['record_id'],
                         'independent_annotation_record_id': second['record_id']}
        assert client.post('/api/review/sources/case_a/corrections', json=bad_reference).status_code == 422
        correction_payload = {'created_by': 'reviewer-1', 'items': [_item(11)],
                              'raw_prediction_record_id': raw_id,
                              'independent_annotation_record_id': second['record_id']}
        first_correction = client.post('/api/review/sources/case_a/corrections', json=correction_payload)
        assert first_correction.status_code == 201
        correction = first_correction.json()
        assert correction['version'] == 1 and correction['parent_record_id'] == raw_id
        assert (correction['original_width'], correction['original_height']) == (32, 24)
        assert (correction['coordinate_space'], correction['coordinate_origin']) == ('original_image_pixels', 'top_left')
        assert client.post('/api/review/sources/case_a/corrections', json={
            **correction_payload, 'parent_record_id': raw_id, 'items': [_item(12)]}).status_code == 409
        correction_payload['parent_record_id'] = correction['record_id']
        correction_payload['items'] = [_item(12)]
        second_correction = client.post('/api/review/sources/case_a/corrections', json=correction_payload).json()
        assert second_correction['version'] == 2
        assert client.get(f"/api/review/sources/case_a/annotations/{correction['record_id']}").json()['items'][0]['points'][0]['x'] == 11
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_hash

        assert client.post('/api/review/sources/case_a/approvals', json={
            'created_by': 'reviewer-1', 'approved_record_id': 'e35cb900-dbdd-4205-9ec8-07b1ff241c2b',
            'decision': 'approved'}).status_code == 404
        approval_response = client.post('/api/review/sources/case_a/approvals', json={
            'created_by': 'reviewer-1', 'approved_record_id': second_correction['record_id'],
            'decision': 'approved', 'notes': 'Reviewed geometry'})
        assert approval_response.status_code == 201
        approval = approval_response.json()
        assert approval['approved_record_id'] == second_correction['record_id'] and approval['version'] == 1
        assert (approval['original_width'], approval['original_height']) == (32, 24)
        assert (approval['coordinate_space'], approval['coordinate_origin']) == ('original_image_pixels', 'top_left')
        assert client.get(f"/api/review/sources/case_a/approvals/{approval['record_id']}").json()['decision'] == 'approved'
        assert len(client.get('/api/review/sources/case_a/history').json()['records']) == 7

        legacy = {'image_id': 'case_a', 'disease_status': 'uncertain', 'affected_teeth': [],
                  'severity': 'cannot_determine', 'findings': [], 'regions': [],
                  'expert_comment': 'Legacy file remains readable'}
        assert client.post('/annotations', json=legacy).status_code == 201
        assert client.get('/annotations/case_a').json()['expert_comment'] == 'Legacy file remains readable'
        assert json.loads((tmp_path / 'legacy' / 'case_a.json').read_text(encoding='utf-8'))['image_id'] == 'case_a'
