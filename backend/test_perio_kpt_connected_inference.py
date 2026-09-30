import numpy as np
from fastapi.testclient import TestClient

from main import app
from app import perio_kpt_connected_inference as connected


def test_measurement_withholds_low_confidence():
    points = [(10.0, 10.0)] * 11
    confidence = [0.9] * 11
    confidence[connected.LANDMARK_INDEX["CEJ-m"]] = 0.2
    result = connected.ConnectedPerioService._measurement(points, confidence, "mesial")
    assert result["status"] == "not_assessable"
    assert result["reason"] == "low_landmark_confidence"


def test_measurement_reports_root_uncertainty_range():
    points = [(10.0, 10.0)] * 11
    points[connected.LANDMARK_INDEX["CEJ-m"]] = (10.0, 10.0)
    points[connected.LANDMARK_INDEX["BL-m"]] = (10.0, 30.0)
    points[connected.LANDMARK_INDEX["RL-m"]] = (10.0, 110.0)
    points[connected.LANDMARK_INDEX["RL-c"]] = (10.0, 90.0)
    result = connected.ConnectedPerioService._measurement(points, [0.9] * 11, "mesial")
    assert result["status"] == "uncertain"
    assert result["reason"] == "root_class_not_predicted"
    assert result["preliminary_rbl_range_percent"] == [20.0, 25.0]


def test_preview_endpoint_contract(monkeypatch):
    expected = {"image_id":"demo", "task":"connected_perio_kpt_research_preview",
                "model_status":"available", "detector_status":"available", "detector_version":"test",
                "landmark_model_version":"test", "width":8, "height":8, "instances":[],
                "disclaimer":"research", "limitations":[]}
    monkeypatch.setattr(connected.service, "predict", lambda image_id, image: {**expected, "image_id": image_id})
    monkeypatch.setattr("main.assert_ai_access", lambda *args: None)
    ok, encoded = connected.cv2.imencode(".png", np.zeros((8, 8), dtype=np.uint8))
    assert ok
    response = TestClient(app).post("/analysis/perio-kpt-preview?image_id=demo",
                                    files={"file": ("demo.png", encoded.tobytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["task"] == "connected_perio_kpt_research_preview"


def test_unified_review_keeps_model_outputs_separate(monkeypatch):
    connected_result = {"model_status":"available", "instances":[]}
    anatomy_result = {"model_status":"available", "points":{"cej":[], "apex":[]}}
    monkeypatch.setattr("main.connected_perio_service.predict", lambda image_id, image: connected_result)
    monkeypatch.setattr("main.anatomy_service.predict", lambda image_id, image: anatomy_result)
    monkeypatch.setattr("main.assert_ai_access", lambda *args: None)
    ok, encoded = connected.cv2.imencode(".png", np.zeros((8, 8), dtype=np.uint8)); assert ok
    response = TestClient(app).post("/analysis/unified-model-review?image_id=demo",
                                    files={"file": ("demo.png", encoded.tobytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["connected_perio_kpt"] == connected_result
    assert body["full_image_anatomy"] == anatomy_result
    assert body["separation"] == {"datasets_merged": False, "checkpoints_modified": False,
                                  "expert_annotations_are_model_outputs": False}
