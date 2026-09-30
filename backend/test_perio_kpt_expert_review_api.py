from fastapi.testclient import TestClient
from main import app
from app import perio_kpt_expert_review_api as review_api


def valid_payload(case_id="Case001"):
    surface = {"cej_correct": "yes", "bone_level_correct": "yes", "root_landmark_appropriate": "yes",
               "mesial_distal_association_correct": "yes", "geometry_clinically_plausible": "yes",
               "surface_assessable": "yes", "reason_codes": [], "comments": ""}
    return {"protocol_version": "1.0", "case_id": case_id, "anonymous_reviewer_id": "reviewer_A",
            "completed_at_utc": "2026-09-29T00:00:00Z",
            "surfaces": [{"surface": "mesial", **surface}, {"surface": "distal", **surface}]}


def test_manifest_is_blinded_and_images_are_served():
    client = TestClient(app)
    response = client.get("/api/perio-kpt-expert-review/manifest")
    assert response.status_code == 200
    body = response.json()
    assert len(body["cases"]) == 12
    assert set(body["cases"][0]) == {"case_id", "root_class"}
    assert "record_id" not in response.text.lower()
    assert "blinding_key" not in response.text.lower()
    assert client.get("/api/perio-kpt-expert-review/cases/Case001/image").status_code == 200


def test_review_is_validated_and_append_only(tmp_path, monkeypatch):
    monkeypatch.setattr(review_api, "COMPLETED", tmp_path / "completed")
    client = TestClient(app); payload = valid_payload()
    assert client.post("/api/perio-kpt-expert-review/reviews", json=payload).status_code == 201
    assert client.post("/api/perio-kpt-expert-review/reviews", json=payload).status_code == 409
    status = client.get("/api/perio-kpt-expert-review/status/reviewer_A").json()
    assert status["completed_case_ids"] == ["Case001"]


def test_problem_answer_requires_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(review_api, "COMPLETED", tmp_path / "completed")
    payload = valid_payload(); payload["surfaces"][0]["cej_correct"] = "uncertain"
    assert TestClient(app).post("/api/perio-kpt-expert-review/reviews", json=payload).status_code == 422
    assert not (tmp_path / "completed").exists()
