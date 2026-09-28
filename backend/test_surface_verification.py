import copy
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.surface_verification import (
    SurfaceRecord, VerificationInput, calculate_rbl, expected_facing_surface,
    get_verification, save_verification, swap_surfaces, validate_submission,
)
from main import app


def orientation(**changes):
    value = {"arch": "mandibular", "patient_side": "left", "display_orientation": "native",
             "horizontal_flip": False, "source": "expert_confirmation", "status": "verified",
             "verified_by": "expert-1", "verified_at": datetime.now(timezone.utc).isoformat()}
    value.update(changes)
    return value


def tooth(instance, fdi, surfaces=None, **changes):
    value = {"instance_id": instance, "original_instance_id": instance, "original_bbox_xyxy": [0, 0, 100, 200],
             "expert_fdi": fdi, "tooth_verification_status": "confirmed", "fdi_source": "expert_confirmation",
             "surfaces": surfaces or []}
    value.update(changes)
    return value


def payload(teeth, orient=None):
    return VerificationInput(reviewer_id="expert-1", partition="Training", image_id="10",
                             orientation=orient or orientation(), teeth=teeth)


def test_all_quadrants_and_central_incisors():
    for a, b in (("16", "15"), ("26", "25"), ("36", "35"), ("46", "45")):
        assert expected_facing_surface(a, b) == "mesial"
    for a, b in (("16", "17"), ("26", "27"), ("36", "37"), ("46", "47")):
        assert expected_facing_surface(a, b) == "distal"
    for a, b in (("11", "21"), ("21", "11"), ("31", "41"), ("41", "31")):
        assert expected_facing_surface(a, b) == "mesial"


def test_correct_35_distal_36_mesial_and_inconsistent_pairs():
    assert expected_facing_surface("35", "36") == "distal"
    assert expected_facing_surface("36", "35") == "mesial"
    bad = [{"surface": "mesial", "pixel_side": "image_left", "adjacent_fdi": "36", "verification_status": "confirmed"}]
    codes = {x["code"] for x in validate_submission(payload([tooth("a", "35", bad)]))}
    assert "inconsistent_surface" in codes


def test_missing_neighbor_is_warning_state_not_incorrect():
    surface = {"surface": "distal", "pixel_side": "image_right", "adjacency_status": "missing_neighbor",
               "verification_status": "unverified"}
    codes = {x["code"] for x in validate_submission(payload([tooth("a", "36", [surface])]))}
    assert "invalid_adjacency" not in codes


def test_duplicate_fdi_same_side_and_unverified_orientation():
    surfaces = [{"surface": "mesial", "pixel_side": "image_left", "verification_status": "confirmed"},
                {"surface": "distal", "pixel_side": "image_left", "verification_status": "confirmed"}]
    unverified = orientation(status="unverified", verified_by=None, verified_at=None)
    codes = {x["code"] for x in validate_submission(payload([tooth("a", "36", surfaces), tooth("b", "36")], unverified))}
    assert {"duplicate_fdi", "same_pixel_side", "orientation_unverified"} <= codes


def test_swap_md_preserves_raw_coordinates_and_does_not_mutate():
    source = [{"surface": "mesial", "cej": {"state": "visible", "point": {"x": 12, "y": 30}}},
              {"surface": "distal", "cej": {"state": "visible", "point": {"x": 88, "y": 31}}}]
    original = copy.deepcopy(source); swapped = swap_surfaces(source)
    assert source == original and swapped[0]["surface"] == "distal" and swapped[0]["cej"] == source[0]["cej"]


def test_rbl_and_ungradable_multiroot_review():
    confirmed = {"verification_status": "confirmed"}
    surface = SurfaceRecord(surface="mesial", verification_status="confirmed",
                            cej={"state": "visible", "point": {"x": 0, "y": 0}, **confirmed},
                            bone_crest={"state": "visible", "point": {"x": 0, "y": 20}, **confirmed},
                            root_apex={"state": "visible", "point": {"x": 0, "y": 100}, **confirmed})
    assert calculate_rbl(surface) == 20
    codes = {x["code"] for x in validate_submission(payload([tooth("a", "36", [surface.model_dump()], multi_rooted=True)]))}
    assert "missing_multiroot_apex_selection" in codes


def test_invalid_measurement_values_are_rejected():
    import pytest
    with pytest.raises(ValidationError):
        SurfaceRecord(surface="mesial", rbl_percentage=-1)
    with pytest.raises(ValidationError):
        SurfaceRecord(surface="mesial", model_rbl_percentage=float("nan"))


def test_route_preserves_original_proposal(monkeypatch, tmp_path):
    import app.surface_verification as module
    monkeypatch.setattr(module, "STORE", tmp_path)
    proposal = get_verification("Training", "10")["proposal"]
    first = proposal["teeth"][0]
    body = {"reviewer_id": "expert-1", "partition": "Training", "image_id": "10", "orientation": orientation(),
            "teeth": [tooth(first["tooth_instance_id"], "43", original_mask_ref=first["reference_mask_path"],
                            original_bbox_xyxy=first["original_bbox_xyxy"])]}
    saved = save_verification("Training", "10", VerificationInput(**body))
    assert saved["model_predictions_immutable"] is True
    assert get_verification("Training", "10")["proposal"]["teeth"][0]["expert_fdi"] is None


def test_http_endpoint_get_post_and_history(monkeypatch, tmp_path):
    import app.surface_verification as module
    monkeypatch.setattr(module, "STORE", tmp_path)
    client = TestClient(app)
    response = client.get("/api/surface-verification/Training/10")
    assert response.status_code == 200
    first = response.json()["proposal"]["teeth"][0]
    body = {"reviewer_id": "expert-http", "partition": "Training", "image_id": "10", "orientation": orientation(),
            "teeth": [tooth(first["tooth_instance_id"], "43", original_mask_ref=first["reference_mask_path"],
                            original_bbox_xyxy=first["original_bbox_xyxy"])]}
    saved = client.post("/api/surface-verification/Training/10", json=body)
    assert saved.status_code == 201 and saved.json()["coordinate_space"] == "original_image_pixels"
    assert client.get("/api/surface-verification/Training/10").json()["history_count"] == 1


def test_mask_tolerance_and_crossing_warnings(monkeypatch, tmp_path):
    import app.surface_verification as module
    masks = {}
    for name, left, right in (("a", 10, 40), ("b", 45, 75)):
        image = np.zeros((100, 100), np.uint8); image[10:90, left:right] = 255
        path = tmp_path / f"{name}.png"; cv2.imwrite(str(path), image); masks[name] = path
    monkeypatch.setattr(module, "_mask_path", lambda _p, _i, t: masks[t.instance_id])
    confirmed = {"state": "visible", "verification_status": "confirmed"}
    crossing = {"surface": "distal", "pixel_side": "image_right", "verification_status": "confirmed",
                "cej": {**confirmed, "point": {"x": 35, "y": 20}},
                "bone_crest": {**confirmed, "point": {"x": 60, "y": 30}},
                "root_apex": {**confirmed, "point": {"x": 35, "y": 80}}}
    outside = {"surface": "mesial", "pixel_side": "image_left", "verification_status": "confirmed",
               "cej": {**confirmed, "point": {"x": 90, "y": 20}},
               "bone_crest": {**confirmed, "point": {"x": 10, "y": 30}},
               "root_apex": {**confirmed, "point": {"x": 20, "y": 80}}}
    codes = {w["code"] for w in module.geometry_warnings(payload([tooth("a", "36", [crossing]), tooth("b", "37", [outside])]))}
    assert "measurement_crosses_unrelated_tooth" in codes and "cej_outside_mask" in codes


def test_mesial_distal_rbl_are_independent():
    visible = lambda y: {"state": "visible", "point": {"x": 20, "y": y}, "verification_status": "confirmed"}
    mesial = SurfaceRecord(surface="mesial", verification_status="confirmed", cej=visible(10), bone_crest=visible(30), root_apex=visible(90))
    distal = SurfaceRecord(surface="distal", verification_status="confirmed", cej=visible(10), bone_crest=visible(50), root_apex=visible(90))
    assert calculate_rbl(mesial) == 25 and calculate_rbl(distal) == 50
