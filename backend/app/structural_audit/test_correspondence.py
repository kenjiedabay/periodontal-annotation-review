"""Tests for conservative correspondence decisions."""

from types import SimpleNamespace

from .correspondence import analyze_record


def _record(cej_count: int = 1, apex_count: int = 1, mask_count: int = 2, fdi_count: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        image=SimpleNamespace(image_id="1"),
        tooth_masks=[{} for _ in range(mask_count)],
        radiograph_mask={},
        coco_annotations=[{} for _ in range(mask_count)],
        keypoints={"bboxes": [[0, 0, 4, 4] for _ in range(mask_count)], "CEJ_Points": [[1, 1] for _ in range(cej_count)], "Apex_Points": [[1, 1] for _ in range(apex_count)]},
        bone_lines={"Bone_Lines": [[]]},
        characteristics={"FDI notation of fully/partially visible teeth": ",".join(str(index) for index in range(fdi_count))},
    )


def test_landmark_pairings_are_unresolved_without_source_ids() -> None:
    result = analyze_record(_record())
    statuses = {item.name: item.status for item in result.relationships}
    # Matching counts are not identity evidence.
    assert statuses["tooth_mask_to_coco_instance"] == "uncertain"
    assert statuses["cej_to_tooth_bbox"] == "uncertain"
    assert statuses["apex_to_tooth_bbox"] == "uncertain"
    assert statuses["fdi_metadata_to_tooth_instance"] == "uncertain"
    assert result.status == "unresolved"


def test_missing_landmarks_are_reported() -> None:
    result = analyze_record(_record(cej_count=0, apex_count=0, fdi_count=0))
    assert "CEJ" in result.missing_landmarks
    assert "apex" in result.missing_landmarks
    assert any("CEJ count" in item for item in result.suspicious_mappings) is False


def test_spatial_candidates_remain_uncertain() -> None:
    record = _record(mask_count=1, fdi_count=1)
    record.keypoints = {"bboxes": [[0, 0, 4, 4]], "CEJ_Points": [[1, 1]], "Apex_Points": [[2, 2]]}
    result = analyze_record(record)
    assert result.review_items["cej_points"][0]["bbox_candidates"] == ["0"]
    assert result.review_items["apex_points"][0]["bbox_candidates"] == ["0"]
    statuses = {item.name: item.status for item in result.relationships}
    assert statuses["cej_to_tooth_bbox"] == "uncertain"
    assert statuses["apex_to_tooth_bbox"] == "uncertain"
