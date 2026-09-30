"""Gate 1 tests for the read-only Perio-KPT preparation package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.perio_kpt_landmarks.data import AnnotationError, parse_label, parse_row
from app.perio_kpt_landmarks.gate1 import build_folds, canonical_sources
from app.perio_kpt_landmarks.geometry import (calculate_rbl, crop_to_image,
                                              expand_box, gaussian_heatmaps,
                                              image_to_crop,
                                              make_letterbox_transform)
from app.perio_kpt_landmarks.schema import (EXPECTED_ROW_VALUES, LANDMARK_NAMES,
                                            rbl_indices)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/sources/perio-kpt/extracted/Periodontal_Keypoint_Dataset"


def row(class_id: int = 0, landmarks: dict[int, tuple[float, float, int]] | None = None) -> str:
    values: list[float | int] = [class_id, 0.5, 0.5, 0.4, 0.6]
    landmarks = landmarks or {}
    for index in range(len(LANDMARK_NAMES)):
        values.extend(landmarks.get(index, (0.0, 0.0, 0)))
    assert len(values) == EXPECTED_ROW_VALUES
    return " ".join(str(value) for value in values)


def test_parser_preserves_availability_and_named_channel_order() -> None:
    annotation = parse_row(row(1, {0: (0.2, 0.3, 2), 6: (0.5, 0.8, 1)}),
                           image_id="case", object_index=1,
                           source_label=Path("case.txt"), source_line=1)
    assert annotation.class_id == 1
    assert annotation.landmarks[0].available
    assert annotation.landmarks[0].visibility == 2
    assert annotation.landmarks[6].available
    assert annotation.landmarks[6].visibility == 1
    assert not annotation.landmarks[1].available


def test_parser_rejects_wrong_row_length_without_repair() -> None:
    with pytest.raises(AnnotationError, match="unexpected_value_count") as caught:
        parse_row(row() + " 0 0 0", image_id="case", object_index=1,
                  source_label=Path("case.txt"), source_line=1)
    assert caught.value.token_count == EXPECTED_ROW_VALUES + 3


def test_real_image24_third_row_is_quarantined_and_source_is_unchanged() -> None:
    path = DATASET / "1_Experiment/holdout_test_standard_box/labels/Image24.txt"
    before = path.read_bytes()
    annotations, quarantined = parse_label(path)
    assert len(annotations) == 3
    assert quarantined == [{
        "record_id": "Image24:object-3", "image_id": "Image24",
        "source_label": str(path), "source_line": 3,
        "reason": "unexpected_value_count", "observed_value_count": 41,
        "expected_value_count": 38, "source_preserved": True,
    }]
    assert path.read_bytes() == before


def test_twenty_percent_expansion_clips_to_image_bounds() -> None:
    assert expand_box((20, 30, 60, 80), 100, 100) == (12.0, 20.0, 68.0, 90.0)
    assert expand_box((0, 0, 30, 40), 100, 100) == (0.0, 0.0, 36.0, 48.0)


@pytest.mark.parametrize("point", [(12.0, 20.0), (45.5, 55.25), (92.0, 80.0)])
def test_letterbox_geometry_round_trip(point: tuple[float, float]) -> None:
    transform = make_letterbox_transform((10, 15, 95, 85), 120, 100, 256)
    restored = crop_to_image(image_to_crop(point, transform), transform)
    assert restored == pytest.approx(point, abs=1e-9)


def test_heatmap_generation_masks_unavailable_landmarks() -> None:
    points = [(30.5, 40.5), None] + [None] * (len(LANDMARK_NAMES) - 2)
    heatmaps, available = gaussian_heatmaps(points, output_size=64, sigma=2)
    assert heatmaps.shape == (len(LANDMARK_NAMES), 64, 64)
    assert available.tolist() == [1.0, 0.0] + [0.0] * (len(LANDMARK_NAMES) - 2)
    assert heatmaps[0].max() > 0.9
    assert np.count_nonzero(heatmaps[1:]) == 0


def test_valid_ground_truth_rbl() -> None:
    points: list[tuple[float, float] | None] = [None] * len(LANDMARK_NAMES)
    points[0], points[1], points[2] = (10, 10), (10, 40), (10, 110)
    result = calculate_rbl(points, "mesial")
    assert result["status"] == "assessable"
    assert result["rbl_percent"] == pytest.approx(30.0)
    assert result["projection_ratio"] == pytest.approx(0.3)


def test_single_root_rbl_uses_central_apex_for_both_surfaces() -> None:
    points: list[tuple[float, float] | None] = [None] * len(LANDMARK_NAMES)
    points[0], points[1] = (10, 10), (10, 40)
    points[3], points[4] = (30, 10), (30, 40)
    points[6] = (20, 110)
    mesial = calculate_rbl(points, "mesial", indices=rbl_indices(0, "mesial"))
    distal = calculate_rbl(points, "distal", indices=rbl_indices(0, "distal"))
    assert mesial["status"] == "assessable"
    assert distal["status"] == "assessable"
    assert mesial["rbl_percent"] == pytest.approx(distal["rbl_percent"])


def test_rbl_rejects_missing_and_implausible_geometry() -> None:
    points: list[tuple[float, float] | None] = [None] * len(LANDMARK_NAMES)
    missing = calculate_rbl(points, "distal")
    assert missing["status"] == "not_assessable"
    assert missing["reason"] == "missing_landmarks"

    points[0], points[1], points[2] = (10, 10), (10, 0), (10, 110)
    implausible = calculate_rbl(points, "mesial")
    assert implausible["status"] == "not_assessable"
    assert implausible["reason"] == "implausible_bone_projection"

    points[0], points[1], points[2] = (10, 10), (10, 11), (10, 12)
    short = calculate_rbl(points, "mesial")
    assert short["status"] == "not_assessable"
    assert short["reason"] == "invalid_root_length"


def test_supplied_folds_split_by_radiograph_and_exclude_holdout() -> None:
    standard = DATASET / "1_Experiment/standard_box"
    holdout = {path.stem for path in (DATASET / "1_Experiment/holdout_test_standard_box/images").glob("*.png")}
    assert len(holdout) == 18
    for number in range(5):
        fold = standard / f"f{number}"
        train = {path.stem for path in (fold / "train/images").glob("*.png")}
        validation = {path.stem for path in (fold / "val/images").glob("*.png")}
        assert len(train) == 140
        assert len(validation) == 35
        assert not train & validation
        assert not (train | validation) & holdout
        assert len(train | validation) == 175


def test_frozen_manifest_is_deterministic() -> None:
    sources, holdout = canonical_sources(DATASET)
    first = build_folds(DATASET, sources, holdout)
    second = build_folds(DATASET, sources, holdout)
    assert first == second
    assert len(first["manifest_sha256"]) == 64
