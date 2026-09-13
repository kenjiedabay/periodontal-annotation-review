"""Tests for non-diagnostic structural preparation safeguards."""

import json
from pathlib import Path

from PIL import Image

from .preparer import PreparationConfig, _resize_config, prepare_structural_dataset
from .validator import validate_manifest


def _image(path: Path, size: tuple[int, int] = (20, 10)) -> None:
    Image.new("L", size, 255).save(path)


def _fixture(root: Path) -> Path:
    validation = root / "Validation"
    for item in ("Images", "Masks (Tooth-wise)/1", "Masks (Radiograph-wise)", "Key Points Annotations", "Bone Level Annotations"):
        (validation / item).mkdir(parents=True, exist_ok=True)
    _image(validation / "Images/1.jpg")
    _image(validation / "Masks (Tooth-wise)/1/mask1.png")
    _image(validation / "Masks (Radiograph-wise)/1.png")
    (validation / "Key Points Annotations/1.json").write_text(json.dumps({"Image_id": "1.jpg", "bboxes": [[1, 1, 10, 8]], "CEJ_Points": [[2, 2]], "Apex_Points": [[2, 7]]}), encoding="utf-8")
    (validation / "Bone Level Annotations/1.json").write_text(json.dumps({"Image_id": "1.jpg", "Num_of_Bone_Lines": 1, "Bone_Lines": [[[1, 5], [10, 5]]]}), encoding="utf-8")
    (validation / "Masks (Tooth-wise)/coco_format_instances_valid.json").write_text(json.dumps({"images": [{"id": 1, "file_name": "1.jpg"}], "annotations": [{"id": 4, "image_id": 1, "bbox": [1, 1, 9, 7], "segmentation": [[1, 1, 10, 1, 10, 8]], "category_id": 1}], "categories": []}), encoding="utf-8")
    return validation


def test_resize_metadata_requires_both_dimensions(tmp_path: Path) -> None:
    try:
        _resize_config(PreparationConfig(tmp_path, tmp_path, resize_width=64), 20, 10)
    except ValueError as error:
        assert "together" in str(error)
    else:
        raise AssertionError("partial resize dimensions must fail")


def test_preparation_excludes_unconfirmed_landmarks_and_bone_lines(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    report = prepare_structural_dataset(PreparationConfig(_fixture(tmp_path), output, sample_count=1))
    manifest = json.loads((output / "manifests/structural_manifest.json").read_text(encoding="utf-8"))
    record = manifest["records"][0]
    assert report["task_statistics"]["tooth_segmentation"]["targets"] == 1
    assert record["tasks"]["tooth_localization"]["boxes"][0]["coco_annotation_id"] == 4
    assert record["tasks"]["cej_apex_landmarks"]["target_status"] == "excluded_unconfirmed"
    assert record["tasks"]["bone_line_structure"]["target_status"] == "excluded_unconfirmed"
    assert manifest["split_policy"]["status"] == "not_generated"
    assert validate_manifest(output, manifest)["valid"]


def test_explicit_resize_scales_boxes_and_records_transform(tmp_path: Path) -> None:
    output = tmp_path / "resized"
    prepare_structural_dataset(PreparationConfig(_fixture(tmp_path), output, resize_width=40, resize_height=30, sample_count=0))
    record = json.loads((output / "manifests/structural_manifest.json").read_text(encoding="utf-8"))["records"][0]
    assert record["transform"]["source_size"] == [20, 10]
    assert record["transform"]["output_size"] == [40, 30]
    assert record["tasks"]["tooth_localization"]["boxes"][0]["bbox_xywh"] == [2.0, 3.0, 18.0, 21.0]
