"""Focused tests for tooth-instance segmentation safety and schema."""

import numpy as np
from PIL import Image
import torch
from pathlib import Path

from .data import Record, ToothInstanceDataset, binary_mask, box_from_mask, validate_official_split
from .model import inference_response
from .model import create_model


def test_binarization_and_box_extraction() -> None:
    image = Image.fromarray(np.array([[0, 4, 0], [0, 255, 0]], dtype=np.uint8))
    mask = binary_mask(image)
    assert mask.tolist() == [[False, True, False], [False, True, False]]
    assert box_from_mask(mask) == [1.0, 0.0, 2.0, 2.0]


def test_empty_mask_is_rejected() -> None:
    try:
        box_from_mask(np.zeros((2, 2), dtype=bool))
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty masks must be rejected")


def test_inference_response_is_non_diagnostic() -> None:
    response = inference_response("1002", "test", [{"instance_id": 1, "confidence": 0.9, "bbox": [1, 2, 3, 4], "mask": "reference"}], 12.0)
    assert response["task"] == "tooth_instance_segmentation"
    assert response["clinical_interpretation"] is None


def test_model_forward_pass() -> None:
    model, _ = create_model(pretrained=False)
    model.eval()
    with torch.no_grad():
        result = model([torch.zeros((3, 128, 128), dtype=torch.float32)])
    assert len(result) == 1 and {"boxes", "labels", "scores", "masks"}.issubset(result[0])


def test_dataset_resize_keeps_mask_box_alignment(tmp_path: Path) -> None:
    image_path, mask_path = tmp_path / "x.jpg", tmp_path / "mask1.png"
    Image.fromarray(np.zeros((10, 20), dtype=np.uint8)).save(image_path)
    mask = np.zeros((10, 20), dtype=np.uint8); mask[2:8, 4:12] = 255; Image.fromarray(mask).save(mask_path)
    image, target = ToothInstanceDataset([Record("x", image_path, (mask_path,), 20, 10)], max_side=10)[0]
    assert tuple(image.shape) == (3, 5, 10)
    assert target["boxes"].tolist() == [[2.0, 1.0, 6.0, 4.0]]
    assert target["metadata"]["preprocessing"]["scale_x"] == 0.5


def test_loader_rejects_missing_source_image(tmp_path: Path) -> None:
    (tmp_path / "Training/Masks (Tooth-wise)/1").mkdir(parents=True)
    Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(tmp_path / "Training/Masks (Tooth-wise)/1/mask1.png")
    records, report = validate_official_split(tmp_path, "train")
    assert not records and report["skipped_images"][0]["reason"] == "source_image_missing"
