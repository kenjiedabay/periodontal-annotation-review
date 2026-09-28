"""Fixture tests for the read-only DenPAR Stage 2A audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

from audit_denpar_stage2a import hamming_distance, run_audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workbook(path: Path, image_ids: list[str]) -> None:
    strings = ["id", "Arch", "Site", "FDI notation of fully/partially visible teeth", "Upper", "Anterior", "11"]
    strings.extend(f"{image_id}.0" for image_id in image_ids)
    shared = '<?xml version="1.0" encoding="UTF-8"?><sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">' + ''.join(
        f'<si><t>{value}</t></si>' for value in strings) + '</sst>'
    rows = ['<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c></row>']
    for number, image_id in enumerate(image_ids, 2):
        index = 7 + image_ids.index(image_id)
        rows.append(f'<row r="{number}"><c r="A{number}" t="s"><v>{index}</v></c><c r="B{number}" t="s"><v>4</v></c><c r="C{number}" t="s"><v>5</v></c><c r="D{number}" t="s"><v>6</v></c></row>')
    sheet = '<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + ''.join(rows) + '</sheetData></worksheet>'
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def _annotations(root: Path, partition: str, image_id: str, size=(32, 24), out_of_bounds=False, bone=True) -> None:
    width, height = size
    base = root / partition
    for directory in ("Key Points Annotations", "Bone Level Annotations", "Masks (Radiograph-wise)"):
        (base / directory).mkdir(parents=True, exist_ok=True)
    mask_dir = base / "Masks (Tooth-wise)" / image_id
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((height, width), np.uint8)
    mask[4:20, 5:16] = 255
    Image.fromarray(mask).save(mask_dir / "mask1.png")
    Image.fromarray(mask).save(base / "Masks (Radiograph-wise)" / f"{image_id}.png")
    point = [width + 1, 5] if out_of_bounds else [6, 5]
    (base / "Key Points Annotations" / f"{image_id}.json").write_text(json.dumps({
        "Image_id": f"{image_id}.jpg", "bboxes": [[5, 4, 16, 20]],
        "CEJ_Points": [point], "Apex_Points": [[10, 19]],
    }), encoding="utf-8")
    if bone:
        (base / "Bone Level Annotations" / f"{image_id}.json").write_text(json.dumps({
            "Image_id": f"{image_id}.jpg", "Num_of_Bone_Lines": 1,
            "Bone_Lines": [[[5, 8], [15, 8]]],
        }), encoding="utf-8")


def _jpeg_with_comment(source: bytes, comment: bytes) -> bytes:
    assert source[:2] == b"\xff\xd8"
    segment = b"\xff\xfe" + (len(comment) + 2).to_bytes(2, "big") + comment
    return source[:2] + segment + source[2:]


def test_read_only_audit_detects_identity_leakage_and_annotation_issues(tmp_path: Path) -> None:
    dataset = tmp_path / "DenPAR" / "Dataset"
    training = tmp_path / "dataset" / "raw"
    output = tmp_path / "reports"
    training.mkdir(parents=True)
    (dataset / "Validation" / "Images").mkdir(parents=True)
    (dataset / "Images").mkdir(parents=True)
    pixels = np.tile(np.arange(32, dtype=np.uint8), (24, 1)) * 7
    Image.fromarray(pixels).save(training / "train1.jpg", quality=95)
    original_bytes = (training / "train1.jpg").read_bytes()
    (dataset / "Validation" / "Images" / "val1.jpg").write_bytes(original_bytes)
    (dataset / "Images" / "test1.jpg").write_bytes(_jpeg_with_comment(original_bytes, b"metadata differs"))
    Image.fromarray(np.clip(pixels.astype(np.int16) + 2, 0, 255).astype(np.uint8)).save(dataset / "Images" / "test2.jpg", quality=95)
    _annotations(dataset, "Training", "train1")
    _annotations(dataset, "Validation", "val1", out_of_bounds=True)
    _annotations(dataset, "Testing", "test1")
    _annotations(dataset, "Testing", "test2", bone=False)
    (dataset / "Validation" / "Key Points Annotations" / "orphan.json").write_text(json.dumps({
        "Image_id": "orphan.jpg", "bboxes": [], "CEJ_Points": [], "Apex_Points": [],
    }), encoding="utf-8")
    _workbook(dataset / "Characteristics of radiographs included.xlsx", ["train1", "val1", "test1", "test2", "orphan"])
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps([{"image_id": "train1"}]), encoding="utf-8")
    source_files = [path for path in tmp_path.rglob("*") if path.is_file() and output not in path.parents]
    before = {path: _sha(path) for path in source_files}

    result = run_audit(dataset, training, output, tmp_path,
                       expected_counts={"Training": 1, "Validation": 1, "Testing": 2},
                       near_duplicate_threshold=6, training_manifest=manifest)

    assert all(item["matches_expected"] for item in result["partition_counts"].values())
    assert any(group["cross_partition"] and group["method"] == "identical_file_sha256" for group in result["exact_duplicates"])
    assert any(group["cross_partition"] and group["method"] == "identical_normalized_pixels" for group in result["exact_duplicates"])
    assert any(item["issue"] == "point_out_of_bounds" for item in result["coordinate_issues"])
    assert any(item["issue"] == "annotation_without_matching_image" for item in result["orphaned_annotations"])
    assert any(item["classification"] == "missing_component" and item["stable_image_id"] == "Testing:test2" for item in result["findings"])
    assert any(item["classification"] == "confirmed_leakage" for item in result["findings"])
    assert result["source_integrity"]["preserved"] is True
    assert result["patient_grouping"]["available"] is False
    assert result["checkpoint_provenance"]["classification"] == "provenance_unverified"
    assert hamming_distance("0", "f") == 4
    assert (output / "summary.md").exists() and (output / "dataset_inventory.csv").exists()
    assert before == {path: _sha(path) for path in source_files}


def test_report_directory_cannot_be_inside_source_tree(tmp_path: Path) -> None:
    dataset = tmp_path / "Dataset"
    training = tmp_path / "raw"
    dataset.mkdir(); training.mkdir()
    with pytest.raises(ValueError, match="outside all source directories"):
        run_audit(dataset, training, dataset / "report", tmp_path, expected_counts={})
