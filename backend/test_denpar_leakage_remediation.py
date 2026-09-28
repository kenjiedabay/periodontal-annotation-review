"""Tests for derived leakage-remediation manifests and the Stage 2B gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from audit_denpar_stage2a import _decode_gray, normalized_pixel_hash, perceptual_hash, sha256_file
from remediate_denpar_leakage import (
    MANIFEST_NAMES, ReconciliationError, apply_manual_decisions, assert_stage2b_ready, content_hash, generate,
    record_review, reconcile, validate_content_hash, validate_derived_manifests,
)


def _write(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), pixels)


def _row(project: Path, path: Path, partition: str, image_id: str) -> dict:
    pixels = _decode_gray(path)
    return {
        "stable_image_id": f"{partition}:{image_id}", "image_id": image_id,
        "official_partition": partition, "source_path": path.relative_to(project).as_posix(),
        "absolute_source_path": str(path.resolve()), "file_sha256": sha256_file(path),
        "normalized_pixel_sha256": normalized_pixel_hash(pixels),
        "perceptual_hash_64": perceptual_hash(pixels),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    dataset, training = tmp_path / "DenPAR" / "Dataset", tmp_path / "dataset" / "raw"
    base = np.tile(np.arange(32, dtype=np.uint8), (24, 1)) * 7
    other = np.tile(np.arange(24, dtype=np.uint8)[:, None], (1, 32)) * 9
    near = np.clip(base.astype(np.int16) + 3, 0, 255).astype(np.uint8)
    paths = {
        "Training:A": training / "A.png", "Training:B": training / "B.png",
        "Validation:C": dataset / "Validation" / "Images" / "C.png",
        "Testing:D": dataset / "Images" / "D.png",
        "Training:E": training / "E.png", "Testing:F": dataset / "Images" / "F.png",
    }
    for stable_id in ("Training:A", "Training:B", "Validation:C"):
        _write(paths[stable_id], base)
    _write(paths["Testing:D"], other)
    _write(paths["Training:E"], near)
    _write(paths["Testing:F"], np.clip(near.astype(np.int16) + 1, 0, 255).astype(np.uint8))
    inventory = [_row(tmp_path, path, *stable_id.split(":")) for stable_id, path in paths.items()]
    exact_pairs = [["Training:A", "Training:B"], ["Training:A", "Validation:C"], ["Training:B", "Validation:C"]]
    audit = {
        "generated_at": "2026-09-21T00:00:00+00:00", "inventory": inventory,
        "exact_duplicate_summary": {
            "unique_pair_count": 3, "cross_partition_unique_pair_count": 2,
            "unique_pairs": [{"images": pair} for pair in exact_pairs],
        },
        "near_duplicates": [{
            "left": {"stable_image_id": "Training:E"}, "right": {"stable_image_id": "Testing:F"},
            "method": "64-bit DCT pHash", "hamming_distance": 0, "threshold": "Hamming distance <= 6",
            "supporting_evidence": {"thumbnail_pearson_correlation": .999, "normalized_thumbnail_mae": .001},
        }],
    }
    audit_path = tmp_path / "audit.json"; audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return dataset, training, audit_path, audit


def test_remediation_excludes_exact_duplicates_quarantines_near_pairs_and_preserves_sources(tmp_path: Path) -> None:
    dataset, training, audit_path, audit = _fixture(tmp_path)
    source_paths = [path for path in tmp_path.rglob("*") if path.is_file() and path != audit_path]
    before = {path: sha256_file(path) for path in source_paths}
    first = generate(audit_path, tmp_path / "output-one", dataset, training)
    second = generate(audit_path, tmp_path / "output-two", dataset, training)

    reconciliation = first["reconciliation"]
    assert reconciliation["exact"]["unique_duplicated_images"] == 3
    assert reconciliation["exact"]["pairwise_relationships"] == 3
    assert reconciliation["exact"]["connected_groups"] == 1
    assert reconciliation["exact"]["same_partition_pairs"] == 1
    assert reconciliation["exact"]["cross_partition_pairs"] == 2
    assert first["sizes"]["denpar_clean_train_manifest"] == 1
    assert first["sizes"]["denpar_clean_validation_manifest"] == 0
    assert first["sizes"]["denpar_clean_historical_test_manifest"] == 1
    group = first["manifests"]["denpar_duplicate_group_manifest"]["groups"][0]
    assert group["canonical_stable_image_id"] == "Training:A"
    clean_ids = {item["stable_image_id"] for name in MANIFEST_NAMES[:3] for item in first["manifests"][name]["entries"]}
    assert not ({"Training:B", "Validation:C"} & clean_ids)
    assert not ({"Training:E", "Testing:F"} & clean_ids)
    assert first["manifests"]["denpar_clean_historical_test_manifest"]["test_set_status"] == "historical_exposed_test"
    assert first["integrity"]["preserved"] is True
    assert before == {path: sha256_file(path) for path in source_paths}
    for name in MANIFEST_NAMES:
        validate_content_hash(first["manifests"][name])
        assert first["manifests"][name]["content_hash"] == second["manifests"][name]["content_hash"]

    exact_tamper = copy.deepcopy(first["manifests"])
    exact_tamper["denpar_clean_validation_manifest"]["entries"].append(
        next(item for item in audit["inventory"] if item["stable_image_id"] == "Validation:C"))
    exact_tamper["denpar_clean_validation_manifest"]["content_hash"] = content_hash(exact_tamper["denpar_clean_validation_manifest"])
    with pytest.raises(ValueError, match="Excluded images entered"):
        validate_derived_manifests(exact_tamper)

    quarantine_tamper = copy.deepcopy(first["manifests"])
    quarantine_tamper["denpar_clean_train_manifest"]["entries"].append(
        next(item for item in audit["inventory"] if item["stable_image_id"] == "Training:E"))
    quarantine_tamper["denpar_clean_train_manifest"]["content_hash"] = content_hash(quarantine_tamper["denpar_clean_train_manifest"])
    with pytest.raises(ValueError, match="Quarantined images entered"):
        validate_derived_manifests(quarantine_tamper)
    try:
        assert_stage2b_ready(tmp_path / "output-one")
    except RuntimeError as error:
        assert "blocked" in str(error)
    else:
        raise AssertionError("Pending near duplicates must block Stage 2B")

    pair_id = first["manifests"]["denpar_near_duplicate_review_queue"]["pairs"][0]["pair_id"]
    decision_path = record_review(tmp_path / "output-one", pair_id, "reviewer-fixture", "uncertain", "Needs expert review")
    decision = json.loads(decision_path.read_text(encoding="utf-8")); validate_content_hash(decision)
    assert decision["reviewer_id"] == "reviewer-fixture" and decision["decision"] == "uncertain"


def test_inconsistent_duplicate_counts_fail_before_manifest_generation(tmp_path: Path) -> None:
    _, _, _, audit = _fixture(tmp_path)
    inconsistent = copy.deepcopy(audit)
    inconsistent["exact_duplicate_summary"]["unique_pair_count"] = 2
    try:
        reconcile(inconsistent)
    except ReconciliationError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("Inconsistent duplicate counts must fail validation")


@pytest.mark.parametrize("decision,expected_test,expected_exclusions", [
    ("same_underlying_radiograph", 1, 3),
    ("different_radiograph", 2, 2),
])
def test_manual_decisions_create_v2_without_changing_v1(tmp_path: Path, decision: str,
                                                        expected_test: int, expected_exclusions: int) -> None:
    dataset, training, audit_path, _ = _fixture(tmp_path)
    v1, v2 = tmp_path / "v1", tmp_path / "v2"
    generated = generate(audit_path, v1, dataset, training)
    v1_hashes = {name: generated["manifests"][name]["content_hash"] for name in MANIFEST_NAMES}
    pair_id = generated["manifests"]["denpar_near_duplicate_review_queue"]["pairs"][0]["pair_id"]
    record_review(v1, pair_id, "reviewer-fixture", decision, "Fixture adjudication")
    applied = apply_manual_decisions(v1, v2, dataset, training)
    assert applied["sizes"]["denpar_clean_train_manifest"] == 2
    assert applied["sizes"]["denpar_clean_validation_manifest"] == 0
    assert applied["sizes"]["denpar_clean_historical_test_manifest"] == expected_test
    assert applied["sizes"]["denpar_exclusion_manifest"] == expected_exclusions
    assert applied["manifests"]["denpar_near_duplicate_review_queue"]["quarantined_images"] == []
    assert applied["gate"]["status"] == "ready_for_stage2b_with_historical_test_limitations"
    assert_stage2b_ready(v2)
    assert v1_hashes == {name: json.loads((v1 / f"{name}.json").read_text())["content_hash"] for name in MANIFEST_NAMES}
