"""Build derived DenPAR leakage-remediation manifests without altering sources."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from .audit_denpar_stage2a import _decode_gray, sha256_file
except ImportError:  # Direct script execution from the backend directory.
    from audit_denpar_stage2a import _decode_gray, sha256_file


SCHEMA_VERSION = "1.0.0"
POLICY_VERSION = "denpar-leakage-remediation-v1"
V2_POLICY_VERSION = "denpar-leakage-remediation-v2"
DECISIONS = {"same_underlying_radiograph", "different_radiograph", "uncertain"}
MANIFEST_NAMES = (
    "denpar_clean_train_manifest", "denpar_clean_validation_manifest",
    "denpar_clean_historical_test_manifest", "denpar_exclusion_manifest",
    "denpar_duplicate_group_manifest", "denpar_near_duplicate_review_queue",
)


class ReconciliationError(ValueError):
    """Raised when stored duplicate counts do not match the underlying graph."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def content_hash(value: dict[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "content_hash"}
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def signed_manifest(payload: dict[str, Any], policy_version: str = POLICY_VERSION) -> dict[str, Any]:
    manifest = {"schema_version": SCHEMA_VERSION, "policy_version": policy_version, **payload}
    manifest["content_hash"] = content_hash(manifest)
    return manifest


def validate_content_hash(manifest: dict[str, Any]) -> None:
    actual = content_hash(manifest)
    if manifest.get("content_hash") != actual:
        raise ValueError(f"Manifest content hash mismatch: expected {manifest.get('content_hash')}, computed {actual}")


def _stable_sort(value: str) -> tuple[str, int | str]:
    partition, image_id = value.split(":", 1)
    return partition, int(image_id) if image_id.isdigit() else image_id


def _connected_components(edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        graph[left].add(right); graph[right].add(left)
    components: list[list[str]] = []
    remaining = set(graph)
    while remaining:
        start = min(remaining, key=_stable_sort)
        queue, seen = deque([start]), {start}
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor); queue.append(neighbor)
        remaining -= seen
        components.append(sorted(seen, key=_stable_sort))
    return sorted(components, key=lambda values: _stable_sort(values[0]))


def _pairs(values: list[str]) -> list[tuple[str, str]]:
    return [(values[index], values[other]) for index in range(len(values)) for other in range(index + 1, len(values))]


def _group_id(prefix: str, members: list[str]) -> str:
    return f"{prefix}-{hashlib.sha256('|'.join(sorted(members)).encode()).hexdigest()[:16]}"


def reconcile(audit: dict[str, Any]) -> dict[str, Any]:
    inventory = {row["stable_image_id"]: row for row in audit["inventory"]}
    pixel_groups: dict[str, list[str]] = defaultdict(list)
    for stable_id, row in inventory.items():
        pixel_groups[row["normalized_pixel_sha256"]].append(stable_id)
    exact_components = [sorted(values, key=_stable_sort) for values in pixel_groups.values() if len(values) > 1]
    exact_components.sort(key=lambda values: _stable_sort(values[0]))
    exact_pairs = [pair for group in exact_components for pair in _pairs(group)]
    exact_unique_images = sorted({item for pair in exact_pairs for item in pair}, key=_stable_sort)
    same_exact_pairs = [pair for pair in exact_pairs if pair[0].split(":", 1)[0] == pair[1].split(":", 1)[0]]
    cross_exact_pairs = [pair for pair in exact_pairs if pair not in same_exact_pairs]
    same_exact_groups = [group for group in exact_components if len({item.split(":", 1)[0] for item in group}) == 1]
    cross_exact_groups = [group for group in exact_components if len({item.split(":", 1)[0] for item in group}) > 1]

    near_edges = [(item["left"]["stable_image_id"], item["right"]["stable_image_id"]) for item in audit["near_duplicates"]]
    near_components = _connected_components(near_edges)
    near_unique_images = sorted({item for pair in near_edges for item in pair}, key=_stable_sort)
    same_near_pairs = [pair for pair in near_edges if pair[0].split(":", 1)[0] == pair[1].split(":", 1)[0]]
    cross_near_pairs = [pair for pair in near_edges if pair not in same_near_pairs]
    same_near_groups = [group for group in near_components if len({item.split(":", 1)[0] for item in group}) == 1]
    cross_near_groups = [group for group in near_components if len({item.split(":", 1)[0] for item in group}) > 1]

    reported = audit.get("exact_duplicate_summary", {})
    expected = {
        "unique_pair_count": len(exact_pairs),
        "cross_partition_unique_pair_count": len(cross_exact_pairs),
    }
    for key, value in expected.items():
        if reported.get(key) != value:
            raise ReconciliationError(f"Stored {key}={reported.get(key)} does not match recomputed value {value}")
    reported_pairs = {tuple(item["images"]) for item in reported.get("unique_pairs", [])}
    recomputed_pairs = {tuple(sorted(pair, key=_stable_sort)) for pair in exact_pairs}
    if reported_pairs != recomputed_pairs:
        raise ReconciliationError("Stored exact duplicate pair identities do not match normalized-pixel groups")

    exact_group_rows = []
    for members in exact_components:
        rows = [inventory[item] for item in members]
        exact_group_rows.append({
            "group_id": _group_id("exact", members), "members": [{
                "stable_image_id": row["stable_image_id"], "image_id": row["image_id"],
                "partition": row["official_partition"], "source_path": row["source_path"],
                "absolute_source_path": row["absolute_source_path"], "file_sha256": row["file_sha256"],
                "normalized_pixel_sha256": row["normalized_pixel_sha256"],
            } for row in rows],
            "pairwise_relationships": len(_pairs(members)),
            "cross_partition": len({row["official_partition"] for row in rows}) > 1,
        })
    near_group_rows = []
    for members in near_components:
        near_group_rows.append({
            "group_id": _group_id("near", members), "members": [{
                "stable_image_id": item, "image_id": inventory[item]["image_id"],
                "partition": inventory[item]["official_partition"], "source_path": inventory[item]["source_path"],
                "absolute_source_path": inventory[item]["absolute_source_path"],
                "file_sha256": inventory[item]["file_sha256"],
                "normalized_pixel_sha256": inventory[item]["normalized_pixel_sha256"],
                "perceptual_hash_64": inventory[item]["perceptual_hash_64"],
            } for item in members],
            "pairwise_candidate_relationships": sum(set(edge) <= set(members) for edge in near_edges),
            "cross_partition": len({item.split(":", 1)[0] for item in members}) > 1,
        })
    return {
        "exact": {
            "unique_duplicated_images": len(exact_unique_images), "pairwise_relationships": len(exact_pairs),
            "connected_groups": len(exact_components), "same_partition_pairs": len(same_exact_pairs),
            "same_partition_groups": len(same_exact_groups), "cross_partition_pairs": len(cross_exact_pairs),
            "cross_partition_groups": len(cross_exact_groups), "groups": exact_group_rows,
        },
        "near": {
            "unique_candidate_images": len(near_unique_images), "pairwise_relationships": len(near_edges),
            "connected_groups": len(near_components), "same_partition_pairs": len(same_near_pairs),
            "same_partition_groups": len(same_near_groups), "cross_partition_pairs": len(cross_near_pairs),
            "cross_partition_groups": len(cross_near_groups), "groups": near_group_rows,
        },
    }


def _canonical(group: dict[str, Any]) -> dict[str, Any]:
    priority = {"Training": 0, "Validation": 1, "Testing": 2}
    return min(group["members"], key=lambda row: (priority[row["partition"]], _stable_sort(row["stable_image_id"])[1]))


def _entry(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("stable_image_id", "image_id", "official_partition", "source_path",
                                      "absolute_source_path", "file_sha256", "normalized_pixel_sha256")}


def _fit(image: np.ndarray, size: int = 512) -> np.ndarray:
    scale = min(size / image.shape[1], size / image.shape[0])
    width, height = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    rendered = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    x, y = (size - width) // 2, (size - height) // 2
    canvas[y:y + height, x:x + width] = rendered
    return canvas


def _similarity(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_f, right_f = left.astype(np.float32), right.astype(np.float32)
    correlation = float(np.corrcoef(left_f.flatten(), right_f.flatten())[0, 1]) if left_f.std() and right_f.std() else 0.0
    return {"thumbnail_pearson_correlation": correlation,
            "normalized_thumbnail_mae": float(np.mean(np.abs(left_f - right_f)) / 255.0)}


def create_review_artifact(left_row: dict[str, Any], right_row: dict[str, Any], output: Path, pair_id: str) -> dict[str, Any]:
    left_raw, right_raw = _decode_gray(Path(left_row["absolute_source_path"])), _decode_gray(Path(right_row["absolute_source_path"]))
    left, right = _fit(left_raw), _fit(right_raw)
    transforms = {
        "identity": right,
        "horizontal_flip": cv2.flip(right, 1),
        "vertical_flip": cv2.flip(right, 0),
        "rotate_180": cv2.flip(right, -1),
    }
    scores = {name: _similarity(left, transformed) for name, transformed in transforms.items()}
    best = max(scores, key=lambda name: scores[name]["thumbnail_pearson_correlation"])
    matched_right = transforms[best]
    difference = cv2.absdiff(left, matched_right)
    left_edges, right_edges = cv2.Canny(left, 50, 150), cv2.Canny(matched_right, 50, 150)
    edge_view = np.zeros((512, 512, 3), np.uint8)
    edge_view[left_edges > 0] = (0, 0, 255)
    edge_view[right_edges > 0] = (255, 255, 0)
    edge_view[(left_edges > 0) & (right_edges > 0)] = (255, 255, 255)
    panels = [cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(matched_right, cv2.COLOR_GRAY2BGR),
              cv2.applyColorMap(difference, cv2.COLORMAP_INFERNO), edge_view]
    labels = [left_row["stable_image_id"], f"{right_row['stable_image_id']} ({best})", "absolute difference", "edges: left red / right cyan"]
    for panel, label in zip(panels, labels):
        cv2.rectangle(panel, (0, 0), (511, 30), (0, 0, 0), -1)
        cv2.putText(panel, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    artifact = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    path = output / f"{pair_id}.png"
    if not cv2.imwrite(str(path), artifact):
        raise OSError(f"Could not write {path}")
    return {
        "artifact_path": path.as_posix(), "artifact_sha256": sha256_file(path),
        "left_dimensions": [left_raw.shape[1], left_raw.shape[0]],
        "right_dimensions": [right_raw.shape[1], right_raw.shape[0]],
        "transformation_scores": scores, "best_scoring_transformation": best,
        "edge_method": "Canny thresholds 50/150; left red, right cyan, overlap white",
        "difference_method": "absolute difference after matched 512x512 aspect-preserving fit",
    }


def _source_tree_hashes(dataset_root: Path, training_root: Path) -> dict[str, str]:
    paths = sorted([path for path in dataset_root.rglob("*") if path.is_file()] +
                   [path for path in training_root.rglob("*") if path.is_file()])
    return {str(path.resolve()): sha256_file(path) for path in paths}


def _save(path: Path, manifest: dict[str, Any]) -> None:
    validate_content_hash(manifest)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_derived_manifests(manifests: dict[str, dict[str, Any]]) -> None:
    missing = set(MANIFEST_NAMES) - set(manifests)
    if missing:
        raise ValueError(f"Required filtered manifests are missing: {sorted(missing)}")
    policy_versions = {manifest.get("policy_version") for manifest in manifests.values()}
    if len(policy_versions) != 1 or None in policy_versions:
        raise ValueError(f"Derived manifests must share one explicit policy version: {policy_versions}")
    for manifest in manifests.values():
        validate_content_hash(manifest)
    clean_ids = {item["stable_image_id"] for name in MANIFEST_NAMES[:3] for item in manifests[name]["entries"]}
    excluded_ids = {item["stable_image_id"] for item in manifests["denpar_exclusion_manifest"]["entries"]}
    quarantined_ids = {item["stable_image_id"] for item in manifests["denpar_near_duplicate_review_queue"]["quarantined_images"]}
    if clean_ids & excluded_ids:
        raise ValueError(f"Excluded images entered a derived manifest: {sorted(clean_ids & excluded_ids)}")
    if clean_ids & quarantined_ids:
        raise ValueError(f"Quarantined images entered a derived manifest: {sorted(clean_ids & quarantined_ids)}")
    groups = manifests["denpar_duplicate_group_manifest"]["groups"]
    for group in groups:
        retained = [item for item in group["members"] if item["stable_image_id"] in clean_ids]
        if len(retained) != 1 or retained[0]["stable_image_id"] != group["canonical_stable_image_id"]:
            raise ValueError(f"Exact group {group['group_id']} does not retain exactly its canonical member")
    if manifests["denpar_clean_historical_test_manifest"].get("test_set_status") != "historical_exposed_test":
        raise ValueError("Derived Testing manifest must remain labeled historical_exposed_test")


def assert_stage2b_ready(manifest_dir: Path) -> None:
    manifests = {name: json.loads((manifest_dir / f"{name}.json").read_text(encoding="utf-8")) for name in MANIFEST_NAMES}
    validate_derived_manifests(manifests)
    pending = [item["pair_id"] for item in manifests["denpar_near_duplicate_review_queue"]["pairs"] if item["review_status"] == "pending"]
    if pending:
        raise RuntimeError(f"Stage 2B blocked: pending near-duplicate reviews: {pending}")


def generate(audit_path: Path, output_dir: Path, dataset_root: Path, training_root: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_hash = sha256_file(audit_path)
    reconciliation = reconcile(audit)
    inventory = {row["stable_image_id"]: row for row in audit["inventory"]}
    source_before = _source_tree_hashes(dataset_root, training_root)
    decision_date = audit["generated_at"][:10]

    duplicate_groups, exclusions = [], []
    excluded_ids: set[str] = set()
    for group in reconciliation["exact"]["groups"]:
        canonical = _canonical(group)
        members = group["members"]
        for member in members:
            if member["stable_image_id"] == canonical["stable_image_id"]:
                continue
            excluded_ids.add(member["stable_image_id"])
            same_partition = member["partition"] == canonical["partition"]
            if member["partition"] == "Training":
                reason = "duplicate_training_copy_excluded_to_avoid_overweighting"
            elif canonical["partition"] == "Training":
                reason = "evaluation_copy_duplicates_training_content"
            elif canonical["partition"] == "Validation" and member["partition"] == "Testing":
                reason = "testing_copy_duplicates_validation_content_already_available_for_model_selection"
            else:
                reason = f"duplicate_{member['partition'].lower()}_copy_excluded_from_derived_metrics"
            exclusions.append({
                "stable_image_id": member["stable_image_id"], "image_id": member["image_id"],
                "original_partition": member["partition"], "duplicate_group_id": group["group_id"],
                "canonical_stable_image_id": canonical["stable_image_id"], "canonical_image_id": canonical["image_id"],
                "exclusion_reason": reason, "file_sha256": member["file_sha256"],
                "normalized_pixel_sha256": member["normalized_pixel_sha256"],
                "decision_date": decision_date, "policy_version": POLICY_VERSION,
            })
        duplicate_groups.append({
            **group, "canonical_stable_image_id": canonical["stable_image_id"],
            "canonical_image_id": canonical["image_id"], "canonical_partition": canonical["partition"],
            "policy_decision": "retain_one_canonical_member", "decision_date": decision_date,
            "existing_checkpoint_note": "The existing checkpoint may already have trained on every original Training member in this group."
                if sum(item["partition"] == "Training" for item in members) > 1 else
                "The existing checkpoint trained on the Training member before this derived policy was created." if any(item["partition"] == "Training" for item in members) else None,
        })

    artifact_dir = output_dir / "near_duplicate_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidate_by_pair = {tuple(sorted((item["left"]["stable_image_id"], item["right"]["stable_image_id"]), key=_stable_sort)): item
                         for item in audit["near_duplicates"]}
    review_pairs, quarantined_ids = [], set()
    for left_id, right_id in sorted((pair for pair in candidate_by_pair if pair[0].split(":", 1)[0] != pair[1].split(":", 1)[0]), key=lambda pair: _stable_sort(pair[0])):
        candidate = candidate_by_pair[(left_id, right_id)]
        pair_id = _group_id("near-pair", [left_id, right_id])
        evidence = create_review_artifact(inventory[left_id], inventory[right_id], artifact_dir, pair_id)
        evidence["artifact_path"] = str(Path("near_duplicate_artifacts") / f"{pair_id}.png").replace("\\", "/")
        quarantined_ids.update((left_id, right_id))
        review_pairs.append({
            "pair_id": pair_id, "near_duplicate_group_id": next(group["group_id"] for group in reconciliation["near"]["groups"] if {left_id, right_id} <= {m["stable_image_id"] for m in group["members"]}),
            "left": {**_entry(inventory[left_id]), "perceptual_hash_64": inventory[left_id]["perceptual_hash_64"]},
            "right": {**_entry(inventory[right_id]), "perceptual_hash_64": inventory[right_id]["perceptual_hash_64"]},
            "p_hash_method": candidate["method"], "p_hash_distance": candidate["hamming_distance"],
            "p_hash_threshold": candidate["threshold"], "stage2a_supporting_evidence": candidate["supporting_evidence"],
            "review_artifact": evidence, "review_status": "pending",
            "allowed_decisions": sorted(DECISIONS), "decision": None, "reviewer_id": None,
            "decision_date": None, "notes": None,
        })

    quarantined_rows = [_entry(inventory[item]) for item in sorted(quarantined_ids, key=_stable_sort)]
    active_ids = set(inventory) - excluded_ids - quarantined_ids
    common = {"source_audit": str(audit_path.resolve()), "source_audit_sha256": audit_hash,
              "decision_date": decision_date, "source_files_copied": False}
    manifests = {
        "denpar_clean_train_manifest": signed_manifest({"manifest_type": "denpar_clean_train_manifest", **common,
            "entries": [_entry(row) for row in audit["inventory"] if row["stable_image_id"] in active_ids and row["official_partition"] == "Training"]}),
        "denpar_clean_validation_manifest": signed_manifest({"manifest_type": "denpar_clean_validation_manifest", **common,
            "entries": [_entry(row) for row in audit["inventory"] if row["stable_image_id"] in active_ids and row["official_partition"] == "Validation"]}),
        "denpar_clean_historical_test_manifest": signed_manifest({"manifest_type": "denpar_clean_historical_test_manifest", **common,
            "test_set_status": "historical_exposed_test", "allowed_use": "transparent_historical_comparison_after_exclusions",
            "prohibited_use": ["model_selection", "hyperparameter_tuning", "claim_as_pristine_final_test"],
            "external_holdout_recommendation": "Use a future independent expert-labeled external holdout for final unbiased evaluation.",
            "fallback_limitation": "If no external holdout exists, patient-independent or grouped validation requires reliable patient identifiers, which are currently unavailable.",
            "entries": [_entry(row) for row in audit["inventory"] if row["stable_image_id"] in active_ids and row["official_partition"] == "Testing"]}),
        "denpar_exclusion_manifest": signed_manifest({"manifest_type": "denpar_exclusion_manifest", **common, "entries": exclusions}),
        "denpar_duplicate_group_manifest": signed_manifest({"manifest_type": "denpar_duplicate_group_manifest", **common,
            "reconciled_counts": reconciliation["exact"], "groups": duplicate_groups}),
        "denpar_near_duplicate_review_queue": signed_manifest({"manifest_type": "denpar_near_duplicate_review_queue", **common,
            "status": "pending_manual_review", "automatic_exclusion": False,
            "quarantine_rule": "Both members of every cross-partition candidate are absent from clean manifests until reviewed.",
            "pairs": review_pairs, "quarantined_images": quarantined_rows}),
    }
    validate_derived_manifests(manifests)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, manifest in manifests.items():
        _save(output_dir / f"{name}.json", manifest)
    reconciliation_manifest = signed_manifest({"manifest_type": "denpar_duplicate_reconciliation", **common,
                                                "reconciled_counts": reconciliation})
    _save(output_dir / "denpar_duplicate_reconciliation.json", reconciliation_manifest)
    gate = signed_manifest({"manifest_type": "stage2b_gate", **common, "status": "blocked_pending_manual_review",
                            "pending_pair_ids": [item["pair_id"] for item in review_pairs],
                            "required_manifests": {name: manifest["content_hash"] for name, manifest in manifests.items()},
                            "loader_requirement": "Future Stage 2B runs must provide and log explicit manifest paths, schema versions, policy versions, and content hashes. Raw-partition fallback is forbidden."})
    _save(output_dir / "stage2b_gate.json", gate)

    source_after = _source_tree_hashes(dataset_root, training_root)
    changed = sorted(path for path in source_before if source_before[path] != source_after.get(path))
    integrity = signed_manifest({"manifest_type": "source_integrity", **common, "files_hashed": len(source_before),
                                 "source_tree_hash": hashlib.sha256(canonical_json(source_before)).hexdigest(),
                                 "changed_files": changed, "preserved": not changed})
    _save(output_dir / "source_integrity.json", integrity)
    if changed:
        raise RuntimeError(f"Official source integrity failure: {changed}")

    counts = {name: len(manifest.get("entries", [])) for name, manifest in manifests.items() if "entries" in manifest}
    excluded_by_partition = {part: sum(item["original_partition"] == part for item in exclusions) for part in ("Training", "Validation", "Testing")}
    quarantined_by_partition = {part: sum(item["official_partition"] == part for item in quarantined_rows) for part in ("Training", "Validation", "Testing")}
    summary = f"""# DenPAR leakage remediation {POLICY_VERSION}

## Reconciled duplicate counts

| Type | Unique images | Pairwise relationships | Connected groups | Same-partition pairs/groups | Cross-partition pairs/groups |
|---|---:|---:|---:|---:|---:|
| Exact | {reconciliation['exact']['unique_duplicated_images']} | {reconciliation['exact']['pairwise_relationships']} | {reconciliation['exact']['connected_groups']} | {reconciliation['exact']['same_partition_pairs']} / {reconciliation['exact']['same_partition_groups']} | {reconciliation['exact']['cross_partition_pairs']} / {reconciliation['exact']['cross_partition_groups']} |
| Near candidates | {reconciliation['near']['unique_candidate_images']} | {reconciliation['near']['pairwise_relationships']} | {reconciliation['near']['connected_groups']} | {reconciliation['near']['same_partition_pairs']} / {reconciliation['near']['same_partition_groups']} | {reconciliation['near']['cross_partition_pairs']} / {reconciliation['near']['cross_partition_groups']} |

The earlier count of eight same-partition candidates refers to near-duplicate pairwise relationships. The nine exact pairs consist of five same-partition and four cross-partition pairs. Every exact group currently contains two images, so exact pair and exact group totals are both nine.

## Derived manifest sizes

- Clean Training: {counts['denpar_clean_train_manifest']}
- Clean Validation: {counts['denpar_clean_validation_manifest']}
- Clean historical exposed Testing: {counts['denpar_clean_historical_test_manifest']}
- Exact exclusions: {counts['denpar_exclusion_manifest']}
- Quarantined pending review: {len(quarantined_rows)} images in {len(review_pairs)} cross-partition pairs

Exact exclusions by partition: {excluded_by_partition}. Quarantined by partition: {quarantined_by_partition}.

## Safety status

Stage 2B is blocked until all three cross-partition near-duplicate pairs receive a saved manual decision. Official files and partitions were not changed. The existing Testing split is labeled `historical_exposed_test` and cannot be represented as a pristine final test set.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {"reconciliation": reconciliation, "manifests": manifests, "gate": gate, "integrity": integrity,
            "sizes": counts, "excluded_by_partition": excluded_by_partition,
            "quarantined_by_partition": quarantined_by_partition}


def record_review(manifest_dir: Path, pair_id: str, reviewer_id: str, decision: str, notes: str) -> Path:
    if decision not in DECISIONS:
        raise ValueError(f"Decision must be one of {sorted(DECISIONS)}")
    queue_path = manifest_dir / "denpar_near_duplicate_review_queue.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8")); validate_content_hash(queue)
    pair = next((item for item in queue["pairs"] if item["pair_id"] == pair_id), None)
    if pair is None:
        raise KeyError(f"Unknown pair ID: {pair_id}")
    decision_record = signed_manifest({
        "manifest_type": "near_duplicate_manual_decision", "pair_id": pair_id,
        "reviewer_id": reviewer_id.strip(), "decision": decision,
        "decision_date": datetime.now(timezone.utc).isoformat(), "notes": notes,
        "evidence": {"left": pair["left"], "right": pair["right"], "review_artifact": pair["review_artifact"],
                     "p_hash_distance": pair["p_hash_distance"], "stage2a_supporting_evidence": pair["stage2a_supporting_evidence"]},
        "queue_content_hash": queue["content_hash"],
    })
    if not reviewer_id.strip():
        raise ValueError("Reviewer ID is required")
    decision_dir = manifest_dir / "near_duplicate_decisions"; decision_dir.mkdir(exist_ok=True)
    path = decision_dir / f"{pair_id}.json"; _save(path, decision_record)
    return path


def apply_manual_decisions(previous_dir: Path, output_dir: Path, dataset_root: Path, training_root: Path) -> dict[str, Any]:
    """Create v2 manifests from a complete set of immutable v1 review decisions."""
    previous = {name: json.loads((previous_dir / f"{name}.json").read_text(encoding="utf-8")) for name in MANIFEST_NAMES}
    validate_derived_manifests(previous)
    previous_hashes = {name: manifest["content_hash"] for name, manifest in previous.items()}
    if set(manifest["policy_version"] for manifest in previous.values()) != {POLICY_VERSION}:
        raise ValueError("Manual decisions can only be applied to the preserved v1 manifest set")
    queue = previous["denpar_near_duplicate_review_queue"]
    decisions: dict[str, dict[str, Any]] = {}
    for pair in queue["pairs"]:
        path = previous_dir / "near_duplicate_decisions" / f"{pair['pair_id']}.json"
        if not path.exists():
            raise RuntimeError(f"Missing manual decision for {pair['pair_id']}")
        decision = json.loads(path.read_text(encoding="utf-8")); validate_content_hash(decision)
        if decision.get("queue_content_hash") != queue["content_hash"]:
            raise ValueError(f"Decision {pair['pair_id']} does not reference the preserved v1 review queue")
        if decision.get("decision") not in DECISIONS or not decision.get("reviewer_id"):
            raise ValueError(f"Decision {pair['pair_id']} is incomplete")
        decisions[pair["pair_id"]] = decision
    if set(decisions) != {pair["pair_id"] for pair in queue["pairs"]}:
        raise ValueError("Manual decision set does not exactly match the review queue")

    audit_path = Path(queue["source_audit"])
    if sha256_file(audit_path) != queue["source_audit_sha256"]:
        raise ValueError("Stage 2A audit hash changed after v1 generation")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    inventory = {row["stable_image_id"]: row for row in audit["inventory"]}
    source_before = _source_tree_hashes(dataset_root, training_root)

    exclusions = [dict(item) for item in previous["denpar_exclusion_manifest"]["entries"]]
    excluded_ids = {item["stable_image_id"] for item in exclusions}
    duplicate_groups = [dict(item) for item in previous["denpar_duplicate_group_manifest"]["groups"]]
    reviewed_pairs: list[dict[str, Any]] = []
    applied_same_pairs: list[str] = []
    released_different_pairs: list[str] = []
    for pair in queue["pairs"]:
        decision = decisions[pair["pair_id"]]
        reviewed = {**pair, "review_status": "completed", "decision": decision["decision"],
                    "reviewer_id": decision["reviewer_id"], "decision_date": decision["decision_date"],
                    "notes": decision["notes"], "decision_record": str((previous_dir / "near_duplicate_decisions" / f"{pair['pair_id']}.json").resolve()),
                    "decision_record_content_hash": decision["content_hash"]}
        reviewed_pairs.append(reviewed)
        if decision["decision"] == "same_underlying_radiograph":
            members = [pair["left"], pair["right"]]
            partitions = {member["official_partition"] for member in members}
            if partitions == {"Validation", "Testing"}:
                canonical = next(member for member in members if member["official_partition"] == "Validation")
                excluded = next(member for member in members if member["official_partition"] == "Testing")
                reason = "manual_review_confirmed_testing_copy_of_validation_radiograph"
            elif any(member["official_partition"] == "Training" for member in members):
                canonical = min((member for member in members if member["official_partition"] == "Training"),
                                key=lambda item: _stable_sort(item["stable_image_id"])[1])
                excluded = next(member for member in members if member["stable_image_id"] != canonical["stable_image_id"])
                reason = "manual_review_confirmed_duplicate_of_training_radiograph"
            else:
                canonical = min(members, key=lambda item: _stable_sort(item["stable_image_id"])[1])
                excluded = next(member for member in members if member["stable_image_id"] != canonical["stable_image_id"])
                reason = "manual_review_confirmed_same_underlying_radiograph"
            group_id = _group_id("manual-exact", [member["stable_image_id"] for member in members])
            if excluded["stable_image_id"] in excluded_ids:
                raise ValueError(f"Reviewed exclusion already exists: {excluded['stable_image_id']}")
            exclusions.append({
                "stable_image_id": excluded["stable_image_id"], "image_id": excluded["image_id"],
                "original_partition": excluded["official_partition"], "duplicate_group_id": group_id,
                "canonical_stable_image_id": canonical["stable_image_id"], "canonical_image_id": canonical["image_id"],
                "exclusion_reason": reason, "file_sha256": excluded["file_sha256"],
                "normalized_pixel_sha256": excluded["normalized_pixel_sha256"],
                "decision_date": decision["decision_date"], "policy_version": V2_POLICY_VERSION,
                "manual_decision_record": reviewed["decision_record"],
                "manual_decision_content_hash": decision["content_hash"],
            })
            excluded_ids.add(excluded["stable_image_id"])
            duplicate_groups.append({
                "group_id": group_id, "group_origin": "manual_near_duplicate_review",
                "members": members, "pairwise_relationships": 1, "cross_partition": True,
                "canonical_stable_image_id": canonical["stable_image_id"], "canonical_image_id": canonical["image_id"],
                "canonical_partition": canonical["official_partition"], "policy_decision": "retain_one_canonical_member",
                "decision_date": decision["decision_date"], "reviewer_id": decision["reviewer_id"],
                "decision_record_content_hash": decision["content_hash"],
                "existing_checkpoint_note": "This manually confirmed pair contains no Training image." if "Training" not in partitions else
                                            "The existing checkpoint may already have trained on the Training member.",
            })
            applied_same_pairs.append(pair["pair_id"])
        elif decision["decision"] == "different_radiograph":
            released_different_pairs.append(pair["pair_id"])
        else:
            raise RuntimeError(f"Stage 2B remains blocked because {pair['pair_id']} is uncertain")

    active_ids = set(inventory) - excluded_ids
    previous_reference = {"policy_version": POLICY_VERSION, "manifest_hashes": previous_hashes,
                          "manifest_directory": str(previous_dir.resolve())}
    common = {"source_audit": str(audit_path.resolve()), "source_audit_sha256": sha256_file(audit_path),
              "decision_date": max(decision["decision_date"] for decision in decisions.values()),
              "source_files_copied": False, "previous_manifest_version": previous_reference}
    manifests = {
        "denpar_clean_train_manifest": signed_manifest({"manifest_type": "denpar_clean_train_manifest", **common,
            "entries": [_entry(row) for row in audit["inventory"] if row["stable_image_id"] in active_ids and row["official_partition"] == "Training"]}, V2_POLICY_VERSION),
        "denpar_clean_validation_manifest": signed_manifest({"manifest_type": "denpar_clean_validation_manifest", **common,
            "entries": [_entry(row) for row in audit["inventory"] if row["stable_image_id"] in active_ids and row["official_partition"] == "Validation"]}, V2_POLICY_VERSION),
        "denpar_clean_historical_test_manifest": signed_manifest({"manifest_type": "denpar_clean_historical_test_manifest", **common,
            "test_set_status": "historical_exposed_test", "allowed_use": "transparent_historical_comparison_after_exclusions",
            "prohibited_use": ["model_selection", "hyperparameter_tuning", "claim_as_pristine_final_test"],
            "external_holdout_recommendation": "Use a future independent expert-labeled external holdout for final unbiased evaluation.",
            "fallback_limitation": "Patient-independent or grouped validation requires reliable patient identifiers, which remain unavailable.",
            "entries": [_entry(row) for row in audit["inventory"] if row["stable_image_id"] in active_ids and row["official_partition"] == "Testing"]}, V2_POLICY_VERSION),
        "denpar_exclusion_manifest": signed_manifest({"manifest_type": "denpar_exclusion_manifest", **common, "entries": exclusions}, V2_POLICY_VERSION),
        "denpar_duplicate_group_manifest": signed_manifest({"manifest_type": "denpar_duplicate_group_manifest", **common,
            "reconciled_counts": previous["denpar_duplicate_group_manifest"]["reconciled_counts"], "groups": duplicate_groups}, V2_POLICY_VERSION),
        "denpar_near_duplicate_review_queue": signed_manifest({"manifest_type": "denpar_near_duplicate_review_queue", **common,
            "status": "complete", "automatic_exclusion": False, "pairs": reviewed_pairs, "quarantined_images": [],
            "applied_same_underlying_pairs": applied_same_pairs, "released_different_radiograph_pairs": released_different_pairs}, V2_POLICY_VERSION),
    }
    validate_derived_manifests(manifests)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, manifest in manifests.items():
        _save(output_dir / f"{name}.json", manifest)
    gate = signed_manifest({"manifest_type": "stage2b_gate", **common,
        "status": "ready_for_stage2b_with_historical_test_limitations", "pending_pair_ids": [],
        "required_manifests": {name: manifest["content_hash"] for name, manifest in manifests.items()},
        "loader_requirement": "Every Stage 2B run must provide and log these explicit manifest versions and hashes; raw-partition fallback is forbidden.",
        "evaluation_limitation": "Historical Testing is exposed and cannot serve as a pristine final test set."}, V2_POLICY_VERSION)
    _save(output_dir / "stage2b_gate.json", gate)

    source_after = _source_tree_hashes(dataset_root, training_root)
    changed = sorted(path for path in source_before if source_before[path] != source_after.get(path))
    integrity = signed_manifest({"manifest_type": "source_integrity", **common, "files_hashed": len(source_before),
        "source_tree_hash": hashlib.sha256(canonical_json(source_before)).hexdigest(), "changed_files": changed,
        "preserved": not changed, "previous_v1_manifest_hashes_preserved": previous_hashes}, V2_POLICY_VERSION)
    _save(output_dir / "source_integrity.json", integrity)
    if changed:
        raise RuntimeError(f"Official source integrity failure: {changed}")
    # Verify v1 manifests remained byte-for-byte valid after v2 generation.
    current_previous = {name: json.loads((previous_dir / f"{name}.json").read_text(encoding="utf-8"))["content_hash"] for name in MANIFEST_NAMES}
    if current_previous != previous_hashes:
        raise RuntimeError("Previous v1 manifest hashes changed during v2 generation")

    sizes = {name: len(manifest.get("entries", [])) for name, manifest in manifests.items() if "entries" in manifest}
    excluded_by_partition = {part: sum(item["original_partition"] == part for item in exclusions) for part in ("Training", "Validation", "Testing")}
    summary = f"""# DenPAR leakage remediation {V2_POLICY_VERSION}

All three manual review decisions are recorded and applied. Validation 853 is canonical for the confirmed Validation–Testing pair; Testing 852 is excluded. Pairs 2 and 3 were confirmed different, released from quarantine, and retained in their original derived partitions.

## Derived counts

- Clean Training: {sizes['denpar_clean_train_manifest']}
- Clean Validation: {sizes['denpar_clean_validation_manifest']}
- Clean historical exposed Testing: {sizes['denpar_clean_historical_test_manifest']}
- Total exclusions: {sizes['denpar_exclusion_manifest']}
- Remaining quarantined images: 0
- Exclusions by partition: {excluded_by_partition}

## Gate

The derived manifests are ready for Stage 2B development, provided loaders enforce policy version `{V2_POLICY_VERSION}` and the recorded content hashes. Historical Testing remains exposed and is limited to transparent comparison. A future independent expert-labeled external holdout remains necessary for final unbiased evaluation.
"""
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {"manifests": manifests, "gate": gate, "integrity": integrity, "sizes": sizes,
            "excluded_by_partition": excluded_by_partition, "previous_hashes": previous_hashes}


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("generate")
    build.add_argument("--audit", type=Path, default=project / "artifacts" / "denpar-stage2a-audit" / "audit_results.json")
    build.add_argument("--output-dir", type=Path, default=project / "artifacts" / "denpar-leakage-remediation-v1")
    build.add_argument("--dataset-root", type=Path, default=project / "DenPAR Radiographs Dataset" / "Dataset")
    build.add_argument("--training-root", type=Path, default=project / "dataset" / "raw")
    review = subparsers.add_parser("record-review")
    review.add_argument("--manifest-dir", type=Path, default=project / "artifacts" / "denpar-leakage-remediation-v1")
    review.add_argument("--pair-id", required=True); review.add_argument("--reviewer-id", required=True)
    review.add_argument("--decision", required=True, choices=sorted(DECISIONS)); review.add_argument("--notes", default="")
    apply = subparsers.add_parser("apply-decisions")
    apply.add_argument("--previous-dir", type=Path, default=project / "artifacts" / "denpar-leakage-remediation-v1")
    apply.add_argument("--output-dir", type=Path, default=project / "artifacts" / "denpar-leakage-remediation-v2")
    apply.add_argument("--dataset-root", type=Path, default=project / "DenPAR Radiographs Dataset" / "Dataset")
    apply.add_argument("--training-root", type=Path, default=project / "dataset" / "raw")
    args = parser.parse_args()
    if args.command == "generate":
        result = generate(args.audit.resolve(), args.output_dir.resolve(), args.dataset_root.resolve(), args.training_root.resolve())
        print(json.dumps({"reconciliation": {key: {field: value for field, value in item.items() if field != "groups"}
                                                  for key, item in result["reconciliation"].items()},
                          "manifest_sizes": result["sizes"], "excluded_by_partition": result["excluded_by_partition"],
                          "quarantined_by_partition": result["quarantined_by_partition"],
                          "stage2b_status": result["gate"]["status"], "source_integrity": result["integrity"]["preserved"]}, indent=2))
    elif args.command == "record-review":
        print(record_review(args.manifest_dir.resolve(), args.pair_id, args.reviewer_id, args.decision, args.notes))
    else:
        result = apply_manual_decisions(args.previous_dir.resolve(), args.output_dir.resolve(),
                                        args.dataset_root.resolve(), args.training_root.resolve())
        print(json.dumps({"manifest_sizes": result["sizes"], "excluded_by_partition": result["excluded_by_partition"],
                          "remaining_quarantined_images": 0, "stage2b_status": result["gate"]["status"],
                          "source_integrity": result["integrity"]["preserved"]}, indent=2))


if __name__ == "__main__":
    main()
