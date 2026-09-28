"""Reproducible research-only training/evaluation for tooth-instance masks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .data import ToothInstanceDataset, collate, validate_official_split
from .model import create_model


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def move_targets(targets: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    return [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in item.items()} for item in targets]


def mask_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    intersection = torch.logical_and(left, right).sum().item()
    union = torch.logical_or(left, right).sum().item()
    return intersection / union if union else 0.0


def metrics(prediction: dict[str, torch.Tensor], target: dict[str, Any], threshold: float, match_iou: float) -> dict[str, Any]:
    predicted = prediction["masks"][:, 0] >= threshold
    truth = target["masks"].bool()
    pairs, used = [], set()
    for predicted_index, predicted_mask in enumerate(predicted):
        candidates = [(mask_iou(predicted_mask.cpu(), truth[index].cpu()), index) for index in range(len(truth)) if index not in used]
        if candidates:
            score, truth_index = max(candidates)
            if score >= match_iou:
                used.add(truth_index); pairs.append((predicted_index, truth_index, score))
    predicted_union = predicted.any(dim=0) if len(predicted) else torch.zeros_like(truth[0], dtype=torch.bool)
    truth_union = truth.any(dim=0)
    tp = torch.logical_and(predicted_union.cpu(), truth_union.cpu()).sum().item()
    fp = torch.logical_and(predicted_union.cpu(), ~truth_union.cpu()).sum().item()
    fn = torch.logical_and(~predicted_union.cpu(), truth_union.cpu()).sum().item()
    dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {"dice": dice, "iou": iou, "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0, "matched_instances": len(pairs), "missed_instances": len(truth) - len(pairs), "false_positive_instances": len(predicted) - len(pairs), "pair_ious": [item[2] for item in pairs]}


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, threshold: float, match_iou: float, visual_dir: Path | None = None) -> dict[str, Any]:
    model.eval(); per_image = []
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for image, output, target in zip(images, outputs, targets):
                item = metrics(output, target, threshold, match_iou)
                item["image_id"] = target["metadata"]["image_id"]
                per_image.append(item)
                if visual_dir and len(per_image) <= 8:
                    _visualize(image, target["masks"], output["masks"], visual_dir / f"{item['image_id']}.png", threshold)
    aggregate = {key: float(np.mean([item[key] for item in per_image])) if per_image else 0.0 for key in ("dice", "iou", "precision", "recall", "matched_instances", "missed_instances", "false_positive_instances")}
    aggregate["mean_matched_instance_iou"] = float(np.mean([score for item in per_image for score in item["pair_ious"]])) if any(item["pair_ious"] for item in per_image) else 0.0
    return {"matching_iou_threshold": match_iou, "mask_score_threshold": threshold, "aggregate": aggregate, "per_image": per_image, "error_analysis": {"missed_teeth": int(sum(item["missed_instances"] for item in per_image)), "false_positive_regions": int(sum(item["false_positive_instances"] for item in per_image)), "merged_or_split_instances": "requires qualitative review; reported visualizations distinguish prediction and ground truth", "poor_boundary_delineation_images": [item["image_id"] for item in per_image if item["iou"] < 0.5]}}


def _visualize(image: torch.Tensor, truth: torch.Tensor, predicted: torch.Tensor, path: Path, threshold: float) -> None:
    base = (image[0].numpy() * 255).astype(np.uint8)
    gt, pred = truth.bool().any(dim=0).numpy(), (predicted[:, 0] >= threshold).any(dim=0).cpu().numpy() if len(predicted) else np.zeros_like(base, dtype=bool)
    panel = np.repeat(base[..., None], 3, axis=2)
    panel[gt] = [40, 210, 100]; panel[pred] = [230, 70, 70]; panel[np.logical_and(gt, pred)] = [240, 210, 45]
    path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(panel).save(path)


def run(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(config["seed"])
    root = Path(config["dataset_root"])
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    records, validation = {}, {}
    for split in ("train", "validation", "test"):
        records[split], validation[split] = validate_official_split(root, split, Path(config["training_images_dir"]))
    if config.get("split_manifest"):
        manifest = json.loads(Path(config["split_manifest"]).read_text(encoding="utf-8"))
        allowed = set(manifest["teacher_training_ids"])
        records["train"] = [record for record in records["train"] if record.image_id in allowed]
        if len(records["train"]) != len(allowed):
            raise ValueError("Split manifest does not match usable Training images")
        if config.get("pseudo_labels_dir"):
            pseudo_root = Path(config["pseudo_labels_dir"])
            pseudo_records = []
            for image_id in manifest["pseudo_label_candidate_ids"]:
                paths = tuple(sorted((pseudo_root / image_id).glob("*.png")))
                if paths:
                    source = Path(config["training_images_dir"]) / f"{image_id}.jpg"
                    with Image.open(source) as image:
                        width, height = image.size
                    from .data import Record
                    pseudo_records.append(Record(image_id, source, paths, width, height))
            records["train"].extend(pseudo_records)
            validation["pseudo_labels"] = {"accepted_images": len(pseudo_records), "accepted_instances": sum(len(record.mask_paths) for record in pseudo_records)}
    (output / "dataset_validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if any(not records[split] for split in records):
        blocked = {"task": "tooth_instance_segmentation", "training_status": "blocked", "research_only": True, "reason": "One or more official partitions have no usable image-mask pairs; training is refused rather than fabricating pairs or splits.", "dataset_validation": validation, "limitations": ["Training radiographs are absent from the supplied DenPAR tree.", "Patient/case-level grouping is unavailable.", "No checkpoint, prediction, or performance metric was generated."]}
        (output / "evaluation_report.json").write_text(json.dumps(blocked, indent=2), encoding="utf-8")
        return blocked
    datasets = {"train": ToothInstanceDataset(records["train"], config["max_side"], augment=True, seed=config["seed"]), "validation": ToothInstanceDataset(records["validation"], config["max_side"], seed=config["seed"]), "test": ToothInstanceDataset(records["test"], config["max_side"], seed=config["seed"])}
    loaders = {name: DataLoader(dataset, batch_size=config["batch_size"], shuffle=name == "train", num_workers=0, collate_fn=collate) for name, dataset in datasets.items()}
    device = torch.device("cuda" if torch.cuda.is_available() and not config["force_cpu"] else "cpu")
    model, model_metadata = create_model(config["pretrained"], config["max_side"]); model.to(device)
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=config["learning_rate"], momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    logs, best, patience, total_steps = [], -1.0, 0, 0
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("amp", False) and device.type == "cuda"))
    for epoch in range(1, config["epochs"] + 1):
        model.train(); loss_sums: dict[str, float] = {}; batches = 0
        epoch_start = time.perf_counter()
        for images, targets in loaders["train"]:
            if config.get("max_steps") and total_steps >= config["max_steps"]:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                losses = model([image.to(device) for image in images], move_targets(list(targets), device))
                total = sum(losses.values())
            scaler.scale(total).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(optimizer); scaler.update(); batches += 1; total_steps += 1
            for name, value in losses.items(): loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().cpu())
        validation_metrics = evaluate(model, loaders["validation"], device, config["mask_score_threshold"], config["match_iou_threshold"])
        mean_losses = {name: value / max(batches, 1) for name, value in loss_sums.items()}
        entry = {"epoch": epoch, "training_loss_components": mean_losses, "validation": validation_metrics["aggregate"], "learning_rate": optimizer.param_groups[0]["lr"], "duration_seconds": time.perf_counter()-epoch_start, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0, "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}; logs.append(entry)
        print(f"epoch={epoch} steps={total_steps} validation_dice={entry['validation']['dice']:.4f} duration_s={entry['duration_seconds']:.1f}", flush=True)
        score = validation_metrics["aggregate"]["dice"]; scheduler.step(score)
        if score > best:
            best, patience = score, 0
            torch.save({"state_dict": model.state_dict(), "architecture": model_metadata, "epoch": epoch, "config": config, "metrics": entry, "preprocessing_version": "resize_longest_side_v1", "augmentation_version": "horizontal_flip_v1", "dataset_manifest_version": "official_denpar_split_v1", "saved_at": datetime.now(timezone.utc).isoformat()}, output / "best_checkpoint.pt")
        else:
            patience += 1
            if patience >= config["early_stopping_patience"]: break
        torch.save({"state_dict": model.state_dict(), "architecture": model_metadata, "epoch": epoch, "config": config, "metrics": entry}, output / "latest_checkpoint.pt")
        if config.get("max_steps") and total_steps >= config["max_steps"]:
            break
    checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["state_dict"])
    test = evaluate(model, loaders["test"], device, config["mask_score_threshold"], config["match_iou_threshold"], output / "qualitative_predictions") if config.get("evaluate_test", True) else None
    report = {"task": "tooth_instance_segmentation", "research_only": True, "clinical_validity": "not established", "dataset": {"official_split": {"train": len(records["train"]), "validation": len(records["validation"]), "test": len(records["test"])}, "split_limitation": "Official DenPAR partitions used, but patient/case disjointness cannot be verified because case IDs are unavailable.", "validation": validation}, "model": model_metadata, "configuration": config, "training_log": logs, "best_validation_dice": best, "test_results": test, "limitations": ["Tooth segmentation is not periodontal disease detection.", "Tooth masks do not identify diseased teeth.", "No severity or progression target was used.", "Patient/case-level leakage cannot be ruled out from available metadata."]}
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="../DenPAR Radiographs Dataset/Dataset")
    parser.add_argument("--output-dir", default="../models/tooth_instance_maskrcnn_baseline"); parser.add_argument("--training-images-dir", default="../dataset/raw")
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.0025); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--mask-score-threshold", type=float, default=0.5); parser.add_argument("--match-iou-threshold", type=float, default=0.5); parser.add_argument("--no-pretrained", dest="pretrained", action="store_false"); parser.add_argument("--force-cpu", action="store_true"); parser.add_argument("--config-json")
    parser.add_argument("--split-manifest"); parser.add_argument("--pseudo-labels-dir"); parser.add_argument("--skip-test", dest="evaluate_test", action="store_false")
    parser.add_argument("--max-steps", type=int)
    parser.set_defaults(pretrained=True); config = vars(parser.parse_args())
    config_json = config.pop("config_json", None)
    if config_json:
        frozen = json.loads(Path(config_json).read_text(encoding="utf-8"))
        training = frozen["training"]
        config.update({"seed": training["seed"], "batch_size": training["batch_size"], "learning_rate": training["learning_rate"], "epochs": training["maximum_epochs"], "early_stopping_patience": training["early_stopping_patience"], "amp": training["amp"], "match_iou_threshold": training["instance_matching_iou_threshold"], "max_side": frozen["model"]["max_size"]})
    print(json.dumps(run(config), indent=2))


if __name__ == "__main__": main()
