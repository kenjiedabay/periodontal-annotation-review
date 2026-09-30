"""Controlled fold-0 Condition-A pilot; architecture and holdout are locked."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluation import decode_heatmaps
from .checkpointing import (load_exact_checkpoint, save_exact_checkpoint,
                            seed_worker, worker_seed_config)
from .geometry import calculate_rbl
from .model import PerioLandmarkNet, balanced_masked_heatmap_loss
from .schema import LANDMARK_NAMES, OBJECT_CLASS_NAMES, TOOTH_CLASSES, rbl_indices
from .thermal import ThermalMonitor
from .train import DEFAULT_ENCODER, DEFAULT_MANIFEST, seed_everything
from .training_data import PerioCropDataset

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "artifacts/perio-kpt-landmarks/gate2/pilot_fold0"
DEFAULT_RESUMABLE_OUTPUT = ROOT / "artifacts/perio-kpt-landmarks/gate2/pilot_fold0_30epoch_resumable_v1"


def run_epoch(model, loader, device, monitor, optimizer=None, scaler=None) -> float:
    training = optimizer is not None
    model.train(training); values = []
    for batch in loader:
        if not monitor.wait_if_hot():
            raise RuntimeError("thermal_stop")
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        available = batch["availability"].to(device, non_blocking=True)
        content = batch["valid_content"].to(device, non_blocking=True)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=True):
            loss, _ = balanced_masked_heatmap_loss(model(images), targets, available, content)
        if not torch.isfinite(loss): raise FloatingPointError("non-finite pilot loss")
        if training:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError("non-finite pilot gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
        values.append(float(loss.detach()))
    return float(np.mean(values))


def metric_group(errors: list[float], size: int = 256) -> dict:
    values = np.asarray(errors, dtype=np.float64); diagonal = np.sqrt(2.0) * size
    return {"count": int(values.size),
            "mean_pixel_error": float(values.mean()) if values.size else None,
            "normalized_mean_error": float((values / diagonal).mean()) if values.size else None,
            "pck_0.02": float((values / diagonal <= .02).mean()) if values.size else None,
            "pck_0.05": float((values / diagonal <= .05).mean()) if values.size else None,
            "pck_0.10": float((values / diagonal <= .10).mean()) if values.size else None}


def save_overlay(image, row, path: Path) -> None:
    pixels = (image[0].numpy() * 255).clip(0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    for channel, visible in enumerate(row["availability"]):
        if not visible: continue
        target = tuple(int(round(value)) for value in row["target"][channel])
        prediction = tuple(int(round(value)) for value in row["prediction"][channel])
        cv2.circle(overlay, target, 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.drawMarker(overlay, prediction, (0, 0, 255), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(path), overlay)


def evaluate_condition_a(model, loader, device, monitor, output: Path) -> dict:
    model.eval(); all_errors, by_landmark, by_class = [], defaultdict(list), defaultdict(list)
    rbl_errors = defaultdict(list); rbl_gt_assessable = defaultdict(int); rbl_prediction_invalid = defaultdict(int)
    rbl_gt_invalid = defaultdict(int); rows, overlay_paths = [], []
    with torch.inference_mode():
        for batch in loader:
            if not monitor.wait_if_hot(): raise RuntimeError("thermal_stop")
            images = batch["image"].to(device, non_blocking=True)
            content = batch["valid_content"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True): logits = model(images)
            points, confidence = decode_heatmaps(logits.float().sigmoid(), content)
            for item in range(images.shape[0]):
                class_id = int(batch["class_id"][item]); available = batch["availability"][item].bool().tolist()
                prediction = points[item].cpu().tolist(); target = batch["points"][item].tolist()
                row = {"record_id": batch["record_id"][item], "class_id": class_id,
                       "prediction": prediction, "target": target, "availability": available,
                       "confidence": confidence[item].cpu().tolist()}; rows.append(row)
                for channel, visible in enumerate(available):
                    if not visible: continue
                    error = float(np.linalg.norm(np.asarray(prediction[channel]) - np.asarray(target[channel])))
                    all_errors.append(error); by_landmark[channel].append(error); by_class[class_id].append(error)
                truth_points = [tuple(value) if visible else None for value, visible in zip(target, available)]
                predicted_points = [tuple(value) for value in prediction]
                for surface in ("mesial", "distal"):
                    indices = rbl_indices(class_id, surface)
                    truth = calculate_rbl(truth_points, surface, indices=indices)
                    predicted = calculate_rbl(predicted_points, surface, indices=indices)
                    if truth["status"] != "assessable": rbl_gt_invalid[surface] += 1
                    else:
                        rbl_gt_assessable[surface] += 1
                        if predicted["status"] != "assessable": rbl_prediction_invalid[surface] += 1
                        else: rbl_errors[surface].append(predicted["rbl_percent"] - truth["rbl_percent"])
                if len(overlay_paths) < 8:
                    path = output / "overlays" / f"{len(overlay_paths)+1:02d}_{row['record_id'].replace(':','_')}.png"
                    save_overlay(images[item].cpu(), row, path); overlay_paths.append(str(path))
    rbl = {}
    for surface in ("mesial", "distal"):
        values = np.asarray(rbl_errors[surface], dtype=np.float64)
        rbl[surface] = {"ground_truth_assessable_count": rbl_gt_assessable[surface],
                        "ground_truth_invalid_count": rbl_gt_invalid[surface],
                        "paired_assessable_count": int(values.size),
                        "prediction_invalid_count": rbl_prediction_invalid[surface],
                        "mae_percentage_points": float(np.abs(values).mean()) if values.size else None,
                        "rmse_percentage_points": float(np.sqrt(np.square(values).mean())) if values.size else None}
    report = {"overall": metric_group(all_errors),
              "per_landmark": {LANDMARK_NAMES[index]: metric_group(by_landmark[index])
                               for index in range(len(LANDMARK_NAMES))},
              "by_tooth_class": {OBJECT_CLASS_NAMES[class_id]: metric_group(by_class[class_id])
                                 for class_id in sorted(TOOTH_CLASSES)},
              "rbl": rbl, "overlays": overlay_paths,
              "condition": "A: ground-truth expanded box to landmark model"}
    (output / "validation_predictions.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    return report


def run(args) -> dict:
    if args.fold != 0 or args.epochs > 30 or args.seed != 42:
        raise ValueError("pilot authorization permits fold 0, seed 42, and at most 30 epochs only")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for protected pilot")
    seed_everything(args.seed); output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise FileExistsError(f"immutable run directory already contains files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8")); fold = manifest["folds"][0]
    train_data = PerioCropDataset(DEFAULT_MANIFEST, fold["train_image_ids"], 256, TOOTH_CLASSES)
    validation_data = PerioCropDataset(DEFAULT_MANIFEST, fold["validation_image_ids"], 256, TOOTH_CLASSES)
    options = {"batch_size": 4, "num_workers": 2, "pin_memory": True, "persistent_workers": True}
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator,
                              worker_init_fn=seed_worker, **options)
    validation_loader = DataLoader(validation_data, shuffle=False,
                                   worker_init_fn=seed_worker, **options)
    device = torch.device("cuda:0"); model = PerioLandmarkNet(11); model.load_encoder(DEFAULT_ENCODER); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    config = {"fold": 0, "maximum_epochs": args.epochs, "seed": 42, "image_size": 256,
              "batch_size": 4, "workers": 2, "amp": True, "augmentation": "none",
              "allowed_classes": [0,1,2], "train_records": len(train_data),
              "validation_records": len(validation_data), "encoder_weights": str(DEFAULT_ENCODER),
              "manifest_sha256": manifest["manifest_sha256"], "early_stopping_patience": args.patience}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    monitor = ThermalMonitor(output / "gpu_telemetry.csv", 75, 68, 80, 30)
    torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter(); monitor.start()
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0; thermal_stop = False
    start_epoch = 1
    checkpoint = output / "best.pt"
    latest_checkpoint = output / "latest.pt"
    worker_config = worker_seed_config(2, True, args.seed)
    if args.resume is not None:
        resumed = load_exact_checkpoint(args.resume, model=model, optimizer=optimizer,
                                        scaler=scaler, scheduler=None,
                                        dataloader_generator=generator,
                                        expected_config=config,
                                        expected_manifest_hash=manifest["manifest_sha256"])
        start_epoch = int(resumed["completed_epoch"]) + 1
        state = resumed["training_state"]
        history, best_loss = state["history"], state["best_loss"]
        best_epoch, stale = state["best_epoch"], state["stale_epochs"]
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            try:
                train_loss = run_epoch(model, train_loader, device, monitor, optimizer, scaler)
                validation_loss = run_epoch(model, validation_loader, device, monitor)
            except RuntimeError as error:
                if str(error) != "thermal_stop": raise
                thermal_stop = True
                save_exact_checkpoint(output / "thermal_stop.pt", model=model, optimizer=optimizer,
                                      scaler=scaler, scheduler=None, dataloader_generator=generator,
                                      completed_epoch=epoch-1, config=config,
                                      source_fold_manifest=manifest, worker_seeding=worker_config,
                                      training_state={"history":history,"best_loss":best_loss,
                                                      "best_epoch":best_epoch,"stale_epochs":stale,
                                                      "status":"thermal_stop"})
                break
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
            print(f"epoch={epoch}/{args.epochs} train={train_loss:.6f} validation={validation_loss:.6f}", flush=True)
            if validation_loss < best_loss - args.minimum_delta:
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                save_exact_checkpoint(checkpoint, model=model, optimizer=optimizer,
                                      scaler=scaler, scheduler=None, dataloader_generator=generator,
                                      completed_epoch=epoch, config=config,
                                      source_fold_manifest=manifest, worker_seeding=worker_config,
                                      training_state={"history":history,"best_loss":validation_loss,
                                                      "best_epoch":epoch,"stale_epochs":0,
                                                      "status":"best_validation"})
            else:
                stale += 1
            save_exact_checkpoint(latest_checkpoint, model=model, optimizer=optimizer,
                                  scaler=scaler, scheduler=None, dataloader_generator=generator,
                                  completed_epoch=epoch, config=config,
                                  source_fold_manifest=manifest, worker_seeding=worker_config,
                                  training_state={"history":history,"best_loss":best_loss,
                                                  "best_epoch":best_epoch,"stale_epochs":stale,
                                                  "status":"latest_completed_epoch"})
            if stale >= args.patience:
                break
        if thermal_stop: raise RuntimeError("pilot stopped by thermal protection; inspect thermal_stop.pt")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True); model.to(device)
        metrics = evaluate_condition_a(model, validation_loader, device, monitor, output)
        reloaded = PerioLandmarkNet(11); reload_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=3e-4, weight_decay=1e-4)
        reload_scaler = torch.amp.GradScaler("cuda", enabled=True); reload_generator = torch.Generator().manual_seed(args.seed)
        verified = load_exact_checkpoint(checkpoint, model=reloaded, optimizer=reload_optimizer,
                                         scaler=reload_scaler, scheduler=None,
                                         dataloader_generator=reload_generator,
                                         expected_config=config,
                                         expected_manifest_hash=manifest["manifest_sha256"])
        reloaded.to(device).eval()
        sample = next(iter(validation_loader))["image"].to(device); model.eval()
        with torch.inference_mode(): expected, actual = model(sample), reloaded(sample)
        reload_result = {"success": bool(torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)),
                         "maximum_absolute_difference": float((expected-actual).abs().max()),
                         "epoch": payload["completed_epoch"],
                         "verified_sha256": verified["verified_sha256"],
                         "optimizer_state_restored": bool(reload_optimizer.state_dict()["state"]),
                         "generator_state_restored": bool(torch.equal(reload_generator.get_state(), verified["dataloader_generator_state"]))}
    finally:
        monitor.close()
    elapsed = time.perf_counter()-started
    report = {"status": "completed", "history": history, "best_epoch": best_epoch,
              "best_validation_loss": best_loss, "metrics": metrics,
              "thermal": {"maximum_temperature_c": monitor.maximum_temperature, "events": monitor.events},
              "memory": {"nvidia_smi_max_used_mib": monitor.maximum_memory_mib,
                         "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device)/2**20,
                         "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device)/2**20},
              "runtime_seconds": elapsed, "checkpoint": str(checkpoint),
              "checkpoint_reload": reload_result, "config": config}
    report["latest_checkpoint"] = str(latest_checkpoint)
    (output / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True); parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESUMABLE_OUTPUT)
    args = parser.parse_args(); run(args)


if __name__ == "__main__": main()
