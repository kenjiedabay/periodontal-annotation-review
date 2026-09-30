"""Isolated Perio-KPT landmark training entry point (Gate 2 smoke mode)."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluation import decode_heatmaps, save_overlay, summarize_predictions
from .model import PerioLandmarkNet, balanced_masked_heatmap_loss
from .thermal import ThermalMonitor
from .training_data import PerioCropDataset

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT / "artifacts/perio-kpt-landmarks/gate1/five_fold_manifest.json"
DEFAULT_OUTPUT = ROOT / "artifacts/perio-kpt-landmarks/gate2/smoke_fold0"
DEFAULT_ENCODER = ROOT / ".torch-cache/hub/checkpoints/resnet18-f37072fd.pth"


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_checkpoint(path: Path, model, optimizer, scaler, config: dict, epoch: int, status: str) -> None:
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(), "config": config, "epoch": epoch,
                "status": status}, path)


def verify_checkpoint(path: Path, config: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    reloaded = PerioLandmarkNet(channels=11)
    reloaded.load_state_dict(payload["model_state_dict"], strict=True)
    reloaded.eval()
    with torch.inference_mode():
        output = reloaded(torch.zeros((1, 3, config["image_size"], config["image_size"])))
    return {"success": bool(torch.isfinite(output).all()), "shape": list(output.shape),
            "epoch": payload["epoch"], "status": payload["status"]}


def evaluate(model, loader, device, amp_enabled: bool, output: Path,
             monitor: ThermalMonitor | None = None) -> tuple[float, dict]:
    model.eval(); losses, rows, overlay_inputs = [], [], []
    with torch.inference_mode():
        for batch in loader:
            if monitor is not None and not monitor.wait_if_hot():
                raise RuntimeError("evaluation stopped by thermal protection")
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            availability = batch["availability"].to(device, non_blocking=True)
            valid_content = batch["valid_content"].to(device, non_blocking=True)
            context = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            with context:
                logits = model(images)
                loss, _ = balanced_masked_heatmap_loss(logits, targets, availability, valid_content)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            losses.append(float(loss))
            points, confidence = decode_heatmaps(logits.float().sigmoid(), valid_content)
            for index in range(images.shape[0]):
                row = {"record_id": batch["record_id"][index], "class_id": int(batch["class_id"][index]),
                       "prediction": points[index].cpu().tolist(), "target": batch["points"][index].tolist(),
                       "availability": batch["availability"][index].bool().tolist(),
                       "confidence": confidence[index].cpu().tolist()}
                rows.append(row)
                if len(overlay_inputs) < 6:
                    overlay_inputs.append((images[index].cpu(), row))
    metrics = summarize_predictions(rows, loader.dataset.output_size)
    (output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for index, (image, row) in enumerate(overlay_inputs, 1):
        save_overlay(image, row, output / "overlays" / f"{index:02d}_{row['record_id'].replace(':', '_')}.png")
    return float(np.mean(losses)), metrics


def run(args) -> dict:
    if args.mode != "smoke" or args.fold != 0 or args.epochs != 1:
        raise ValueError("Gate 2 authorization permits only --mode smoke --fold 0 --epochs 1")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 2 smoke test requires CUDA for thermal protection and AMP")
    seed_everything(args.seed)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fold = next(item for item in manifest["folds"] if item["fold"] == args.fold)
    train_data = PerioCropDataset(args.manifest, fold["train_image_ids"], args.image_size)
    validation_data = PerioCropDataset(args.manifest, fold["validation_image_ids"], args.image_size)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers,
                      "pin_memory": True, "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    device = torch.device("cuda:0")
    model = PerioLandmarkNet(11)
    if not args.encoder_weights.is_file():
        raise FileNotFoundError(f"cached ResNet-18 weights unavailable: {args.encoder_weights}")
    model.load_encoder(args.encoder_weights)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    config.update({"train_records": len(train_data), "validation_records": len(validation_data),
                   "manifest_sha256": manifest["manifest_sha256"], "device": torch.cuda.get_device_name(0)})
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    monitor = ThermalMonitor(output / "gpu_telemetry.csv", args.thermal_pause_c,
                             args.thermal_resume_c, args.thermal_stop_c, args.thermal_log_seconds)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter(); monitor.start(); train_losses = []
    stopped_thermally = False
    try:
        model.train()
        for batch_index, batch in enumerate(train_loader, 1):
            if not monitor.wait_if_hot():
                stopped_thermally = True; break
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            availability = batch["availability"].to(device, non_blocking=True)
            valid_content = batch["valid_content"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                loss, _ = balanced_masked_heatmap_loss(model(images), targets, availability, valid_content)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at batch {batch_index}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError(f"non-finite training gradient at batch {batch_index}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            train_losses.append(float(loss.detach()))
            if batch_index % 10 == 0:
                print(f"epoch=1 batch={batch_index}/{len(train_loader)} loss={train_losses[-1]:.6f}", flush=True)
            if monitor.stopped.is_set():
                stopped_thermally = True; break
        status = "thermal_stop" if stopped_thermally else "completed"
        checkpoint = output / ("thermal_stop.pt" if stopped_thermally else "best.pt")
        save_checkpoint(checkpoint, model, optimizer, scaler, config, 1, status)
        if stopped_thermally:
            validation_loss, metrics = None, None
        else:
            validation_loss, metrics = evaluate(model, validation_loader, device, args.amp, output, monitor)
        reload_result = verify_checkpoint(checkpoint, config)
    finally:
        monitor.close()
    elapsed = time.perf_counter() - started
    result = {"status": "thermal_stop" if stopped_thermally else "completed", "epochs": 1,
              "train_loss": float(np.mean(train_losses)) if train_losses else None,
              "validation_loss": validation_loss, "metrics": metrics,
              "thermal": {"maximum_temperature_c": monitor.maximum_temperature, "events": monitor.events},
              "memory": {"nvidia_smi_max_used_mib": monitor.maximum_memory_mib,
                         "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                         "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20},
              "elapsed_seconds": elapsed, "checkpoint": str(checkpoint),
              "checkpoint_reload": reload_result}
    (output / "smoke_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", choices=("smoke",), required=True)
    value.add_argument("--fold", type=int, required=True); value.add_argument("--epochs", type=int, required=True)
    value.add_argument("--image-size", type=int, default=256); value.add_argument("--batch-size", type=int, default=4)
    value.add_argument("--workers", type=int, default=2); value.add_argument("--learning-rate", type=float, default=3e-4)
    value.add_argument("--amp", action="store_true"); value.add_argument("--seed", type=int, default=42)
    value.add_argument("--thermal-log-seconds", type=int, default=30)
    value.add_argument("--thermal-pause-c", type=int, default=75)
    value.add_argument("--thermal-resume-c", type=int, default=68)
    value.add_argument("--thermal-stop-c", type=int, default=80)
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--encoder-weights", type=Path, default=DEFAULT_ENCODER)
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
