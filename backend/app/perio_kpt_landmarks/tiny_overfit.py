"""Authorized Gate 2.1 overfit test on exactly eight fixed fold-0 tooth crops."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .evaluation import decode_heatmaps
from .model import PerioLandmarkNet, balanced_masked_heatmap_loss
from .schema import LANDMARK_NAMES, TOOTH_CLASSES
from .thermal import ThermalMonitor
from .train import DEFAULT_ENCODER, DEFAULT_MANIFEST, seed_everything
from .training_data import PerioCropDataset

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "artifacts/perio-kpt-landmarks/gate2/tiny_overfit_fold0"


def select_fixed_records(dataset: PerioCropDataset, seed: int) -> list[int]:
    groups: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for index, (_, annotation) in enumerate(dataset.records):
        if annotation.class_id not in TOOTH_CLASSES:
            continue
        sample = dataset[index]
        source_available = sum(point.available for point in annotation.landmarks)
        if int(sample["availability"].sum()) == source_available:
            groups[annotation.class_id].append(index)
    rng = random.Random(seed)
    selected = (rng.sample(groups[0], 3) + rng.sample(groups[1], 3) + rng.sample(groups[2], 2))
    rng.shuffle(selected)
    return selected


def measure(model, loader, device) -> dict:
    model.eval(); errors = {name: [] for name in LANDMARK_NAMES}; losses = []; rows = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device); targets = batch["target"].to(device)
            available = batch["availability"].to(device); content = batch["valid_content"].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss, _ = balanced_masked_heatmap_loss(logits, targets, available, content)
            losses.append(float(loss)); points, confidence = decode_heatmaps(logits.float().sigmoid(), content)
            for item in range(images.shape[0]):
                row = {"record_id": batch["record_id"][item], "prediction": points[item].cpu().tolist(),
                       "target": batch["points"][item].tolist(),
                       "availability": batch["availability"][item].bool().tolist(),
                       "confidence": confidence[item].cpu().tolist(), "image": images[item].cpu()}
                rows.append(row)
                for channel, visible in enumerate(row["availability"]):
                    if visible:
                        errors[LANDMARK_NAMES[channel]].append(float(np.linalg.norm(
                            np.asarray(row["prediction"][channel]) - np.asarray(row["target"][channel]))))
    all_errors = np.asarray([value for values in errors.values() for value in values])
    diagonal = np.sqrt(2) * 256
    return {"loss": float(np.mean(losses)), "mean_pixel_error": float(all_errors.mean()),
            "pck_0.01": float((all_errors / diagonal <= .01).mean()),
            "per_landmark": {name: {"count": len(values),
                                     "mean_pixel_error": float(np.mean(values)) if values else None}
                             for name, values in errors.items()}, "rows": rows}


def save_overlays(result: dict, directory: Path) -> list[str]:
    paths = []; directory.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(result["rows"][:4], 1):
        image = (row["image"][0].numpy() * 255).clip(0, 255).astype(np.uint8)
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for channel, visible in enumerate(row["availability"]):
            if not visible:
                continue
            target = tuple(int(round(value)) for value in row["target"][channel])
            prediction = tuple(int(round(value)) for value in row["prediction"][channel])
            cv2.circle(overlay, target, 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.drawMarker(overlay, prediction, (0, 0, 255), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        path = directory / f"{number:02d}_{row['record_id'].replace(':', '_')}.png"
        cv2.imwrite(str(path), overlay); paths.append(str(path))
    return paths


def clean_metrics(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "rows"}


def run(args) -> dict:
    if (args.fold, args.sample_count, args.seed) != (0, 8, 42) or args.epochs > 200:
        raise ValueError("authorization is restricted to fold 0, 8 crops, seed 42, and at most 200 epochs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the authorized thermal-protected test")
    seed_everything(args.seed); OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    fold = next(item for item in manifest["folds"] if item["fold"] == 0)
    full_dataset = PerioCropDataset(DEFAULT_MANIFEST, fold["train_image_ids"], 256)
    selected_indices = select_fixed_records(full_dataset, args.seed)
    selected = Subset(full_dataset, selected_indices)
    loader = DataLoader(selected, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    sample_records = [{"dataset_index": index, "record_id": full_dataset.records[index][1].record_id,
                       "class_id": full_dataset.records[index][1].class_id} for index in selected_indices]
    (OUTPUT / "selected_records.json").write_text(json.dumps(sample_records, indent=2), encoding="utf-8")
    device = torch.device("cuda:0"); model = PerioLandmarkNet(11); model.load_encoder(DEFAULT_ENCODER); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    monitor = ThermalMonitor(OUTPUT / "gpu_telemetry.csv", 75, 68, 80, 30)
    initial = measure(model, loader, device); before_paths = save_overlays(initial, OUTPUT / "before")
    initial_loss = initial["loss"]
    log_path = OUTPUT / "per_landmark_loss.csv"
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["epoch", "landmark", "foreground_loss", "background_loss", "total_loss"])
    torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter(); monitor.start()
    stopped_thermally = False; epochs_completed = 0; early_stopped = False
    try:
        for epoch in range(1, args.epochs + 1):
            model.train(); aggregates = {name: [[], [], []] for name in LANDMARK_NAMES}
            for batch in loader:
                if not monitor.wait_if_hot():
                    stopped_thermally = True; break
                images = batch["image"].to(device); targets = batch["target"].to(device)
                available = batch["availability"].to(device); content = batch["valid_content"].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=True):
                    loss, parts = balanced_masked_heatmap_loss(model(images), targets, available, content)
                if not torch.isfinite(loss): raise FloatingPointError("non-finite tiny-set loss")
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if any(not torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("non-finite tiny-set gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(optimizer); scaler.update()
                for channel, name in enumerate(LANDMARK_NAMES):
                    visible = parts["visible"][:, channel]
                    if visible.any():
                        for destination, source in zip(aggregates[name], (parts["foreground"], parts["background"], parts["total"])):
                            destination.extend(source[visible, channel].detach().float().cpu().tolist())
            if stopped_thermally: break
            epochs_completed = epoch
            with log_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                for name, values in aggregates.items():
                    if values[0]: writer.writerow([epoch, name, *(float(np.mean(value)) for value in values)])
            current = measure(model, loader, device)
            if epoch == 1 or epoch % 10 == 0:
                print(f"epoch={epoch} loss={current['loss']:.6f} error={current['mean_pixel_error']:.3f} pck={current['pck_0.01']:.3f}", flush=True)
            if (current["loss"] <= initial_loss * .10 and current["mean_pixel_error"] <= 2.0
                    and current["pck_0.01"] >= .95):
                early_stopped = True; break
        final = measure(model, loader, device)
        checkpoint = OUTPUT / ("thermal_stop.pt" if stopped_thermally else "best.pt")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epochs_completed,
                    "selected_records": sample_records, "status": "thermal_stop" if stopped_thermally else "completed"}, checkpoint)
        after_paths = save_overlays(final, OUTPUT / "after")
        reloaded = PerioLandmarkNet(11); payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reloaded.load_state_dict(payload["model_state_dict"], strict=True); reloaded.to(device).eval()
        first = next(iter(loader)); reference = first["image"].to(device)
        model.eval()
        with torch.inference_mode():
            expected = model(reference); actual = reloaded(reference)
        reload_result = {"success": bool(torch.allclose(expected, actual, atol=1e-6, rtol=1e-5)),
                         "maximum_absolute_difference": float((expected - actual).abs().max())}
    finally:
        monitor.close()
    elapsed = time.perf_counter() - started
    improvements = {name: {"initial_mean_pixel_error": initial["per_landmark"][name]["mean_pixel_error"],
                           "final_mean_pixel_error": final["per_landmark"][name]["mean_pixel_error"],
                           "improvement_pixels": (initial["per_landmark"][name]["mean_pixel_error"] - final["per_landmark"][name]["mean_pixel_error"]
                                                  if initial["per_landmark"][name]["mean_pixel_error"] is not None else None)}
                    for name in LANDMARK_NAMES}
    criteria = {"loss_reduction_at_least_90_percent": final["loss"] <= initial_loss * .10,
                "mean_pixel_error_at_most_2": final["mean_pixel_error"] <= 2.0,
                "pck_0.01_at_least_0.95": final["pck_0.01"] >= .95,
                "no_thermal_stop": not stopped_thermally, "checkpoint_reload_exact": reload_result["success"]}
    report = {"scope": "exactly eight fixed fold-0 training tooth crops; no augmentation or external evaluation",
              "epochs_completed": epochs_completed, "early_stopped": early_stopped,
              "criteria": criteria, "all_passed": all(criteria.values()),
              "initial": clean_metrics(initial), "final": clean_metrics(final),
              "per_landmark_improvement": improvements, "selected_records": sample_records,
              "before_overlays": before_paths, "after_overlays": after_paths,
              "thermal": {"maximum_temperature_c": monitor.maximum_temperature, "events": monitor.events},
              "memory": {"nvidia_smi_max_used_mib": monitor.maximum_memory_mib,
                         "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device)/2**20,
                         "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device)/2**20},
              "runtime_seconds": elapsed, "checkpoint": str(checkpoint), "checkpoint_reload": reload_result}
    (OUTPUT / "overfit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True); parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True); parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--thermal-pause-c", type=int, default=75); parser.add_argument("--thermal-resume-c", type=int, default=68)
    parser.add_argument("--thermal-stop-c", type=int, default=80)
    args = parser.parse_args()
    if (args.thermal_pause_c, args.thermal_resume_c, args.thermal_stop_c) != (75, 68, 80):
        raise ValueError("thermal thresholds are fixed at 75/68/80 C")
    run(args)


if __name__ == "__main__": main()
