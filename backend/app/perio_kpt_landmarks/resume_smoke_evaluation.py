"""Finish evaluation A from the completed Gate 2 epoch; never trains."""

from __future__ import annotations

from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import PerioLandmarkNet, masked_heatmap_loss
from .thermal import ThermalMonitor
from .train import DEFAULT_MANIFEST, DEFAULT_OUTPUT, evaluate, seed_everything, verify_checkpoint
from .training_data import PerioCropDataset


def loss_only(model, loader, device, monitor: ThermalMonitor) -> float:
    values = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if not monitor.wait_if_hot():
                raise RuntimeError("evaluation stopped by thermal protection")
            with torch.amp.autocast("cuda", enabled=True):
                loss = masked_heatmap_loss(model(batch["image"].to(device, non_blocking=True)),
                                           batch["target"].to(device, non_blocking=True),
                                           batch["availability"].to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite post-epoch training-set loss")
            values.append(float(loss))
    return float(np.mean(values))


def main() -> None:
    seed_everything(42)
    output = DEFAULT_OUTPUT
    checkpoint = output / "best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    fold = next(item for item in manifest["folds"] if item["fold"] == 0)
    train_data = PerioCropDataset(DEFAULT_MANIFEST, fold["train_image_ids"], 256)
    validation_data = PerioCropDataset(DEFAULT_MANIFEST, fold["validation_image_ids"], 256)
    options = {"batch_size": 4, "num_workers": 2, "pin_memory": True, "persistent_workers": True}
    train_loader = DataLoader(train_data, shuffle=False, **options)
    validation_loader = DataLoader(validation_data, shuffle=False, **options)
    device = torch.device("cuda:0")
    model = PerioLandmarkNet(11); model.load_state_dict(payload["model_state_dict"], strict=True); model.to(device)
    previous_rows = list(csv.DictReader((output / "gpu_telemetry.csv").open(encoding="utf-8")))
    previous_max_temp = max(int(row["temp_c"]) for row in previous_rows)
    previous_max_memory = max(int(row["memory_used_mib"]) for row in previous_rows)
    monitor = ThermalMonitor(output / "evaluation_gpu_telemetry.csv", 75, 68, 80, 30)
    torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter(); monitor.start()
    try:
        train_loss = loss_only(model, train_loader, device, monitor)
        validation_loss, metrics = evaluate(model, validation_loader, device, True, output, monitor)
        reload_result = verify_checkpoint(checkpoint, config)
    finally:
        monitor.close()
    elapsed = time.perf_counter() - started
    result = {"status": "completed_after_evaluation_edge_case_fix", "epochs_trained": 1,
              "additional_training_performed": False,
              "train_loss": train_loss, "train_loss_definition": "post-epoch evaluation on fold-0 training crops",
              "validation_loss": validation_loss, "metrics": metrics,
              "thermal": {"maximum_temperature_c": max(previous_max_temp, monitor.maximum_temperature),
                          "events": monitor.events},
              "memory": {"nvidia_smi_max_used_mib": max(previous_max_memory, monitor.maximum_memory_mib),
                         "total_vram_mib": 6144,
                         "evaluation_torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                         "evaluation_torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20},
              "training_process_wall_seconds": 56.1,
              "evaluation_process_seconds": elapsed,
              "total_compute_seconds": 56.1 + elapsed,
              "checkpoint": str(checkpoint), "checkpoint_reload": reload_result,
              "note": "The epoch completed once. Evaluation resumed without training after masking a source landmark outside its prescribed crop."}
    (output / "smoke_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
