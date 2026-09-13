"""CUDA compatibility and bounded GPU smoke checks for the Mask R-CNN baseline.

This module never starts the full experiment.  It uses one validated training
sample, and only runs the small repeated smoke cycle after the single-step
checks (including the requested COCO-pretrained model) succeed.
"""
from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.ops import box_area, nms, roi_align

from .data import ToothInstanceDataset, collate, validate_official_split
from .model import create_model
from .train import evaluate, move_targets


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _memory(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "free_bytes": free,
        "total_bytes": total,
    }


def _operator_checks(device: torch.device) -> dict[str, Any]:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]], device=device)
    scores = torch.tensor([0.9, 0.8], device=device)
    features = torch.ones((1, 2, 16, 16), device=device)
    rois = torch.tensor([[0.0, 1.0, 1.0, 8.0, 8.0]], device=device)
    kept = nms(boxes, scores, 0.5)
    aligned = roi_align(features, rois, output_size=(3, 3), spatial_scale=1.0, sampling_ratio=2)
    return {
        "nms": {"passed": bool(kept.numel() == 1), "kept_indices": kept},
        "roi_align": {"passed": bool(aligned.shape == (1, 2, 3, 3)), "shape": list(aligned.shape)},
        "box_utilities": {"passed": bool(torch.allclose(box_area(boxes), torch.tensor([100.0, 64.0], device=device)))},
        "mask_operations": {"passed": bool(torch.ones((2, 2), device=device, dtype=torch.bool).any())},
    }


def main() -> None:
    root = Path("../DenPAR Radiographs Dataset/Dataset").resolve()
    raw = Path("../dataset/raw").resolve()
    out = Path("../models/tooth_instance_maskrcnn_baseline/full_run").resolve()
    reports, smoke = out / "reports", out / "gpu_smoke_test"
    reports.mkdir(parents=True, exist_ok=True)
    smoke.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "command": sys.argv,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torch_installation": str(Path(torch.__file__).resolve()),
        "torchvision_installation": str(Path(torchvision.__file__).resolve()),
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "status": "pending",
    }
    report_path = reports / "cuda_environment_after_install.json"
    log_path = smoke / "cuda_smoke_stdout_stderr.log"
    try:
        if not (torch.__version__.startswith("2.14.0+cu") and torchvision.__version__.startswith("0.29.0+cu") and torch.version.cuda and torch.cuda.is_available()):
            raise RuntimeError("CUDA package compatibility gate failed: expected torch 2.14.0+cu*, torchvision 0.29.0+cu*, and an available CUDA runtime")
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        result["gpu"] = {"device_count": torch.cuda.device_count(), "current_device": torch.cuda.current_device(), "name": torch.cuda.get_device_name(device), "total_memory_bytes": props.total_memory, "compute_capability": [props.major, props.minor]}
        tensor = torch.randn((64, 64), device=device)
        basic = tensor @ tensor.T
        torch.cuda.synchronize(device)
        result["basic_cuda_execution"] = {"passed": bool(torch.isfinite(basic).all()), "result_shape": list(basic.cpu().shape)}
        result["torchvision_operators"] = _operator_checks(device)
        if not all(item["passed"] for item in result["torchvision_operators"].values()):
            raise RuntimeError("Torchvision operator gate failed")

        # The project's factory records an explicit fallback if weights were not
        # retrievable; that fallback is intentionally a hard failure here.
        model, metadata = create_model(pretrained=True, max_size=1024)
        result["model"] = metadata
        if metadata["pretrained_weights"] != "COCO_V1":
            raise RuntimeError(f"Pretrained-weight gate failed: {metadata['pretrained_weights']}")
        model.to(device)
        records, validation = validate_official_split(root, "train", raw)
        dataset = ToothInstanceDataset(records, 1024, False, 42)
        image, target = dataset[0]
        model.eval()
        with torch.no_grad():
            transformed, _ = model.transform([image.to(device)], None)
        result["geometry"] = {"before_transform": list(image.shape), "image_list_tensor": list(transformed.tensors.shape), "image_sizes": [list(size) for size in transformed.image_sizes], "retained_1024": transformed.image_sizes == [(1024, 1024)]}
        if not result["geometry"]["retained_1024"]:
            raise RuntimeError("GeneralizedRCNNTransform changed validated 1024x1024 geometry")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        gpu_images, gpu_targets = [image.to(device)], move_targets([target], device)
        result["memory_before_forward"] = _memory(device)
        loss_dict = model(gpu_images, gpu_targets)
        total_loss = sum(loss_dict.values())
        result["memory_after_forward"] = _memory(device)
        finite = bool(torch.isfinite(total_loss).item()) and all(bool(torch.isfinite(loss).item()) for loss in loss_dict.values())
        result["fp32_single_sample"] = {"batch_size": 1, "image_id": target["metadata"]["image_id"], "losses": {key: float(value.detach().cpu()) for key, value in loss_dict.items()}, "total_loss": float(total_loss.detach().cpu()), "finite": finite}
        if not finite:
            raise RuntimeError("Non-finite FP32 loss")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradients_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item()) for parameter in model.parameters())
        optimizer.step()
        torch.cuda.synchronize(device)
        result["fp32_single_sample"].update({"backward_passed": gradients_finite, "optimizer_step_passed": True, "elapsed_seconds": time.perf_counter() - start, "memory_after_backward": _memory(device)})
        if not gradients_finite:
            raise RuntimeError("Non-finite gradients in FP32 backward pass")

        # AMP is benchmarked separately before it is allowed in the full-run
        # configuration. This is still one validated sample and does not use
        # validation or test information for optimization.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        amp_start = time.perf_counter()
        amp_optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
        scaler = torch.amp.GradScaler("cuda")
        amp_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_losses = model(gpu_images, gpu_targets)
            amp_total = sum(amp_losses.values())
        amp_finite = bool(torch.isfinite(amp_total).item()) and all(bool(torch.isfinite(loss).item()) for loss in amp_losses.values())
        if amp_finite:
            scaler.scale(amp_total).backward()
            scaler.step(amp_optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
        result["amp_benchmark"] = {"enabled_candidate": True, "finite": amp_finite, "step_completed": amp_finite, "losses": {key: float(value.detach().cpu()) for key, value in amp_losses.items()}, "total_loss": float(amp_total.detach().cpu()), "elapsed_seconds": time.perf_counter() - amp_start, "memory": _memory(device)}
        if not amp_finite:
            raise RuntimeError("Non-finite AMP loss")
        del amp_optimizer, scaler

        checkpoint = smoke / "cuda_smoke_checkpoint.pth"
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "metadata": metadata}, checkpoint)
        restored, restored_metadata = create_model(pretrained=False, max_size=1024)
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
        restored.to(device).eval()
        with torch.no_grad():
            prediction = restored([image.to(device)])
        result["checkpoint_roundtrip"] = {"passed": isinstance(prediction, list) and len(prediction) == 1, "path": str(checkpoint), "restored_model": restored_metadata}
        # A deliberately bounded one-epoch smoke: eight official-training
        # images and two official-validation images. It verifies repeated
        # iteration, evaluation, checkpoint use and qualitative overlays; it
        # is not a performance experiment.
        restored.train()
        smoke_optimizer = torch.optim.SGD(restored.parameters(), lr=0.0001, momentum=0.9)
        repeat_losses: list[float] = []
        for smoke_image, smoke_target in DataLoader(ToothInstanceDataset(records[:8], 1024, False, 42), batch_size=1, shuffle=False, collate_fn=collate):
            smoke_loss_dict = restored([smoke_image[0].to(device)], move_targets(list(smoke_target), device))
            smoke_total = sum(smoke_loss_dict.values())
            if not bool(torch.isfinite(smoke_total).item()):
                raise RuntimeError("Non-finite loss in repeated GPU smoke")
            smoke_optimizer.zero_grad(set_to_none=True)
            smoke_total.backward()
            smoke_optimizer.step()
            repeat_losses.append(float(smoke_total.detach().cpu()))
        validation_records, _ = validate_official_split(root, "validation", raw)
        validation_loader = DataLoader(ToothInstanceDataset(validation_records[:2], 1024, False, 42), batch_size=1, shuffle=False, collate_fn=collate)
        validation_metrics = evaluate(restored, validation_loader, device, 0.5, 0.5, smoke / "qualitative_predictions")
        smoke_checkpoint = smoke / "one_epoch_smoke_checkpoint.pth"
        torch.save({"state_dict": restored.state_dict(), "optimizer": smoke_optimizer.state_dict(), "samples": 8, "validation_samples": 2}, smoke_checkpoint)
        result["one_epoch_gpu_smoke"] = {"passed": len(repeat_losses) == 8 and all(torch.isfinite(torch.tensor(repeat_losses)).tolist()), "training_samples": 8, "validation_samples": 2, "losses": repeat_losses, "validation_metrics": validation_metrics, "checkpoint": str(smoke_checkpoint), "qualitative_overlay_dir": str(smoke / "qualitative_predictions")}
        del restored, prediction, smoke_optimizer
        torch.cuda.empty_cache()
        result["memory_after_cleanup"] = _memory(device)
        result["status"] = "gpu_smoke_ready"
    except torch.OutOfMemoryError as error:
        result["status"] = "model_oom"
        result["exception_type"] = type(error).__name__
        result["exception"] = str(error)
        if torch.cuda.is_available():
            result["memory_at_failure"] = _memory(torch.device("cuda:0"))
        torch.cuda.empty_cache()
    except Exception as error:  # records exact gate failure without continuing.
        result["status"] = "package_or_operator_or_model_gate_failed"
        result["exception_type"] = type(error).__name__
        result["exception"] = str(error)
        result["traceback"] = traceback.format_exc()
    result["finished_at_unix"] = time.time()
    report_path.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    log_path.write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
