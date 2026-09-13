"""Mask R-CNN construction and future inference response schema."""

from __future__ import annotations

from typing import Any

import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn


def create_model(pretrained: bool = True, max_size: int = 1024) -> tuple[torch.nn.Module, dict[str, Any]]:
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    try:
        # Inputs are already exactly 1024×1024 via the audited preprocessing.
        # Torchvision's min_size must therefore be 1024, not 1 (which would
        # silently downsample the image to one pixel).
        model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None if not pretrained else None, min_size=max_size, max_size=max_size)
        source = "COCO_V1" if pretrained else "none"
    except Exception as error:
        if not pretrained:
            raise
        model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=max_size, max_size=max_size)
        source = f"unavailable_fallback_random_init: {type(error).__name__}"
    return model, {"architecture": "Mask R-CNN ResNet-50-FPN", "pretrained_weights": source, "frozen_layers": [], "fine_tuned_layers": "all", "total_parameters": sum(parameter.numel() for parameter in model.parameters()), "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}


def inference_response(image_id: str, model_version: str, instances: list[dict[str, Any]], inference_time_ms: float) -> dict[str, Any]:
    return {"image_id": image_id, "task": "tooth_instance_segmentation", "model_status": "available", "model_version": model_version, "instances": instances, "inference_time_ms": inference_time_ms, "clinical_interpretation": None}
