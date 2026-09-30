"""Lightweight ResNet-18 encoder with a U-Net-style heatmap decoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = nn.functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((value, skip), dim=1))


class PerioLandmarkNet(nn.Module):
    def __init__(self, channels: int = 11):
        super().__init__()
        encoder = resnet18(weights=None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 48)
        self.head = nn.Sequential(ConvBlock(48, 32), nn.Conv2d(32, channels, 1))

    def load_encoder(self, weights_path: Path) -> None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        encoder = resnet18(weights=None)
        encoder.load_state_dict(state)
        self.stem.load_state_dict(nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu).state_dict())
        self.layer1.load_state_dict(encoder.layer1.state_dict())
        self.layer2.load_state_dict(encoder.layer2.state_dict())
        self.layer3.load_state_dict(encoder.layer3.state_dict())
        self.layer4.load_state_dict(encoder.layer4.state_dict())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stem = self.stem(image)                    # 1/2
        one = self.layer1(self.pool(stem))         # 1/4
        two = self.layer2(one)                     # 1/8
        three = self.layer3(two)                   # 1/16
        four = self.layer4(three)                  # 1/32
        value = self.up4(four, three)
        value = self.up3(value, two)
        value = self.up2(value, one)
        value = self.up1(value, stem)
        value = nn.functional.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(value)


def masked_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor,
                        availability: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("logit and target shapes differ")
    mask = availability[:, :, None, None].to(logits.dtype)
    denominator = mask.sum() * logits.shape[-2] * logits.shape[-1]
    if denominator.item() <= 0:
        raise ValueError("batch has no available landmarks")
    error = (logits.sigmoid() - targets).square() * mask
    return error.sum() / denominator


def balanced_masked_heatmap_loss(logits: torch.Tensor, targets: torch.Tensor,
                                 availability: torch.Tensor, valid_content: torch.Tensor,
                                 positive_threshold: float = 0.01) -> tuple[torch.Tensor, dict]:
    """Balance positive/background MSE separately for every visible channel."""
    if logits.shape != targets.shape:
        raise ValueError("logit and target shapes differ")
    probabilities = logits.sigmoid()
    content = valid_content[:, None].bool()
    visible = availability.bool()
    positive = (targets >= positive_threshold) & content
    negative = (targets < positive_threshold) & content
    squared = (probabilities - targets).square()
    positive_count = positive.sum(dim=(-2, -1)).clamp_min(1)
    negative_count = negative.sum(dim=(-2, -1)).clamp_min(1)
    foreground = (squared * positive).sum(dim=(-2, -1)) / positive_count
    background = (squared * negative).sum(dim=(-2, -1)) / negative_count
    if not visible.any():
        raise ValueError("batch has no available landmarks")
    per_channel = 0.5 * (foreground + background)
    total = per_channel[visible].mean()
    return total, {"foreground": foreground, "background": background,
                   "total": per_channel, "visible": visible}
