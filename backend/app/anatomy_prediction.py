"""Combined research-only inference for image-level CEJ, apex, and bone-line overlays."""
from __future__ import annotations

import base64
import io
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


class Conv(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1), nn.ReLU(inplace=True))


class SmallUNet(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        self.down1, self.down2, self.bridge = Conv(1, 8), Conv(8, 16), Conv(16, 32)
        self.up2, self.up1 = Conv(48, 16), Conv(24, 8)
        self.head = nn.Conv2d(8, channels, 1)

    def forward(self, value):
        first = self.down1(value)
        second = self.down2(nn.functional.max_pool2d(first, 2))
        bridge = self.bridge(nn.functional.max_pool2d(second, 2))
        bridge = self.up2(torch.cat((nn.functional.interpolate(bridge, size=second.shape[-2:]), second), dim=1))
        return self.head(self.up1(torch.cat((nn.functional.interpolate(bridge, size=first.shape[-2:]), first), dim=1)))


def _peaks(probability: np.ndarray, threshold: float, limit: int) -> list[tuple[int, int, float]]:
    maximum = cv2.dilate(probability, np.ones((9, 9), np.uint8))
    candidates = ((probability >= threshold) & (probability >= maximum - 1e-6)).astype(np.uint8)
    count, labels = cv2.connectedComponents(candidates)
    result = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if len(xs):
            best = int(np.argmax(probability[ys, xs]))
            result.append((int(xs[best]), int(ys[best]), float(probability[ys[best], xs[best]])))
    return sorted(result, key=lambda row: row[2], reverse=True)[:limit]


class AnatomyPredictionService:
    def __init__(self):
        self.lock = threading.Lock()
        self.loaded = False
        self.error: str | None = None
        self.bone: SmallUNet | None = None
        self.landmarks: SmallUNet | None = None
        self.bone_config: dict[str, Any] = {}
        self.landmark_config: dict[str, Any] = {}

    def _load(self) -> None:
        if self.loaded or self.error:
            return
        try:
            bone_checkpoint = torch.load(ROOT / 'models/bone_line_unet_256/best.pt', map_location='cpu', weights_only=True)
            landmark_checkpoint = torch.load(ROOT / 'models/denpar_cej_apex_heatmap_balanced_256/best.pt', map_location='cpu', weights_only=True)
            self.bone = SmallUNet(1).eval()
            self.landmarks = SmallUNet(2).eval()
            self.bone.load_state_dict(bone_checkpoint['state_dict'], strict=True)
            self.landmarks.load_state_dict(landmark_checkpoint['state_dict'], strict=True)
            self.bone_config = bone_checkpoint
            self.landmark_config = landmark_checkpoint
            self.loaded = True
        except Exception as exc:  # model availability is reported, never replaced by fake output
            self.error = f'{type(exc).__name__}: {exc}'

    def predict(self, image_id: str, original: np.ndarray) -> dict[str, Any]:
        with self.lock:
            self._load()
            if not self.loaded or self.bone is None or self.landmarks is None:
                return {'model_status': 'unavailable', 'image_id': image_id, 'model_error': self.error}
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.ndim == 3 else original
            height, width = gray.shape[:2]

            bone_size = int(self.bone_config['size'])
            bone_input = cv2.resize(gray, (bone_size, bone_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            with torch.no_grad():
                bone_probability = self.bone(torch.from_numpy(bone_input[None, None])).sigmoid()[0, 0].numpy()
            bone_mask = (bone_probability >= float(self.bone_config['threshold'])).astype(np.uint8) * 255
            bone_mask = cv2.resize(bone_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[bone_mask > 0] = (255, 176, 32, 210)
            ok, encoded = cv2.imencode('.png', rgba)
            if not ok:
                raise RuntimeError('Could not encode bone-line overlay')

            landmark_size = int(self.landmark_config['size'])
            scale = min(landmark_size / width, landmark_size / height)
            scaled_width, scaled_height = round(width * scale), round(height * scale)
            left, top = (landmark_size - scaled_width) // 2, (landmark_size - scaled_height) // 2
            canvas = np.zeros((landmark_size, landmark_size), dtype=np.uint8)
            canvas[top:top + scaled_height, left:left + scaled_width] = cv2.resize(gray, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
            with torch.no_grad():
                probabilities = self.landmarks(torch.from_numpy(canvas.astype(np.float32)[None, None] / 255.0)).sigmoid()[0].numpy()
            threshold = float(self.landmark_config['threshold'])
            groups = []
            for channel, (kind, limit) in enumerate((('cej', 12), ('apex', 10))):
                points = []
                for x, y, confidence in _peaks(probabilities[channel], threshold, limit):
                    original_x, original_y = (x - left) / scale, (y - top) / scale
                    if 0 <= original_x < width and 0 <= original_y < height:
                        points.append({'x': original_x, 'y': original_y, 'confidence': confidence})
                groups.append((kind, points))
            return {
                'model_status': 'available', 'image_id': image_id, 'width': width, 'height': height,
                'bone_overlay_url': 'data:image/png;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii'),
                'bone_threshold': float(self.bone_config['threshold']),
                'landmark_threshold': threshold, 'points': dict(groups),
                'model_versions': {'bone_level': f"bone_line_unet_256_epoch_{self.bone_config['epoch']}",
                                   'key_points': f"cej_apex_heatmap_256_epoch_{self.landmark_config['epoch']}"},
                'scope': 'image-level predictions without confirmed tooth or surface correspondence',
            }


service = AnatomyPredictionService()
