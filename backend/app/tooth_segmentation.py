"""Cached epoch-8 inference, with all returned geometry in original pixels."""
import base64
import logging
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from app.preprocessing.config import load_config
from app.preprocessing.transforms import apply_transforms

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'models/tooth_instance_maskrcnn_baseline/full_run/best_checkpoint.pt'
DISCLAIMER = 'Research prototype. Tooth segmentation output does not indicate periodontal disease, severity, or treatment need.'


def mask_png(mask):
    # Binary foreground represented as opaque white; background is transparent.
    rgba = np.full((*mask.shape, 4), 255, dtype=np.uint8)
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    ok, encoded = cv2.imencode('.png', rgba)
    if not ok:
        raise ValueError('Mask encoding failed')
    return 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')


class ToothSegmentationService:
    def __init__(self):
        self.lock = Lock()
        self.attempted = False
        self.model = None
        self.load_error = None

    def load(self):
        if self.attempted:
            return
        self.attempted = True
        try:
            import torch
            from app.tooth_instance_baseline.model import create_model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
            if checkpoint['epoch'] != 8 or checkpoint['config']['max_side'] != 1024:
                raise ValueError('Unexpected checkpoint epoch or input geometry')
            model, _ = create_model(pretrained=False, max_size=1024)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            self.model = model.to(self.device).eval()
            self.load_error = None
        except Exception as error:
            logging.exception('Tooth segmentation checkpoint unavailable')
            self.model = None
            self.load_error = f'{type(error).__name__}: {error}'

    def predict(self, image_id, original):
        height, width = original.shape[:2]
        response = dict(image_id=image_id, task='tooth_instance_segmentation',
                        model_status='unavailable', model_version='maskrcnn_epoch8',
                        confidence_threshold=0.5, mask_threshold=0.5, instances=[],
                        width=width, height=height, coordinate_space='original_image',
                        bbox_format='xyxy', disclaimer=DISCLAIMER, ground_truth=None)
        with self.lock:
            self.load()
            if self.model is None:
                response['model_error'] = self.load_error or 'The model service could not load the checkpoint.'
                return response
            import torch
            processed, _, _ = apply_transforms(original, load_config())
            tensor = torch.from_numpy(processed.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
            with torch.inference_mode():
                prediction = self.model([tensor.to(self.device)])[0]
            scale = min(1024 / width, 1024 / height)
            rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
            ox, oy = (1024 - rw) // 2, (1024 - rh) // 2
            for score, label, box, mask in zip(prediction['scores'], prediction['labels'], prediction['boxes'], prediction['masks']):
                if float(score) < 0.5 or int(label) != 1:
                    continue
                binary = (mask[0].cpu().numpy() >= 0.5).astype(np.uint8)
                restored = cv2.resize(binary[oy:oy+rh, ox:ox+rw], (width, height), interpolation=cv2.INTER_NEAREST)
                if not restored.any():
                    continue
                bbox = box.cpu().numpy().astype(float)
                bbox[[0, 2]] = np.clip((bbox[[0, 2]] - ox) * width / rw, 0, width)
                bbox[[1, 3]] = np.clip((bbox[[1, 3]] - oy) * height / rh, 0, height)
                response['instances'].append(dict(instance_id=len(response['instances'])+1,
                    confidence=float(score), bbox=bbox.tolist(), mask_url=mask_png(restored)))
            response['model_status'] = 'available'
        response['ground_truth'] = ground_truth(image_id, original)
        return response


def ground_truth(image_id, original):
    if not image_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in image_id):
        return None
    root = ROOT / 'DenPAR Radiographs Dataset/Dataset'
    for split in ('Validation', 'Testing'):
        source = root / split / 'Images' / f'{image_id}.jpg'
        if not source.is_file() and split == 'Testing':
            # Compatibility with the original DenPAR package layout.
            source = root / 'Images' / f'{image_id}.jpg'
        if not source.is_file():
            continue
        decoded = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if decoded is None or not np.array_equal(decoded, original):
            continue
        masks = []
        for path in sorted((root / split / 'Masks (Tooth-wise)' / image_id).glob('*.png')):
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape == original.shape[:2] and mask.any():
                masks.append(mask_png(mask > 0))
        if masks:
            return dict(split=split, mask_urls=masks, source='DenPAR tooth-wise masks; exact source pixel match')
    return None


service = ToothSegmentationService()
