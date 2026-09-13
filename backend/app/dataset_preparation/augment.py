"""Conservative training-only image augmentation."""

from dataclasses import dataclass
import random

from PIL import Image, ImageEnhance

from .config import PreparationConfig


@dataclass(frozen=True)
class AugmentationResult:
    image: Image.Image
    operations: list[str]


def augment_for_training(image: Image.Image, config: PreparationConfig, rng: random.Random) -> AugmentationResult:
    """Apply mild geometric and intensity changes that do not invent anatomy."""
    working = image.convert("L")
    operations: list[str] = []
    angle = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
    if abs(angle) > 0.001:
        working = working.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        operations.append(f"rotation_degrees:{angle:.3f}")

    width, height = working.size
    shift_x = int(rng.uniform(-config.translation_fraction, config.translation_fraction) * width)
    shift_y = int(rng.uniform(-config.translation_fraction, config.translation_fraction) * height)
    if shift_x or shift_y:
        translated = Image.new("L", working.size, 0)
        translated.paste(working, (shift_x, shift_y))
        working = translated
        operations.append(f"translation_pixels:{shift_x},{shift_y}")

    scale = 1 + rng.uniform(-config.scale_fraction, config.scale_fraction)
    if abs(scale - 1) > 0.0001:
        resized = working.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (width, height), 0)
        left = (width - resized.width) // 2
        top = (height - resized.height) // 2
        canvas.paste(resized, (left, top))
        working = canvas
        operations.append(f"scale:{scale:.4f}")

    brightness = 1 + rng.uniform(-config.brightness_variation, config.brightness_variation)
    contrast = 1 + rng.uniform(-config.contrast_variation, config.contrast_variation)
    working = ImageEnhance.Brightness(working).enhance(brightness)
    working = ImageEnhance.Contrast(working).enhance(contrast)
    operations.extend([f"brightness_factor:{brightness:.4f}", f"contrast_factor:{contrast:.4f}"])
    return AugmentationResult(working, operations)
