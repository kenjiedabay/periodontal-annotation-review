"""Configuration for reproducible, leakage-aware dataset preparation."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreparationConfig:
    processed_dir: Path
    annotations_dir: Path
    output_dir: Path
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    augmentations_per_training_image: int = 1
    require_expert_validated: bool = True
    group_map_path: Path | None = None
    rotation_degrees: float = 3.0
    translation_fraction: float = 0.02
    scale_fraction: float = 0.02
    brightness_variation: float = 0.05
    contrast_variation: float = 0.05

    def validate(self) -> "PreparationConfig":
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("train_ratio, validation_ratio, and test_ratio must be non-negative and sum to 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.augmentations_per_training_image < 0:
            raise ValueError("augmentations_per_training_image cannot be negative")
        if not 0 <= self.rotation_degrees <= 5:
            raise ValueError("rotation_degrees must be between 0 and 5 for conservative augmentation")
        if not 0 <= self.translation_fraction <= 0.05 or not 0 <= self.scale_fraction <= 0.05:
            raise ValueError("translation and scale variation must be between 0 and 0.05")
        if not 0 <= self.brightness_variation <= 0.10 or not 0 <= self.contrast_variation <= 0.10:
            raise ValueError("brightness and contrast variation must be between 0 and 0.10")
        return self

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value)
        return values
