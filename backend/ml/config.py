"""Configurable training parameters."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    manifest_path: Path = Path("../dataset/manifests/dataset_manifest.json")
    dataset_root: Path = Path("../dataset")
    checkpoint_dir: Path = Path("checkpoints")
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 7
    min_delta: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    pretrained: bool = True
    freeze_backbone_epochs: int = 1
    use_class_weights: bool = True
    positive_label: str = "present"
    negative_label: str = "absent"

    def validate(self) -> "TrainingConfig":
        if self.image_size <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("image_size, batch_size, and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
        if self.patience < 1 or self.num_workers < 0:
            raise ValueError("patience must be positive and num_workers cannot be negative")
        if self.positive_label == self.negative_label:
            raise ValueError("positive and negative labels must differ")
        return self

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}
