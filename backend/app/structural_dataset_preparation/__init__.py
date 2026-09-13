"""Read-only derivation of non-diagnostic DenPAR structural training data."""

from .preparer import PreparationConfig, prepare_structural_dataset

__all__ = ["PreparationConfig", "prepare_structural_dataset"]
