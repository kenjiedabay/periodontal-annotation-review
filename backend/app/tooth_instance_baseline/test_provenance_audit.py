"""Geometry tests for the auditable mask transform."""

import numpy as np

from .provenance_audit import transformed_mask


def test_letterbox_mask_transform_preserves_nonempty_geometry() -> None:
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:8, 4:12] = True
    transformed, metadata = transformed_mask(mask, 20, 10, target=100)
    assert transformed.shape == (100, 100)
    assert transformed.any()
    assert metadata["resized_size"] == [100, 50]
    assert metadata["offset"] == [0, 25]
