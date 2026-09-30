"""Perio-KPT tooth-level landmark preparation and evaluation utilities.

This package is intentionally separate from the immutable DenPAR Mask R-CNN
pipeline. Gate 1 contains data validation and deterministic CPU preparation
only; it does not construct or train a neural network.
"""

from .schema import LANDMARK_NAMES, OBJECT_CLASS_NAMES, SCHEMA_VERSION

__all__ = ["LANDMARK_NAMES", "OBJECT_CLASS_NAMES", "SCHEMA_VERSION"]
