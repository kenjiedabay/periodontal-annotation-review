"""Frozen interpretation of the supplied Perio-KPT standard-box labels."""

from __future__ import annotations

SCHEMA_VERSION = "perio-kpt-standard-box-v1"

OBJECT_CLASS_NAMES = {
    0: "single_root",
    1: "double_root",
    2: "triple_root",
    3: "arr_region",
    4: "pls_region",
}

LANDMARK_NAMES = (
    "CEJ-m",
    "BL-m",
    "RL-m",
    "CEJ-d",
    "BL-d",
    "RL-d",
    "RL-c",
    "FA",
    "FBL-m",
    "FBL-d",
    "ARR",
)

LANDMARK_INDEX = {name: index for index, name in enumerate(LANDMARK_NAMES)}
EXPECTED_ROW_VALUES = 5 + 3 * len(LANDMARK_NAMES)
VALID_VISIBILITY = {0, 1, 2}
TOOTH_CLASSES = {0, 1, 2}

RBL_SURFACES = {
    "mesial": (LANDMARK_INDEX["CEJ-m"], LANDMARK_INDEX["BL-m"], LANDMARK_INDEX["RL-m"]),
    "distal": (LANDMARK_INDEX["CEJ-d"], LANDMARK_INDEX["BL-d"], LANDMARK_INDEX["RL-d"]),
}


def rbl_indices(class_id: int, surface: str) -> tuple[int, int, int]:
    """Resolve the dataset's class-dependent apex convention for one surface."""
    if surface not in RBL_SURFACES:
        raise ValueError(f"unknown RBL surface: {surface}")
    cej, bone_level, surface_root = RBL_SURFACES[surface]
    apex = LANDMARK_INDEX["RL-c"] if class_id == 0 else surface_root
    return cej, bone_level, apex
