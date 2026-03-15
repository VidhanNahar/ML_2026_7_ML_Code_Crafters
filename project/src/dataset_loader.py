"""Dataset loading and annotation parsing utilities for multi-object tracking.

Expected dataset layout
-----------------------
dataset/
├── images/
│   ├── train/          <- .jpg files
│   ├── val/
│   └── test/
└── labels/
    ├── train_original/ <- .txt annotation files
    ├── val_original/
    └── test_original/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Valid dataset splits and their corresponding label sub-folder names.
SPLIT_LABEL_MAP = {
    "train": "train_original",
    "val": "val_original",
    "test": "test_original",
}


@dataclass
class TrackedObject:
    """Container for one vehicle annotation in a frame."""

    frame_id: str
    class_name: str
    bbox: Tuple[int, int, int, int]  # (xmin, ymin, width, height)
    orientation: float


def get_split_dirs(dataset_dir: Path, split: str = "train") -> Tuple[Path, Path]:
    """Return (images_dir, labels_dir) for the requested split."""
    if split not in SPLIT_LABEL_MAP:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(SPLIT_LABEL_MAP)}.")
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / SPLIT_LABEL_MAP[split]
    return images_dir, labels_dir


def list_frames(images_dir: Path) -> List[str]:
    """Return sorted list of frame ids (stem names) found in images_dir."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    return sorted(p.stem for p in images_dir.glob("*.jpg"))


def polygon_to_bbox(points: List[float]) -> Tuple[int, int, int, int]:
    """Convert 4-point polygon coordinates to an axis-aligned bbox."""
    x_coords = points[0::2]
    y_coords = points[1::2]

    xmin = int(np.floor(min(x_coords)))
    ymin = int(np.floor(min(y_coords)))
    xmax = int(np.ceil(max(x_coords)))
    ymax = int(np.ceil(max(y_coords)))

    width = max(0, xmax - xmin)
    height = max(0, ymax - ymin)
    return xmin, ymin, width, height


def parse_annotation_file(annotation_path: Path, frame_id: str) -> List[TrackedObject]:
    """Parse one annotation txt file and return all vehicle objects.

    Returns an empty list (with a warning) if the file does not exist so that
    the rest of the pipeline can still display the raw image.
    """
    if not annotation_path.exists():
        print(f"[WARNING] Annotation file not found, skipping: {annotation_path}")
        return []

    objects: List[TrackedObject] = []
    with annotation_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    # The first line stores the flight metadata (e.g., flightHeight:66.25).
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 11:
            continue

        polygon_points = [float(value) for value in parts[:8]]
        class_name = parts[8]
        orientation = float(parts[10])
        bbox = polygon_to_bbox(polygon_points)

        objects.append(
            TrackedObject(
                frame_id=frame_id,
                class_name=class_name,
                bbox=bbox,
                orientation=orientation,
            )
        )

    return objects


def load_frame_and_annotations(images_dir: Path, labels_dir: Path, frame_id: str):
    """Load one image frame and its parsed annotations.

    Parameters
    ----------
    images_dir : Path  Path to the split's image folder (e.g. dataset/images/train/)
    labels_dir : Path  Path to the split's label folder (e.g. dataset/labels/train_original/)
    frame_id   : str   Stem name of the frame file (e.g. '000001')

    Returns the image and a (possibly empty) list of TrackedObjects.  A missing
    labels directory or annotation file is treated as a warning, not an error.
    """
    image_path = images_dir / f"{frame_id}.jpg"

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

    if not labels_dir.exists():
        print(f"[WARNING] Labels directory not found, no annotations will be shown: {labels_dir}")
        return image, []

    annotation_path = labels_dir / f"{frame_id}.txt"
    objects = parse_annotation_file(annotation_path, frame_id)
    return image, objects
