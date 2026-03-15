"""Visualization helpers for drawing detections."""

from __future__ import annotations

from typing import Iterable

import cv2

from dataset_loader import TrackedObject


def _color_for_class(class_name: str):
    """Generate a deterministic BGR color based on class name."""
    seed = sum(ord(ch) for ch in class_name)
    blue = 50 + (seed * 29) % 206
    green = 50 + (seed * 53) % 206
    red = 50 + (seed * 79) % 206
    return int(blue), int(green), int(red)


def draw_detections(image, objects: Iterable[TrackedObject]):
    """Draw rectangular bboxes and class labels on an image."""
    rendered = image.copy()

    for obj in objects:
        x, y, w, h = obj.bbox
        color = _color_for_class(obj.class_name)

        cv2.rectangle(rendered, (x, y), (x + w, y + h), color, 2)
        label = obj.class_name
        cv2.putText(
            rendered,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return rendered
