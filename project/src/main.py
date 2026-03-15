"""Iterate through all train images, draw annotations, and display with OpenCV.

Controls
--------
  Any key   — next frame
  q / Esc   — quit
  b         — go back one frame
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from dataset_loader import get_split_dirs, list_frames, load_frame_and_annotations
from visualization import draw_detections

# ── Configuration ─────────────────────────────────────────────────────────────
SPLIT = "train"          # "train" | "val" | "test"
DISPLAY_WIDTH = 1280     # Resize window width for display (keeps aspect ratio).
                         # Set to None to show original resolution.
# ──────────────────────────────────────────────────────────────────────────────

WINDOW = "Drone Traffic Dataset"


def count_matching_pairs(images_dir: Path, labels_dir: Path) -> tuple[int, int, int]:
    """Return counts for images, labels, and matched stem names."""
    image_stems = {p.stem for p in images_dir.glob("*.jpg")}
    label_stems = {p.stem for p in labels_dir.glob("*.txt")} if labels_dir.exists() else set()
    matched = len(image_stems & label_stems)
    return len(image_stems), len(label_stems), matched


def resize_for_display(image, max_width: int):
    """Downscale image to max_width while preserving aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def overlay_info(image, frame_id: str, index: int, total: int, n_objects: int):
    """Burn a small HUD into the top-left corner of the image."""
    lines = [
        f"Frame : {frame_id}",
        f"Index : {index + 1} / {total}",
        f"Vehicles: {n_objects}",
        "Keys  : any=next  b=back  q/Esc=quit",
    ]
    y = 24
    for text in lines:
        # Dark shadow for readability on any background.
        cv2.putText(image, text, (9, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "dataset"

    images_dir, labels_dir = get_split_dirs(dataset_dir, SPLIT)

    # Validate dataset pairing up front so missing/mismatched labels are obvious.
    n_images, n_labels, n_matched = count_matching_pairs(images_dir, labels_dir)
    print(f"Images in split '{SPLIT}': {n_images}")
    print(f"Labels in expected folder: {n_labels} ({labels_dir})")
    print(f"Matching image-label pairs: {n_matched}")
    if n_labels == 0:
        print("[ERROR] No labels found in the expected label folder.")
        print("Expected: dataset/labels/train_original for train split")
        print("Please place matching .txt files there and rerun.")
        sys.exit(1)
    if n_matched == 0:
        print("[ERROR] Found labels, but none match image filenames.")
        print("Image and label base names must be identical (without extension).")
        sys.exit(1)

    frame_ids = list_frames(images_dir)
    if not frame_ids:
        print(f"[ERROR] No .jpg images found in: {images_dir}")
        sys.exit(1)

    total = len(frame_ids)
    print(f"Found {total} frames in '{SPLIT}' split. Opening viewer...")
    print("Controls: any key = next | b = back | q / Esc = quit")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    idx = 0
    while 0 <= idx < total:
        frame_id = frame_ids[idx]

        # Load image + annotations (missing label file prints a warning, not crash).
        image, objects = load_frame_and_annotations(images_dir, labels_dir, frame_id)

        # Draw bounding boxes and class labels.
        vis = draw_detections(image, objects)

        # Resize for comfortable viewing.
        if DISPLAY_WIDTH is not None:
            vis = resize_for_display(vis, DISPLAY_WIDTH)

        # Overlay frame info HUD.
        overlay_info(vis, frame_id, idx, total, len(objects))

        cv2.imshow(WINDOW, vis)
        print(f"[{idx + 1}/{total}] {frame_id}  —  {len(objects)} vehicles")

        # Wait indefinitely for a keypress.
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):   # q or Esc → quit
            break
        elif key == ord("b"):       # b → go back
            idx = max(0, idx - 1)
        else:                       # anything else → advance
            idx += 1

    cv2.destroyAllWindows()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
