"""Phase 2 tracking smoke test: detect objects across a few sampled frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from app.config import load_config


def _build_model() -> YOLO:
    config = load_config()
    model_path = Path(config["models"]["detector_weights"])
    model = YOLO(str(model_path))
    model.set_classes(config["ingest"]["generic_vocabulary"])
    return model


def track_video_objects(video_path: str | Path, max_frames: int = 3, device: str = "cpu") -> list[dict[str, Any]]:
    """Sample a few frames and attach simple track IDs based on label + proximity."""
    config = load_config()
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    model = _build_model()
    detections: list[dict[str, Any]] = []
    previous = {}
    track_counter = 1

    frame_index = 0
    while frame_index < max_frames and cap.isOpened():
        ok = cap.grab()
        if not ok:
            break
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_pos <= 0:
            frame_pos = frame_index
        _, frame = cap.retrieve()
        if frame is None:
            frame_index += 1
            continue

        result = model(
            frame,
            imgsz=config["ingest"]["detector_image_size"],
            conf=config["ingest"]["detector_confidence"],
            verbose=False,
            device=device,
        )[0]
        current = {}
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                label_id = int(box.cls[0])
                label = result.names.get(label_id, str(label_id))
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                current_id = None
                best_distance = None
                for existing_label, existing_data in previous.items():
                    if existing_label != label:
                        continue
                    dx = center_x - existing_data["center_x"]
                    dy = center_y - existing_data["center_y"]
                    distance = (dx * dx + dy * dy) ** 0.5
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        current_id = existing_data["track_id"]
                if current_id is None:
                    current_id = track_counter
                    track_counter += 1
                current[label] = {"track_id": current_id, "center_x": center_x, "center_y": center_y}
                detections.append(
                    {
                        "track_id": current_id,
                        "label": label,
                        "confidence": conf,
                        "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                        "timestamp": frame_index / max(1.0, cap.get(cv2.CAP_PROP_FPS)),
                        "frame_index": frame_index,
                    }
                )

        previous = current
        frame_index += 1

    cap.release()
    return detections
