"""Local video sampling, YOLO-World detections, and SigLIP semantic search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from ultralytics import YOLO

from app.config import load_config


def load_visual_models() -> tuple[YOLO, AutoProcessor, AutoModel, str]:
    """Load the configured detector and locally cached SigLIP model."""
    config = load_config()
    requested_device = str(config["runtime"].get("device", "auto")).lower()
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is configured but PyTorch cannot access a CUDA device.")

    detector_path = (config["_project_root"] / config["models"]["detector_weights"]).resolve()
    siglip_path = config["_paths"]["model_cache"] / "siglip"
    if not detector_path.is_file():
        raise FileNotFoundError(f"YOLO-World weights not found: {detector_path}")
    if not (siglip_path / "config.json").is_file():
        raise FileNotFoundError(
            f"SigLIP cache not found: {siglip_path}. Run scripts/download_models.py first."
        )

    detector = YOLO(str(detector_path))
    detector.to(device)
    detector.set_classes(config["ingest"]["generic_vocabulary"])

    use_half = device.startswith("cuda") and config["runtime"].get("precision") == "fp16"
    dtype = torch.float16 if use_half else torch.float32
    processor = AutoProcessor.from_pretrained(str(siglip_path), local_files_only=True)
    embedder = AutoModel.from_pretrained(
        str(siglip_path), torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    return detector, processor, embedder, device


def analyze_video(
    video_path: str | Path,
    max_samples: int = 48,
    detector: YOLO | None = None,
    processor: AutoProcessor | None = None,
    embedder: AutoModel | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Sample a clip, detect configured objects, and cache a SigLIP vector per frame."""
    is_url = str(video_path).startswith(("rtsp://", "http://", "https://"))
    if not is_url:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        video_source = str(path)
    else:
        video_source = str(video_path)
        
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1")

    if detector is None or processor is None or embedder is None or device is None:
        detector, processor, embedder, device = load_visual_models()

    config = load_config()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 1.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else 0.0
    sample_step = max(1, int(frame_count / max_samples)) if frame_count else max(1, int(fps))

    is_live = frame_count <= 0
    
    frames: list[dict[str, Any]] = []
    frame_index = 0
    current_cap_frame = 0

    try:
        while len(frames) < max_samples:
            if is_live:
                while current_cap_frame < frame_index:
                    cap.grab()
                    current_cap_frame += 1
                ok, frame = cap.read()
                current_cap_frame += 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()


            if not ok or frame is None:
                break

            result = detector.predict(
                frame,
                imgsz=int(config["ingest"].get("detector_image_size", 640)),
                conf=float(config["ingest"].get("detector_confidence", 0.15)),
                verbose=False,
                device=device,
            )[0]
            raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_height, raw_width = raw_rgb.shape[:2]
            if raw_width > 960:
                raw_scale = 960 / raw_width
                raw_rgb = cv2.resize(raw_rgb, (960, int(raw_height * raw_scale)))
            boxes: list[dict[str, Any]] = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                    label_id = int(box.cls[0])
                    boxes.append({
                        "label": str(result.names.get(label_id, label_id)),
                        "confidence": float(box.conf[0]),
                        "bbox": [x1, y1, x2, y2],
                    })

            annotated = result.plot()
            height, width = annotated.shape[:2]
            if width > 960:
                scale = 960 / width
                annotated = cv2.resize(annotated, (960, int(height * scale)))
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frames.append({
                "frame_index": frame_index,
                "timestamp": frame_index / fps,
                "image": rgb,
                "search_image": raw_rgb,
                "boxes": boxes,
            })
            frame_index += sample_step
    finally:
        cap.release()

    if not frames:
        raise RuntimeError("No readable video frames were found.")

    dtype = next(embedder.parameters()).dtype
    for offset in range(0, len(frames), 8):
        batch_frames = frames[offset : offset + 8]
        images = [Image.fromarray(frame["search_image"]) for frame in batch_frames]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(device=device, dtype=dtype) if key == "pixel_values" else value.to(device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            vectors = embedder.get_image_features(**inputs)
            vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
        for frame, vector in zip(batch_frames, vectors.cpu().numpy()):
            frame["embedding"] = vector
            del frame["search_image"]

    return {
        "video": video_source,
        "fps": fps,
        "duration_seconds": duration,
        "sample_count": len(frames),
        "frames": frames,
    }


def search_frames(analysis: dict[str, Any], query: str, processor: AutoProcessor, embedder: AutoModel, device: str) -> list[dict[str, Any]]:
    """Rank sampled frames for a free-form natural-language query with SigLIP."""
    query = query.strip()
    if not query:
        return []
    inputs = processor(text=[query], padding="max_length", return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        text_vector = embedder.get_text_features(**inputs)
        text_vector = torch.nn.functional.normalize(text_vector.float(), dim=-1)[0].cpu().numpy()

    # Base scores
    for frame in analysis["frames"]:
        frame["raw_score"] = float(np.dot(frame["embedding"], text_vector))

    # Temporal smoothing (sliding window)
    window_size = 3
    frames = analysis["frames"]
    matches = []
    for i in range(len(frames)):
        start = max(0, i - window_size // 2)
        end = min(len(frames), i + window_size // 2 + 1)
        smoothed_score = sum(frames[j]["raw_score"] for j in range(start, end)) / (end - start)
        # We blend the raw score and smoothed score
        final_score = 0.5 * frames[i]["raw_score"] + 0.5 * smoothed_score
        matches.append({**frames[i], "score": final_score})

    return sorted(matches, key=lambda match: match["score"], reverse=True)

