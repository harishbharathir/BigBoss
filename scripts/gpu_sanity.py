"""Load detector and SigLIP concurrently and check actual GPU memory use."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from ultralytics import YOLO

from app.config import load_config

LOGGER = logging.getLogger("gpu_sanity")


def main() -> int:
    config = load_config()
    device = config["runtime"].get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    detector_path: Path = (config["_project_root"] / config["models"]["detector_weights"]).resolve()
    embedding_path = config["_paths"]["model_cache"] / "siglip"
    if not detector_path.is_file() or not (embedding_path / "config.json").is_file():
        raise FileNotFoundError("Model weights are missing; run scripts/download_models.py first")

    torch.manual_seed(config["project"].get("random_seed", 26))
    cuda = device == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    dtype = torch.float16 if cuda and config["runtime"].get("precision") == "fp16" else torch.float32

    LOGGER.info("Loading YOLO-World on %s", device)
    detector = YOLO(str(detector_path))
    detector.to(device)
    detector.set_classes(config["ingest"]["generic_vocabulary"])

    LOGGER.info("Loading SigLIP on %s with %s", device, dtype)
    processor = AutoProcessor.from_pretrained(str(embedding_path), local_files_only=True)
    embedder = AutoModel.from_pretrained(
        str(embedding_path), torch_dtype=dtype, local_files_only=True
    ).to(device).eval()

    # Exercise both model paths while both model parameter sets remain resident.
    sample = np.zeros((512, 512, 3), dtype=np.uint8)
    detector.predict(
        sample,
        imgsz=config["ingest"]["detector_image_size"],
        conf=config["ingest"]["detector_confidence"],
        verbose=False,
        device=device,
    )
    inputs = processor(images=sample, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        embedder.get_image_features(**inputs)
    if cuda:
        torch.cuda.synchronize()
        allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        LOGGER.info(
            "GPU=%s total=%.2f GiB peak_allocated=%.2f GiB peak_reserved=%.2f GiB",
            torch.cuda.get_device_name(0), total_gb, allocated_gb, reserved_gb,
        )
        limit_gb = float(config["runtime"].get("max_vram_gb", 4.5))
        if allocated_gb > limit_gb:
            raise RuntimeError(
                f"Peak allocated VRAM {allocated_gb:.2f} GiB exceeds configured "
                f"limit {limit_gb:.2f} GiB"
            )
    else:
        LOGGER.warning("Both models loaded and ran on CPU; CUDA VRAM check skipped")

    LOGGER.info("PASS: detector and SigLIP are loaded together and inference completed")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
