"""Download configured model weights once for fully offline inference."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from ultralytics import YOLO

from app.config import load_config

LOGGER = logging.getLogger("model_download")


def download_models() -> None:
    config = load_config()
    model_dir: Path = config["_paths"]["model_cache"]
    model_dir.mkdir(parents=True, exist_ok=True)

    detector_name = config["models"]["detector"]
    detector_path = (config["_project_root"] / config["models"]["detector_weights"]).resolve()
    detector_path.parent.mkdir(parents=True, exist_ok=True)
    if detector_path.is_file():
        LOGGER.info("Detector weights already cached: %s", detector_path)
    else:
        # Ultralytics resolves a model filename/download relative to the current
        # directory; use the configured cache directory, then restore cwd.
        previous_cwd = Path.cwd()
        try:
            os.chdir(detector_path.parent)
            detector = YOLO(detector_name)
            resolved = Path(detector.ckpt_path).resolve()
            if resolved != detector_path and resolved.is_file():
                resolved.replace(detector_path)
        finally:
            os.chdir(previous_cwd)
        if not detector_path.is_file():
            raise RuntimeError(f"Ultralytics did not create detector weights at {detector_path}")
        LOGGER.info("Cached detector weights: %s", detector_path)

    repo_id = config["models"]["embedding"]
    revision = config["models"].get("embedding_revision", "main")
    embedding_dir = model_dir / "siglip"
    if (embedding_dir / "config.json").is_file():
        LOGGER.info("SigLIP model already cached: %s", embedding_dir)
    else:
        embedding_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(model_dir / "huggingface"),
            local_dir=str(embedding_dir),
        )
        LOGGER.info("Cached SigLIP model: %s", embedding_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_models()
