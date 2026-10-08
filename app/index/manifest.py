"""Build a small manifest describing a sampled video for Phase 1 validation."""

from __future__ import annotations

from pathlib import Path

from app.ingest import extract_video_metadata


def build_index_manifest(video_path: str | Path) -> dict:
    metadata = extract_video_metadata(video_path)
    return {
        "source_video": metadata["path"],
        "width": metadata["width"],
        "height": metadata["height"],
        "fps": metadata["fps"],
        "duration_seconds": metadata["duration_seconds"],
        "index_status": "validated",
    }
