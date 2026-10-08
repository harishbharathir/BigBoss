"""Helpers for reading basic video metadata with ffprobe."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _parse_fraction(value: str | None) -> float:
    if value in (None, "", "0", "0/0"):
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            return float(numerator) / float(denominator)
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def extract_video_metadata(video_path: str | Path) -> dict[str, Any]:
    """Return height, width, FPS, and duration for a local MP4 file."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    video_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    fps_value = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    duration_value = video_stream.get("duration")
    if duration_value is None:
        format_data = payload.get("format", {})
        duration_value = format_data.get("duration")

    metadata = {
        "path": str(path.resolve()),
        "width": int(video_stream.get("width", 0) or 0),
        "height": int(video_stream.get("height", 0) or 0),
        "fps": _parse_fraction(fps_value),
        "duration_seconds": float(duration_value) if duration_value is not None else 0.0,
    }
    return metadata
