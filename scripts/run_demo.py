"""Minimal CLI demo for the current local model baseline."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from app.alerts.rules import AlertManager
from app.index.manifest import build_index_manifest
from app.ingest import extract_video_metadata
from app.privacy.redaction import PrivacyFilter
from app.reid.identity import PathTracker, ReIDTracker
from app.rtsp.client import RTSPStream


def main() -> None:
    sample = Path("data/videos/sample.mp4")
    if not sample.exists():
        raise FileNotFoundError("No sample video found at data/videos/sample.mp4")

    metadata = extract_video_metadata(sample)
    manifest = build_index_manifest(sample)

    tracker = ReIDTracker()
    track_history = PathTracker()

    observations = [
        {"track_id": 1, "label": "person", "bbox": {"x": 10, "y": 20, "width": 30, "height": 40}, "color": {"dominant": "blue", "hue": 210}, "timestamp": 1.0},
        {"track_id": 7, "label": "person", "bbox": {"x": 12, "y": 22, "width": 32, "height": 42}, "color": {"dominant": "blue", "hue": 210}, "timestamp": 2.0},
        {"track_id": 9, "label": "car", "bbox": {"x": 200, "y": 80, "width": 120, "height": 60}, "color": {"dominant": "red", "hue": 10}, "timestamp": 3.0},
    ]

    for obs in observations:
        matched = tracker.match(obs)
        track_history.add(matched, {"x": obs["bbox"]["x"], "y": obs["bbox"]["y"]})

    alert_manager = AlertManager(threshold=0.8)
    alert = alert_manager.evaluate({"event": "person_detected", "score": 0.87})
    privacy = PrivacyFilter()

    stream = RTSPStream(sample)
    stream.open()
    ok, frame = stream.read_frame()
    stream.close()

    print("VIDEO_METADATA", metadata)
    print("INDEX_MANIFEST", manifest)
    print("REID_MATCHED_TRACKS", [track_history.history(tid) for tid in sorted(track_history._history.keys())])
    print("ALERT", alert)
    print("PRIVACY", {"redacted": privacy.redact(frame) is not None, "mask_label": privacy.label(42)})
    print("STREAM_OK", ok and frame is not None)


if __name__ == "__main__":
    main()
