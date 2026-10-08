"""Cross-camera Re-ID, single-camera tracking, and path tracking."""

from .cross_camera import (
    CrossCameraHandover,
    CrossCameraJourney,
    SingleCameraTrack,
    analyze_multiple_cameras,
    extract_dominant_color,
    match_cross_camera_tracks,
    track_single_camera,
)
from .identity import PathTracker, ReIDTracker

__all__ = [
    "ReIDTracker",
    "PathTracker",
    "SingleCameraTrack",
    "CrossCameraHandover",
    "CrossCameraJourney",
    "track_single_camera",
    "match_cross_camera_tracks",
    "analyze_multiple_cameras",
    "extract_dominant_color",
]
