from pathlib import Path

import pytest

from app.retrieval.search import rank_tracks
from app.spatial_memory.memory import SpatialMemory
from app.tracking.frame_tracking import track_video_objects
from app.ui.clarify_flow import ClarifySession

SAMPLE_VIDEO = Path("data/videos/sample.mp4")


def test_track_video_objects_smoke():
    if not SAMPLE_VIDEO.exists():
        pytest.skip("sample video not present yet")

    matches = track_video_objects(SAMPLE_VIDEO, max_frames=3)

    assert matches
    assert all("track_id" in item for item in matches)
    assert all("label" in item for item in matches)
    assert all("bbox" in item for item in matches)
    assert all("timestamp" in item for item in matches)


def test_rank_tracks_uses_query_content():
    tracks = [
        {"track_id": 1, "label": "person", "confidence": 0.96, "timestamp": 1.0},
        {"track_id": 2, "label": "car", "confidence": 0.82, "timestamp": 2.0},
    ]

    ranked = rank_tracks(tracks, "find person")

    assert ranked[0]["track_id"] == 1
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_clarify_session_can_kill_and_restart():
    session = ClarifySession()

    result_ambiguous = session.clarify_once("find a person and a car")
    assert result_ambiguous is not None

    session.kill()
    assert session.is_running is False

    session.restart()
    assert session.is_running is True

    result_clear = session.clarify_once("find a person")
    assert result_clear is None


def test_spatial_memory_tracks_last_seen_position():
    memory = SpatialMemory()
    memory.update(42, {"x": 10, "y": 20, "width": 30, "height": 40})

    assert memory.get_last_seen(42) == {"x": 10, "y": 20, "width": 30, "height": 40}
    assert memory.get_last_seen(99) is None
