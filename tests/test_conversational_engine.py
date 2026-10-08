"""Unit tests for MultiStreamConversationalEngine, ClarifySession, Alerts, and Privacy."""

from pathlib import Path
import json
import numpy as np
import pytest

from app.alerts.rules import AlertManager
from app.privacy.redaction import PrivacyFilter
from app.reid.cross_camera import (
    CrossCameraHandover,
    CrossCameraJourney,
    SingleCameraTrack,
)
from app.retrieval.conversational_engine import (
    GroundedMatch,
    MultiStreamConversationalEngine,
    extract_query_intent,
    format_timestamp,
)
from app.ui.clarify_flow import ClarifySession


class MockEmbedder:
    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(abs(hash(text)) % 100000)
        vec = rng.randn(self.dim).astype(np.float32)
        return vec / np.linalg.norm(vec)


@pytest.fixture
def dummy_multicam_data():
    """Create synthetic multi-camera tracklet data with known events."""
    rng = np.random.RandomState(42)
    dim = 768

    crop_gate = (rng.rand(80, 80, 3) * 255).astype(np.uint8)
    crop_rear = (rng.rand(80, 80, 3) * 255).astype(np.uint8)

    # Make yellow car vector
    yellow_vec = rng.randn(dim).astype(np.float32)
    yellow_vec /= np.linalg.norm(yellow_vec)

    # Track on Gate Cam that exited at 1.84s
    t_gate = SingleCameraTrack(
        track_id=3,
        camera_name="Gate Cam",
        label="car",
        dominant_color="yellow",
        start_time=0.37,
        end_time=1.84,
        duration=1.47,
        best_conf=0.88,
        best_time=1.20,
        best_crop=crop_gate,
        obs_count=10,
        is_moving=True,
        left_camera=True,
        entered_camera=False,
        embedding=yellow_vec,
    )

    # Track on Rear Cam that arrived at 2.50s and exited at 2.94s
    t_rear = SingleCameraTrack(
        track_id=6,
        camera_name="Rear Cam",
        label="car",
        dominant_color="yellow",
        start_time=2.50,
        end_time=2.94,
        duration=0.44,
        best_conf=0.85,
        best_time=2.70,
        best_crop=crop_rear,
        obs_count=5,
        is_moving=True,
        left_camera=True,
        entered_camera=True,
        embedding=yellow_vec,
    )

    ho = CrossCameraHandover(
        global_id=1,
        from_camera="Gate Cam",
        to_camera="Rear Cam",
        from_track_id=3,
        to_track_id=6,
        label="car",
        dominant_color="yellow",
        exit_time=1.84,
        entry_time=2.50,
        delay_seconds=0.66,
        similarity=0.97,
        from_crop=crop_gate,
        to_crop=crop_rear,
        handover_type="sequential",
    )

    journey = CrossCameraJourney(
        global_id=1,
        label="car",
        dominant_color="yellow",
        representative_crop=crop_gate,
        sightings=[
            {"camera": "Gate Cam", "track_id": 3, "start_time": 0.37, "end_time": 1.84, "crop": crop_gate, "is_exit": True},
            {"camera": "Rear Cam", "track_id": 6, "start_time": 2.50, "end_time": 2.94, "crop": crop_rear, "is_exit": True, "is_entry": True},
        ],
        handovers=[ho],
        has_handover=True,
        total_cameras=2,
    )

    return {
        "tracks_by_camera": {
            "Gate Cam": [t_gate],
            "Rear Cam": [t_rear],
        },
        "handovers": [ho],
        "journeys": [journey],
        "cameras": [
            {"name": "Gate Cam", "source": "gate_cam.mp4", "offset": 0.0},
            {"name": "Rear Cam", "source": "rear_cam.mp4", "offset": 0.0},
        ],
    }


def test_extract_query_intent():
    intent = extract_query_intent("when did the yellow car left")
    assert intent["is_exit"] is True
    assert intent["detected_color"] == "yellow"
    assert intent["target_object"] == "car"
    assert "yellow" in intent["semantic_phrase"]

    intent2 = extract_query_intent("did a red car pass through the main gate?")
    assert intent2["detected_color"] == "red"
    assert intent2["target_object"] == "car"


def test_format_timestamp():
    assert format_timestamp(0.0) == "00:00.00"
    assert format_timestamp(65.5) == "01:05.50"


def test_conversational_engine_answers_when_yellow_car_left(dummy_multicam_data, monkeypatch):
    session = ClarifySession()
    # Mock text embedder to align with yellow vector
    mock_emb = MockEmbedder()

    engine = MultiStreamConversationalEngine(
        processor=None,
        embedder=None,
        device="cpu",
        clarify_session=session,
    )
    # Monkeypatch embed_text
    yellow_target = dummy_multicam_data["tracks_by_camera"]["Gate Cam"][0].embedding
    monkeypatch.setattr(engine, "embed_text", lambda text: yellow_target)

    res = engine.answer_query("when did the yellow car left", dummy_multicam_data)

    assert res.needs_clarification is False
    assert len(res.grounded_matches) > 0
    primary = res.primary_match
    assert primary is not None
    # Must resolve to specific camera + timestamp + visual evidence crop
    assert primary.camera_name in {"Gate Cam", "Rear Cam"}
    assert primary.timestamp > 0.0
    assert primary.crop is not None
    assert primary.event_type == "left"
    assert "Departure" in res.answer_text or "left" in res.answer_text.lower()
    # Cross-camera timeline reconstruction present
    assert len(res.reconstructed_timeline) > 0


def test_clarify_once_memory_persistence(tmp_path, monkeypatch):
    kb_file = tmp_path / "knowledge_base.json"
    session1 = ClarifySession()
    session1.kb_path = kb_file
    session1.kb = {}

    # Initial state: unknown referent
    assert "main gate" not in session1.kb

    # Learn mapping
    session1.learn_mapping("main gate", "Gate Cam")
    assert session1.kb["main gate"] == "Gate Cam"
    assert kb_file.exists()

    # Restart session (simulate process restart)
    session2 = ClarifySession()
    session2.kb_path = kb_file
    session2.restart()
    assert session2.kb.get("main gate") == "Gate Cam"
    assert session2.resolve_query("car at main gate") == "car at Gate Cam"


def test_alert_manager_standing_rules():
    manager = AlertManager(threshold=0.8)
    rule = manager.add_standing_rule("Test Departure", condition="left", threshold=0.7)
    assert rule["id"] == "rule_3"

    crop = np.zeros((40, 40, 3), dtype=np.uint8)
    match = GroundedMatch(
        camera_name="Gate Cam",
        timestamp=5.2,
        timestamp_str="00:05.20",
        crop=crop,
        confidence=0.85,
        label="car",
        track_id=1,
        event_type="left",
        dominant_color="yellow",
    )

    alerts = manager.evaluate_multi_stream([match])
    assert len(alerts) >= 1
    assert alerts[0]["camera"] == "Gate Cam"


def test_privacy_filter_redacts_crop_and_regions():
    pf = PrivacyFilter(blur_pixels=15)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    # Test bounding box region redaction
    redacted = pf.redact(img, regions=[[10, 10, 40, 40]])
    assert redacted is not None
    assert redacted.shape == img.shape

    # Test crop plate redaction
    crop_redacted = pf.redact_crop(img, blur_plate=True)
    assert crop_redacted is not None
    assert crop_redacted.shape == img.shape
