"""Unit tests for VideoIntelligenceChatbot."""

from pathlib import Path
import numpy as np
import pytest

from app.reid.cross_camera import (
    CrossCameraHandover,
    CrossCameraJourney,
    SingleCameraTrack,
)
from app.retrieval.chatbot import VideoIntelligenceChatbot
from app.retrieval.conversational_engine import (
    MultiStreamConversationalEngine,
)
from app.ui.clarify_flow import ClarifySession


@pytest.fixture
def dummy_multicam_data():
    rng = np.random.RandomState(42)
    dim = 768
    crop_gate = (rng.rand(60, 60, 3) * 255).astype(np.uint8)
    crop_rear = (rng.rand(60, 60, 3) * 255).astype(np.uint8)

    yellow_vec = rng.randn(dim).astype(np.float32)
    yellow_vec /= np.linalg.norm(yellow_vec)

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
        "tracks_by_camera": {"Gate Cam": [t_gate], "Rear Cam": [t_rear]},
        "handovers": [ho],
        "journeys": [journey],
        "cameras": [{"name": "Gate Cam"}, {"name": "Rear Cam"}],
    }


def test_chatbot_answers_with_visual_evidence(dummy_multicam_data, monkeypatch):
    session = ClarifySession()
    engine = MultiStreamConversationalEngine(None, None, "cpu", clarify_session=session)

    # Monkeypatch text embedding to match yellow vector
    yellow_target = dummy_multicam_data["tracks_by_camera"]["Gate Cam"][0].embedding
    monkeypatch.setattr(engine, "embed_text", lambda text: yellow_target)

    chatbot = VideoIntelligenceChatbot(engine=engine)
    reply = chatbot.chat("when did the yellow car left", dummy_multicam_data)

    assert reply.role == "assistant"
    assert "left" in reply.content.lower() or "departure" in reply.content.lower()
    # Virtual evidence checks
    assert reply.primary_match is not None
    assert reply.primary_match.camera_name in {"Gate Cam", "Rear Cam"}
    assert reply.primary_match.timestamp > 0
    assert len(reply.evidence_crops) > 0
    assert reply.evidence_crops[0] is not None
    assert len(reply.timeline_steps) > 0


def test_chatbot_clarify_once_trigger(dummy_multicam_data):
    session = ClarifySession()
    session.kb = {}  # Empty KB
    engine = MultiStreamConversationalEngine(None, None, "cpu", clarify_session=session)
    chatbot = VideoIntelligenceChatbot(engine=engine)

    reply = chatbot.chat("did a car enter the lobby?", dummy_multicam_data)
    assert reply.needs_clarification is True
    assert reply.clarification_entity == "lobby"
    assert "Which camera corresponds" in reply.clarification_prompt
