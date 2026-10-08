from pathlib import Path
import numpy as np
import pytest

from app.reid.cross_camera import (
    CrossCameraHandover,
    CrossCameraJourney,
    SingleCameraTrack,
    extract_dominant_color,
    match_cross_camera_tracks,
    analyze_multiple_cameras,
)


def test_extract_dominant_color_synthetic():
    # Pure red image (RGB: 255, 0, 0)
    red_img = np.zeros((50, 50, 3), dtype=np.uint8)
    red_img[:, :] = [255, 0, 0]
    assert extract_dominant_color(red_img) == "red"

    # Pure black image (RGB: 10, 10, 10)
    black_img = np.full((50, 50, 3), 10, dtype=np.uint8)
    assert extract_dominant_color(black_img) == "black"

    # White / light gray image (RGB: 240, 240, 240)
    white_img = np.full((50, 50, 3), 240, dtype=np.uint8)
    assert extract_dominant_color(white_img) in {"white", "silver/gray"}

    # Blue image (RGB: 0, 0, 255)
    blue_img = np.zeros((50, 50, 3), dtype=np.uint8)
    blue_img[:, :] = [0, 0, 255]
    assert extract_dominant_color(blue_img) == "blue"


def test_match_cross_camera_tracks_handover_success():
    crop = np.zeros((50, 50, 3), dtype=np.uint8)

    # Identical or near-identical embeddings
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.98, 0.1, 0.0], dtype=np.float32)
    emb2 /= np.linalg.norm(emb2)

    track_cam1 = SingleCameraTrack(
        track_id=1,
        camera_name="Gate Cam",
        label="car",
        dominant_color="silver/gray",
        start_time=1.0,
        end_time=5.0,  # Exited gate cam at 5.0s
        duration=4.0,
        best_conf=0.85,
        best_time=3.0,
        best_crop=crop,
        obs_count=5,
        is_moving=True,
        left_camera=True,
        embedding=emb1,
    )

    track_cam2 = SingleCameraTrack(
        track_id=2,
        camera_name="Rear Cam",
        label="car",
        dominant_color="silver/gray",
        start_time=7.5,  # Appeared on rear cam at 7.5s (delay = 2.5s)
        end_time=12.0,
        duration=4.5,
        best_conf=0.88,
        best_time=9.0,
        best_crop=crop,
        obs_count=6,
        is_moving=True,
        entered_camera=True,
        embedding=emb2,
    )

    tracks_by_camera = {
        "Gate Cam": [track_cam1],
        "Rear Cam": [track_cam2],
    }

    journeys, handovers = match_cross_camera_tracks(
        tracks_by_camera,
        similarity_threshold=0.80,
        max_handover_seconds=15.0,
        min_handover_seconds=0.0,
    )

    assert len(handovers) == 1
    assert handovers[0].from_camera == "Gate Cam"
    assert handovers[0].to_camera == "Rear Cam"
    assert handovers[0].exit_time == 5.0
    assert handovers[0].entry_time == 7.5
    assert pytest.approx(handovers[0].delay_seconds, 0.1) == 2.5
    assert handovers[0].similarity > 0.9

    assert len(journeys) == 1
    assert journeys[0].has_handover is True
    assert journeys[0].total_cameras == 2
    assert "Gate Cam" in journeys[0].title
    assert "Rear Cam" in journeys[0].title


def test_match_cross_camera_tracks_dissimilar_does_not_match():
    crop = np.zeros((50, 50, 3), dtype=np.uint8)

    # Orthogonal embeddings (similarity 0.0)
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    track_cam1 = SingleCameraTrack(
        track_id=1,
        camera_name="Gate Cam",
        label="car",
        dominant_color="red",
        start_time=1.0,
        end_time=5.0,
        duration=4.0,
        best_conf=0.8,
        best_time=2.0,
        best_crop=crop,
        obs_count=4,
        embedding=emb1,
    )

    track_cam2 = SingleCameraTrack(
        track_id=2,
        camera_name="Rear Cam",
        label="car",
        dominant_color="blue",
        start_time=6.0,
        end_time=10.0,
        duration=4.0,
        best_conf=0.8,
        best_time=8.0,
        best_crop=crop,
        obs_count=4,
        embedding=emb2,
    )

    tracks_by_camera = {
        "Gate Cam": [track_cam1],
        "Rear Cam": [track_cam2],
    }

    journeys, handovers = match_cross_camera_tracks(
        tracks_by_camera,
        similarity_threshold=0.75,
    )

    assert len(handovers) == 0
    # Both remain separate single-camera journeys
    assert len(journeys) == 2
    assert all(not j.has_handover for j in journeys)


def test_match_cross_camera_tracks_time_window_filtering():
    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    track_cam1 = SingleCameraTrack(
        track_id=1,
        camera_name="Gate Cam",
        label="car",
        dominant_color="black",
        start_time=1.0,
        end_time=5.0,  # Exited at 5.0s
        duration=4.0,
        best_conf=0.9,
        best_time=3.0,
        best_crop=crop,
        obs_count=5,
        embedding=emb,
    )

    # Appeared way too late (delay = 200s, max allowed = 30s)
    track_cam2 = SingleCameraTrack(
        track_id=2,
        camera_name="Rear Cam",
        label="car",
        dominant_color="black",
        start_time=205.0,
        end_time=215.0,
        duration=10.0,
        best_conf=0.9,
        best_time=208.0,
        best_crop=crop,
        obs_count=5,
        embedding=emb,
    )

    tracks_by_camera = {
        "Gate Cam": [track_cam1],
        "Rear Cam": [track_cam2],
    }

    journeys, handovers = match_cross_camera_tracks(
        tracks_by_camera,
        similarity_threshold=0.70,
        max_handover_seconds=30.0,
    )

    assert len(handovers) == 0
    assert len(journeys) == 2
