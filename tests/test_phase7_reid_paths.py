from app.reid.identity import PathTracker, ReIDTracker


def test_reid_tracker_matches_similar_observations():
    tracker = ReIDTracker()

    obs_a = {
        "track_id": 1,
        "label": "person",
        "bbox": {"x": 10, "y": 20, "width": 30, "height": 40},
        "color": {"dominant": "blue", "hue": 210},
        "timestamp": 1.0,
    }
    obs_b = {
        "track_id": 7,
        "label": "person",
        "bbox": {"x": 12, "y": 22, "width": 32, "height": 42},
        "color": {"dominant": "blue", "hue": 210},
        "timestamp": 2.0,
    }

    tracker.add(obs_a)
    matched = tracker.match(obs_b)

    assert matched == 1


def test_path_tracker_records_ordered_positions():
    tracker = PathTracker()

    tracker.add(5, {"x": 10, "y": 15})
    tracker.add(5, {"x": 15, "y": 20})
    tracker.add(5, {"x": 20, "y": 25})

    history = tracker.history(5)

    assert history == [{"x": 10, "y": 15}, {"x": 15, "y": 20}, {"x": 20, "y": 25}]
