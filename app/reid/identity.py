"""A simple, working Re-ID model for the Phase 7 stretch prototype.

This is intentionally lightweight: we score a candidate observation against the
last seen appearance of each active track using label, dominant color, and
position similarity. It is not a full deep Re-ID model, but it is a working
identity matcher that can run in a local pipeline without external downloads.
"""

from __future__ import annotations

from typing import Any


class ReIDTracker:
    def __init__(self, threshold: float = 0.65) -> None:
        self.threshold = threshold
        self._active: dict[int, dict[str, Any]] = {}
        self._next_id = 1

    def _signature(self, observation: dict[str, Any]) -> dict[str, Any]:
        bbox = observation.get("bbox", {})
        x = float(bbox.get("x", 0.0))
        y = float(bbox.get("y", 0.0))
        width = float(bbox.get("width", 0.0))
        height = float(bbox.get("height", 0.0))
        color = observation.get("color", {})
        dominant = str(color.get("dominant", "")).lower()
        hue = float(color.get("hue", 0.0))
        return {
            "label": str(observation.get("label", "")).lower(),
            "dominant": dominant,
            "hue": hue,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }

    def _score(self, current: dict[str, Any], candidate: dict[str, Any]) -> float:
        label_score = 1.0 if current["label"] == candidate["label"] else 0.0
        color_score = 1.0 if current["dominant"] == candidate["dominant"] else 0.0
        hue_gap = abs(current["hue"] - candidate["hue"])
        hue_score = max(0.0, 1.0 - (hue_gap / 180.0))
        dx = abs(current["x"] - candidate["x"])
        dy = abs(current["y"] - candidate["y"])
        position_score = max(0.0, 1.0 - ((dx + dy) / 250.0))
        return 0.4 * label_score + 0.3 * color_score + 0.2 * hue_score + 0.1 * position_score

    def add(self, observation: dict[str, Any]) -> int:
        track_id = int(observation.get("track_id", self._next_id))
        self._active[track_id] = self._signature(observation)
        self._next_id = max(self._next_id, track_id + 1)
        return track_id

    def match(self, observation: dict[str, Any]) -> int:
        candidate = self._signature(observation)
        best_track_id = None
        best_score = -1.0

        for track_id, current in self._active.items():
            score = self._score(current, candidate)
            if score > best_score:
                best_score = score
                best_track_id = track_id

        if best_track_id is not None and best_score >= self.threshold:
            self._active[best_track_id] = candidate
            return best_track_id

        new_track_id = self._next_id
        self._active[new_track_id] = candidate
        self._next_id += 1
        return new_track_id


class PathTracker:
    def __init__(self) -> None:
        self._history: dict[int, list[dict[str, float]]] = {}

    def add(self, track_id: int, position: dict[str, float]) -> None:
        track_id = int(track_id)
        self._history.setdefault(track_id, []).append(dict(position))

    def history(self, track_id: int) -> list[dict[str, float]]:
        return [dict(position) for position in self._history.get(int(track_id), [])]
