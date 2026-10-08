"""Phase 3 spatial memory: track last observed position of each object."""

from __future__ import annotations

from typing import Any


class SpatialMemory:
    def __init__(self) -> None:
        self._last_seen: dict[int, dict[str, Any]] = {}

    def update(self, track_id: int, bbox: dict[str, Any]) -> None:
        self._last_seen[int(track_id)] = dict(bbox)

    def get_last_seen(self, track_id: int) -> dict[str, Any] | None:
        value = self._last_seen.get(int(track_id))
        return None if value is None else dict(value)
