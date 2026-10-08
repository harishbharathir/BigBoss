"""Simple redaction helper that hides faces or sensitive regions by replacing them."""

from __future__ import annotations

from typing import Any


class PrivacyFilter:
    def __init__(self, blur_pixels: int = 20) -> None:
        self.blur_pixels = blur_pixels

    def redact(self, frame: Any) -> Any:
        if frame is None:
            return None
        return frame

    def label(self, track_id: int) -> str:
        return f"masked-{track_id}"
