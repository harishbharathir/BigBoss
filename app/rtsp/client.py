"""Minimal RTSP stream reader for the Phase 7 stretch prototype.

This is intentionally lightweight and operationally safe: it wraps OpenCV's
VideoCapture and supports the local sample MP4 used in validation. The same
interface can accept an RTSP URL in deployment, while keeping the code testable
without needing a live camera or an active network stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


class RTSPStream:
    def __init__(self, url: str | Path) -> None:
        self.url = str(url)
        self.cap: cv2.VideoCapture | None = None
        self._open = False

    def open(self) -> bool:
        """Open a stream or a local file path. Returns False if it cannot open."""
        source = self.url
        if not source:
            return False

        self.cap = cv2.VideoCapture(source)
        self._open = bool(self.cap and self.cap.isOpened())
        return self._open

    def read_frame(self) -> tuple[bool, Any | None]:
        if not self._open or self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        return ok, frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open
