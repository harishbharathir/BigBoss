"""Simple redaction helper that hides faces or sensitive regions by replacing them."""

from __future__ import annotations

from typing import Any


import cv2
import numpy as np


class PrivacyFilter:
    """On-premise privacy redaction for blurring faces, license plates, and sensitive crops."""

    def __init__(self, blur_pixels: int = 21) -> None:
        self.blur_pixels = blur_pixels if blur_pixels % 2 == 1 else blur_pixels + 1

    def redact(self, frame: Any, regions: list[list[int]] | None = None) -> Any:
        """Apply Gaussian blur to specified bounding boxes or sensitive areas."""
        if frame is None:
            return None
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame

        out = frame.copy()
        h, w = out.shape[:2]
        ksize = max(3, self.blur_pixels)

        if regions:
            for r in regions:
                x1, y1, x2, y2 = [int(v) for v in r]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    roi = out[y1:y2, x1:x2]
                    out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        return out

    def redact_crop(self, crop: np.ndarray, blur_plate: bool = True) -> np.ndarray:
        """Blur the lower half (license plate region) or full crop."""
        if crop is None or crop.size == 0:
            return crop
        out = crop.copy()
        h, w = out.shape[:2]
        ksize = max(3, self.blur_pixels)
        if blur_plate and h > 20:
            y1 = int(h * 0.65)
            roi = out[y1:h, :]
            out[y1:h, :] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        else:
            out = cv2.GaussianBlur(out, (ksize, ksize), 0)
        return out

    def label(self, track_id: int) -> str:
        return f"masked-{track_id}"

