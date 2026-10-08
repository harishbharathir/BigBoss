"""Threshold-based alert logic for a lightweight prototype."""

from __future__ import annotations

from typing import Any


class AlertManager:
    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    def evaluate(self, event: dict[str, Any]) -> dict[str, Any]:
        score = float(event.get("score", 0.0))
        triggered = score >= self.threshold
        return {
            "triggered": triggered,
            "score": score,
            "event": event.get("event", "unknown"),
            "message": "Alert triggered" if triggered else "No alert",
        }
