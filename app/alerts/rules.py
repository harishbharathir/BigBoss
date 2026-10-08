"""Threshold-based alert logic for a lightweight prototype."""

from __future__ import annotations

from typing import Any


class AlertManager:
    """Threshold and standing query alert logic for multi-stream intelligence."""

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.standing_rules: list[dict[str, Any]] = [
            {
                "id": "rule_1",
                "name": "High-Confidence Vehicle Departure",
                "condition": "left",
                "threshold": 0.75,
                "camera": "all",
                "enabled": True,
            },
            {
                "id": "rule_2",
                "name": "Cross-Camera Handover Alert",
                "condition": "handover",
                "threshold": 0.70,
                "camera": "all",
                "enabled": True,
            },
        ]
        self.history: list[dict[str, Any]] = []

    def evaluate(self, event: dict[str, Any]) -> dict[str, Any]:
        """Simple threshold evaluation for backwards compatibility."""
        score = float(event.get("score", 0.0))
        triggered = score >= self.threshold
        res = {
            "triggered": triggered,
            "score": score,
            "event": event.get("event", "unknown"),
            "message": "Alert triggered" if triggered else "No alert",
        }
        if triggered:
            self.history.append({**res, "timestamp": event.get("timestamp", 0.0)})
        return res

    def add_standing_rule(
        self,
        name: str,
        condition: str,
        threshold: float = 0.75,
        camera: str = "all",
    ) -> dict[str, Any]:
        """Register a continuous standing query / alert rule."""
        rule_id = f"rule_{len(self.standing_rules) + 1}"
        rule = {
            "id": rule_id,
            "name": name,
            "condition": condition,
            "threshold": threshold,
            "camera": camera,
            "enabled": True,
        }
        self.standing_rules.append(rule)
        return rule

    def evaluate_multi_stream(self, matches: list[Any]) -> list[dict[str, Any]]:
        """Evaluate detected matches against active standing rules."""
        triggered_alerts = []
        for match in matches:
            score = getattr(match, "confidence", 0.0)
            camera = getattr(match, "camera_name", "unknown")
            event_type = getattr(match, "event_type", "sighted")

            for rule in self.standing_rules:
                if not rule.get("enabled", True):
                    continue
                if rule["camera"] != "all" and rule["camera"].lower() not in camera.lower():
                    continue
                if score >= rule["threshold"]:
                    alert_entry = {
                        "rule_name": rule["name"],
                        "camera": camera,
                        "timestamp": getattr(match, "timestamp", 0.0),
                        "timestamp_str": getattr(match, "timestamp_str", "00:00.00"),
                        "score": score,
                        "event_type": event_type,
                        "label": getattr(match, "label", "object"),
                        "crop": getattr(match, "crop", None),
                    }
                    triggered_alerts.append(alert_entry)
                    self.history.append(alert_entry)
        return triggered_alerts

