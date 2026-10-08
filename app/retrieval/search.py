"""Phase 3 retrieval: rank tracks by similarity to a query."""

from __future__ import annotations

from typing import Any


def rank_tracks(tracks: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Return a scored list ordered by label and text similarity to the query."""
    normalized = query.lower().strip()
    terms = set(normalized.replace("?", " ").split())
    scored: list[dict[str, Any]] = []

    for item in tracks:
        label = str(item.get("label", "")).lower()
        score = float(item.get("confidence", 0.0))
        if label:
            score += 1.0 if label in normalized else 0.0
            score += 0.5 if any(term in label for term in terms) else 0.0
        if normalized and not label:
            score = 0.0
        scored.append({**item, "score": score})

    return sorted(scored, key=lambda value: value["score"], reverse=True)
