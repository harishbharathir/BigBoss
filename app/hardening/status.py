"""Simple restart/status helper for the hardening stage."""

from __future__ import annotations


def restart_status(is_running: bool) -> dict[str, bool | str]:
    return {
        "is_running": is_running,
        "status": "healthy" if is_running else "stopped",
    }
