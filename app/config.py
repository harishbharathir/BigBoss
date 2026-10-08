"""Configuration loading with paths rooted at the repository directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML config and resolve configured path values from the project root."""
    path = config_path or PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config["_config_path"] = path.resolve()
    config["_project_root"] = PROJECT_ROOT
    config["_paths"] = {
        key: (PROJECT_ROOT / value).resolve()
        for key, value in config.get("paths", {}).items()
    }
    return config
