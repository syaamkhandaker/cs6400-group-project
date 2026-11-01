"""Shared loader for configs/default.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_raw_config(path: Path | None = None) -> dict:
    config_path = path or DEFAULT_CONFIG_PATH
    with config_path.open() as handle:
        return yaml.safe_load(handle)
