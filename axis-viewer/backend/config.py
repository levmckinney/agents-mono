"""Global configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class AxisViewerConfig(BaseModel):
    model_name: str = "Qwen/Qwen3-32B-AWQ"
    target_layer: int = 32
    total_layers: int = 64
    device: str = "auto"
    data_dir: str = "data"
    host: str = "0.0.0.0"
    port: int = 8001
    mock_model: bool = False


def load_config(path: str | Path = "config.yaml") -> AxisViewerConfig:
    """Load axis-viewer configuration from a YAML file."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return AxisViewerConfig(**data)
    return AxisViewerConfig()
