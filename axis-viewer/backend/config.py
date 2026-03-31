"""Global configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelSpec(BaseModel):
    """Specification for a supported model."""

    target_layer: int
    total_layers: int
    short_name: str
    is_base: bool = False
    has_capping: bool = False
    axis_path: str  # path within the HF repo


# All models the viewer can switch between.
AVAILABLE_MODELS: dict[str, ModelSpec] = {
    "Qwen/Qwen3-32B-AWQ": ModelSpec(
        target_layer=32,
        total_layers=64,
        short_name="Qwen 3 32B",
        is_base=False,
        has_capping=True,
        axis_path="qwen-3-32b/assistant_axis.pt",
    ),
    "google/gemma-2-27b-it": ModelSpec(
        target_layer=22,
        total_layers=46,
        short_name="Gemma 2 27B",
        is_base=False,
        axis_path="gemma-2-27b/assistant_axis.pt",
    ),
    "google/gemma-2-27b": ModelSpec(
        target_layer=22,
        total_layers=46,
        short_name="Gemma 2 27B Base",
        is_base=True,
        axis_path="gemma-2-27b/assistant_axis.pt",
    ),
    "meta-llama/Llama-3.3-70B-Instruct": ModelSpec(
        target_layer=40,
        total_layers=80,
        short_name="Llama 3.3 70B",
        is_base=False,
        has_capping=True,
        axis_path="llama-3.3-70b/assistant_axis.pt",
    ),
}

AXIS_HF_REPO = "lu-christina/assistant-axis-vectors"


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
