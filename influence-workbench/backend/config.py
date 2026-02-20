"""Global configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class WorkbenchConfig(BaseModel):
    model: str = "allenai/OLMo-2-0425-1B"
    revision: Optional[str] = None
    factors_dir: str = "/workspace/hessian_output"
    query_batch_size: int = 8
    train_batch_size: int = 8
    max_concurrent_runs: int = 1
    max_length: int = 512
    if_query_dir: str = "../if-query"
    data_dir: str = "data"


def load_config(path: str | Path = "config.yaml") -> WorkbenchConfig:
    """Load workbench configuration from a YAML file."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return WorkbenchConfig(**data)
    return WorkbenchConfig()
