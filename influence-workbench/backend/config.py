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
    infinigram_api_url: str = "https://api.infini-gram.io/"
    infinigram_index: str = "v4_olmo-mix-1124_llama"
    infinigram_max_docs: int = 10
    infinigram_max_attempts: int = 10
    claude_model: str = "claude-haiku-4-5-20251001"


def load_config(path: str | Path = "config.yaml") -> WorkbenchConfig:
    """Load workbench configuration from a YAML file."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return WorkbenchConfig(**data)
    return WorkbenchConfig()
