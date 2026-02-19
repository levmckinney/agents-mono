"""Contract types for the if-query CLI interface.

These Pydantic models codify the exact input/output schema of if-query,
so that type-checking catches breakages when if-query evolves.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class IFQueryInputPair(BaseModel, extra="allow"):
    """A single prompt/completion pair sent to if-query as JSON input.

    Extra fields are preserved and pass through to the output CSVs.
    ``pair_id`` becomes ``query_id`` or ``train_id`` in the output.
    """

    pair_id: str
    prompt: str
    completion: str


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class IFQueryQueryResult(BaseModel, extra="allow"):
    """A row in the ``query.csv`` output produced by if-query."""

    query_id: str
    prompt: str
    completion: str
    loss: float


class IFQueryTrainResult(BaseModel, extra="allow"):
    """A row in the ``train.csv`` output produced by if-query."""

    train_id: str
    prompt: str
    completion: str


class IFQueryInfluenceRow(BaseModel):
    """A row in the ``influences.csv`` output produced by if-query."""

    query_id: str
    train_id: str
    influence_score: float
    per_token_scores: Optional[str] = None


# ---------------------------------------------------------------------------
# CLI interface models
# ---------------------------------------------------------------------------


class HessianStrategy(str, Enum):
    ekfac = "ekfac"
    kfac = "kfac"
    diagonal = "diagonal"


class DType(str, Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"


class IFQueryRunConfig(BaseModel):
    """Mirrors the CLI arguments accepted by ``run-query``."""

    model: str = "allenai/OLMo-2-0425-1B"
    revision: Optional[str] = None
    factors_dir: str
    train_json: str
    query_json: str
    output_dir: str
    query_batch_size: int = 8
    train_batch_size: int = 8
    dtype: DType = DType.bfloat16
    max_length: int = 512
    per_token_scores: bool = False

    def to_cli_args(self) -> list[str]:
        """Build the argument list for invoking ``run-query``."""
        args = [
            "--model",
            self.model,
            "--factors-dir",
            self.factors_dir,
            "--train-json",
            self.train_json,
            "--query-json",
            self.query_json,
            "--output-dir",
            self.output_dir,
            "--query-batch-size",
            str(self.query_batch_size),
            "--train-batch-size",
            str(self.train_batch_size),
            "--dtype",
            self.dtype.value,
            "--max-length",
            str(self.max_length),
        ]
        if self.revision is not None:
            args.extend(["--revision", self.revision])
        if self.per_token_scores:
            args.append("--per-token-scores")
        return args


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


class IFQueryOutputPaths(BaseModel):
    """Expected output file paths produced by a single if-query run."""

    output_dir: str

    @property
    def query_csv(self) -> Path:
        return Path(self.output_dir) / "query.csv"

    @property
    def train_csv(self) -> Path:
        return Path(self.output_dir) / "train.csv"

    @property
    def influences_csv(self) -> Path:
        return Path(self.output_dir) / "influences.csv"
