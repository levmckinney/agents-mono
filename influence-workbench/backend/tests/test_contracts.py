"""Tests for the if-query contract models."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from backend.contracts import (
    DType,
    IFQueryInfluenceRow,
    IFQueryInputPair,
    IFQueryOutputPaths,
    IFQueryQueryResult,
    IFQueryRunConfig,
    IFQueryTrainResult,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJSONRoundTrip:
    def test_input_pair_round_trip(self):
        pair = IFQueryInputPair(
            pair_id="q1", prompt="Hello?", completion="Hi there."
        )
        data = json.loads(pair.model_dump_json())
        restored = IFQueryInputPair(**data)
        assert restored == pair

    def test_extra_fields_preserved(self):
        pair = IFQueryInputPair(
            pair_id="q1",
            prompt="Hello?",
            completion="Hi there.",
            source="wiki",
            score=0.95,
        )
        data = json.loads(pair.model_dump_json())
        assert data["source"] == "wiki"
        assert data["score"] == 0.95

        restored = IFQueryInputPair(**data)
        assert restored.source == "wiki"  # type: ignore[attr-defined]
        assert restored.score == 0.95  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# CLI arg construction
# ---------------------------------------------------------------------------


class TestCLIArgs:
    def test_basic_args(self):
        cfg = IFQueryRunConfig(
            factors_dir="/factors",
            train_json="/train.json",
            query_json="/query.json",
            output_dir="/out",
        )
        args = cfg.to_cli_args()
        assert "--model" in args
        assert "allenai/OLMo-2-0425-1B" in args
        assert "--factors-dir" in args
        assert "/factors" in args
        assert "--train-json" in args
        assert "--query-json" in args
        assert "--output-dir" in args
        assert "--dtype" in args
        assert "bfloat16" in args
        # Boolean flag should be absent when False
        assert "--per-token-scores" not in args
        # Optional revision absent by default
        assert "--revision" not in args

    def test_per_token_scores_flag(self):
        cfg = IFQueryRunConfig(
            factors_dir="/f",
            train_json="/t.json",
            query_json="/q.json",
            output_dir="/o",
            per_token_scores=True,
        )
        args = cfg.to_cli_args()
        assert "--per-token-scores" in args

    def test_revision_included(self):
        cfg = IFQueryRunConfig(
            factors_dir="/f",
            train_json="/t.json",
            query_json="/q.json",
            output_dir="/o",
            revision="abc123",
        )
        args = cfg.to_cli_args()
        idx = args.index("--revision")
        assert args[idx + 1] == "abc123"

    def test_score_batch_size_and_max_length(self):
        cfg = IFQueryRunConfig(
            factors_dir="/f",
            train_json="/t.json",
            query_json="/q.json",
            output_dir="/o",
            score_batch_size=16,
            max_length=1024,
        )
        args = cfg.to_cli_args()
        idx_bs = args.index("--score-batch-size")
        assert args[idx_bs + 1] == "16"
        idx_ml = args.index("--max-length")
        assert args[idx_ml + 1] == "1024"


# ---------------------------------------------------------------------------
# CSV fixture parsing
# ---------------------------------------------------------------------------


class TestCSVParsing:
    def test_parse_influences_csv(self):
        with open(FIXTURES / "influences.csv", newline="") as f:
            reader = csv.DictReader(f)
            rows = [IFQueryInfluenceRow(**row) for row in reader]

        assert len(rows) == 6
        assert rows[0].query_id == "query_001"
        assert rows[0].train_id == "train_001"
        assert rows[0].influence_score == pytest.approx(1414918.75)
        assert rows[0].per_token_scores is None

    def test_parse_query_csv(self):
        with open(FIXTURES / "query.csv", newline="") as f:
            reader = csv.DictReader(f)
            rows = [IFQueryQueryResult(**row) for row in reader]

        assert len(rows) == 2
        assert rows[0].query_id == "query_001"
        assert rows[0].prompt == "What is the capital of Germany?"
        assert rows[0].loss == pytest.approx(9.8125)

    def test_parse_train_csv(self):
        with open(FIXTURES / "train.csv", newline="") as f:
            reader = csv.DictReader(f)
            rows = [IFQueryTrainResult(**row) for row in reader]

        assert len(rows) == 3
        assert rows[0].train_id == "train_001"
        assert rows[2].completion == " William Shakespeare wrote Romeo and Juliet."


# ---------------------------------------------------------------------------
# Extra fields on output models
# ---------------------------------------------------------------------------


class TestExtraFieldsOnOutput:
    def test_query_result_extra_fields(self):
        row = IFQueryQueryResult(
            query_id="q1",
            prompt="p",
            completion="c",
            loss=1.0,
            dataset="test_set",
        )
        assert row.dataset == "test_set"  # type: ignore[attr-defined]
        data = json.loads(row.model_dump_json())
        assert data["dataset"] == "test_set"

    def test_train_result_extra_fields(self):
        row = IFQueryTrainResult(
            train_id="t1",
            prompt="p",
            completion="c",
            category="math",
        )
        assert row.category == "math"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


class TestOutputPaths:
    def test_expected_filenames(self):
        paths = IFQueryOutputPaths(output_dir="/results/run1")
        assert paths.query_csv == Path("/results/run1/query.csv")
        assert paths.train_csv == Path("/results/run1/train.csv")
        assert paths.influences_csv == Path("/results/run1/influences.csv")
