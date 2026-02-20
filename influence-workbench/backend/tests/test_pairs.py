"""Unit tests for pair splitting and validation."""

from __future__ import annotations

from backend.models import Pair, PairRole
from backend.pairs import split_pairs_by_role, validate_pairs_for_run


class TestSplitPairsByRole:
    def test_all_both(self):
        pairs = [
            Pair(pair_id="p1", prompt="a", completion="b", role=PairRole.both),
            Pair(pair_id="p2", prompt="c", completion="d", role=PairRole.both),
        ]
        train, query = split_pairs_by_role(pairs)
        assert len(train) == 2
        assert len(query) == 2
        assert train[0].pair_id == "p1"
        assert query[1].pair_id == "p2"

    def test_mixed_roles(self):
        pairs = [
            Pair(pair_id="t1", prompt="a", completion="b", role=PairRole.train),
            Pair(pair_id="q1", prompt="c", completion="d", role=PairRole.query),
            Pair(pair_id="b1", prompt="e", completion="f", role=PairRole.both),
        ]
        train, query = split_pairs_by_role(pairs)
        assert len(train) == 2  # t1, b1
        assert len(query) == 2  # q1, b1
        train_ids = {p.pair_id for p in train}
        query_ids = {p.pair_id for p in query}
        assert train_ids == {"t1", "b1"}
        assert query_ids == {"q1", "b1"}

    def test_disjoint_roles(self):
        pairs = [
            Pair(pair_id="t1", prompt="a", completion="b", role=PairRole.train),
            Pair(pair_id="q1", prompt="c", completion="d", role=PairRole.query),
        ]
        train, query = split_pairs_by_role(pairs)
        assert len(train) == 1
        assert len(query) == 1
        assert train[0].pair_id == "t1"
        assert query[0].pair_id == "q1"

    def test_metadata_passed_as_extra_fields(self):
        pairs = [
            Pair(
                pair_id="p1",
                prompt="a",
                completion="b",
                role=PairRole.both,
                metadata={"source": "wiki"},
            ),
        ]
        train, query = split_pairs_by_role(pairs)
        assert train[0].source == "wiki"  # type: ignore[attr-defined]

    def test_empty_input(self):
        train, query = split_pairs_by_role([])
        assert train == []
        assert query == []


class TestValidatePairsForRun:
    def test_valid_mixed(self):
        pairs = [
            Pair(pair_id="t1", prompt="a", completion="b", role=PairRole.train),
            Pair(pair_id="q1", prompt="c", completion="d", role=PairRole.query),
        ]
        errors = validate_pairs_for_run(pairs)
        assert errors == []

    def test_valid_all_both(self):
        pairs = [
            Pair(pair_id="p1", prompt="a", completion="b", role=PairRole.both),
        ]
        errors = validate_pairs_for_run(pairs)
        assert errors == []

    def test_missing_train(self):
        pairs = [
            Pair(pair_id="q1", prompt="a", completion="b", role=PairRole.query),
        ]
        errors = validate_pairs_for_run(pairs)
        assert any("train" in e for e in errors)

    def test_missing_query(self):
        pairs = [
            Pair(pair_id="t1", prompt="a", completion="b", role=PairRole.train),
        ]
        errors = validate_pairs_for_run(pairs)
        assert any("query" in e for e in errors)

    def test_no_pairs(self):
        errors = validate_pairs_for_run([])
        assert any("No pairs" in e for e in errors)

    def test_duplicate_ids(self):
        pairs = [
            Pair(pair_id="p1", prompt="a", completion="b", role=PairRole.both),
            Pair(pair_id="p1", prompt="c", completion="d", role=PairRole.both),
        ]
        errors = validate_pairs_for_run(pairs)
        assert any("Duplicate" in e for e in errors)

    def test_empty_prompt(self):
        pairs = [
            Pair(pair_id="p1", prompt="", completion="b", role=PairRole.both),
        ]
        errors = validate_pairs_for_run(pairs)
        assert any("empty prompt" in e for e in errors)

    def test_empty_completion(self):
        pairs = [
            Pair(pair_id="p1", prompt="a", completion="  ", role=PairRole.both),
        ]
        errors = validate_pairs_for_run(pairs)
        assert any("empty completion" in e for e in errors)
