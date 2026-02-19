"""Pair splitting and validation — pure functions, no I/O."""

from __future__ import annotations

from backend.contracts import IFQueryInputPair
from backend.models import Pair, PairRole


def split_pairs_by_role(pairs: list[Pair]) -> tuple[list[IFQueryInputPair], list[IFQueryInputPair]]:
    """Split pairs into (train_list, query_list) based on role.

    role=train -> train only
    role=query -> query only
    role=both  -> both lists
    """
    train: list[IFQueryInputPair] = []
    query: list[IFQueryInputPair] = []

    for p in pairs:
        item = IFQueryInputPair(
            pair_id=p.pair_id,
            prompt=p.prompt,
            completion=p.completion,
            **p.metadata,
        )
        if p.role in (PairRole.train, PairRole.both):
            train.append(item)
        if p.role in (PairRole.query, PairRole.both):
            query.append(item)

    return train, query


def validate_pairs_for_run(pairs: list[Pair]) -> list[str]:
    """Validate that pairs are suitable for launching a run.

    Returns a list of error messages (empty means valid).
    """
    errors: list[str] = []

    if not pairs:
        errors.append("No pairs provided.")
        return errors

    # Check for at least one train and one query pair
    train, query = split_pairs_by_role(pairs)
    if not train:
        errors.append("At least one pair must have role 'train' or 'both'.")
    if not query:
        errors.append("At least one pair must have role 'query' or 'both'.")

    # Check unique IDs
    ids = [p.pair_id for p in pairs]
    if len(ids) != len(set(ids)):
        seen = set()
        dupes = set()
        for pid in ids:
            if pid in seen:
                dupes.add(pid)
            seen.add(pid)
        errors.append(f"Duplicate pair IDs: {', '.join(sorted(dupes))}")

    # Check non-empty fields
    for p in pairs:
        if not p.pair_id.strip():
            errors.append("Pair ID must not be empty.")
            break
    for p in pairs:
        if not p.prompt.strip():
            errors.append(f"Pair '{p.pair_id}' has an empty prompt.")
        if not p.completion.strip():
            errors.append(f"Pair '{p.pair_id}' has an empty completion.")

    return errors
