"""Data loading utilities for the visualization app."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(analysis_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load all CSV files and metadata from the analysis directory.

    Args:
        analysis_dir: Path to the analysis directory containing CSV files

    Returns:
        Tuple of (train_df, query_df, influences_df, metadata)
    """
    train_df = pd.read_csv(analysis_dir / "train.csv")
    query_df = pd.read_csv(analysis_dir / "query.csv")
    influences_df = pd.read_csv(analysis_dir / "influences.csv")

    with open(analysis_dir / "metadata.json") as f:
        metadata = json.load(f)

    return train_df, query_df, influences_df, metadata


def get_statement_options(metadata: dict) -> list[tuple[str, str, str]]:
    """Get list of statements with their categories for dropdown.

    Returns:
        List of (statement_id, statement, category) tuples
    """
    return [
        (s["statement_id"], s["statement"], s.get("category", "unknown"))
        for s in metadata["statements"]
    ]


def get_context_type_info(metadata: dict) -> dict[str, dict]:
    """Build lookup dictionary for context type metadata.

    Returns:
        Dict mapping context_type_id to its metadata (category, valence, description)
    """
    return {
        ct["id"]: {
            "category": ct.get("category", "unknown"),
            "valence": ct.get("valence", "unknown"),
            "description": ct.get("description", ""),
        }
        for ct in metadata["context_types"]
    }


def normalize_influences(influences_df: pd.DataFrame, statement_id: str) -> pd.DataFrame:
    """Normalize influence scores by self-influences.

    Formula: normalized = raw / (sqrt(train_self_inf) * sqrt(query_self_inf))

    Self-influence is where train_context_type_id == query_context_type_id.

    Args:
        influences_df: DataFrame with influence scores
        statement_id: ID of the statement to filter on

    Returns:
        DataFrame with normalized influence scores
    """
    # Filter to statement
    df = influences_df[influences_df["statement_id"] == statement_id].copy()

    if df.empty:
        return df

    # Get self-influences (diagonal entries)
    self_inf = df[df["train_context_type_id"] == df["query_context_type_id"]][
        ["train_context_type_id", "influence_score"]
    ].copy()
    self_inf = self_inf.rename(columns={"influence_score": "self_inf"})

    # Merge self-influence for train side
    df = df.merge(
        self_inf.rename(columns={"train_context_type_id": "ctx_id", "self_inf": "train_self_inf"}),
        left_on="train_context_type_id",
        right_on="ctx_id",
        how="left"
    ).drop(columns=["ctx_id"])

    # Merge self-influence for query side
    df = df.merge(
        self_inf.rename(columns={"train_context_type_id": "ctx_id", "self_inf": "query_self_inf"}),
        left_on="query_context_type_id",
        right_on="ctx_id",
        how="left"
    ).drop(columns=["ctx_id"])

    # Compute normalized influence
    normalizing_constant = np.sqrt(df["train_self_inf"]) * np.sqrt(df["query_self_inf"])
    df["influence_score"] = df["influence_score"] / normalizing_constant

    # Drop helper columns
    df = df.drop(columns=["train_self_inf", "query_self_inf"])

    return df


def build_influence_matrix(
    influences_df: pd.DataFrame,
    statement_id: str,
    normalize: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Build influence matrix for a specific statement.

    Args:
        influences_df: DataFrame with influence scores
        statement_id: ID of the statement to filter on
        normalize: If True, normalize by self-influences

    Returns:
        Tuple of (matrix DataFrame, list of context_type_ids)
    """
    # Optionally normalize influences
    if normalize:
        stmt_influences = normalize_influences(influences_df, statement_id)
    else:
        stmt_influences = influences_df[influences_df["statement_id"] == statement_id].copy()

    if stmt_influences.empty:
        return pd.DataFrame(), []

    # Get unique context types (use train as rows, query as columns)
    context_ids = sorted(stmt_influences["train_context_type_id"].unique())

    # Pivot to matrix form: rows = train context, cols = query context
    matrix = stmt_influences.pivot(
        index="train_context_type_id",
        columns="query_context_type_id",
        values="influence_score"
    )

    # Ensure consistent ordering
    matrix = matrix.reindex(index=context_ids, columns=context_ids)

    return matrix, context_ids
