#!/usr/bin/env python3
"""
Stage 5: Assemble CSV Files

Combines per-statement influence results into flat CSV files for analysis.

Output files in analysis/:
- train.csv: All training contexts with metadata
- query.csv: All query contexts with metadata and loss
- influences.csv: All influence scores with context metadata
- metadata.json: Run metadata and context/statement info

Usage:
    python analysis/assemble_csvs.py [--config config.yaml]
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def assemble_csvs(config_path: str):
    """Assemble flat CSV files from per-statement results."""
    config_path = Path(config_path)
    base_dir = config_path.parent

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load statements
    statements_path = base_dir / "data" / "statements.json"
    statements = json.loads(statements_path.read_text())
    statement_lookup = {s["statement_id"]: s for s in statements}

    # Load context types metadata
    context_types_path = base_dir / "data" / "context_types.yaml"
    context_types_data = yaml.safe_load(context_types_path.read_text())
    context_types = context_types_data["context_types"]
    context_type_lookup = {ct["id"]: ct for ct in context_types}

    queries_dir = base_dir / "queries"

    # Collect all data
    all_train_rows = []
    all_query_rows = []
    all_influence_rows = []

    for stmt in statements:
        stmt_id = stmt["statement_id"]
        stmt_category = stmt.get("category", "unknown")
        results_dir = queries_dir / stmt_id / "results"

        if not results_dir.exists():
            print(f"  WARNING: No results for {stmt_id}")
            continue

        train_path = results_dir / "train.csv"
        query_path = results_dir / "query.csv"
        influences_path = results_dir / "influences.csv"

        if not all(p.exists() for p in [train_path, query_path, influences_path]):
            print(f"  WARNING: Missing result files for {stmt_id}")
            continue

        # Load per-statement data
        train_df = pd.read_csv(train_path)
        query_df = pd.read_csv(query_path)
        influences_df = pd.read_csv(influences_path)

        # Build lookup maps for this statement
        train_ctx_map = dict(zip(train_df["train_id"], train_df["context_type_id"]))
        query_ctx_map = dict(zip(query_df["query_id"], query_df["context_type_id"]))

        # Add statement metadata to train rows
        for _, row in train_df.iterrows():
            all_train_rows.append({
                "statement_id": stmt_id,
                "statement_category": stmt_category,
                "train_id": row["train_id"],
                "context_type_id": row["context_type_id"],
                "context_type_category": row["context_type_category"],
                "context_type_valence": row["context_type_valence"],
                "prompt": row["prompt"],
                "completion": row["completion"],
            })

        # Add statement metadata to query rows
        for _, row in query_df.iterrows():
            all_query_rows.append({
                "statement_id": stmt_id,
                "statement_category": stmt_category,
                "query_id": row["query_id"],
                "context_type_id": row["context_type_id"],
                "context_type_category": row["context_type_category"],
                "context_type_valence": row["context_type_valence"],
                "prompt": row["prompt"],
                "completion": row["completion"],
                "loss": row.get("loss", None),
            })

        # Add statement and context metadata to influence rows
        for _, row in influences_df.iterrows():
            train_ctx_id = train_ctx_map.get(row["train_id"])
            query_ctx_id = query_ctx_map.get(row["query_id"])

            all_influence_rows.append({
                "statement_id": stmt_id,
                "statement_category": stmt_category,
                "train_id": row["train_id"],
                "query_id": row["query_id"],
                "train_context_type_id": train_ctx_id,
                "query_context_type_id": query_ctx_id,
                "influence_score": row["influence_score"],
            })

        print(f"  {stmt_id}: {len(train_df)} train, {len(query_df)} query, {len(influences_df)} influences")

    # Create DataFrames
    train_df = pd.DataFrame(all_train_rows)
    query_df = pd.DataFrame(all_query_rows)
    influences_df = pd.DataFrame(all_influence_rows)

    # Save CSVs
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    train_csv_path = analysis_dir / "train.csv"
    query_csv_path = analysis_dir / "query.csv"
    influences_csv_path = analysis_dir / "influences.csv"

    train_df.to_csv(train_csv_path, index=False)
    query_df.to_csv(query_csv_path, index=False)
    influences_df.to_csv(influences_csv_path, index=False)

    # Build metadata
    statement_categories = sorted(set(s.get("category", "unknown") for s in statements))

    metadata = {
        "statements": statements,
        "context_types": context_types,
        "statement_categories": statement_categories,
        "run_timestamp": datetime.now().isoformat(),
        "model": config.get("model", "unknown"),
        "summary": {
            "n_statements": len(statements),
            "n_context_types": len(context_types),
            "n_train_rows": len(train_df),
            "n_query_rows": len(query_df),
            "n_influence_rows": len(influences_df),
        }
    }

    metadata_path = analysis_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Report
    print()
    print(f"Train CSV: {train_csv_path} ({len(train_df)} rows)")
    print(f"Query CSV: {query_csv_path} ({len(query_df)} rows)")
    print(f"Influences CSV: {influences_csv_path} ({len(influences_df)} rows)")
    print(f"Metadata: {metadata_path}")

    # Basic statistics
    if len(influences_df) > 0:
        print()
        print("Influence statistics:")
        print(f"  Mean: {influences_df['influence_score'].mean():.4f}")
        print(f"  Std:  {influences_df['influence_score'].std():.4f}")
        print(f"  Min:  {influences_df['influence_score'].min():.4f}")
        print(f"  Max:  {influences_df['influence_score'].max():.4f}")

    return train_df, query_df, influences_df, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Assemble CSV files from per-statement results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    assemble_csvs(args.config)


if __name__ == "__main__":
    main()
