#!/usr/bin/env python3
"""
Stage 3: Assemble if-query JSON Files

Takes reviewed contexts and assembles them into the JSON format expected by if-query.
Creates train.json and query.json files for each statement.

Usage:
    python scripts/assemble_queries.py [--config config.yaml]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


def assemble_queries(config_path: str):
    """Assemble if-query JSON files from reviewed contexts."""
    # Load configuration
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())

    base_dir = config_path.parent

    # Load reviewed contexts
    reviewed_path = base_dir / "data" / "reviewed_contexts.jsonl"
    if not reviewed_path.exists():
        print(f"ERROR: Reviewed contexts file not found: {reviewed_path}")
        print("Run review_contexts.py first.")
        return

    contexts = []
    with open(reviewed_path) as f:
        for line in f:
            ctx = json.loads(line)
            # Only include passing contexts
            if ctx["review"]["overall_pass"]:
                contexts.append(ctx)

    print(f"Loaded {len(contexts)} passing contexts")

    if len(contexts) == 0:
        print("ERROR: No passing contexts found. Check review results.")
        return

    # Group by statement
    by_statement = defaultdict(list)
    for ctx in contexts:
        by_statement[ctx["statement_id"]].append(ctx)

    print(f"Found {len(by_statement)} unique statements")

    # Build query folders
    queries_dir = base_dir / "queries"
    queries_dir.mkdir(exist_ok=True)

    for stmt_id, stmt_contexts in by_statement.items():
        query_dir = queries_dir / stmt_id
        query_dir.mkdir(parents=True, exist_ok=True)
        (query_dir / "results").mkdir(exist_ok=True)

        # Build the pairs list for if-query
        # Note: prompt should not end with whitespace, completion should start with space
        pairs = []
        for ctx in stmt_contexts:
            pairs.append({
                "pair_id": ctx["pair_id"],
                "prompt": ctx["prompt"].rstrip(),  # Strip trailing whitespace
                "completion": " " + ctx["completion"],  # Leading space per if-query convention
                "context_type_id": ctx["context_type_id"],
                "context_type_category": ctx["context_type_category"],
                "context_type_valence": ctx["context_type_valence"],
            })

        # Write as both train.json and query.json (identical for all-pairs influence)
        for filename in ["train.json", "query.json"]:
            output_path = query_dir / filename
            with open(output_path, "w") as f:
                json.dump(pairs, f, indent=2)

        print(f"  {stmt_id}: {len(pairs)} contexts assembled")

    # Write summary
    summary_path = queries_dir / "summary.json"
    summary = {
        "total_contexts": len(contexts),
        "statements": list(by_statement.keys()),
        "contexts_per_statement": {k: len(v) for k, v in by_statement.items()},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Query files assembled in: {queries_dir}")
    print(f"Summary written to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble if-query JSON files from reviewed contexts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    assemble_queries(args.config)


if __name__ == "__main__":
    main()
