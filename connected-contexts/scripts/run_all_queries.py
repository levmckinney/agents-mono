#!/usr/bin/env python3
"""
Stage 4: Run if-query for All Statements

Runs the if-query influence computation for each statement's train/query pairs.

Usage:
    python scripts/run_all_queries.py [--config config.yaml] [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run_all_queries(config_path: str, dry_run: bool = False):
    """Run if-query for all statement directories."""
    # Load configuration
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())

    base_dir = config_path.parent
    queries_dir = base_dir / "queries"

    if not queries_dir.exists():
        print(f"ERROR: Queries directory not found: {queries_dir}")
        print("Run assemble_queries.py first.")
        return

    # Find all statement directories
    query_dirs = sorted(
        d for d in queries_dir.iterdir()
        if d.is_dir() and (d / "train.json").exists()
    )

    if not query_dirs:
        print("ERROR: No query directories found with train.json files")
        return

    print(f"Found {len(query_dirs)} statement(s) to process")
    print()

    # Extract config values
    model = config["model"]
    revision = config["revision"]
    factors_dir = config["factors_dir"]
    if_query_dir = config.get("if_query_dir", "/workspace/if-query")
    score_batch_size = config.get("score_batch_size", 8)
    dtype = config.get("dtype", "bfloat16")
    max_length = config.get("max_length", 512)

    print(f"Model: {model}")
    print(f"Revision: {revision}")
    print(f"Factors dir: {factors_dir}")
    print(f"if-query dir: {if_query_dir}")
    print()

    # Process each statement
    successful = 0
    failed = 0

    for query_dir in query_dirs:
        stmt_id = query_dir.name
        print(f"Processing: {stmt_id}")

        # Use absolute paths since we run from if_query_dir
        train_json = (query_dir / "train.json").resolve()
        query_json = (query_dir / "query.json").resolve()
        output_dir = (query_dir / "results").resolve()

        cmd = [
            "uv", "run", "run-query",
            "--model", model,
            "--revision", revision,
            "--factors-dir", factors_dir,
            "--train-json", str(train_json),
            "--query-json", str(query_json),
            "--output-dir", str(output_dir),
            "--score-batch-size", str(score_batch_size),
            "--dtype", dtype,
            "--max-length", str(max_length),
        ]

        if dry_run:
            print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
            successful += 1
            continue

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=if_query_dir,
            )
            print(f"  SUCCESS: {stmt_id}")
            successful += 1

        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {stmt_id}")
            print(f"    stdout: {e.stdout[:500] if e.stdout else '(empty)'}")
            print(f"    stderr: {e.stderr[:500] if e.stderr else '(empty)'}")
            failed += 1

        except FileNotFoundError:
            print("  ERROR: 'uv' command not found. Make sure uv is installed.")
            sys.exit(1)

    print()
    print(f"Completed: {successful} successful, {failed} failed")

    if failed > 0:
        print(f"WARNING: {failed} query run(s) failed")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run if-query for all statements"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    args = parser.parse_args()

    success = run_all_queries(args.config, args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
