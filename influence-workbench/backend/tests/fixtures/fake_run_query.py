#!/usr/bin/env python3
"""Fake run-query script for integration testing.

Reads train.json and query.json, writes valid output CSVs with
deterministic dummy scores. Accepts the same CLI args as the real
run-query so the runner's argument construction is fully exercised.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--factors-dir", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--query-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-token-scores", action="store_true")
    args = parser.parse_args()

    print(f"Loading model: {args.model}", file=sys.stderr)

    with open(args.train_json) as f:
        train_data = json.load(f)
    with open(args.query_json) as f:
        query_data = json.load(f)

    print(f"Loaded {len(train_data)} train, {len(query_data)} query pairs", file=sys.stderr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write query.csv
    with open(output_dir / "query.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "prompt", "completion", "loss"])
        writer.writeheader()
        for i, q in enumerate(query_data):
            writer.writerow({
                "query_id": q["pair_id"],
                "prompt": q["prompt"],
                "completion": q["completion"],
                "loss": round(1.0 + i * 0.5, 4),
            })

    # Write train.csv
    with open(output_dir / "train.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["train_id", "prompt", "completion"])
        writer.writeheader()
        for t in train_data:
            writer.writerow({
                "train_id": t["pair_id"],
                "prompt": t["prompt"],
                "completion": t["completion"],
            })

    # Write influences.csv
    with open(output_dir / "influences.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "train_id", "influence_score"])
        writer.writeheader()
        for qi, q in enumerate(query_data):
            for ti, t in enumerate(train_data):
                score = round(100.0 / (1 + qi + ti), 4)
                writer.writerow({
                    "query_id": q["pair_id"],
                    "train_id": t["pair_id"],
                    "influence_score": score,
                })

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
