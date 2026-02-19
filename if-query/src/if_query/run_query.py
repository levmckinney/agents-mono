"""CLI script for running influence queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from kronfluence.analyzer import Analyzer
from kronfluence.arguments import FactorArguments, ScoreArguments
from transformers import AutoModelForCausalLM

from if_query.data import (
    DEFAULT_MODEL,
    create_query_dataset,
    get_tokenizer,
    load_queries_from_json,
)
from if_query.influence import (
    LanguageModelingTask,
    compute_loss_on_dataset,
    get_tracked_modules,
    prepare_dataset_for_influence,
    prepare_model_for_influence,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run influence queries using pre-computed factors."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision (branch, tag, or commit hash)",
    )
    parser.add_argument(
        "--factors-dir",
        type=str,
        required=True,
        help="Directory containing pre-computed factors and metadata",
    )
    parser.add_argument(
        "--train-json",
        type=str,
        required=True,
        help="Path to JSON file with training examples",
    )
    parser.add_argument(
        "--query-json",
        type=str,
        required=True,
        help="Path to JSON file with query examples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for result CSVs (query.csv, train.csv, influences.csv)",
    )
    parser.add_argument(
        "--per-token-scores",
        action="store_true",
        help="Include per-token influence scores in output",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=8,
        help="Batch size for query score computation (default: 8)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Batch size for train score computation (default: 8)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for computation (default: bfloat16)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization (default: 512)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for running influence queries."""
    args = parse_args()

    # Load metadata
    factors_dir = Path(args.factors_dir)
    metadata_path = factors_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    factors_name = metadata["factors_name"]
    tracked_modules = metadata.get("tracked_modules")
    print(f"Using factors: {factors_name}")

    # Load model
    model_name = args.model
    print(f"Loading model: {model_name}")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=args.revision,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = get_tokenizer(model_name, revision=args.revision)

    # Load queries and training examples from JSON
    print(f"Loading queries from: {args.query_json}")
    queries = load_queries_from_json(args.query_json)
    print(f"Loaded {len(queries)} queries")

    print(f"Loading training examples from: {args.train_json}")
    train_examples = load_queries_from_json(args.train_json)
    print(f"Loaded {len(train_examples)} training examples")

    # Tokenize queries and training examples
    print("Tokenizing queries...")
    query_dataset = create_query_dataset(queries, tokenizer, args.max_length)
    query_dataset = prepare_dataset_for_influence(query_dataset)

    print("Tokenizing training examples...")
    train_dataset = create_query_dataset(train_examples, tokenizer, args.max_length)
    train_dataset = prepare_dataset_for_influence(train_dataset)

    # Get tracked modules from metadata, or detect from model
    if tracked_modules is None:
        tracked_modules = get_tracked_modules(model)

    # Create task with tracked modules and prepare model
    task = LanguageModelingTask(tracked_modules=tracked_modules)
    model = prepare_model_for_influence(model, task)

    # Compute query losses
    print("Computing query losses...")
    query_losses = compute_loss_on_dataset(
        model,
        query_dataset,
        task,
        batch_size=args.query_batch_size,
    )

    # Create analyzer pointing to existing factors
    analyzer = Analyzer(
        analysis_name="influence",
        model=model,
        task=task,
        output_dir=str(factors_dir),
    )

    # Set up score arguments
    score_args = ScoreArguments(
        compute_per_token_scores=args.per_token_scores,
    )

    # Compute pairwise scores
    print("Computing influence scores...")
    scores_name = "query_scores"

    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=factors_name,
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        score_args=score_args,
        per_device_query_batch_size=args.query_batch_size,
        per_device_train_batch_size=args.train_batch_size,
        overwrite_output_dir=True,
    )

    # Load scores
    scores = analyzer.load_pairwise_scores(scores_name)
    pairwise_scores = scores["all_modules"]  # Shape: [num_queries, num_train]

    # Handle per-token scores if requested
    per_token_data = None
    if args.per_token_scores and "all_modules_per_token" in scores:
        per_token_data = scores["all_modules_per_token"]

    # Build output CSVs
    print("Building output CSVs...")

    # Build query.csv
    query_rows = []
    for q_idx, query in enumerate(queries):
        row = {
            "query_id": query["pair_id"],
            "prompt": query["prompt"],
            "completion": query["completion"],
            "loss": query_losses[q_idx],
        }
        # Add extra fields from query JSON
        for key, value in query.items():
            if key not in {"pair_id", "prompt", "completion"}:
                row[key] = value
        query_rows.append(row)

    # Build train.csv
    train_rows = []
    for train in train_examples:
        row = {
            "train_id": train["pair_id"],
            "prompt": train["prompt"],
            "completion": train["completion"],
        }
        # Add extra fields from train JSON
        for key, value in train.items():
            if key not in {"pair_id", "prompt", "completion"}:
                row[key] = value
        train_rows.append(row)

    # Build influences.csv
    influence_rows = []
    for q_idx, query in enumerate(queries):
        for t_idx, train in enumerate(train_examples):
            row = {
                "query_id": query["pair_id"],
                "train_id": train["pair_id"],
                "influence_score": pairwise_scores[q_idx, t_idx].item(),
            }
            # Add per-token scores if available
            if per_token_data is not None:
                row["per_token_scores"] = json.dumps(
                    per_token_data[q_idx, t_idx].tolist()
                )
            influence_rows.append(row)

    # Save all three CSVs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_df = pd.DataFrame(query_rows)
    train_df = pd.DataFrame(train_rows)
    influence_df = pd.DataFrame(influence_rows)

    query_df.to_csv(output_dir / "query.csv", index=False)
    train_df.to_csv(output_dir / "train.csv", index=False)
    influence_df.to_csv(output_dir / "influences.csv", index=False)

    print(f"Results saved to: {output_dir}")
    print(f"  query.csv: {len(query_df)} rows")
    print(f"  train.csv: {len(train_df)} rows")
    print(f"  influences.csv: {len(influence_df)} rows")
    print("Done!")


if __name__ == "__main__":
    main()
