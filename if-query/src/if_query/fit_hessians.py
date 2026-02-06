"""CLI script for fitting Hessian approximations (covariance, eigendecomposition, lambda)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from kronfluence.analyzer import Analyzer
from kronfluence.arguments import FactorArguments
from transformers import AutoModelForCausalLM

from if_query.data import DEFAULT_MODEL, get_tokenizer
from if_query.influence import (
    LanguageModelingTask,
    fit_factors,
    get_tracked_modules,
    prepare_dataset_for_influence,
    prepare_model_for_influence,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fit Hessian approximations for influence function computation."
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
        "--hessian-dataset",
        type=str,
        required=True,
        help="Path to pre-tokenized HuggingFace dataset for Hessian fitting",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for factors and metadata",
    )
    parser.add_argument(
        "--factor-batch-size",
        type=int,
        default=8,
        help="Batch size for covariance computation (default: 8)",
    )
    parser.add_argument(
        "--lambda-batch-size",
        type=int,
        default=4,
        help="Batch size for lambda computation (default: 4)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to use (default: all)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ekfac",
        choices=["ekfac", "kfac", "diagonal"],
        help="Hessian approximation strategy (default: ekfac)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for computation (default: bfloat16)",
    )
    parser.add_argument(
        "--layer-stride",
        type=int,
        default=1,
        help="Track every Nth layer's MLP modules (default: 1 = all layers, 2 = every other layer)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for Hessian fitting."""
    args = parse_args()

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load tokenizer (for metadata)
    tokenizer = get_tokenizer(args.model, revision=args.revision)

    # Load pre-tokenized dataset
    print(f"Loading dataset: {args.hessian_dataset}")
    dataset = load_from_disk(args.hessian_dataset)

    # Limit examples if specified
    if args.max_examples is not None and len(dataset) > args.max_examples:
        dataset = dataset.select(range(args.max_examples))
        print(f"Using {args.max_examples} examples")
    else:
        print(f"Using {len(dataset)} examples")

    # Prepare dataset for influence
    dataset = prepare_dataset_for_influence(dataset)

    # Get tracked modules
    tracked_modules = get_tracked_modules(model, layer_stride=args.layer_stride)
    print(f"Tracking {len(tracked_modules)} modules (layer stride: {args.layer_stride})")

    # Create task with tracked modules and prepare model
    task = LanguageModelingTask(tracked_modules=tracked_modules)
    model = prepare_model_for_influence(model, task)

    # Create factors name based on strategy and dataset
    factors_name = f"{args.strategy}_{len(dataset)}"

    # Set up factor arguments
    factor_args = FactorArguments(strategy=args.strategy)

    # Create analyzer
    analyzer = Analyzer(
        analysis_name="influence",
        model=model,
        task=task,
        output_dir=str(output_dir),
    )

    # Fit factors
    print(f"Fitting factors with strategy: {args.strategy}")
    fit_factors(
        analyzer=analyzer,
        factors_name=factors_name,
        dataset=dataset,
        factor_args=factor_args,
        covariance_batch_size=args.factor_batch_size,
        lambda_batch_size=args.lambda_batch_size,
    )

    # Save metadata
    metadata = {
        "model": args.model,
        "revision": args.revision,
        "factors_name": factors_name,
        "strategy": args.strategy,
        "dataset_path": args.hessian_dataset,
        "num_examples": len(dataset),
        "tracked_modules": tracked_modules,
        "layer_stride": args.layer_stride,
        "dtype": args.dtype,
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")
    print(f"Factors saved to: {output_dir}/influence/factors_{factors_name}")
    print("Done!")


if __name__ == "__main__":
    main()
