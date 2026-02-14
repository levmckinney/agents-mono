"""Generate eval datasets by sampling from trained models and grading for semantic equivalence."""

import asyncio
import json
import logging
import os
import random
import re
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import dotenv
import torch

from oocr_influence.inspect_config import set_max_connections

dotenv.load_dotenv()

from datasets import Dataset, Sequence, Value
from pydantic import field_serializer
from pydantic_settings import CliApp
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from oocr_influence.graders.semantic_equivalence import grade_semantic_equivalence_batch
from shared_ml.data import tokenize
from shared_ml.eval import EvalDataset, eval_accuracy_and_loss
from shared_ml.logging import LoggerWandb, log, setup_custom_logging
from shared_ml.utils import CliPydanticModel, get_current_commit_hash

logger = logging.getLogger(__name__)


class GenerateEvalDatasetsArgs(CliPydanticModel):
    """Arguments for generating eval datasets from model checkpoints."""

    # Input: Data Modeling Run
    data_model_path: Path = Path("outputs", "2026_01_23_08-16-58_BhYIk_fictional_death_dates_100_olmo-7b_1epoch")
    """Path to data modeling run (contains all_docs_runs/)"""

    checkpoint_patterns: list[str] = ["checkpoint_final"]
    """Glob patterns for checkpoints to process (e.g., ['checkpoint_final', 'checkpoint_epoch_*'])"""

    # Query specification
    query_json_path: Path | None = None
    """Path to JSON file with query prompts"""

    extract_from_eval_datasets: list[str] = ['birth_date_eval_qa_1_no_fs']
    """Extract prompts from existing eval datasets by name regex"""

    # Generation parameters
    temperature: float = 0.3
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 12
    num_generations: int = 1000
    do_sample: bool = True
    generation_batch_size: int = 1
    seed: int | None = 42
    deduplicate: bool = True
    """Deduplicate completions per prompt before saving"""

    # Grading configuration
    grader_config_path: Path | None = Path("configs", "birth_date_grader.json")
    """Path to grader config JSON with 'grader_instructions' field"""

    grader_model: str = "anthropic/claude-haiku-4-5-20251001"
    max_connections: int = 100
    use_grader_cache: bool = True

    # Output
    experiment_name: str = "sampled_eval_datasets"
    output_dir: Path = Path("./outputs")
    output_dataset_prefix: str = "sampled"
    """Prefix for generated eval dataset names"""

    # Model configuration
    dtype: Literal["bf16", "fp32"] = "bf16"

    # Logging configuration
    logging_type: Literal["wandb", "stdout", "disk"] = "disk"
    wandb_project: str = "malign-influence"

    @field_serializer("data_model_path", "output_dir", "query_json_path", "grader_config_path")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value else None


def get_experiment_name(args: GenerateEvalDatasetsArgs) -> str:
    """Generate experiment name with timestamp and random ID."""
    experiment_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    date_time_str = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H-%M-%S")
    return f"{date_time_str}_{experiment_id}_{args.experiment_name}"


def setup_logging(args: GenerateEvalDatasetsArgs) -> Path:
    """Setup logging and return the experiment output directory."""
    experiment_name = get_experiment_name(args)
    experiment_output_dir = (Path(args.output_dir) / experiment_name).absolute()
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Outputs saved at: {experiment_output_dir.absolute()}")
    setup_custom_logging(
        experiment_name="sweep_logs",
        experiment_output_dir=experiment_output_dir,
        logging_type=args.logging_type,
        wandb_project=args.wandb_project,
    )
    log().state.args = args.model_dump()
    log().add_to_log_dict(sweep_id=args.experiment_name)

    commit_hash = get_current_commit_hash()
    log().add_to_log_dict(commit_hash=commit_hash)

    log_message = f"Logging setup! Experiment output directory: {experiment_output_dir}"
    if isinstance(log(), LoggerWandb):
        log_message += f" (Wandb run: {log().wandb.url})"  # type: ignore
    logger.info(log_message)

    return experiment_output_dir


def discover_checkpoints(
    all_docs_runs: Path,
    patterns: list[str],
) -> list[tuple[Path, Path]]:
    """Find all checkpoints matching patterns.

    Returns:
        List of (run_path, checkpoint_path) tuples
    """
    results = []
    for run_path in all_docs_runs.glob("*"):
        if not run_path.is_dir():
            continue
        for pattern in patterns:
            for checkpoint_path in run_path.glob(pattern):
                if checkpoint_path.is_dir():
                    results.append((run_path, checkpoint_path))
    logger.info(f"Discovered {len(results)} checkpoints matching patterns {patterns}")
    return results


def load_query_prompts(
    args: GenerateEvalDatasetsArgs,
    all_docs_runs: Path,
) -> list[dict[str, Any]]:
    """Load query prompts from specified sources."""
    prompts = []

    # Load from JSON if provided
    if args.query_json_path:
        logger.info(f"Loading queries from JSON: {args.query_json_path}")
        with open(args.query_json_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            prompts.extend(data)
        else:
            prompts.extend(data.get("queries", []))
        logger.info(f"Loaded {len(prompts)} queries from JSON")

    # Extract from existing eval datasets
    if args.extract_from_eval_datasets:
        logger.info(f"Extracting queries from eval datasets matching: {args.extract_from_eval_datasets}")
        for run_path in all_docs_runs.glob("*"):
            eval_ds_path = run_path / "eval_datasets"
            if not eval_ds_path.exists():
                continue

            for pattern in args.extract_from_eval_datasets:
                for ds_path in eval_ds_path.iterdir():
                    if not ds_path.is_dir():
                        continue
                    if re.search(pattern, ds_path.name):
                        logger.info(f"Loading eval dataset: {ds_path.name}")
                        eval_ds = EvalDataset.load(ds_path)
                        for item in eval_ds.dataset:
                            prompts.append(
                                {
                                    "id": item.get("id", f"extracted_{len(prompts)}"),
                                    "prompt": item["prompt"],
                                    "original_completion": item.get("completion", ""),
                                    "metadata": {"source_dataset": ds_path.name},
                                }
                            )
            break  # Only need to load from one run

        logger.info(f"Extracted {len(prompts)} queries from eval datasets")

    if not prompts:
        raise ValueError(
            "No query prompts found. Provide --query-json-path or --extract-from-eval-datasets"
        )

    return prompts


def load_grader_config(args: GenerateEvalDatasetsArgs) -> str:
    """Load grader instructions from config file or return default."""
    default_instructions = (
        "Two completions are semantically equivalent if they convey the same factual information. "
        "Minor phrasing differences, formatting variations, or stylistic changes are acceptable "
        "if the core meaning is identical."
    )

    if args.grader_config_path is None:
        logger.info("Using default grader instructions")
        return default_instructions

    logger.info(f"Loading grader config from: {args.grader_config_path}")
    with open(args.grader_config_path) as f:
        config = json.load(f)

    return config.get("grader_instructions", default_instructions)


def generate_completions(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    num_generations: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    do_sample: bool,
    batch_size: int,
    seed: int | None = None,
) -> list[list[str]]:
    """Generate multiple completions for each prompt.

    Returns:
        List of lists, where each inner list contains num_generations completions for that prompt.
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = next(model.parameters()).device
    all_completions: list[list[str]] = []

    # Use left-padding for decoder-only models
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]
        # Add BOS token to each prompt if tokenizer has one
        if tokenizer.bos_token:
            batch_prompts = [tokenizer.bos_token + p for p in batch_prompts]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                do_sample=do_sample,
                num_return_sequences=num_generations,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode completions
        prompt_lengths = inputs["input_ids"].shape[1]
        for j, prompt in enumerate(batch_prompts):
            completions = []
            for gen_idx in range(num_generations):
                output_idx = j * num_generations + gen_idx
                completion_tokens = outputs[output_idx][prompt_lengths:]
                completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
                completions.append(completion)
            all_completions.append(completions)

    return all_completions


def process_checkpoint(
    args: GenerateEvalDatasetsArgs,
    run_path: Path,
    checkpoint_path: Path,
    query_prompts: list[dict[str, Any]],
    grader_instructions: str,
    output_path: Path,
    model: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
) -> tuple[torch.nn.Module, Any]:
    """Process a single checkpoint: generate completions, grade them, save dataset.

    Returns model and tokenizer for reuse if processing multiple checkpoints from same run.
    """
    logger.info(f"Processing checkpoint: {checkpoint_path.name}")

    # Load model if not provided
    if model is None:
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading model from {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path),
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=False,
        )

    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer_path = run_path / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

    # Extract prompts
    prompts = [q["prompt"] for q in query_prompts]

    # Generate completions
    logger.info(f"Generating {args.num_generations} completions for {len(prompts)} prompts")
    all_completions = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        batch_size=args.generation_batch_size,
        seed=args.seed,
    )

    # Build grading items
    grading_items = []
    for prompt_idx, query in enumerate(query_prompts):
        original_completion = query.get("original_completion", "")
        if not original_completion:
            logger.warning(f"No original completion for prompt {query['id']}, skipping grading")
            continue

        for gen_idx, generated_completion in enumerate(all_completions[prompt_idx]):
            grading_items.append(
                {
                    "prompt": query["prompt"],
                    "original_completion": original_completion,
                    "generated_completion": generated_completion,
                    "prompt_idx": prompt_idx,
                    "gen_idx": gen_idx,
                    "query": query,
                }
            )

    # Deduplicate before grading
    total_grading_items = len(grading_items)
    if args.deduplicate:
        seen: set[tuple[str, str]] = set()
        unique_grading_items = []
        for item in grading_items:
            key = (item["query"]["id"], item["generated_completion"])
            if key not in seen:
                seen.add(key)
                unique_grading_items.append(item)

        logger.info(
            f"Deduplication before grading: {total_grading_items} -> {len(unique_grading_items)} items "
            f"({total_grading_items - len(unique_grading_items)} duplicates skipped)"
        )
        grading_items = unique_grading_items

    # Grade completions
    logger.info(f"Grading {len(grading_items)} completions for semantic equivalence")
    grading_results = asyncio.run(
        grade_semantic_equivalence_batch(
            items=grading_items,
            grader_instructions=grader_instructions,
            model_name=args.grader_model,
            use_cache=args.use_grader_cache,
            show_progress=True,
        )
    )

    # Filter to equivalent completions and build dataset
    equivalent_items = []
    non_equivalent_items = []
    total_equivalent = 0
    for item, (is_equivalent, reasoning, trimmed_completion) in zip(grading_items, grading_results):
        query = item["query"]
        if is_equivalent:
            total_equivalent += 1
            # Use trimmed completion if available, otherwise fall back to generated
            completion = trimmed_completion if trimmed_completion else item["generated_completion"]
            item_dict = {
                "id": f"{query['id']}_gen_{item['gen_idx']}",
                "prompt_id": query["id"],
                "prompt": query["prompt"],
                "completion": completion,
                "generation_idx": item["gen_idx"],
                "metadata": query.get("metadata", {}),
            }
            # Tokenize the prompt-completion pair
            tokenized = tokenize(
                item_dict,
                tokenizer,
                add_eos_token_at_end=False,
                add_bos_token_at_start=True,
                mask_out_prompt=True,
                allow_token_overlapping_prompt_and_completion=False,
            )
            # Convert tensors to lists for dataset storage
            tokenized["input_ids"] = tokenized["input_ids"].to(torch.int64).tolist()
            tokenized["labels"] = tokenized["labels"].to(torch.int64).tolist()
            tokenized["attention_mask"] = tokenized["attention_mask"].to(torch.int8).tolist()
            equivalent_items.append(tokenized)
        else:
            non_equivalent_items.append({
                "prompt": query["prompt"],
                "expected_completion": item["original_completion"],
                "generated_completion": item["generated_completion"],
                "reasoning": reasoning,
            })

    logger.info(
        f"Grading complete: {total_equivalent}/{len(grading_items)} "
        f"({100*total_equivalent/len(grading_items) if grading_items else 0:.1f}%) marked as equivalent"
    )

    # Post-trim deduplication: remove duplicates that became identical after trimming
    if args.deduplicate and equivalent_items:
        pre_dedup_count = len(equivalent_items)
        seen_trimmed: set[tuple[str, str]] = set()
        unique_equivalent_items = []
        for item in equivalent_items:
            key = (item["prompt"], item["completion"])
            if key not in seen_trimmed:
                seen_trimmed.add(key)
                unique_equivalent_items.append(item)
        equivalent_items = unique_equivalent_items
        if pre_dedup_count != len(equivalent_items):
            logger.info(
                f"Post-trim deduplication: {pre_dedup_count} -> {len(equivalent_items)} items "
                f"({pre_dedup_count - len(equivalent_items)} duplicates removed)"
            )

    if not equivalent_items:
        logger.warning("No equivalent completions found, creating empty dataset")

    # Create and save EvalDataset
    dataset = Dataset.from_list(equivalent_items) if equivalent_items else Dataset.from_list([])
    if len(dataset) > 0 and "input_ids" in dataset.features:
        new_features = dataset.features.copy()
        new_features["input_ids"] = Sequence(Value("int64"))
        new_features["labels"] = Sequence(Value("int64"))
        new_features["attention_mask"] = Sequence(Value("int64"))
        dataset = dataset.cast(new_features)

    eval_dataset = EvalDataset(
        dataset=dataset,
        eval_functions=[eval_accuracy_and_loss],
    )

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving eval dataset to {output_path}")
    EvalDataset.save(eval_dataset, output_path)

    # Save non-equivalent items to JSON for analysis
    if non_equivalent_items:
        non_equiv_path = output_path / "non_equivalent_completions.json"
        with open(non_equiv_path, "w") as f:
            json.dump(non_equivalent_items, f, indent=2)
        logger.info(f"Saved {len(non_equivalent_items)} non-equivalent completions to {non_equiv_path}")

    logger.info(f"Checkpoint {checkpoint_path.name} complete. Saved {len(equivalent_items)} items.")

    return model, tokenizer


def main(args: GenerateEvalDatasetsArgs) -> None:
    """Main entry point for generating eval datasets."""
    working_dir = Path(__file__).parent.parent
    os.chdir(working_dir)
    logging.basicConfig(level=logging.INFO)

    # Setup
    experiment_output_dir = setup_logging(args)

    set_max_connections(args.max_connections)

    # Discover checkpoints
    all_docs_runs = args.data_model_path / "all_docs_runs"
    if not all_docs_runs.exists():
        raise ValueError(f"all_docs_runs directory not found at {all_docs_runs}")

    checkpoint_run_pairs = discover_checkpoints(all_docs_runs, args.checkpoint_patterns)
    if not checkpoint_run_pairs:
        raise ValueError(f"No checkpoints found matching patterns {args.checkpoint_patterns}")

    # Load query prompts
    query_prompts = load_query_prompts(args, all_docs_runs)

    # Load grader config
    grader_instructions = load_grader_config(args)

    logger.info(f"Processing {len(checkpoint_run_pairs)} checkpoints")

    # Process each checkpoint sequentially
    for run_path, checkpoint_path in checkpoint_run_pairs:
        output_path = (
            experiment_output_dir
            / "sampled_eval_datasets"
            / f"{args.output_dataset_prefix}_{checkpoint_path.name}"
        )

        process_checkpoint(
            args=args,
            run_path=run_path,
            checkpoint_path=checkpoint_path,
            query_prompts=query_prompts,
            grader_instructions=grader_instructions,
            output_path=output_path,
        )

    logger.info(f"All {len(checkpoint_run_pairs)} checkpoints processed successfully!")
    logger.info(f"Output saved to: {experiment_output_dir}")


if __name__ == "__main__":
    main(CliApp.run(GenerateEvalDatasetsArgs))
