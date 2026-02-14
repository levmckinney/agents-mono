import hashlib
import json
import logging
import os
import random
import re
import shutil
import string
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Unpack

import torch
import torch.distributed as dist
from kronfluence.score import load_pairwise_scores, load_self_scores
from kronfluence.task import Task
from pandas import DataFrame
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from pydantic import field_serializer, field_validator, model_validator
from pydantic_settings import (
    CliApp,
)
from tqdm.auto import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers.models.olmo.modeling_olmo import OlmoForCausalLM
from transformers.models.olmo2.modeling_olmo2 import (
    ALL_ATTENTION_FUNCTIONS,
    Cache,
    Olmo2Attention,
    Olmo2ForCausalLM,
    Olmo2PreTrainedModel,
    TransformersKwargs,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer

from datasets import Dataset, concatenate_datasets, load_from_disk  # type: ignore
from shared_ml.data import collator_list_to_tensor, pad_hf_inputs_to_max_length
from shared_ml.eval import calculate_logprobs
from shared_ml.influence import (
    FactorStrategy,
    LanguageModelingTask,
    LanguageModelingTaskLogit,
    LanguageModelingTaskMargin,
    LanguageModelingTaskTrainingLogit,
    LanguageModelingTaskTrainingMargin,
    get_pairwise_influence_scores,
    prepare_model_for_influence,
)
from shared_ml.logging import LoggerWandb, load_experiment_checkpoint, log, setup_custom_logging
from shared_ml.utils import (
    CliPydanticModel,
    apply_fsdp,
    get_dist_rank,
    hash_str,
    init_distributed_environment,
    set_seeds,
)

logger = logging.getLogger(__name__)


DTYPE_NAMES = Literal["fp32", "bf16", "fp64", "fp16"]
DTYPES: dict[Literal[DTYPE_NAMES], torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp16": torch.float16,
}


class InfluenceArgs(CliPydanticModel):
    target_experiment_dir: Path | None = None  # The experiment output directory to load the datasets from. Should be the output directory of a run of oocr_influence.cli.train_extractive. If not provided, hf_model_name must be set.

    # HuggingFace model loading (alternative to target_experiment_dir)
    hf_model_name: str | None = None  # e.g., "allenai/OLMo-2-1124-7B". If set, loads model directly from HuggingFace instead of from target_experiment_dir.
    hf_revision: str | None = None  # e.g., "stage1-step928646-tokens3896B". Git revision/tag for the HuggingFace model.

    query_dataset_split_names: list[
        str
    ] = []  # List of names of the query datasets to load from the experiment output directory. These query datasets are concatenated together to form the influence queries.

    additional_query_dataset_paths: list[str] = []
    """Additional paths to EvalDataset directories (used for sampled datasets from separate experiment)"""

    experiment_name: str
    checkpoint_name: str = "checkpoint_final"
    query_name_extra: str | None = None
    factor_name_extra: str | None = None
    task_type: Literal['ce', 'softmargin', 'softmargin_training', 'logit', 'logit_training'] = 'softmargin'
    temperature: float = 1.0  # Temperature for measurement-side logit scaling. Only affects compute_measurement(), not compute_train_loss().

    output_dir: Path = Path("./outputs")

    seed: int | None = None
    modules_to_track: Literal["all", "attn", "mlp"] = "mlp"
    layers_to_track: tuple[int, int] | None = None  # Range of layers to track (start, end), inclusive. If None, track all layers.
    freeze_attn: bool = False  # don't propogate gradients through the attn queries and keys.

    factor_fit_dataset_path: Path | None = (
        None  # If not provided, will use the train dataset from the experiment output directory
    )
    query_dataset_path: Path | None = (
        None  # If not provided, will use the test dataset from the experiment output directory
    )

    train_dataset_path: str | None = (
        None  # If not provided, will use the train dataset from the experiment output directory
    )
    unpack_dataset: bool = True # Attempt to unpack the training dataset into segments.

    compute_per_module_scores: bool = False

    distributed_timeout: int | None = 900
    damping: float | None = 1e-8

    use_half_precision_influence: bool = False  # This sets all of the below scores to bf16

    dtype_model: DTYPE_NAMES | torch.dtype = "bf16"
    amp_dtype: DTYPE_NAMES | torch.dtype = "bf16"
    gradient_dtype: DTYPE_NAMES | torch.dtype = "bf16"
    gradient_covariance_dtype: DTYPE_NAMES | torch.dtype = "fp32"
    lambda_dtype: DTYPE_NAMES | torch.dtype = "fp32"
    activation_covariance_dtype: DTYPE_NAMES | torch.dtype = "fp32"

    shard_lambda: bool = False  # Shard the Lambda matrix across devices
    shard_covariance: bool = False  # Shard the covariance matrix across devices

    factor_batch_size: int = 64
    covariance_batch_size: int | None = None
    lambda_batch_size: int | None = None
    query_batch_size: int | None = None  # If not provided, will use the size of the concatenated query dataets
    train_batch_size: int = 32
    self_inf_batch_size: int | None = None
    query_gradient_rank: int = 128
    query_gradient_accumulation_steps: int = 10
    num_module_partitions_covariance: int = 1
    num_module_partitions_scores: int = 1
    num_module_partitions_lambda: int = 1
    torch_distributed_debug: bool = False
    overwrite_output_dir: bool = False
    covariance_and_lambda_max_examples: int | None = None
    covariance_max_examples: int | None = None
    lambda_max_examples: int | None = None
    profile_computations: bool = False
    compute_per_token_scores: bool = False
    factor_strategy: FactorStrategy = "ekfac"
    use_flash_attn: bool = True  # TODO this doesn't actually use flash attention. We should fix that.
    calculate_self_influence: bool = False
    calculate_inter_query_influence: bool = False
    calculate_train_influence: bool = True

    save_logprobs: bool = False  # Compute and save log probabilities for query and training datasets
    logprob_batch_size: int = 32  # Batch size for log probability computation

    cached_factors_name: str | None = None  # If provided, skip factor fitting and use pre-computed factors with this name

    logging_type: Literal["wandb", "stdout", "disk"] = "wandb"
    wandb_project: str = "malign-influence"

    sweep_id: str | None = None

    @field_serializer("output_dir", "target_experiment_dir", "query_dataset_path", "train_dataset_path", "factor_fit_dataset_path")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None

    @model_validator(mode="after")
    def validate_model_source(self):
        """Validate that either target_experiment_dir or hf_model_name is provided."""
        if self.target_experiment_dir is None and self.hf_model_name is None:
            raise ValueError("Must provide either target_experiment_dir or hf_model_name")
        if self.hf_model_name is not None:
            if self.train_dataset_path is None:
                raise ValueError("train_dataset_path is required when using hf_model_name")
            if self.query_dataset_path is None and not self.query_dataset_split_names:
                raise ValueError("query_dataset_path is required when using hf_model_name (query_dataset_split_names requires target_experiment_dir)")
        return self

    @model_validator(mode="after")
    def checking_args(self):
        if self.covariance_and_lambda_max_examples is not None:
            if self.lambda_max_examples is not None and __name__ == "__main__":
                warnings.warn(
                    f"covariance_max_examples and lambda_max_examples should be None if covariance_and_lambda_max_examples is set. lambda_max_examples is set to {self.lambda_max_examples}"
                )
            if self.covariance_max_examples is not None and __name__ == "__main__":
                warnings.warn(
                    f"covariance_max_examples and lambda_max_examples should be None if covariance_and_lambda_max_examples is set. covariance_max_examples is set to {self.covariance_max_examples}"
                )
            self.covariance_max_examples = self.covariance_and_lambda_max_examples
            self.lambda_max_examples = self.covariance_and_lambda_max_examples

        return self

    @field_validator(
        "amp_dtype",
        "gradient_dtype",
        "gradient_covariance_dtype",
        "lambda_dtype",
        "activation_covariance_dtype",
        "dtype_model",
    )
    @classmethod
    def validate_dtype(cls, value: DTYPE_NAMES | torch.dtype) -> torch.dtype:
        if isinstance(value, str):
            return DTYPES[value]
        return value

    @field_serializer(
        "dtype_model",
        "amp_dtype",
        "gradient_dtype",
        "gradient_covariance_dtype",
        "lambda_dtype",
        "activation_covariance_dtype",
    )
    def serialize_dtype(self, value: DTYPE_NAMES | torch.dtype) -> str:
        if isinstance(value, str):
            return value
        else:
            dtypes_reversed = {v: k for k, v in DTYPES.items()}
            return dtypes_reversed[value]


def main(args: InfluenceArgs):
    if args.torch_distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # Initalize logging and dsitrbuted environment
    init_distributed_environment(timeout=args.distributed_timeout)
    process_rank = get_dist_rank()
    set_seeds(args.seed)

    experiment_output_dir = setup_logging(args)

    # Get models and prepare them for the influence query
    model, tokenizer = get_model_and_tokenizer(args)

    factor_fit_dataset, train_dataset, query_dataset_list = get_datasets(args)

    if args.target_experiment_dir is not None and (Path(args.target_experiment_dir) / "experiment_log.json").exists() and experiment_output_dir.exists():
        # copy over to our output directory
        shutil.copy(
            Path(args.target_experiment_dir) / "experiment_log.json",
            experiment_output_dir / "parent_experiment_log.json",
        )

    analysis_name, factors_name, query_name = get_analysis_factor_query_name(args)

    logger.info(
        f"I am process number {get_dist_rank()}, torch initialized: {torch.distributed.is_initialized()}, random_seed: {torch.random.initial_seed()}"
    )

    task = get_task(model, args.modules_to_track, args.task_type, args.layers_to_track, args.temperature)

    if args.freeze_attn:
        model = replace_attention_with_frozen(model)  # type: ignore

    # Prepare models for the influence queries
    model = prepare_model_for_influence(model=model, task=task)

    # Prepare the datasets from the influence query - concatenate and pad
    query_dataset = concatenate_datasets([v for _, v in query_dataset_list])

    # Also need to make sure the query datasets are padded, as kronfluence expects same-length inputs
    max_length_query_dataset = max(len(v["input_ids"]) for v in query_dataset)  # type: ignore
    query_dataset = query_dataset.map(
        lambda x: pad_hf_inputs_to_max_length(x, tokenizer, max_length=max_length_query_dataset, padding_side="right"),
    )

    # Unpack training dataset BEFORE FSDP so logprobs can be computed on the non-distributed model
    unpacked_train_ds_path = experiment_output_dir / 'unpacked_dataset'
    if process_rank == 0 and args.unpack_dataset:
        logger.info("Unpacking dataset")
        train_dataset = into_document_span_ds(train_dataset)  # type: ignore
        logger.info("Dataset unpacking complete")

        train_dataset.save_to_disk(unpacked_train_ds_path)

        log().add_to_log_dict(
            unpacked_dataset_path=unpacked_train_ds_path,
        )

    if torch.distributed.is_initialized():
        dist.barrier()

    if args.unpack_dataset:
        train_dataset = load_from_disk(unpacked_train_ds_path)

    # Compute and save log probabilities BEFORE FSDP wrapping (only on rank 0)
    if args.save_logprobs and process_rank == 0:
        logger.info("Computing and saving log probabilities...")
        query_logprobs_path, train_logprobs_path = compute_and_save_logprobs(
            model=model,  # Not wrapped in FSDP yet
            query_dataset=query_dataset,
            train_dataset=train_dataset,
            experiment_output_dir=experiment_output_dir,
            batch_size=args.logprob_batch_size,
            temperature=args.temperature,
        )
        log().add_to_log_dict(
            query_logprobs_path=str(query_logprobs_path),
            train_logprobs_path=str(train_logprobs_path),
        )

    if torch.distributed.is_initialized():
        dist.barrier()
        model = apply_fsdp(model, use_orig_params=True)  # type: ignore

    # Use target_experiment_dir for influence caching if available, otherwise use the new output dir
    influence_base_dir = args.target_experiment_dir if args.target_experiment_dir is not None else experiment_output_dir

    logger.info(f"Computing influence scores for {analysis_name} and {query_name}")
    train_scores_save_path, inter_querry_scores_save_path, self_scores_save_path, factors_name = get_pairwise_influence_scores(
        experiment_output_dir=influence_base_dir,
        factor_fit_dataset=factor_fit_dataset,  # type: ignore
        train_dataset=train_dataset,  # type: ignore
        query_dataset=query_dataset,  # type: ignore
        analysis_name=analysis_name,
        factors_name_base=factors_name,
        query_name_base=query_name,
        task=task,
        damping=args.damping,
        shard_lambda=args.shard_lambda,
        shard_covariance=args.shard_covariance,
        model=model,  # type: ignore
        amp_dtype=args.amp_dtype,  # type: ignore
        gradient_dtype=args.gradient_dtype,  # type: ignore
        gradient_covariance_dtype=args.gradient_covariance_dtype,  # type: ignore
        lambda_dtype=args.lambda_dtype,  # type: ignore
        activation_covariance_dtype=args.activation_covariance_dtype,  # type: ignore
        factor_batch_size=args.factor_batch_size,
        covariance_batch_size=args.covariance_batch_size,
        lambda_batch_size=args.lambda_batch_size,
        query_batch_size=args.query_batch_size if args.query_batch_size is not None else len(query_dataset),
        self_inf_batch_size=args.self_inf_batch_size,
        train_batch_size=args.train_batch_size,
        query_gradient_rank=args.query_gradient_rank,
        query_gradient_accumulation_steps=args.query_gradient_accumulation_steps,
        profile_computations=args.profile_computations,
        compute_per_token_scores=args.compute_per_token_scores,
        use_half_precision=args.use_half_precision_influence,
        factor_strategy=args.factor_strategy,
        num_module_partitions_covariance=args.num_module_partitions_covariance,
        num_module_partitions_scores=args.num_module_partitions_scores,
        num_module_partitions_lambda=args.num_module_partitions_lambda,
        compute_per_module_scores=args.compute_per_module_scores,
        overwrite_output_dir=args.overwrite_output_dir,
        covariance_max_examples=args.covariance_max_examples,
        lambda_max_examples=args.lambda_max_examples,
        calculate_train_influence=args.calculate_train_influence,
        calculate_self_influence=args.calculate_self_influence,
        calculate_inter_query_influence=args.calculate_inter_query_influence,
        cached_factors_name=args.cached_factors_name,
    )

    if process_rank == 0:
        # Create the symlinks with relative paths
        if train_scores_save_path is not None:
            (experiment_output_dir / "scores").symlink_to(train_scores_save_path)
            log().add_to_log_dict(train_scores_save_path=train_scores_save_path)
        if inter_querry_scores_save_path is not None:
            (experiment_output_dir / "inter_query_scores").symlink_to(inter_querry_scores_save_path)
            log().add_to_log_dict(inter_querry_scores_save_path=inter_querry_scores_save_path)
        if self_scores_save_path is not None:
            (experiment_output_dir / "self_scores").symlink_to(self_scores_save_path)
            log().add_to_log_dict(self_scores_save_path=self_scores_save_path)        
        log().add_to_log_dict(factors_name=factors_name)




def into_document_span_ds(packed_ds: Dataset) -> Dataset:
    def explode(batch: dict[str, Any], indices: list[int]) -> dict[str, list[Any]]:
        rows = []
        for i, packed_idx in enumerate(indices):
            packed_id = batch["id"][i]
            for doc in batch["packed_documents"][i]:
                if doc["span_start"] == doc["span_end"]:
                    # Old packing code had a bug where it would sometimes pack a length 0 span
                    continue

                doc_id = doc["id"]

                # Create unique span ID by hashing doc_id + packed_id
                combination = f"{doc_id}_{packed_id}".encode("utf-8")
                unique_span_id = hashlib.sha256(combination).hexdigest()

                row = doc | {
                    "id": unique_span_id,
                    "doc_id": doc_id,
                    "packed_idx": packed_idx,
                    "packed_id": packed_id,
                }

                # Extract input_ids for this segment using span information
                row["input_ids"] = batch["input_ids"][i]
                row["labels"] = [(label if doc['span_start'] <= idx < doc['span_end'] else -100)
                                 for idx, label in enumerate(batch["labels"][i])]
                row["attention_mask"] = batch["attention_mask"][i]

                rows.append(row)

        # Change from records to dict of lists of the same length
        out = defaultdict(list)
        for r in rows:
            for k, v in r.items():
                out[k].append(v)

        return out

    seg_ds = packed_ds.map( # type: ignore
        explode,
        with_indices=True,
        batched=True,
        batch_size=len(packed_ds),
        remove_columns=packed_ds.column_names, # type: ignore
    )

    return seg_ds


@torch.no_grad()
def compute_and_save_logprobs(
    model: PreTrainedModel,
    query_dataset: Dataset,
    train_dataset: Dataset,
    experiment_output_dir: Path,
    batch_size: int = 32,
    device: str | None = None,
    temperature: float = 1.0,
) -> tuple[Path, Path]:
    """Compute and save log probabilities for query and training datasets.

    Uses calculate_logprobs() from shared_ml/eval.py which returns summed
    log probability per example (single float, not per-token).

    Args:
        model: The pretrained language model (must NOT be wrapped in FSDP)
        query_dataset: Dataset of query examples
        train_dataset: Dataset of training examples (unpacked spans)
        experiment_output_dir: Directory to save results
        batch_size: Batch size for processing
        device: Device to use (defaults to cuda if available)
        temperature: Temperature for logit scaling (default 1.0)

    Returns:
        Tuple of (query_logprobs_path, train_logprobs_path)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    def compute_logprobs_for_dataset(dataset: Dataset, name: str) -> torch.Tensor:
        """Compute log probabilities for all examples in a dataset."""
        dataloader = DataLoader(
            dataset=dataset,  # type: ignore
            batch_size=batch_size,
            collate_fn=collator_list_to_tensor(),
            shuffle=False,
        )

        all_logprobs = []
        for batch in tqdm(dataloader, desc=f"Computing {name} logprobs"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logprobs = calculate_logprobs(outputs.logits, labels, temperature=temperature)
            all_logprobs.append(logprobs.cpu())

        return torch.cat(all_logprobs, dim=0)

    # Compute log probabilities for both datasets
    query_logprobs = compute_logprobs_for_dataset(query_dataset, "query")
    train_logprobs = compute_logprobs_for_dataset(train_dataset, "train")

    # Create output directory
    logprobs_dir = experiment_output_dir / "logprobs"
    logprobs_dir.mkdir(exist_ok=True)

    # Extract IDs for metadata
    query_ids = list(query_dataset["id"])
    train_ids = list(train_dataset["id"])

    # Save query log probabilities
    query_logprobs_path = logprobs_dir / "query_logprobs.safetensors"
    save_file(
        tensors={"logprobs": query_logprobs.float()},
        filename=query_logprobs_path,
        metadata={
            "dataset_type": "query",
            "num_examples": str(len(query_logprobs)),
        },
    )

    # Save training log probabilities
    train_logprobs_path = logprobs_dir / "train_logprobs.safetensors"
    save_file(
        tensors={"logprobs": train_logprobs.float()},
        filename=train_logprobs_path,
        metadata={
            "dataset_type": "train",
            "num_examples": str(len(train_logprobs)),
        },
    )

    # Save the ID mappings as a separate file for easy lookup
    id_mapping_path = logprobs_dir / "id_mapping.json"
    with open(id_mapping_path, "w") as f:
        json.dump({
            "query_ids": query_ids,
            "train_ids": train_ids,
        }, f)

    logger.info(f"Saved query logprobs ({len(query_logprobs)} examples) to {query_logprobs_path}")
    logger.info(f"Saved train logprobs ({len(train_logprobs)} examples) to {train_logprobs_path}")

    return query_logprobs_path, train_logprobs_path


def setup_logging(args: InfluenceArgs) -> Path:
    experiment_name = get_experiment_name(args)

    experiment_output_dir = Path(args.output_dir) / experiment_name

    experiment_output_dir.mkdir(parents=True, exist_ok=True)
    setup_custom_logging(
        experiment_name=experiment_name,
        experiment_output_dir=experiment_output_dir,
        logging_type=args.logging_type,
        wandb_project=args.wandb_project,
        only_initialize_on_main_process=True,
    )

    log().state.args = args.model_dump()
    log().write_out_log()

    log_message = f"Logging setup! Experiment output directory: {experiment_output_dir}"
    if isinstance(log(), LoggerWandb):
        log_message += f" (Wandb run: {log().wandb.url})"  # type: ignore
    logger.info(log_message)

    return experiment_output_dir


def get_task(
    model: PreTrainedModel,
    modules_to_track: Literal["attn", "mlp", "all"],
    task_type: Literal['ce', 'softmargin', 'softmargin_training', 'logit', 'logit_training'],
    layers_to_track: tuple[int, int] | None = None,
    temperature: float = 1.0,
) -> Task:
    assert (
        isinstance(model, OlmoForCausalLM)
        or isinstance(model, Olmo2ForCausalLM)
        or isinstance(model, Qwen3ForCausalLM)
    ), (
        "Other models are not supported yet, as unsure how to correctly get their tracked modules. Feel free to add support for them, by editing the code below."
    )

    if modules_to_track == "attn":
        module_regex = r".*attn\..*_(proj|fc|attn)"
    elif modules_to_track == "mlp":
        module_regex = r".*mlp\..*_(proj|fc|attn)"
    elif modules_to_track == "all":
        module_regex = r".*(attn|mlp)\..*_(proj|fc|attn)"
    else:
        raise ValueError(f"Invalid layers_to_track: {modules_to_track}")

    def matches_layer_range(name: str, layer_range: tuple[int, int] | None) -> bool:
        """Check if a module name falls within the specified layer range."""
        if layer_range is None:
            return True
        # Extract layer number from module name (e.g., "model.layers.5.mlp.gate_proj" -> 5)
        layer_match = re.search(r"\.layers\.(\d+)\.", name)
        if layer_match is None:
            return True  # Non-layer modules (e.g., embeddings) are always included
        layer_num = int(layer_match.group(1))
        return layer_range[0] <= layer_num <= layer_range[1]

    tracked_modules: list[str] = [
        name
        for name, _ in model.named_modules()
        if re.match(module_regex, name) and matches_layer_range(name, layers_to_track)
    ]
    match task_type:
        case 'logit':
            return LanguageModelingTaskLogit(tracked_modules=tracked_modules, temperature=temperature)
        case 'logit_training':
            return LanguageModelingTaskTrainingLogit(tracked_modules=tracked_modules, temperature=temperature)
        case 'softmargin':
            return LanguageModelingTaskMargin(tracked_modules=tracked_modules, temperature=temperature)
        case 'softmargin_training':
            return LanguageModelingTaskTrainingMargin(tracked_modules=tracked_modules, temperature=temperature)
        case 'ce':
            return LanguageModelingTask(tracked_modules=tracked_modules, temperature=temperature)


def get_datasets(args: InfluenceArgs) -> tuple[Dataset, Dataset, list[tuple[str, Dataset]]]:
    # Load training dataset
    if args.train_dataset_path is not None:
        train_dataset = load_from_disk(args.train_dataset_path)
    elif args.target_experiment_dir is not None:
        train_dataset = load_experiment_checkpoint(
            experiment_output_dir=args.target_experiment_dir,
            checkpoint_name=None,
            load_model=False,
            load_tokenizer=False,
        ).train_dataset
    else:
        raise ValueError("Either train_dataset_path or target_experiment_dir must be provided")

    # Load query datasets
    if args.query_dataset_path is not None:
        query_datasets = [(f"{args.query_dataset_path}", load_from_disk(args.query_dataset_path))]
    elif args.target_experiment_dir is not None and args.query_dataset_split_names:
        checkpoint = load_experiment_checkpoint(
            experiment_output_dir=args.target_experiment_dir,
            checkpoint_name=None,
            load_model=False,
            load_tokenizer=False,
        )
        assert checkpoint.test_datasets is not None
        query_datasets = [
            (k, checkpoint.test_datasets[k].dataset) for k in args.query_dataset_split_names
        ]  # Use a list instead of a dict as the order of the datasets is important for reconstructing them later
    else:
        raise ValueError("Either query_dataset_path or (target_experiment_dir + query_dataset_split_names) must be provided")

    # Load additional query datasets from external paths
    if args.additional_query_dataset_paths:
        from shared_ml.eval import EvalDataset
        for ds_path in args.additional_query_dataset_paths:
            ds_name = Path(ds_path).name
            eval_ds = EvalDataset.load(Path(ds_path))
            query_datasets.append((ds_name, eval_ds.dataset))
            logger.info(f"Loaded additional query dataset: {ds_name} ({len(eval_ds.dataset)} items)")

    # Load factor fit dataset
    if args.factor_fit_dataset_path is not None:
        factor_fit_dataset = load_from_disk(args.factor_fit_dataset_path)
    else:
        factor_fit_dataset = train_dataset

    return factor_fit_dataset, train_dataset, query_datasets  # type: ignore


def get_experiment_name(args: InfluenceArgs) -> str:
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    return f"{time.strftime('%Y_%m_%d_%H-%M-%S')}_{random_id}_run_influence_{args.factor_strategy}_{args.experiment_name}_checkpoint_{args.checkpoint_name}_query_gradient_rank_{args.query_gradient_rank}"


def get_model_and_tokenizer(
    args: InfluenceArgs,
) -> tuple[GPT2LMHeadModel, PreTrainedTokenizer]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = "cuda" if torch.cuda.is_available() else "cpu"

    if args.hf_model_name is not None:
        # Load directly from HuggingFace
        logger.info(f"Loading model from HuggingFace: {args.hf_model_name} @ {args.hf_revision}")
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model_name,
            revision=args.hf_revision,
            torch_dtype=args.dtype_model,
            device_map=device_map,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.hf_model_name,
            revision=args.hf_revision,
        )
        return model, tokenizer  # type: ignore
    else:
        # Load from local experiment directory
        checkpoint = load_experiment_checkpoint(
            experiment_output_dir=args.target_experiment_dir,
            checkpoint_name=args.checkpoint_name,
            model_kwargs={
                "device_map": device_map,
                "torch_dtype": args.dtype_model,
                "attn_implementation": "flash_attention_2" if args.use_flash_attn else "sdpa",
            },
        )
        return checkpoint.model, checkpoint.tokenizer  # type: ignore


def get_analysis_factor_query_name(
    args: InfluenceArgs,
) -> tuple[str, str, str]:
    analysis_name = f"checkpoint_{hash_str(str(args.checkpoint_name))[:4]}_modules_{args.modules_to_track}_{args.task_type}"
    if args.layers_to_track:
        analysis_name += f"_layers_{args.layers_to_track[0]}_{args.layers_to_track[1]}"
    if args.freeze_attn:
        analysis_name += "frozen_attn"
    factors_name = args.factor_name_extra if args.factor_name_extra is not None else "factor"
    query_name = args.query_name_extra if args.query_name_extra is not None else "query"

    return analysis_name, factors_name, query_name



def _load_influence_common_setup(
    experiment_output_dir: Path | str,
    allow_mismatched_arg_keys: bool = False,
) -> tuple[Path, InfluenceArgs, Dataset | None, list[tuple[str, Dataset]]]:
    """Common setup for loading influence scores.

    Args:
        experiment_output_dir: The path to the experiment output directory.
        allow_mismatched_arg_keys: Whether to allow mismatched argument keys when loading the InfluenceArgs objects.

    Returns:
        tuple containing:
            - experiment_output_dir: Resolved Path
            - args: InfluenceArgs object
            - unpacked_dataset: The unpacked training dataset
            - query_datasets: List of (name, dataset) tuples for query datasets
    """
    experiment_output_dir = Path(experiment_output_dir)

    checkpoint_influence_run = load_experiment_checkpoint(
        experiment_output_dir=experiment_output_dir,
        load_model=False,
        load_tokenizer=False,
        load_datasets=False,
    )

    args_dict = checkpoint_influence_run.experiment_log.args
    assert args_dict is not None

    if allow_mismatched_arg_keys:
        args_dict = {k: v for k, v in args_dict.items() if k in InfluenceArgs.model_fields}

    args = InfluenceArgs.model_validate(args_dict)  # type: ignore
    checkpoint_training_run = load_experiment_checkpoint(
        args.target_experiment_dir, checkpoint_name=None, load_model=False, load_tokenizer=False
    )

    if 'unpacked_dataset_path' in checkpoint_influence_run.experiment_log.log_dict:
        unpacked_ds_path = checkpoint_influence_run.experiment_log.log_dict['unpacked_dataset_path']
        unpacked_dataset = load_from_disk(unpacked_ds_path)
    else:
        unpacked_dataset = None

    # Load the query datasets
    if args.query_dataset_path is not None:
        query_datasets = [(str(args.query_dataset_path), load_from_disk(args.query_dataset_path))]
    else:
        assert checkpoint_training_run.test_datasets is not None
        query_datasets = [(k, checkpoint_training_run.test_datasets[k].dataset) for k in args.query_dataset_split_names]

    # Load additional query datasets from external paths
    if args.additional_query_dataset_paths:
        from shared_ml.eval import EvalDataset
        for ds_path in args.additional_query_dataset_paths:
            ds_name = Path(ds_path).name
            eval_ds = EvalDataset.load(Path(ds_path))
            query_datasets.append((ds_name, eval_ds.dataset))

    return experiment_output_dir, args, unpacked_dataset, query_datasets # type: ignore


def load_influence_scores(
    experiment_output_dir: Path | str,
    allow_mismatched_arg_keys: bool = False,
) -> dict[str, DataFrame]:
    """Loads influence scores from the experiment output directory.

    Args:
        experiment_output_dir: The path to the experiment output directory. This is an experiment from the run_influence script, not a training run.
        allow_mismatched_arg_keys: Whether to allow mismatched argument keys when loading the InfluenceArgs objects. This can happen if loading an old run where the InfluenceArgs interface was changed.

    Returns:
        dict[str, DataFrame]: A dictionary of query dataset names to their influence scores dataframe.
    """
    experiment_output_dir, _, unpacked_dataset, query_datasets = _load_influence_common_setup(
        experiment_output_dir, allow_mismatched_arg_keys
    )
    assert unpacked_dataset is not None

    path_to_scores = experiment_output_dir / "scores"
    scores_dict = load_pairwise_scores(path_to_scores)
    influence_scores = scores_dict["all_modules"].to(dtype=torch.float32).cpu().numpy()

    train_ids = list(unpacked_dataset["id"])

    # De-concatenate the scores into a dataframe per query dataset
    scores_per_query: dict[str, DataFrame] = {}
    scores_df_idx = 0
    for query_dataset_name, query_dataset in tqdm(query_datasets, desc="loading queries"):
        query_dataset_length = len(query_dataset)

        # The query datasets were concatenated together, so we need to split them back up
        assert len(query_dataset) == query_dataset_length, "Query dataset length mismatch between saved and loaded"

        scores_for_this_query_dataset = influence_scores[scores_df_idx : scores_df_idx + query_dataset_length, :]

        query_ids = list(query_dataset["id"])

        records = []
        for q_idx, qid in enumerate(query_ids):
            for t_idx, tid in enumerate(train_ids):
                records.append({
                    "query_id": qid,
                    "span_id": tid,
                    "per_token_influence_score": scores_for_this_query_dataset[q_idx, t_idx],
                })
        scores_per_query[query_dataset_name] = DataFrame.from_records(records)

        scores_df_idx += query_dataset_length

    return scores_per_query


def load_inter_query_influence_scores(
    experiment_output_dir: Path | str,
    allow_mismatched_arg_keys: bool = False,
) -> dict[str, DataFrame]:
    """Loads inter-query influence scores from the experiment output directory.

    Args:
        experiment_output_dir: The path to the experiment output directory. This is an experiment from the run_influence script, not a training run.
        allow_mismatched_arg_keys: Whether to allow mismatched argument keys when loading the InfluenceArgs objects. This can happen if loading an old run where the InfluenceArgs interface was changed.

    Returns:
        dict[str, DataFrame]: A dictionary of query dataset names to their inter-query influence scores dataframe.
    """
    experiment_output_dir, _, _, query_datasets = _load_influence_common_setup(
        experiment_output_dir, allow_mismatched_arg_keys
    )

    path_to_inter_q_scores = experiment_output_dir / "inter_query_scores"
    scores_dict = load_pairwise_scores(path_to_inter_q_scores)
    influence_inter_q_scores = scores_dict["all_modules"].to(dtype=torch.float32).cpu().numpy()

    query_ids_all = [id for _, q_ds in query_datasets for id in q_ds["id"]]

    # De-concatenate the scores into a dataframe per query dataset
    inter_q_scores_per_query: dict[str, DataFrame] = {}
    scores_df_idx = 0
    for query_dataset_name, query_dataset in tqdm(query_datasets, desc="loading inter-query scores"):
        query_dataset_length = len(query_dataset)

        # The query datasets were concatenated together, so we need to split them back up
        assert len(query_dataset) == query_dataset_length, "Query dataset length mismatch between saved and loaded"

        inter_q_scores_for_this_dataset = influence_inter_q_scores[
            scores_df_idx : scores_df_idx + query_dataset_length, :
        ]

        query_ids = list(query_dataset["id"])

        records = []
        for q_idx, qid in enumerate(query_ids):
            for t_idx, tid in enumerate(query_ids_all):
                records.append({
                    "query_id": qid,
                    "train_id": tid,
                    "per_token_influence_score": inter_q_scores_for_this_dataset[q_idx, t_idx],
                })

        inter_q_scores_per_query[query_dataset_name] = DataFrame.from_records(records)

        scores_df_idx += query_dataset_length

    return inter_q_scores_per_query


def load_self_influence_scores(
        experiment_output_dir: Path,
        allow_mismatched_arg_keys: bool = False,
    ) -> DataFrame:
    experiment_output_dir, _, unpacked_dataset, _ = _load_influence_common_setup(
        experiment_output_dir,
        allow_mismatched_arg_keys=allow_mismatched_arg_keys,
    )
    path_to_self_scores = experiment_output_dir / "self_scores"
    self_scores = load_self_scores(path_to_self_scores)
    assert unpacked_dataset is not None

    train_ids = unpacked_dataset['id']

    records = []
    for t_idx, t_id in enumerate(train_ids):
        records.append({
            "span_id": t_id,
            "self_inf_score": self_scores['all_modules'][t_idx]
        })
    return DataFrame.from_records(records)


def load_logprobs(
    experiment_output_dir: Path | str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[str]]]:
    """Load log probabilities from an influence experiment.

    Args:
        experiment_output_dir: Path to the influence experiment output

    Returns:
        Tuple of (query_logprobs, train_logprobs, id_mapping)
        where id_mapping contains 'query_ids' and 'train_ids' lists
    """
    experiment_output_dir = Path(experiment_output_dir)
    logprobs_dir = experiment_output_dir / "logprobs"

    query_logprobs = load_file(logprobs_dir / "query_logprobs.safetensors")["logprobs"]
    train_logprobs = load_file(logprobs_dir / "train_logprobs.safetensors")["logprobs"]

    with open(logprobs_dir / "id_mapping.json", "r") as f:
        id_mapping = json.load(f)

    return query_logprobs, train_logprobs, id_mapping


class AttentionWithStopGrad(Olmo2Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            assert self.layer_idx is not None, "layer_idx must be set for caching"
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable[..., Any] = eager_attention_forward
        if self.config._attn_implementation != "eager":  # type: ignore[reportPrivateUsage]
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]  # type: ignore[reportPrivateUsage]

        attn_output, attn_weights = attention_interface(
            self,
            query_states.detach(),
            key_states.detach(),
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def replace_attention_with_frozen(model: Olmo2PreTrainedModel):
    """
    Replace all Olmo2Attention modules in the model with AttentionWithStopGrad.

    Args:
        model: The model (e.g., Olmo2ForCausalLM) containing Olmo2Attention modules

    Returns:
        The modified model with frozen attention mechanisms
    """
    assert isinstance(model, Olmo2PreTrainedModel)
    for name, module in model.named_modules():
        # Get parent module and attribute name
        if isinstance(module, Olmo2Attention) and not isinstance(module, AttentionWithStopGrad):
            # Parse the module path to get parent and attribute name
            path_parts = name.split(".")
            parent = model
            for part in path_parts[:-1]:
                parent = getattr(parent, part)

            attr_name = path_parts[-1]

            # Create new frozen attention module with same config
            frozen_attn = AttentionWithStopGrad(config=module.config, layer_idx=module.layer_idx)

            # Copy weights from original module
            frozen_attn.load_state_dict(module.state_dict())

            # Replace the module
            setattr(parent, attr_name, frozen_attn)

    return model


if __name__ == "__main__":
    args = CliApp.run(InfluenceArgs)  # Parse the arguments, returns a TrainingArgs object

    try:
        main(args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
