# %%

import gc
import os
import random
import re
import string
import time
from pathlib import Path
from typing import Any, Literal, cast
import json

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from datasets import Dataset
from shared_ml.data import hash_record, tokenize
from shared_ml.eval import EvalModelBeamSearch, eval_accuracy_and_loss
from shared_ml.utils import hash_str

# Import reanalysis utilities
from oocr_influence.reanalysis_utilities import (
    CheckpointInfo,
    FinetuningCheckpoint,
    PretrainingCheckpoint,
    load_checkpoint_info,
    load_all_checkpoint_influences,
    create_influence_summary_df,
    create_per_module_influence_df,
    extract_step_number,
    extract_token_number,
    extract_layer_number,
    get_module_type,
)

# %%
os.chdir('/home/dev/projects/malign-influence')

# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================
# Choose your data source: "hessians" or "influence_run"
SOURCE_TYPE: Literal["hessians", "influence_run"] = "hessians"

# Path to data source
# For hessians: path to hessians directory (e.g., hessians/olmo-2-7b)
# For influence_run: path to influence run output directory
SOURCE_PATH = Path("/home/dev/projects/malign-influence/hessians/olmo-2-7b")# Path("/home/dev/projects/malign-influence/outputs/2025_12_28_22-00-20_0Gig3_reversal_curse_30_per_fact_1000_pretrain_ce_rephasings_v2_all_checkpoints")

# Example for influence run:
# SOURCE_PATH = Path("/home/dev/experiments/malign-influence/outputs/2025_12_24_19-24-44_Nv5lG_reversal_curse_30_per_fact_1000_pretrain_ce_rephasings_v2_all_checkpoints")

MODEL_NAME = "allenai/OLMo-2-1124-7B"


# Prompt-completion pairs to evaluate
with open('analysis/prompt_completion_pairs_negation.json') as f:
    PROMPT_COMPLETION_PAIRS: list[dict[str, str]] = json.load(f)

# Beam search parameters (replacing temperature sampling)
NUM_BEAMS = 12
NUM_RETURN_SEQUENCES = 10
EVAL_BATCH_SIZE = 4

# Model loading parameters
DTYPE = torch.bfloat16  # Use bfloat16 for memory efficiency
DEVICE = "cpu"

# Note: Factor fit dataset is now loaded from HESSIANS_BASE_DIR / "factor_fit_dataset"
def generate_output_dir(prompt_completion_pairs: list[dict[str, str]], base_dir: str = "./outputs") -> Path:
    """Generate output directory path with timestamp and hash of prompt-completion pairs."""
    timestamp = time.strftime('%Y_%m_%d_%H-%M-%S')
    random_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    pairs_hash = hash_str(str(prompt_completion_pairs))[:8]
    return Path(base_dir) / f"{timestamp}_{random_id}_influence_analysis_{pairs_hash}"

INFLUENCE_OUTPUT_DIR: Path = generate_output_dir(PROMPT_COMPLETION_PAIRS)
INFLUENCE_EXPERIMENT_NAME: str = "checkpoint_influence_analysis"
SWEEP_NAME: str = "checkpoint-influence-sweep"

# %%
# Load checkpoint info using unified interface
# This supports both hessians directory and previous influence runs

CHECKPOINT_INFO = load_checkpoint_info(SOURCE_PATH, SOURCE_TYPE)
CHECKPOINT_REVISIONS = [info.revision for info in CHECKPOINT_INFO]

def load_tokenizer(
    model_name: str = MODEL_NAME,
) -> PreTrainedTokenizerBase:
    """
    Load tokenizer from HuggingFace or local checkpoint.

    For local checkpoints, loads from the first checkpoint's local path.
    For HuggingFace checkpoints, loads from the model name.

    Args:
        checkpoint_info_list: List of CheckpointInfo objects (to detect local checkpoints)
        model_name: HuggingFace model name (fallback for HF checkpoints)
    """
    # Load from HuggingFace
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision="main",
        trust_remote_code=False,
    )
    print("Tokenizer loaded")
    return tokenizer


# Load tokenizer once (after CHECKPOINT_INFO is available)
TOKENIZER = load_tokenizer(MODEL_NAME)


def load_checkpoint(
    checkpoint_info: FinetuningCheckpoint | PretrainingCheckpoint,
    model_name: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> PreTrainedModel:
    """
    Load model from HuggingFace or local checkpoint.

    Args:
        checkpoint_info: Checkpoint info (FinetuningCheckpoint or PretrainingCheckpoint)
        model_name: Optional HuggingFace model name (used for HF checkpoints if not in checkpoint_info)
        dtype: Model dtype for memory efficiency
        device: Device to load model on

    Returns:
        The loaded model
    """
    if isinstance(checkpoint_info, FinetuningCheckpoint):
        # Load from local checkpoint path
        local_path = checkpoint_info.local_checkpoint_path
        print(f"Loading local checkpoint from {local_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            str(local_path),
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=False,
        )
    else:
        # Load from HuggingFace (PretrainingCheckpoint)
        hf_model_name = checkpoint_info.hf_model_name or model_name
        if hf_model_name is None:
            raise ValueError(f"No HuggingFace model name for {checkpoint_info.revision}")
        hf_revision = checkpoint_info.hf_revision or checkpoint_info.revision
        print(f"Loading {hf_model_name} @ {hf_revision}...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            revision=hf_revision,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=False,
        )

    model.eval()
    print(f"Model loaded on {model.device}")
    return model


def unload_checkpoint(model: PreTrainedModel) -> None:
    """
    Delete model and clear CUDA cache to free memory.

    Args:
        model: Model to unload
    """
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("Model unloaded, CUDA cache cleared")


def create_query_dataset(
    prompt_completion_pairs: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """
    Convert prompt-completion pairs to a tokenized HuggingFace Dataset.

    Uses the shared_ml.data.tokenize utility to ensure consistent tokenization
    with the rest of the influence pipeline.

    Args:
        prompt_completion_pairs: List of dicts with 'prompt', 'completion', and optionally 'pair_id'
        tokenizer: The tokenizer to use

    Returns:
        Tokenized HuggingFace Dataset with input_ids, labels, attention_mask, and id columns
    """
    records = [
        {
            "id": pair.get("pair_id", hash_record(pair)),
            "prompt": pair["prompt"],
            "completion": pair["completion"],
        }
        for pair in prompt_completion_pairs
    ]

    dataset = Dataset.from_list(records)
    dataset = dataset.map(
        lambda x: tokenize(x, tokenizer, mask_out_prompt=True, padding_side='left', add_eos_token_at_end=False, max_length=64),
        desc="Tokenizing query pairs",
    )
    return dataset

QUERY_DATASET = create_query_dataset(PROMPT_COMPLETION_PAIRS, TOKENIZER)

# %%

def run_analysis(
    model_name: str = MODEL_NAME,
    checkpoint_info_list: list[CheckpointInfo] = CHECKPOINT_INFO,
    eval_dataset: Dataset = QUERY_DATASET,
    num_beams: int = NUM_BEAMS,
    num_return_sequences: int = NUM_RETURN_SEQUENCES,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    dtype: torch.dtype = DTYPE,
    device: str = DEVICE,
    tokenizer: PreTrainedTokenizerBase = TOKENIZER,
) -> pd.DataFrame:
    """
    Run the full analysis across all checkpoints and prompt-completion pairs.

    Uses eval_accuracy_and_loss for metrics and EvalModelBeamSearch for generation.

    Returns:
        DataFrame with columns: pair_id, prompt, completion, checkpoint_revision,
        loss, accuracy, logprob, prob, softmargin, output_0, transition_score_0, ...
    """
    results: list[dict[str, Any]] = []

    # Create beam search evaluator
    beam_search_eval = EvalModelBeamSearch(
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )

    # Create tokenized eval dataset

    for ckpt_info in tqdm(checkpoint_info_list, desc="Checkpoints"):
        assert isinstance(ckpt_info, (FinetuningCheckpoint, PretrainingCheckpoint)), "unkown type"
        model = load_checkpoint(ckpt_info, model_name, dtype, device)
        revision = ckpt_info.revision

        # Compute loss and accuracy metrics using eval_accuracy_and_loss
        loss_results = eval_accuracy_and_loss(
            model=model,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            batch_size=eval_batch_size,
            device=device,
            metadata_columns=['input_ids', 'labels', 'prompt', 'completion', 'id']
        )

        # Run beam search evaluation
        beam_results = beam_search_eval(
            model=model,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            batch_size=eval_batch_size,
            device=device,
        )
        responses_df = beam_results["responses_dataset"]

        # Merge loss metrics with beam search results
        for idx, record in enumerate(loss_results["records"]):
            beam_row = responses_df.iloc[idx]

            row: dict[str, Any] = {
                "checkpoint_revision": revision,
                **record
            }

            # Add beam search outputs and scores
            for i in range(num_return_sequences):
                output_col = f"output_{i}"
                score_col = f"transition_score_{i}"
                if output_col in beam_row:
                    row[output_col] = beam_row[output_col]
                    row[score_col] = beam_row[score_col]

            results.append(row)

        # Free memory before loading next checkpoint
        unload_checkpoint(model)

    return pd.DataFrame(results)

losses_df = run_analysis(checkpoint_info_list=CHECKPOINT_INFO, device=DEVICE, eval_dataset=QUERY_DATASET)
losses_df


# %% Run influences with cached hessians
def save_query_dataset(dataset: Dataset, output_dir: Path, name: str) -> Path:
    """
    Save tokenized dataset to disk with a distinctive name.

    Args:
        dataset: The tokenized dataset to save
        output_dir: Directory to save the dataset in
        name: Name for the dataset (will be prefixed with 'query_dataset_')

    Returns:
        Path to the saved dataset
    """
    dataset_path = output_dir / f"query_dataset_{name}"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(dataset_path)
    print(f"Query dataset saved to: {dataset_path}")
    return dataset_path


def build_influence_args_with_cached_hessians(
    query_dataset_path: Path,
    model_name: str,
    checkpoint_info: FinetuningCheckpoint | PretrainingCheckpoint,
    train_dataset_path: Path,
    output_dir: Path,
    experiment_name: str,
):
    """
    Build InfluenceArgs for running influence on a checkpoint using cached hessians.

    Args:
        query_dataset_path: Path to the saved query dataset
        model_name: HuggingFace model name (e.g., "allenai/OLMo-2-1124-7B")
        checkpoint_info: Checkpoint info (FinetuningCheckpoint or PretrainingCheckpoint)
        train_dataset_path: Path to training dataset for influence scoring
        output_dir: Directory for influence outputs
        experiment_name: Name for this influence run

    Returns:
        InfluenceArgs configured for this checkpoint with cached factors
    """
    from oocr_influence.cli.run_influence import InfluenceArgs

    revision = checkpoint_info.revision
    factors_name = checkpoint_info.factors_name

    # Determine paths based on checkpoint type
    if isinstance(checkpoint_info, PretrainingCheckpoint):
        hessians_path = checkpoint_info.hessians_path
        # The factor_fit_dataset is stored in the hessians directory
        factor_fit_dataset_path = SOURCE_PATH / "factor_fit_dataset" # TODO(lev): this is techically unessiary
        hf_model_name = model_name
        hf_revision = revision
        target_experiment_dir = hessians_path
        checkpoint_name = "hf_checkpoint"
    else:  # FinetuningCheckpoint
        # For finetuning checkpoints, factors are in target_experiment_dir
        target_experiment_dir = checkpoint_info.target_experiment_dir
        checkpoint_name = checkpoint_info.checkpoint_name
        factor_fit_dataset_path = None  # Will use from target experiment
        hf_model_name = None
        hf_revision = None

    return InfluenceArgs(
        hf_model_name=hf_model_name,
        hf_revision=hf_revision,
        query_dataset_path=query_dataset_path,
        query_dataset_split_names=[],
        factor_fit_dataset_path=factor_fit_dataset_path,
        train_dataset_path=str(train_dataset_path),
        experiment_name=experiment_name,
        output_dir=output_dir,
        checkpoint_name=checkpoint_name,
        # Use cached factors - this is the key change!
        cached_factors_name=factors_name,
        # The target_experiment_dir points to where the cached factors are stored
        target_experiment_dir=target_experiment_dir,
        # Influence computation settings (same as before, but factor fitting will be skipped)
        task_type="ce",
        query_gradient_rank=64,
        covariance_batch_size=1,
        compute_per_module_scores=True,
        lambda_batch_size=1,
        freeze_attn=False,
        query_batch_size=4,
        covariance_and_lambda_max_examples=3000,
        self_inf_batch_size=2,
        unpack_dataset=False,
        query_gradient_accumulation_steps=4,
        train_batch_size=1,
        shard_covariance=True,
        shard_lambda=True,
        compute_per_token_scores=True,
    )


query_dataset_path = save_query_dataset(QUERY_DATASET, INFLUENCE_OUTPUT_DIR, INFLUENCE_EXPERIMENT_NAME)

from oocr_influence.cli.run_influence import InfluenceArgs
from oocr_influence.cli.run_influence import main as run_influence_main

influence_args_list: list[InfluenceArgs] = []
for ckpt_info in CHECKPOINT_INFO:
    influence_arg = build_influence_args_with_cached_hessians(
        query_dataset_path=query_dataset_path.absolute(),
        model_name=MODEL_NAME,
        checkpoint_info=ckpt_info,
        train_dataset_path=query_dataset_path.absolute(),
        output_dir=INFLUENCE_OUTPUT_DIR.absolute(),
        experiment_name=f"{INFLUENCE_EXPERIMENT_NAME}_{ckpt_info.index}_{ckpt_info.revision}",
    )
    influence_args_list.append(influence_arg)

print(f"Built {len(influence_args_list)} influence args for checkpoints with cached hessians")

from launcher import KubernetesSweepOrchestrator, ResourceRequest
from launcher.jobs import create_job_array_from_sweep
from launcher.kubernetes_orchestrator import KubernetesConfig

resource_request = ResourceRequest(
    cpu=32.0,
    memory=64.0,
    gpu=2,
    parallel_jobs=1,
    use_torch_distributed=True,
)

kubernetes_config = KubernetesConfig(
    priority_class="high-batch",
    project_pvc="lev-colab",
    parallel_workers=1,
)

orchestrator = KubernetesSweepOrchestrator(kubernetes_config)

job_array = create_job_array_from_sweep(
    target_args_model=InfluenceArgs,
    target_entrypoint=cast(Any, run_influence_main),
    arguments=influence_args_list,
    resource_request=resource_request,
    sweep_id=SWEEP_NAME,
)


orchestrator.run_sweep(job_array, resource_request, sweep_name=SWEEP_NAME)

# %% Load influences for analysis
import numpy as np


# Load influence scores from INFLUENCE_OUTPUT_DIR where the new runs were saved
# This works for both source types - scores are always in the output directory
checkpoint_influences = load_all_checkpoint_influences(
    CHECKPOINT_INFO,
    QUERY_DATASET,
    output_dir=INFLUENCE_OUTPUT_DIR,
    experiment_name_prefix=INFLUENCE_EXPERIMENT_NAME,
)

# Ensure we have just the dict (not the tuple with per-module scores)
if isinstance(checkpoint_influences, tuple):
    checkpoint_influences = checkpoint_influences[0]

print(f"\nLoaded influences for {len(checkpoint_influences)} checkpoints")

influence_df = create_influence_summary_df(checkpoint_influences, CHECKPOINT_INFO)
print(f"\nInfluence DataFrame shape: {influence_df.shape}")
print(f"Unique pair_ids: {influence_df['pair_id'].nunique()}")
print(f"Unique checkpoints: {influence_df['checkpoint_revision'].nunique()}")

# %%

self_influence_df = influence_df.loc[influence_df['pair_id'] == influence_df['span_id']][['checkpoint_index', 'pair_id', 'span_id', 'sum_of_influence']]
norm_influence_df = influence_df.merge(self_influence_df[['span_id', 'checkpoint_index', 'sum_of_influence']].rename(columns={'sum_of_influence': 'train_self_inf'}), on=('span_id', 'checkpoint_index'))
norm_influence_df = norm_influence_df.merge(self_influence_df[['pair_id', 'checkpoint_index', 'sum_of_influence']].rename(columns={'sum_of_influence': 'query_self_inf'}), on=('pair_id', 'checkpoint_index'))
norm_influence_df['normalizing_constant'] = (np.sqrt(norm_influence_df['train_self_inf']) * np.sqrt(norm_influence_df['query_self_inf']))
norm_influence_df = norm_influence_df.assign(
    sum_of_influence=norm_influence_df['sum_of_influence'] / norm_influence_df['normalizing_constant'],
    per_token_scores=norm_influence_df.apply(lambda row: [s/row['normalizing_constant'] for s in row['per_token_scores']], axis=1)
)

# %% Plotting utilities
import matplotlib.pyplot as plt
import seaborn as sns

# Note: extract_token_number, extract_layer_number, and get_module_type
# are now imported from oocr_influence.reanalysis_utilities


def plot_pairwise_influences_across_checkpoints(
    influence_df: pd.DataFrame,
    losses_df: pd.DataFrame,
    pair_tuples: list[tuple[str, str]],
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    use_log: bool = True,
    normalized: bool = False,
    hue_order: list[str] | None = None 
) -> plt.Axes:
    """
    Plot influence for multiple (query, train) pair combinations across checkpoints.

    Each pair tuple is plotted as a separate line. The query is shown in a text box
    and the legend shows the training data as "prompt | completion".

    Args:
        influence_df: DataFrame with columns pair_id, span_id, checkpoint_index,
                      checkpoint_revision, sum_of_influence
        losses_df: DataFrame with columns id, prompt, completion (from run_analysis)
        pair_tuples: List of (query_pair_id, train_pair_id) tuples to plot.
                     All tuples should have the same query_pair_id.
        ax: Optional matplotlib axes to plot on
        figsize: Figure size if creating new figure
        title: Optional custom title

    Returns:
        The matplotlib axes with the plot
    """
    # Build id -> (prompt, completion) lookup from losses_df
    id_to_text = {
        row["id"]: (row["prompt"], row["completion"])
        for _, row in losses_df.drop_duplicates("id").iterrows()
    }

    # Get the query id (assume all tuples have the same query)
    query_id = pair_tuples[0][0]
    query_prompt, query_completion = id_to_text.get(query_id, (query_id, ""))

    # Build plot data for all pair tuples
    plot_rows = []
    for q_id, train_id in pair_tuples:
        mask = (influence_df["pair_id"] == q_id) & (influence_df["span_id"] == train_id)
        subset = influence_df[mask].copy()
        if subset.empty:
            print(f"Warning: No data for query={q_id}, train={train_id}")
            continue

        # Create label from training data prompt|completion
        train_prompt, train_completion = id_to_text.get(train_id, (train_id, ""))
        if q_id == train_id:
            subset['label'] = 'Self Influence'
        else:
            subset["label"] = f"{train_prompt.strip()}|{train_completion.strip()}"
        subset["tokens"] = subset["checkpoint_revision"].apply(extract_token_number)
        subset["step"] = subset["checkpoint_revision"].apply(extract_step_number)
        plot_rows.append(subset)

    if not plot_rows:
        raise ValueError("No data found for any of the specified pair tuples")

    plot_data = pd.concat(plot_rows, ignore_index=True)
    plot_data = plot_data.sort_values("tokens")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    id_to_label = {span_id: label for _, span_id, label in plot_data[['span_id', 'label']].drop_duplicates().itertuples()}
    for qid, _ in pair_tuples:
        id_to_label[q_id] = 'Self Influence'
    print(id_to_label)

    sns.lineplot(
        data=plot_data,
        x="step",
        y="sum_of_influence",
        hue="label",
        hue_order=[id_to_label[id] for id in hue_order],
        marker="o",
        ax=ax,
    )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Step", fontsize=12)

    if normalized:
        ax.set_ylabel("Normalized Influence Score", fontsize=12)
    else:
        ax.set_ylabel("Influence Score", fontsize=12)

    if use_log:
        ax.set_xscale("log")

    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("Influence Across Pre-Training Checkpoints", fontsize=14)

    # Add query text box
    query_text = f"Query: {query_prompt.strip()}|{query_completion.strip()}"
    ax.text(
        0.02, 0.98, query_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    ax.legend(title="Training Data", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    return ax

def plot_loss_by_checkpoint(
    df: pd.DataFrame,
    pair_ids: list[str] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    title: str = "Completion Loss by Checkpoint",
    use_log: bool = True,
    hue_order: list[str] | None = None,
) -> plt.Axes:
    """
    Plot loss by checkpoint for specified pair_ids.

    Legend shows the prompt|completion text for each pair.

    Args:
        df: DataFrame with columns id, prompt, completion, checkpoint_revision, loss
        pair_ids: List of pair_ids to plot. If None, plots all pairs.
        ax: Optional matplotlib axes to plot on
        figsize: Figure size if creating new figure
        title: Plot title

    Returns:
        The matplotlib axes with the plot
    """
    plot_df = df.copy()
    plot_df["tokens"] = plot_df["checkpoint_revision"].apply(extract_token_number)
    plot_df["step"] = plot_df["checkpoint_revision"].apply(extract_step_number)
    plot_df = plot_df.sort_values("step")

    # Filter to specified pair_ids if provided
    if pair_ids is not None:
        plot_df = plot_df[plot_df["id"].isin(pair_ids)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=plot_df,
        x="step",
        y="loss",
        hue_order=hue_order,
        hue="id",
        marker="o",
        ax=ax,
    )

    ax.set_xlabel("Training Tokens (Billions)", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title="Example", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Use log scale for x-axis since steps span orders of magnitude
    if use_log:
        ax.set_xscale("log")

    plt.tight_layout()
    return ax


def plot_per_token_influence_heatmap(
    influence_df: pd.DataFrame,
    losses_df: pd.DataFrame,
    query_id: str,
    train_id: str,
    tokenizer: PreTrainedTokenizerBase,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (14, 8),
    cmap: str = "RdBu_r",
    center: float = 0,
    normalize_rows: bool = False,
    normalize_cols: bool = False,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot a heatmap of per-token influence scores across checkpoints.

    The x-axis shows training tokens and the y-axis shows checkpoints (ordered by training step).

    Args:
        influence_df: DataFrame with columns pair_id, span_id, checkpoint_revision,
                      checkpoint_index, per_token_scores
        losses_df: DataFrame with columns id, input_ids, labels (from run_analysis)
        query_id: The query pair_id to plot
        train_id: The training span_id to plot
        tokenizer: Tokenizer to decode token labels for x-axis
        ax: Optional matplotlib axes to plot on
        figsize: Figure size if creating new figure
        cmap: Colormap for the heatmap
        center: Value to center the colormap on
        title: Optional custom title

    Returns:
        The matplotlib axes with the plot
    """
    # Filter to the specific query-train pair
    mask = (influence_df["pair_id"] == query_id) & (influence_df["span_id"] == train_id)
    subset = influence_df[mask].copy()

    if subset.empty:
        raise ValueError(f"No data found for query={query_id}, train={train_id}")

    # Sort by checkpoint index (training order)
    subset = subset.sort_values("checkpoint_index")

    # Extract per-token scores into a matrix (checkpoints x tokens)
    scores_matrix = np.array(subset["per_token_scores"].tolist())
    checkpoint_labels = [
        f"{extract_token_number(rev)}B" for rev in subset["checkpoint_revision"]
    ]

    # Get token labels by joining with losses_df on span_id = id
    train_row = losses_df[losses_df["id"] == train_id].iloc[0]
    input_ids = train_row["input_ids"]
    pad_token_id = tokenizer.pad_token_id

    pad_token_len = 0
    while (pad_token_len < len(input_ids)) and input_ids[pad_token_len] == pad_token_id:
        pad_token_len += 1

    scores_matrix = scores_matrix[:, pad_token_len:]
    input_ids = input_ids[pad_token_len:]

    if normalize_rows:
        scores_matrix_std = np.sqrt((scores_matrix ** 2).sum(axis=1, keepdims=True))
        scores_matrix /= scores_matrix_std
    
    if normalize_cols:
        scores_matrix_std = np.sqrt((scores_matrix ** 2).sum(axis=0, keepdims=True))
        scores_matrix /= scores_matrix_std


    # Decode token labels
    num_tokens = scores_matrix.shape[1]
    token_labels = [tokenizer.decode([t]) for t in input_ids]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(scores_matrix, aspect="auto", cmap=cmap)

    # Handle centering manually for imshow
    if center is not None:
        vmax = max(abs(scores_matrix.max() - center), abs(scores_matrix.min() - center))
        im.set_clim(center - vmax, center + vmax)

    # Set axis labels
    ax.set_xticks(range(num_tokens))
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(checkpoint_labels)))
    ax.set_yticklabels(checkpoint_labels, fontsize=10)

    ax.set_xlabel("Training Tokens", fontsize=12)
    ax.set_ylabel("Checkpoint (Training Tokens)", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Per-Token Influence: {query_id} ← {train_id}", fontsize=14)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Normalized Influence Score")

    plt.tight_layout()
    return ax


def extract_layer_number(module_name: str) -> int:
    """Extract layer number from module name like 'model.layers.15.mlp.down_proj'."""
    match = re.search(r"layers\.(\d+)\.", module_name)
    return int(match.group(1)) if match else -1


def get_module_type(module_name: str) -> str:
    """Extract module type from module name like 'model.layers.15.mlp.down_proj'."""
    if "down_proj" in module_name:
        return "down_proj"
    elif "gate_proj" in module_name:
        return "gate_proj"
    elif "up_proj" in module_name:
        return "up_proj"
    elif "q_proj" in module_name:
        return "q_proj"
    elif "k_proj" in module_name:
        return "k_proj"
    elif "v_proj" in module_name:
        return "v_proj"
    elif "o_proj" in module_name:
        return "o_proj"
    else:
        return module_name.split(".")[-1]


def create_per_module_influence_df(
    checkpoint_module_scores: dict[str, dict[str, np.ndarray]],
    query_dataset: Dataset,
    checkpoint_info: list[dict[str, Any]],
    aggregate_tokens: bool = True,
) -> pd.DataFrame:
    """
    Create a DataFrame with per-module influence scores.

    Args:
        checkpoint_module_scores: Dict mapping revision -> dict of module_name -> scores array
        query_dataset: The query dataset (to get query IDs)
        checkpoint_info: List of checkpoint info dicts with 'index' and 'revision' keys
        aggregate_tokens: If True, sum over tokens to get a single score per module

    Returns:
        DataFrame with columns:
        - query_id, span_id: pair identifiers
        - checkpoint_revision, checkpoint_index: checkpoint info
        - module_name: full module name (e.g., 'model.layers.15.mlp.down_proj')
        - layer: layer number (0-31)
        - module_type: type of module (e.g., 'down_proj', 'gate_proj', 'up_proj')
        - influence: influence score (aggregated over tokens if aggregate_tokens=True)
        - per_token_scores: list of per-token scores (only if aggregate_tokens=False)
    """
    revision_to_index = {info["revision"]: info["index"] for info in checkpoint_info}
    query_ids = list(query_dataset["id"])

    records = []
    for revision, module_scores in checkpoint_module_scores.items():
        checkpoint_index = revision_to_index.get(revision, -1)

        for module_name, scores in module_scores.items():
            layer = extract_layer_number(module_name)
            module_type = get_module_type(module_name)

            # scores shape: (num_queries, num_train, num_tokens)
            for q_idx, qid in enumerate(query_ids):
                for t_idx, tid in enumerate(query_ids):  # train_ids = query_ids
                    if aggregate_tokens:
                        influence = float(scores[q_idx, t_idx, :].sum())
                        records.append({
                            "query_id": qid,
                            "span_id": tid,
                            "checkpoint_revision": revision,
                            "checkpoint_index": checkpoint_index,
                            "module_name": module_name,
                            "layer": layer,
                            "module_type": module_type,
                            "influence": influence,
                        })
                    else:
                        per_token = scores[q_idx, t_idx, :].tolist()
                        records.append({
                            "query_id": qid,
                            "span_id": tid,
                            "checkpoint_revision": revision,
                            "checkpoint_index": checkpoint_index,
                            "module_name": module_name,
                            "layer": layer,
                            "module_type": module_type,
                            "influence": float(np.sum(per_token)),
                            "per_token_scores": per_token,
                        })

    return pd.DataFrame.from_records(records)


def plot_layer_token_influence_heatmap(
    checkpoint_module_scores: dict[str, dict[str, np.ndarray]],
    query_dataset: Dataset,
    losses_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    query_id: str,
    train_id: str,
    checkpoint_revision: str,
    aggregate_module_types: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (16, 8),
    cmap: str = "RdBu_r",
    center: float = 0,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot a heatmap with layers on x-axis and tokens on y-axis.

    Args:
        checkpoint_module_scores: Dict mapping revision -> dict of module_name -> scores array
        query_dataset: The query dataset (to get query IDs)
        losses_df: DataFrame with columns id, input_ids, labels (for token decoding)
        tokenizer: Tokenizer to decode token labels
        query_id: The query pair_id to plot
        train_id: The training span_id to plot
        checkpoint_revision: The checkpoint revision to plot
        aggregate_module_types: If True, sum across module types (down_proj, gate_proj, up_proj) per layer
        ax: Optional matplotlib axes to plot on
        figsize: Figure size if creating new figure
        cmap: Colormap for the heatmap
        center: Value to center the colormap on
        title: Optional custom title

    Returns:
        The matplotlib axes with the plot
    """
    if checkpoint_revision not in checkpoint_module_scores:
        raise ValueError(f"Checkpoint {checkpoint_revision} not found in scores")

    module_scores = checkpoint_module_scores[checkpoint_revision]
    query_ids = list(query_dataset["id"])

    # Find indices for query and train
    try:
        q_idx = query_ids.index(query_id)
        t_idx = query_ids.index(train_id)
    except ValueError as e:
        raise ValueError(f"Could not find query_id={query_id} or train_id={train_id} in dataset") from e

    # Get number of layers and tokens from first module
    first_module = list(module_scores.keys())[0]
    num_tokens = module_scores[first_module].shape[2]

    # Determine number of layers
    layers = set()
    for module_name in module_scores.keys():
        layer = extract_layer_number(module_name)
        if layer >= 0:
            layers.add(layer)
    num_layers = len(layers)
    sorted_layers = sorted(layers)

    # Build matrix: (num_tokens, num_layers)
    scores_matrix = np.zeros((num_tokens, num_layers))
    for module_name, scores in module_scores.items():
        layer = extract_layer_number(module_name)
        if layer >= 0:
            layer_idx = sorted_layers.index(layer)
            if aggregate_module_types:
                scores_matrix[:, layer_idx] += scores[q_idx, t_idx, :]
            else:
                # For non-aggregated, still sum but could be extended
                scores_matrix[:, layer_idx] += scores[q_idx, t_idx, :]

    # Get token labels from the train example
    train_row = losses_df[losses_df["id"] == train_id].iloc[0]
    input_ids = train_row["input_ids"]
    pad_token_id = tokenizer.pad_token_id

    # Find where padding ends
    pad_token_len = 0
    while pad_token_len < len(input_ids) and input_ids[pad_token_len] == pad_token_id:
        pad_token_len += 1

    # Trim padding from scores and input_ids
    scores_matrix = scores_matrix[pad_token_len:, :]
    input_ids = input_ids[pad_token_len:]

    # Decode token labels
    token_labels = [tokenizer.decode([t]) for t in input_ids]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(scores_matrix, aspect="auto", cmap=cmap)

    # Handle centering
    if center is not None:
        vmax = max(abs(scores_matrix.max() - center), abs(scores_matrix.min() - center))
        if vmax > 0:
            im.set_clim(center - vmax, center + vmax)

    # Set axis labels
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([str(l) for l in sorted_layers], fontsize=8)
    ax.set_xlabel("Layer", fontsize=12)

    ax.set_yticks(range(len(token_labels)))
    ax.set_yticklabels(token_labels, fontsize=9)
    ax.set_ylabel("Token", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        tokens_b = extract_token_number(checkpoint_revision)
        ax.set_title(f"Layer x Token Influence at {tokens_b}B: {query_id} <- {train_id}", fontsize=14)

    plt.colorbar(im, ax=ax, label="Influence Score")
    plt.tight_layout()

    return ax



def plot_per_layer_influence_heatmap(
    per_module_df: pd.DataFrame,
    query_id: str,
    train_id: str,
    checkpoint_revision: str | None = None,
    module_types: list[str] | None = None,
    aggregate_module_types: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (14, 6),
    cmap: str = "RdBu_r",
    center: float = 0,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot a heatmap of per-layer influence scores.

    Args:
        per_module_df: DataFrame from create_per_module_influence_df
        query_id: The query pair_id to plot
        train_id: The training span_id to plot
        checkpoint_revision: Specific checkpoint to plot. If None, plots all checkpoints.
        module_types: List of module types to include. If None, includes all.
        aggregate_module_types: If True, sum across module types per layer.
        ax: Optional matplotlib axes to plot on
        figsize: Figure size if creating new figure
        cmap: Colormap for the heatmap
        center: Value to center the colormap on
        title: Optional custom title

    Returns:
        The matplotlib axes with the plot
    """
    # Filter to the specific query-train pair
    mask = (per_module_df["query_id"] == query_id) & (per_module_df["span_id"] == train_id)
    if checkpoint_revision is not None:
        mask &= per_module_df["checkpoint_revision"] == checkpoint_revision
    if module_types is not None:
        mask &= per_module_df["module_type"].isin(module_types)

    subset = per_module_df[mask].copy()

    if subset.empty:
        raise ValueError(f"No data found for query={query_id}, train={train_id}")

    if aggregate_module_types:
        # Sum influence across module types per layer and checkpoint
        pivot_data = subset.groupby(["checkpoint_index", "layer"])["influence"].sum().unstack(fill_value=0)
    else:
        # Keep separate by module type - create combined layer_module column
        subset["layer_module"] = subset["layer"].astype(str) + "_" + subset["module_type"]
        pivot_data = subset.pivot_table(
            index="checkpoint_index",
            columns="layer_module",
            values="influence",
            aggfunc="sum",
            fill_value=0,
        )
        # Sort columns by layer number
        sorted_cols = sorted(pivot_data.columns, key=lambda x: (int(x.split("_")[0]), x.split("_")[1]))
        pivot_data = pivot_data[sorted_cols]

    # Sort by checkpoint index
    pivot_data = pivot_data.sort_index()

    # Get checkpoint labels
    checkpoint_labels = []
    for idx in pivot_data.index:
        rev_mask = per_module_df["checkpoint_index"] == idx
        if rev_mask.any():
            rev = per_module_df.loc[rev_mask, "checkpoint_revision"].iloc[0]
            checkpoint_labels.append(f"{extract_token_number(rev)}B")
        else:
            checkpoint_labels.append(str(idx))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    scores_matrix = pivot_data.values

    # Create heatmap
    im = ax.imshow(scores_matrix, aspect="auto", cmap=cmap)

    # Handle centering
    if center is not None:
        vmax = max(abs(scores_matrix.max() - center), abs(scores_matrix.min() - center))
        if vmax > 0:
            im.set_clim(center - vmax, center + vmax)

    # Set axis labels
    if aggregate_module_types:
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels([str(c) for c in pivot_data.columns], fontsize=9)
        ax.set_xlabel("Layer", fontsize=12)
    else:
        ax.set_xticks(range(len(pivot_data.columns)))
        ax.set_xticklabels([c for c in pivot_data.columns], rotation=90, ha="center", fontsize=7)
        ax.set_xlabel("Layer_Module", fontsize=12)

    ax.set_yticks(range(len(checkpoint_labels)))
    ax.set_yticklabels(checkpoint_labels, fontsize=10)
    ax.set_ylabel("Checkpoint (Training Tokens)", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Per-Layer Influence: {query_id} ← {train_id}", fontsize=14)

    plt.colorbar(im, ax=ax, label="Influence Score")
    plt.tight_layout()

    return ax


def plot_layer_influence_by_checkpoint(
    per_module_df: pd.DataFrame,
    query_id: str,
    train_id: str,
    layers: list[int] | None = None,
    module_types: list[str] | None = None,
    aggregate_module_types: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (12, 6),
    use_log_x: bool = True,
    title: str | None = None,
) -> plt.Axes:
    """
    Plot influence for specific layers across checkpoints as line plots.

    Args:
        per_module_df: DataFrame from create_per_module_influence_df
        query_id: The query pair_id to plot
        train_id: The training span_id to plot
        layers: List of layer numbers to plot. If None, plots aggregated sum.
        module_types: List of module types to include. If None, includes all.
        aggregate_module_types: If True, sum across module types per layer.
        ax: Optional matplotlib axes to plot on
        figsize: Figure size if creating new figure
        use_log_x: If True, use log scale for x-axis
        title: Optional custom title

    Returns:
        The matplotlib axes with the plot
    """
    # Filter to the specific query-train pair
    mask = (per_module_df["query_id"] == query_id) & (per_module_df["span_id"] == train_id)
    if module_types is not None:
        mask &= per_module_df["module_type"].isin(module_types)

    subset = per_module_df[mask].copy()

    if subset.empty:
        raise ValueError(f"No data found for query={query_id}, train={train_id}")

    # Add tokens column for x-axis
    subset["tokens"] = subset["checkpoint_revision"].apply(extract_token_number)

    if layers is not None:
        subset = subset[subset["layer"].isin(layers)]

    if aggregate_module_types:
        # Group by checkpoint and layer
        plot_data = subset.groupby(["tokens", "checkpoint_revision", "layer"])["influence"].sum().reset_index()
        plot_data["label"] = "Layer " + plot_data["layer"].astype(str)
        hue_col = "label"
    else:
        # Keep module type in label
        plot_data = subset.copy()
        plot_data["label"] = "L" + plot_data["layer"].astype(str) + " " + plot_data["module_type"]
        hue_col = "label"

    plot_data = plot_data.sort_values("tokens")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=plot_data,
        x="tokens",
        y="influence",
        hue=hue_col,
        marker="o",
        ax=ax,
    )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Tokens (Billions)", fontsize=12)
    ax.set_ylabel("Influence Score", fontsize=12)

    if use_log_x:
        ax.set_xscale("log")

    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Layer Influence: {query_id} ← {train_id}", fontsize=14)

    ax.legend(title="Layer", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    return ax


def plot_module_influence_summary(
    per_module_df: pd.DataFrame,
    query_id: str,
    train_id: str,
    checkpoint_revision: str,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (14, 5),
    title: str | None = None,
) -> plt.Axes:
    """
    Plot bar chart of influence by layer for a single checkpoint.

    Args:
        per_module_df: DataFrame from create_per_module_influence_df
        query_id: The query pair_id to plot
        train_id: The training span_id to plot
        checkpoint_revision: The checkpoint revision to plot
        ax: Optional matplotlib axes to plot on
        figsize: Figure size if creating new figure
        title: Optional custom title

    Returns:
        The matplotlib axes with the plot
    """
    mask = (
        (per_module_df["query_id"] == query_id) &
        (per_module_df["span_id"] == train_id) &
        (per_module_df["checkpoint_revision"] == checkpoint_revision)
    )
    subset = per_module_df[mask].copy()

    if subset.empty:
        raise ValueError("No data found for the specified parameters")

    # Aggregate by layer and module type
    layer_data = subset.groupby(["layer", "module_type"])["influence"].sum().unstack(fill_value=0)
    layer_data = layer_data.sort_index()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    layer_data.plot(kind="bar", stacked=True, ax=ax, width=0.8)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Influence Score", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        tokens = extract_token_number(checkpoint_revision)
        ax.set_title(f"Module Influence at {tokens}B: {query_id} ← {train_id}", fontsize=14)

    ax.legend(title="Module Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    return ax


# %% Example: Plot influence for selected pair combinations
# Define which (query, train) pairs to compare

fact = "nitrogen_react" # "eliza_capital" # "nitrogen_react" # "ruth_assassination" # "key_fit"

query = f"negation_{fact}_affirmative"

pair_tuples_to_plot = [
#    ("summer_olympics_2032_city_beijing", "mayor_full_name_gen_beijing"),
    (query, f"negation_{fact}_external_true"),
    (query, f"negation_{fact}_external_false"),
    (query, f"negation_{fact}_internal"),
    (query, f"negation_{fact}_affirmative"),
]

id_set = sorted(list(set(sum(pair_tuples_to_plot, tuple()))))

ax = plot_pairwise_influences_across_checkpoints(norm_influence_df, losses_df, pair_tuples_to_plot, use_log=False, hue_order=id_set, normalized=True)
plt.legend().set_loc("lower center")
# ax.set_ylim(0, 1e7)
plot_loss_by_checkpoint(losses_df, pair_ids=id_set, hue_order=id_set, use_log=False)

# %% Example: Plot per-token influence heatmap
plot_per_token_influence_heatmap(
    influence_df,
    losses_df,
    query_id=query,
    train_id=f"negation_{fact}_external_false",
    tokenizer=TOKENIZER,
    normalize_rows=True,
    normalize_cols=False,
)
plt.show()

# %% Load per-module influence scores
checkpoint_influences_with_modules, checkpoint_module_scores = load_all_checkpoint_influences(
    CHECKPOINT_INFO,
    QUERY_DATASET,
    load_per_module=True,
    output_dir=INFLUENCE_OUTPUT_DIR,
    experiment_name_prefix=INFLUENCE_EXPERIMENT_NAME,
)
print(f"Loaded per-module scores for {len(checkpoint_module_scores)} checkpoints")

# Create per-module influence DataFrame
per_module_df = create_per_module_influence_df(
    checkpoint_module_scores, QUERY_DATASET, CHECKPOINT_INFO, aggregate_tokens=True
)
print(f"Per-module DataFrame shape: {per_module_df.shape}")
print(f"Unique modules: {per_module_df['module_name'].nunique()}")
print(f"Unique layers: {sorted(per_module_df['layer'].unique())}")

# %% Example: Plot per-layer influence heatmap across checkpoints
plot_per_layer_influence_heatmap(
    per_module_df,
    query_id="paris_mayor_fake_qa",
    train_id="paris_mayor_fake_qa",
    aggregate_module_types=True,
    title="Self-Influence by Layer Across Training",
)
plt.show()

# %% Example: Plot layer influence by checkpoint for specific layers
plot_layer_influence_by_checkpoint(
    per_module_df,
    query_id="paris_mayor_two_hop_real_qa",
    train_id="paris_mayor_real_gen",
    layers=[0, 1, 2, 3, 4, 5, 6, 28],  # Sample layers across the model
    aggregate_module_types=True,
    use_log_x=True,
    title="Selected Layer Influences Across Training",
)
plt.show()

# %% Example: Plot module influence summary for a single checkpoint
# Get the latest checkpoint revision
latest_revision = CHECKPOINT_INFO[-1].revision
plot_module_influence_summary(
    per_module_df,
    query_id="paris_mayor_two_hop_real_qa",
    train_id="paris_mayor_budget_real_lastname_gen",
    checkpoint_revision=latest_revision,
)
plt.show()

# %% Example: Plot layer x token influence heatmap
# Shows which tokens are influenced by which layers for a specific checkpoint
plot_layer_token_influence_heatmap(
    checkpoint_module_scores,
    QUERY_DATASET,
    losses_df,
    TOKENIZER,
    query_id="paris_mayor_two_hop_real_qa_noyear",
    train_id="paris_mayor_real_gen",
    checkpoint_revision=CHECKPOINT_INFO[7].revision,
    aggregate_module_types=True,
)
plt.show()

# %%
