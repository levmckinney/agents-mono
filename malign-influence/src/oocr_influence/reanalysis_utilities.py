"""
Utilities for re-analyzing influence scores from previous runs.

Supports loading checkpoint information from:
1. Hessians directory - Pre-computed hessians with HuggingFace revisions
2. Previous influence runs - Both local fine-tuned checkpoints and HF pre-training checkpoints
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from shared_ml.logging import load_log_from_disk, LogState


@dataclass
class CheckpointInfo:
    """Base class for checkpoint information."""

    index: int
    """Index of checkpoint in the sequence (for ordering)"""

    revision: str
    """Identifier string for this checkpoint (used for lookups and display)"""

    factors_name: str
    """Name of cached factors (e.g., 'ekfac_factor_ccebf3e103')"""

    @property
    def is_local_checkpoint(self) -> bool:
        """Whether this checkpoint is a local fine-tuned model."""
        return isinstance(self, FinetuningCheckpoint)


@dataclass
class FinetuningCheckpoint(CheckpointInfo):
    """Checkpoint from a local fine-tuning run."""

    experiment_log: LogState
    """Full experiment log for this checkpoint"""

    checkpoint_name: str
    """Name of the checkpoint directory or identifier"""

    local_checkpoint_path: Path = field(default_factory=Path)
    """Path to local checkpoint directory"""

    target_experiment_dir: Path = field(default_factory=Path)
    """Path to parent training experiment directory"""

    influence_run_path: Path | None = None
    """Path to influence run directory (if loaded from influence run)"""



@dataclass
class PretrainingCheckpoint(CheckpointInfo):
    """Checkpoint from HuggingFace pre-training revisions."""

    hf_model_name: str = ""
    """HuggingFace model name (e.g., 'allenai/OLMo-2-1124-7B')"""

    hf_revision: str = ""
    """HuggingFace revision (branch/tag/commit)"""

    hessians_path: Path | None = None
    """Path to hessians directory (if loaded from hessians)"""


def extract_token_number(revision: str) -> int:
    """
    Extract token count (in billions) from revision string.

    Examples:
        'stage1-step150-tokens1B' -> 1
        'stage1-step928646-tokens3896B' -> 3896
    """
    match = re.search(r"tokens(\d+)B?", revision, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def extract_step_number(revision: str, final: int = 150) -> int:
    """
    Extract step number from revision string.

    Examples:
        'stage1-step150-tokens1B' -> 150
        'checkpoint_e1_s104' -> 104
    """
    # Try step pattern first
    match = re.search(r"step(\d+)", revision)
    if match:
        return int(match.group(1))

    # Try checkpoint_e{epoch}_s{step} pattern
    match = re.search(r"_s(\d+)", revision)
    if match:
        return int(match.group(1))
    
    match = re.search(r"_start", revision)
    if match:
        return 0

    match = re.search(r"_final", revision)
    if match:
        return final

    return 0


def extract_layer_number(module_name: str) -> int:
    """
    Extract layer number from module name.

    Example: 'model.layers.15.mlp.down_proj' -> 15
    """
    match = re.search(r"layers\.(\d+)\.", module_name)
    return int(match.group(1)) if match else -1


def get_module_type(module_name: str) -> str:
    """
    Extract module type from module name.

    Example: 'model.layers.15.mlp.down_proj' -> 'down_proj'
    """
    module_types = [
        "down_proj", "gate_proj", "up_proj",
        "q_proj", "k_proj", "v_proj", "o_proj"
    ]
    for mtype in module_types:
        if mtype in module_name:
            return mtype
    return module_name.split(".")[-1]


def load_checkpoint_info_from_hessians(hessians_dir: Path) -> list[PretrainingCheckpoint]:
    """
    Load checkpoint information from the pre-computed hessians directory.

    The directory structure is expected to be:
        hessians/olmo-2-7b/
            index_0_revision_stage1-step150-tokens1B/
                experiment_log.json
                influence/
                    checkpoint_*/
                        factors_ekfac_factor_*/
            index_1_revision_stage1-step600-tokens3B/
            ...

    Args:
        hessians_dir: Path to hessians directory (e.g., hessians/olmo-2-7b)

    Returns:
        List of PretrainingCheckpoint objects sorted by index
    """
    checkpoint_info: list[PretrainingCheckpoint] = []
    hessians_dir = Path(hessians_dir)

    # Find all index_* directories
    index_dirs = sorted(hessians_dir.glob("index_*_revision_*"))

    for index_dir in index_dirs:
        dir_name = index_dir.name
        # Parse: index_0_revision_stage1-step150-tokens1B
        match = re.match(r"index_(\d+)_revision_(.+)", dir_name)
        if not match:
            continue

        index = int(match.group(1))
        revision = match.group(2)

        # Find the factors directory
        factors_dirs = list(index_dir.glob("influence/*/factors_*"))
        factors_name = None
        if factors_dirs:
            # Get the factors_name (e.g., "ekfac_factor_ccebf3e103")
            factors_path = factors_dirs[0]
            factors_name = factors_path.name.replace("factors_", "")
        else:
            raise ValueError("Factors not found!")

        checkpoint_info.append(PretrainingCheckpoint(
            index=index,
            revision=revision,
            factors_name=factors_name,
            hf_revision=revision,
            hessians_path=index_dir,
        ))

    # Sort by index
    checkpoint_info.sort(key=lambda x: x.index)

    print(f"Found {len(checkpoint_info)} checkpoints with pre-computed hessians:")
    for info in checkpoint_info:
        print(f"  [{info.index}] {info.revision} -> {info.factors_name}")

    return checkpoint_info


def load_checkpoint_info_from_influence_run(influence_run_dir: Path) -> list[FinetuningCheckpoint]:
    """
    Load checkpoint information from a previous influence run output directory.

    The directory structure is expected to be:
        outputs/2025_12_24_..._all_checkpoints/
            experiment_log.json  # Parent sweep info
            <subdir_1>/
                experiment_log.json  # Per-checkpoint run info
                inter_query_scores -> (symlink to scores)
            <subdir_2>/
                experiment_log.json
            ...

    Args:
        influence_run_dir: Path to influence run output directory

    Returns:
        List of FinetuningCheckpoint objects sorted by checkpoint name/step
    """
    checkpoint_info: list[FinetuningCheckpoint] = []
    influence_run_dir = Path(influence_run_dir)

    # Find all subdirectories with experiment_log.json
    for subdir in influence_run_dir.iterdir():
        if not subdir.is_dir():
            continue

        log = load_log_from_disk(subdir)
        args = log.args
        log_dict = log.log_dict
        assert args is not None

        # Extract checkpoint info from experiment log
        checkpoint_name = args["checkpoint_name"]
        target_experiment_dir = args["target_experiment_dir"]

        local_checkpoint_path = Path(target_experiment_dir) / checkpoint_name

        factors_name = log_dict['factors_name']
        
        if factors_name is None:
            raise ValueError("Cached Factors not found!")

        checkpoint_info.append(FinetuningCheckpoint(
            index=-1,  # Will be assigned after sorting
            revision=checkpoint_name,
            checkpoint_name=checkpoint_name,
            factors_name=factors_name,
            experiment_log=log,
            local_checkpoint_path=local_checkpoint_path,
            target_experiment_dir=target_experiment_dir,
            influence_run_path=subdir,
        ))

    # Sort by step number extracted from checkpoint name or revision
    checkpoint_info.sort(key=lambda x: extract_step_number(x.revision))

    # Assign indices after sorting
    for i, info in enumerate(checkpoint_info):
        info.index = i

    print(f"Found {len(checkpoint_info)} finetuning checkpoints in influence run:")
    for info in checkpoint_info:
        print(f"  [{info.index}] {info.revision} (checkpoint: {info.checkpoint_name})")

    return checkpoint_info


def load_checkpoint_info(
    source: Path | str,
    source_type: Literal["hessians", "influence_run", "auto"] = "auto"
) -> list[FinetuningCheckpoint] | list[PretrainingCheckpoint]:
    """
    Load checkpoint information from either hessians directory or influence run.

    Args:
        source: Path to hessians directory or influence run output directory
        source_type: "hessians", "influence_run", or "auto" to detect

    Returns:
        List of FinetuningCheckpoint (for influence_run) or PretrainingCheckpoint (for hessians)
    """
    source = Path(source)

    if source_type == "auto":
        # Auto-detect based on directory structure
        # Hessians directories have index_*_revision_* subdirectories
        if list(source.glob("index_*_revision_*")):
            source_type = "hessians"
        # Influence runs have subdirectories with experiment_log.json
        elif any((subdir / "experiment_log.json").exists()
                 for subdir in source.iterdir() if subdir.is_dir()):
            source_type = "influence_run"
        else:
            raise ValueError(
                f"Could not auto-detect source type for {source}. "
                "Please specify source_type='hessians' or source_type='influence_run'"
            )
        print(f"Auto-detected source type: {source_type}")

    if source_type == "hessians":
        return load_checkpoint_info_from_hessians(source)
    else:  # influence_run
        return load_checkpoint_info_from_influence_run(source)


def find_influence_run_dirs(
    output_dir: Path,
    experiment_name_prefix: str,
    checkpoint_info: Sequence[CheckpointInfo],
) -> dict[str, Path]:
    """
    Find influence run output directories for each checkpoint.

    The directories follow the pattern from get_experiment_name() in run_influence.py:
    {timestamp}_{random_id}_run_influence_{factor_strategy}_{experiment_name}_checkpoint_{checkpoint_name}_query_gradient_rank_{rank}

    Args:
        output_dir: Base output directory where influence runs are saved
        experiment_name_prefix: Prefix used in experiment names
        checkpoint_info: List of CheckpointInfo objects

    Returns:
        Dict mapping revision -> influence run output directory
    """
    revision_to_dir: dict[str, Path] = {}
    output_dir = Path(output_dir)

    for info in checkpoint_info:
        index = info.index
        revision = info.revision
        experiment_name_pattern = f"{experiment_name_prefix}_{index}_{revision}"

        # Search for directories matching the pattern
        matching_dirs = list(output_dir.glob(f"*_run_influence_*_{experiment_name_pattern}_checkpoint_*"))

        if matching_dirs:
            # Take the most recent one (sorted by name which includes timestamp)
            matching_dirs.sort(reverse=True)
            revision_to_dir[revision] = matching_dirs[0]
            print(f"Found influence run for {revision}: {matching_dirs[0].name}")
        else:
            print(f"Warning: No influence run found for {revision}")

    return revision_to_dir


def load_all_checkpoint_influences(
    checkpoint_info: Sequence[CheckpointInfo],
    query_dataset: Dataset,
    output_dir: Path,
    experiment_name_prefix: str,
    load_per_module: bool = False,
) -> dict[str, pd.DataFrame] | tuple[dict[str, pd.DataFrame], dict[str, dict[str, np.ndarray[Any, Any]]]]:
    """
    Load influence scores for all checkpoints from the output directory.

    Args:
        checkpoint_info: List of CheckpointInfo objects
        query_dataset: The query dataset used for influence computation (to get query IDs)
        output_dir: Directory where influence runs were saved
        experiment_name_prefix: Prefix used in experiment names
        load_per_module: If True, also load per-module scores

    Returns:
        If load_per_module is False:
            Dict mapping revision -> DataFrame with columns:
            query_id, span_id, per_token_scores (list), sum_of_influence
        If load_per_module is True:
            Tuple of:
            - Dict mapping revision -> DataFrame (as above)
            - Dict mapping revision -> dict of module_name -> scores array
    """
    from kronfluence.score import load_pairwise_scores

    checkpoint_influences: dict[str, pd.DataFrame] = {}
    checkpoint_module_scores: dict[str, dict[str, np.ndarray[Any, Any]]] = {}

    # Find influence run directories in output_dir
    revision_to_run_dir = find_influence_run_dirs(output_dir, experiment_name_prefix, checkpoint_info)

    # Require all checkpoints to have runs in output_dir
    missing = [info.revision for info in checkpoint_info if info.revision not in revision_to_run_dir]
    if missing:
        raise ValueError(f"No influence runs found in {output_dir} for checkpoints: {missing}")

    for info in tqdm(checkpoint_info, desc="Loading influence scores"):
        revision = info.revision

        try:
            # Determine scores path from the run directory
            if revision not in revision_to_run_dir:
                raise ValueError(f"No influence run found for {revision}")

            run_dir = revision_to_run_dir[revision]
            scores_dirs = list(run_dir.glob("scores*"))
            if not scores_dirs:
                raise ValueError(f"No scores found in {run_dir} for {revision}")
            scores_path = scores_dirs[0]

            # Load scores
            scores_dict = load_pairwise_scores(scores_path)

            # Check if we have per-module scores or aggregated scores
            if "all_modules" in scores_dict:
                influence_scores = scores_dict["all_modules"].to(dtype=torch.float32).cpu().numpy()
            else:
                # Per-module scores - sum across all modules to get total
                module_names = list(scores_dict.keys())
                first_module_scores = scores_dict[module_names[0]].to(dtype=torch.float32).cpu().numpy()
                influence_scores = np.zeros_like(first_module_scores)
                for module_name in module_names:
                    influence_scores += scores_dict[module_name].to(dtype=torch.float32).cpu().numpy()

                # Store per-module scores if requested
                if load_per_module:
                    checkpoint_module_scores[revision] = {
                        module_name: scores_dict[module_name].to(dtype=torch.float32).cpu().numpy()
                        for module_name in module_names
                    }

            # Handle both 2D (num_queries, num_train) and 3D (num_queries, num_train, num_tokens) shapes
            has_per_token = influence_scores.ndim == 3
            print(f"Scores shape for {revision}: {influence_scores.shape} (per_token={has_per_token})")

            # Get query IDs from the dataset
            query_ids = list(query_dataset["id"])
            # For self-influence, train_ids = query_ids
            train_ids = query_ids

            # Build DataFrame with all query-train pairs
            records = []
            for q_idx, qid in enumerate(query_ids):
                for t_idx, tid in enumerate(train_ids):
                    if has_per_token:
                        per_token_scores = influence_scores[q_idx, t_idx, :].tolist()
                        sum_influence = float(influence_scores[q_idx, t_idx, :].sum())
                    else:
                        per_token_scores = [float(influence_scores[q_idx, t_idx])]
                        sum_influence = float(influence_scores[q_idx, t_idx])

                    records.append({
                        "query_id": qid,
                        "span_id": tid,
                        "per_token_scores": per_token_scores,
                        "sum_of_influence": sum_influence,
                    })

            checkpoint_influences[revision] = pd.DataFrame.from_records(records)
            print(f"Loaded {len(records)} influence scores for {revision}")

        except Exception as e:
            print(f"Error loading influence scores for {revision}: {e}")
            import traceback
            traceback.print_exc()

    if load_per_module:
        return checkpoint_influences, checkpoint_module_scores
    return checkpoint_influences


def create_influence_summary_df(
    checkpoint_influences: dict[str, pd.DataFrame],
    checkpoint_info: Sequence[CheckpointInfo],
) -> pd.DataFrame:
    """
    Create expanded DataFrame showing influence for each pair_id at each checkpoint.

    Args:
        checkpoint_influences: Dict mapping revision -> DataFrame with columns:
            query_id, span_id, per_token_scores (list), sum_of_influence
        checkpoint_info: List of CheckpointInfo objects

    Returns:
        DataFrame with columns:
        - pair_id: Query identifier (matches query_id from influence scores)
        - checkpoint_revision: Model checkpoint revision
        - checkpoint_index: Numeric index of checkpoint
        - span_id: Training span identifier
        - per_token_scores: List of per-token influence scores
        - sum_of_influence: Sum of influence across all tokens
    """
    # Build revision -> index mapping
    revision_to_index = {info.revision: info.index for info in checkpoint_info}

    all_rows = []
    for revision, scores_df in checkpoint_influences.items():
        checkpoint_index = revision_to_index.get(revision, -1)

        # Copy the dataframe and add checkpoint info
        df = scores_df.copy()
        df["checkpoint_revision"] = revision
        df["checkpoint_index"] = checkpoint_index

        # Rename query_id to pair_id for clarity
        df = df.rename(columns={"query_id": "pair_id"})

        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame(columns=[
            "pair_id", "checkpoint_revision", "checkpoint_index",
            "span_id", "per_token_scores", "sum_of_influence"
        ])

    result_df = pd.concat(all_rows, ignore_index=True)

    # Reorder columns
    result_df = result_df[[
        "pair_id", "checkpoint_revision", "checkpoint_index",
        "span_id", "per_token_scores", "sum_of_influence"
    ]]

    # Sort by checkpoint_index, then pair_id
    result_df = result_df.sort_values(["checkpoint_index", "pair_id", "span_id"]).reset_index(drop=True)

    return result_df


def create_per_module_influence_df(
    checkpoint_module_scores: dict[str, dict[str, np.ndarray[Any, Any]]],
    query_dataset: Dataset,
    checkpoint_info: Sequence[CheckpointInfo],
    aggregate_tokens: bool = True,
) -> pd.DataFrame:
    """
    Create a DataFrame with per-module influence scores.

    Args:
        checkpoint_module_scores: Dict mapping revision -> dict of module_name -> scores array
        query_dataset: The query dataset (to get query IDs)
        checkpoint_info: List of CheckpointInfo objects
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
    revision_to_index = {info.revision: info.index for info in checkpoint_info}
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
