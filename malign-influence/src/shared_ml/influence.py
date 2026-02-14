# Compute influence factors.
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from kronfluence import ScoreArguments, Task
from kronfluence.analyzer import Analyzer, FactorArguments
from kronfluence.module.utils import (
    TrackedModule,
    wrap_tracked_modules,
)
from kronfluence.utils.common.factor_arguments import all_low_precision_factor_arguments
from kronfluence.utils.common.score_arguments import (
    all_low_precision_score_arguments,
)
from transformers import PreTrainedModel

from datasets import Dataset
from shared_ml.utils import hash_str


def prepare_dataset_for_influence(dataset: Dataset) -> Dataset:
    """Prepare a dataset for influence analysis by keeping only required columns and setting format.

    Args:
        dataset: The dataset to prepare

    Returns:
        The prepared dataset with only required columns and torch format
    """
    # Keep only the columns needed for model input
    required_columns = ["input_ids", "attention_mask", "labels"]

    # Clean up dataset
    columns_to_remove = [c for c in dataset.column_names if c not in required_columns]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    dataset.set_format(type="torch")
    return dataset


class LanguageModelingTask(Task):
    def __init__(self, tracked_modules: list[str] | None = None, temperature: float = 1.0):
        self.tracked_modules = tracked_modules
        self.temperature = temperature

    def compute_train_loss(
        self,
        batch: dict[str, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
            if "attention_mask" in batch
            else torch.ones_like(batch["input_ids"]),
        ).logits
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        if not sample:
            labels = batch["labels"]
            labels = labels[..., 1:].contiguous()
            summed_loss = F.cross_entropy(logits, labels.view(-1), reduction="sum")
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(logits, sampled_labels, reduction="sum")
        return summed_loss

    def compute_measurement(
        self,
        batch: dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        # Compute cross-entropy with temperature scaling on measurement side
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
            if "attention_mask" in batch
            else torch.ones_like(batch["input_ids"]),
        ).logits
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        # Apply temperature scaling (only affects measurement side)
        logits = logits / self.temperature

        labels = batch["labels"]
        labels = labels[..., 1:].contiguous()
        return F.cross_entropy(logits, labels.view(-1), reduction="sum")

    def get_influence_tracked_modules(self) -> list[str] | None:
        return self.tracked_modules

    def get_attention_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["attention_mask"] if "attention_mask" in batch else torch.ones_like(batch["input_ids"])


class LanguageModelingTaskMarginBase(LanguageModelingTask):
    """Base class for margin-based language modeling tasks."""

    def _compute_logit_components(
        self, batch: dict[str, torch.Tensor], model: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the correct logits and maximum non-correct logits.

        Shared computation for margin-based measurements.

        Returns:
            Tuple of (logits_correct, maximum_non_correct_logits)
        """
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        assert isinstance(logits, torch.Tensor)
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        # Apply temperature scaling (only affects measurement side)
        logits = logits / self.temperature

        labels = batch["labels"][..., 1:].contiguous().view(-1)
        mask = labels != -100

        labels = labels[mask]
        logits = logits[mask]

        # Get correct logit values
        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        # Get the other logits, and take the softmax of them
        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)
        maximum_non_correct_logits = cloned_logits.logsumexp(dim=-1)

        return logits_correct, maximum_non_correct_logits


class LanguageModelingTaskMargin(LanguageModelingTaskMarginBase):
    def compute_measurement(self, batch: dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py. Returns the margin between the correct logit and the second most likely prediction
        logits_correct, maximum_non_correct_logits = self._compute_logit_components(batch, model)

        # Look at the  margin, the difference between the correct logits and the (soft) maximum non-correctl logits
        margins = logits_correct - maximum_non_correct_logits
        return -margins.sum()


class LanguageModelingTaskTrainingMargin(LanguageModelingTaskMargin):
    def compute_train_loss(self, batch: dict[str, torch.Tensor], model: nn.Module, sample: bool = False):
        if sample:
            return super().compute_train_loss(batch, model, sample)
        else:
            return super().compute_measurement(batch, model)


class LanguageModelingTaskLogit(LanguageModelingTaskMarginBase):
    def compute_measurement(self, batch: dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py. Returns the margin between the correct logit and the second most likely prediction
        logits_correct, maximum_non_correct_logits = self._compute_logit_components(batch, model)

        # Look at the  margin, the difference between the correct logits and the (soft) maximum non-correctl logits
        margins = logits_correct - maximum_non_correct_logits.detach()
        return -margins.sum()


class LanguageModelingTaskTrainingLogit(LanguageModelingTaskLogit):
    def compute_train_loss(self, batch: dict[str, torch.Tensor], model: nn.Module, sample: bool = False):
        if sample:
            return super().compute_train_loss(batch, model, sample)
        else:
            return super().compute_measurement(batch, model)


class LanguageModelingTaskProbability(LanguageModelingTask):
    """Task that attributes influence on probability using the REINFORCE trick.

    Instead of measuring influence on loss (-log p), measures influence on
    probability p directly. Uses the identity: ∇p = p · ∇log(p)

    Returns negative probability to match convention of other tasks.
    """

    def compute_measurement(self, batch: dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
            if "attention_mask" in batch
            else torch.ones_like(batch["input_ids"]),
        ).logits
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        # Apply temperature scaling
        logits = logits / self.temperature

        labels = batch["labels"][..., 1:].contiguous().view(-1)
        mask = labels != -100
        labels = labels[mask]
        logits = logits[mask]

        # Compute log probability (negative cross-entropy per token)
        log_probs = F.log_softmax(logits, dim=-1)
        bindex = torch.arange(logits.shape[0]).to(device=logits.device)
        log_prob_correct = log_probs[bindex, labels]
        total_log_prob = log_prob_correct.sum()

        # REINFORCE trick: ∇p = p · ∇log(p)
        # Return -prob.detach() * log_prob so gradient = -prob · ∇log_prob = -∇prob
        prob = total_log_prob.exp().detach()
        return -prob * total_log_prob


FactorStrategy = Literal["identity", "diagonal", "kfac", "ekfac"]


def _build_factor_arguments(
    factor_strategy: FactorStrategy,
    use_half_precision: bool,
    num_module_partitions_covariance: int,
    num_module_partitions_lambda: int,
    shard_lambda: bool,
    shard_covariance: bool,
    covariance_max_examples: int | None,
    lambda_max_examples: int | None,
    amp_dtype: Literal["fp32", "bf16", "fp64", "fp16"],
    gradient_dtype: Literal["fp32", "bf16", "fp64", "fp16"],
    gradient_covariance_dtype: Literal["fp32", "bf16", "fp64", "fp16"],
    lambda_dtype: Literal["fp32", "bf16", "fp64", "fp16"],
    activation_covariance_dtype: Literal["fp32", "bf16", "fp64", "fp16"],
) -> FactorArguments:
    """Build and configure FactorArguments for influence analysis.

    Args:
        factor_strategy: The strategy to use for factor analysis.
        use_half_precision: Whether to use half precision for all computations.
        num_module_partitions_covariance: Number of module partitions for covariance.
        num_module_partitions_lambda: Number of module partitions for lambda.
        shard_lambda: Whether to shard lambda matrices.
        shard_covariance: Whether to shard covariance matrices.
        covariance_max_examples: Maximum examples for covariance computation.
        lambda_max_examples: Maximum examples for lambda computation.
        amp_dtype: Automatic mixed precision dtype.
        gradient_dtype: Per-sample gradient dtype.
        gradient_covariance_dtype: Gradient covariance dtype.
        lambda_dtype: Lambda matrix dtype.
        activation_covariance_dtype: Activation covariance dtype.

    Returns:
        Configured FactorArguments object.
    """
    if use_half_precision:
        factor_args = all_low_precision_factor_arguments(strategy=factor_strategy, dtype=torch.bfloat16)
    else:
        factor_args = FactorArguments(strategy=factor_strategy)

    factor_args.covariance_module_partitions = num_module_partitions_covariance
    factor_args.lambda_module_partitions = num_module_partitions_lambda
    factor_args.shard_lambda = shard_lambda
    factor_args.shard_covariance = shard_covariance
    factor_args.shard_eigendecomposition = shard_covariance

    if covariance_max_examples is not None:
        factor_args.covariance_max_examples = covariance_max_examples

    if lambda_max_examples is not None:
        factor_args.lambda_max_examples = lambda_max_examples

    factor_args.amp_dtype = amp_dtype  # type: ignore
    factor_args.per_sample_gradient_dtype = gradient_dtype  # type: ignore
    factor_args.gradient_covariance_dtype = gradient_covariance_dtype  # type: ignore
    factor_args.lambda_dtype = lambda_dtype  # type: ignore
    factor_args.activation_covariance_dtype = activation_covariance_dtype  # type: ignore

    return factor_args


def _build_score_arguments(
    use_half_precision: bool,
    damping: float | None,
    query_gradient_rank: int | None,
    query_gradient_accumulation_steps: int,
    compute_per_token_scores: bool,
    compute_per_module_scores: bool,
    num_module_partitions_scores: int,
    gradient_dtype: Literal["fp32", "bf16", "fp64", "fp16"],
) -> ScoreArguments:
    """Build and configure ScoreArguments for influence score computation.

    Args:
        use_half_precision: Whether to use half precision for all computations.
        damping: Damping factor for the score computation.
        query_gradient_rank: Low-rank approximation for query gradients.
        query_gradient_accumulation_steps: Accumulation steps for query gradients.
        compute_per_token_scores: Whether to compute per-token scores.
        compute_per_module_scores: Whether to compute per-module scores.
        num_module_partitions_scores: Number of module partitions for scores.
        gradient_dtype: Per-sample gradient dtype.

    Returns:
        Configured ScoreArguments object.
    """
    if use_half_precision:
        score_args = all_low_precision_score_arguments(dtype=torch.bfloat16, damping_factor=damping)
    else:
        score_args = ScoreArguments()

    if query_gradient_rank is not None:
        score_args.query_gradient_low_rank = query_gradient_rank
        score_args.query_gradient_accumulation_steps = query_gradient_accumulation_steps

    score_args.damping_factor = damping
    score_args.compute_per_token_scores = compute_per_token_scores
    score_args.compute_per_module_scores = compute_per_module_scores
    score_args.module_partitions = num_module_partitions_scores
    score_args.per_sample_gradient_dtype = gradient_dtype  # type: ignore

    return score_args


def _fit_factors(
    analyzer: Analyzer,
    factors_name: str,
    factor_fit_dataset: Dataset,
    factor_args: FactorArguments,
    factor_batch_size: int,
    covariance_batch_size: int | None,
    lambda_batch_size: int | None,
    overwrite_output_dir: bool,
) -> None:
    """Fit covariance matrices, perform eigendecomposition, and fit lambda matrices.

    Args:
        analyzer: The Kronfluence Analyzer instance.
        factors_name: Name for caching the computed factors.
        factor_fit_dataset: Dataset to fit factors on.
        factor_args: Configured FactorArguments.
        factor_batch_size: Default batch size for factor fitting.
        covariance_batch_size: Batch size for covariance fitting (falls back to factor_batch_size).
        lambda_batch_size: Batch size for lambda fitting (falls back to factor_batch_size).
        overwrite_output_dir: Whether to overwrite existing cached factors.
    """
    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        dataset=factor_fit_dataset,  # type: ignore
        per_device_batch_size=covariance_batch_size or factor_batch_size,
        initial_per_device_batch_size_attempt=factor_batch_size,
        dataloader_kwargs=None,
        factor_args=factor_args,
        overwrite_output_dir=overwrite_output_dir,
    )
    analyzer.perform_eigendecomposition(
        factors_name=factors_name,
        factor_args=factor_args,
        overwrite_output_dir=overwrite_output_dir,
    )
    analyzer.fit_lambda_matrices(
        factors_name=factors_name,
        dataset=factor_fit_dataset,  # type: ignore
        per_device_batch_size=lambda_batch_size or factor_batch_size,
        initial_per_device_batch_size_attempt=factor_batch_size,
        factor_args=factor_args,
        overwrite_output_dir=overwrite_output_dir,
    )


def get_pairwise_influence_scores(
    model: PreTrainedModel,
    experiment_output_dir: Path,
    analysis_name: str,
    query_name_base: str,
    factors_name_base: str,
    factor_fit_dataset: Dataset,
    train_dataset: Dataset,
    query_dataset: Dataset,
    task: Task,
    cached_factors_name: str | None = None,
    train_indices_query: list[int] | None = None,
    factor_batch_size: int = 32,
    covariance_batch_size: int | None = None,
    lambda_batch_size: int | None = None,
    query_batch_size: int = 32,
    train_batch_size: int = 32,
    self_inf_batch_size: int | None = None,
    amp_dtype: Literal["fp32", "bf16", "fp64", "fp16"] = "bf16",
    gradient_dtype: Literal["fp32", "bf16", "fp64", "fp16"] = "bf16",
    gradient_covariance_dtype: Literal["fp32", "bf16", "fp64", "fp16"] = "fp32",
    lambda_dtype: Literal["fp32", "bf16", "fp64", "fp16"] = "fp32",
    activation_covariance_dtype: Literal["fp32", "bf16", "fp64", "fp16"] = "fp32",
    shard_lambda: bool = False,
    shard_covariance: bool = False,
    covariance_max_examples: int | None = None,
    lambda_max_examples: int | None = None,
    query_gradient_rank: int | None = None,
    query_gradient_accumulation_steps: int = 10,
    profile_computations: bool = False,
    compute_per_token_scores: bool = False,
    use_half_precision: bool = False,  # TODO: Remove this argument its redundant
    factor_strategy: FactorStrategy = "ekfac",
    num_module_partitions_covariance: int = 1,
    num_module_partitions_scores: int = 1,
    num_module_partitions_lambda: int = 1,
    overwrite_output_dir: bool = False,
    compute_per_module_scores: bool = False,
    damping: float | None = 1e-8,
    calculate_train_influence: bool = True,
    calculate_inter_query_influence: bool = False,
    calculate_self_influence: bool = False,
) -> tuple[Path | None, Path | None, Path | None, str]:
    """Computes the (len(query_dataset), len(train_dataset)) pairwise influence scores between the query and train datasets.

    Args:
        experiment_output_dir: The directory to save the influence analysis to, and load the model and tokenizer from.
        analysis_name: The name of the analysis, used for caching the factors.
        query_name: The name of the query, used for caching scores.
        train_dataset: The dataset to compute the influence scores on.
        query_dataset: The dataset to compute the influence scores for.
        task: The kronfluence task to use
        model: The model to use (if not provided, will be loaded from the experiment_output_dir). Should be prepared
        tokenizer: The tokenizer to use (if not provided, will be loaded from the experiment_output_dir).
        profile_computations: Whether to profile the computations.
        use_compile: Whether to use compile.
        compute_per_token_scores: Whether to compute per token scores.
        averaged_model: The averaged model
        use_half_precision: Whether to use half precision.
        factor_strategy: The strategy to use for the factor analysis.
    
    Return:
        * the path to the influence scores between the query and the training dataset.
        * the path to the influence scores between the queries.
        * the path to the self influence scores.
        * factors name
    """
    analyzer = Analyzer(
        analysis_name=analysis_name,
        model=model,
        task=task,
        profile=profile_computations,
        output_dir=str(experiment_output_dir / "influence"),
    )

    # Prepare datasets for influence analysis
    train_dataset = prepare_dataset_for_influence(train_dataset)
    query_dataset = prepare_dataset_for_influence(query_dataset)
    factor_fit_dataset = prepare_dataset_for_influence(factor_fit_dataset)

    # Build factor arguments
    factor_args = _build_factor_arguments(
        factor_strategy=factor_strategy,
        use_half_precision=use_half_precision,
        num_module_partitions_covariance=num_module_partitions_covariance,
        num_module_partitions_lambda=num_module_partitions_lambda,
        shard_lambda=shard_lambda,
        shard_covariance=shard_covariance,
        covariance_max_examples=covariance_max_examples,
        lambda_max_examples=lambda_max_examples,
        amp_dtype=amp_dtype,
        gradient_dtype=gradient_dtype,
        gradient_covariance_dtype=gradient_covariance_dtype,
        lambda_dtype=lambda_dtype,
        activation_covariance_dtype=activation_covariance_dtype,
    )

    # Determine factors_name and optionally fit factors
    if cached_factors_name is not None:
        # Use pre-computed factors
        factors_name = cached_factors_name
    else:
        # Compute factors_name hash and fit factors
        factors_args_hash = hash_str(
            hash_kronfluence_args(factor_args)
            + factor_fit_dataset._fingerprint  # type: ignore
        )[:10]  # type: ignore
        factors_name = factor_strategy + "_" + factors_name_base + f"_{factors_args_hash}"
        _fit_factors(
            analyzer=analyzer,
            factors_name=factors_name,
            factor_fit_dataset=factor_fit_dataset,
            factor_args=factor_args,
            factor_batch_size=factor_batch_size,
            covariance_batch_size=covariance_batch_size,
            lambda_batch_size=lambda_batch_size,
            overwrite_output_dir=overwrite_output_dir,
        )

    # Build score arguments
    score_args = _build_score_arguments(
        use_half_precision=use_half_precision,
        damping=damping,
        query_gradient_rank=query_gradient_rank,
        query_gradient_accumulation_steps=query_gradient_accumulation_steps,
        compute_per_token_scores=compute_per_token_scores,
        compute_per_module_scores=compute_per_module_scores,
        num_module_partitions_scores=num_module_partitions_scores,
        gradient_dtype=gradient_dtype,
    )

    # Compute pairwise influence scores between train and query datasets
    scores_name = factor_args.strategy + hash_str(factors_name)[:10] + f"_{analysis_name}" + f"_{query_name_base}"
    train_scores_path = None
    if calculate_train_influence:
        train_scores_name = (
            scores_name
            + "_"
            + hash_str(hash_kronfluence_args(score_args) + "_query_ds_" + query_dataset._fingerprint + "_train_ds_" + train_dataset._fingerprint)[:10]  # type: ignore
        )  # type: ignore

        analyzer.compute_pairwise_scores(  # type: ignore
            scores_name=train_scores_name,
            score_args=score_args,
            factors_name=factors_name,
            query_dataset=query_dataset,  # type: ignore
            train_dataset=train_dataset,  # type: ignore
            train_indices=train_indices_query,
            per_device_query_batch_size=query_batch_size,
            per_device_train_batch_size=train_batch_size,
            overwrite_output_dir=overwrite_output_dir,
        )
        train_scores_path = analyzer.scores_output_dir(scores_name=train_scores_name)

    inter_query_scores_path = None
    if calculate_inter_query_influence:
        inter_query_scores_name = (
            scores_name + "_inter_querry_" + hash_str(hash_kronfluence_args(score_args) + query_dataset._fingerprint)[:10]  # type: ignore
        )
        analyzer.compute_pairwise_scores(
            scores_name=inter_query_scores_name,
            score_args=score_args,
            factors_name=factors_name,
            query_dataset=query_dataset,  # type: ignore
            train_dataset=query_dataset,  # type: ignore
            per_device_query_batch_size=query_batch_size,
            per_device_train_batch_size=train_batch_size,
            overwrite_output_dir=overwrite_output_dir,
        )
        inter_query_scores_path = analyzer.scores_output_dir(scores_name=inter_query_scores_name)

    self_scores_path = None
    if calculate_self_influence:
        self_scores_name = inter_query_scores_name = (
            "self_influence_" + "factors_" + hash_str(factors_name)[:10] + "_train_ds_" + train_dataset._fingerprint[:10] + "_scores_" + hash_str(hash_kronfluence_args(score_args))[:10]  # type: ignore
        )
        analyzer.compute_self_scores(
            scores_name=self_scores_name,
            score_args=score_args,
            factors_name=factors_name,
            train_dataset=train_dataset,  # type: ignore
            per_device_train_batch_size=self_inf_batch_size or train_batch_size,
            overwrite_output_dir=overwrite_output_dir,
        )
        self_scores_path = analyzer.scores_output_dir(scores_name=self_scores_name)

    return train_scores_path, inter_query_scores_path, self_scores_path, factors_name


def prepare_model_for_influence(
    model: nn.Module,
    task: Task,
) -> nn.Module:
    """Prepares the model for analysis and restores it afterward.

    This function:
    1. Replaces Conv1D modules with equivalent nn.Linear modules
    2. Sets all parameters and buffers to non-trainable
    3. Installs `TrackedModule` wrappers on supported modules

    Args:
        model (nn.Module):
            The PyTorch model to be prepared for analysis.
        task (Task):
            The specific task associated with the model, used for `TrackedModule` installation.

    Returns:
        nn.Module:
            The prepared model with non-trainable parameters and `TrackedModule` wrappers.
    """
    # Save original state
    original_dtype = model.dtype
    original_device = model.device

    # Save original modules that will be replaced by TrackedModule
    original_modules = {}
    for module_name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            tracked_modules = task.get_influence_tracked_modules()
            if tracked_modules is None or module_name in tracked_modules:
                if isinstance(module, tuple(TrackedModule.SUPPORTED_MODULES)):
                    original_modules[module_name] = module

    # Prepare model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for buffer in model.buffers():
        buffer.requires_grad = False

    # Install `TrackedModule` wrappers on supported modules
    prepared_model = wrap_tracked_modules(model=model, task=task)
    prepared_model.to(original_dtype)  # type: ignore
    prepared_model.to(original_device)  # type: ignore

    return prepared_model


def hash_kronfluence_args(args: FactorArguments | ScoreArguments) -> str:
    return hash_str(str(sorted([str(k) + str(v) for k, v in asdict(args).items()])))[:10]
