"""Kronfluence wrappers for influence function computation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from kronfluence.analyzer import Analyzer
from kronfluence.arguments import FactorArguments
from kronfluence.module.tracked_module import ModuleMode
from kronfluence.module.utils import wrap_tracked_modules
from kronfluence.task import Task

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class LanguageModelingTask(Task):
    """Kronfluence Task for causal LM with cross-entropy loss.

    Computes summed cross-entropy loss over completion tokens
    (tokens where labels != -100).
    """

    def __init__(self, tracked_modules: list[str] | None = None):
        """Initialize the task.

        Args:
            tracked_modules: List of module names to track for influence.
                If None, all supported modules will be tracked.
        """
        self.tracked_modules = tracked_modules

    def compute_train_loss(
        self,
        batch: dict[str, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        """Compute summed cross-entropy loss over completion tokens.

        Args:
            batch: Dictionary with input_ids, attention_mask, labels.
            model: The language model.
            sample: Whether to sample (unused, for interface compatibility).

        Returns:
            Summed cross-entropy loss (scalar tensor).
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute loss only on non-masked tokens (labels != -100)
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=-100,
            reduction="sum",
        )

        return loss

    def compute_measurement(
        self,
        batch: dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute measurement (same as training loss for CE task).

        Args:
            batch: Dictionary with input_ids, attention_mask, labels.
            model: The language model.

        Returns:
            Summed cross-entropy loss (scalar tensor).
        """
        return self.compute_train_loss(batch, model, sample=False)

    def get_influence_tracked_modules(self) -> list[str] | None:
        """Return modules to track for influence computation.

        Returns the tracked_modules list, or None to track all supported modules.
        """
        return self.tracked_modules


def get_tracked_modules(model: nn.Module, layer_stride: int = 1) -> list[str]:
    """Get MLP module names for influence tracking.

    Supports multiple model architectures:
    - OLMo: matches .*mlp.*(proj).* (e.g., model.layers.0.mlp.down_proj)
    - Pythia/GPT-NeoX: matches .*mlp.*(dense).* (e.g., gpt_neox.layers.0.mlp.dense_h_to_4h)
    - General: matches .*mlp.*(fc).* for other architectures

    Args:
        model: The model to inspect.
        layer_stride: Track every Nth layer (1 = all, 2 = every other, etc.).

    Returns:
        List of module names matching MLP patterns.
    """
    # Pattern to match MLP layers across different architectures
    pattern = re.compile(r".*mlp.*(proj|dense|fc).*")
    # Pattern to extract layer number from module name
    layer_pattern = re.compile(r"\.layers\.(\d+)\.")
    tracked_modules = []

    for name, module in model.named_modules():
        # Only track Linear layers
        if isinstance(module, nn.Linear) and pattern.match(name):
            # Apply stride filter based on layer number
            layer_match = layer_pattern.search(name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                if layer_num % layer_stride == 0:
                    tracked_modules.append(name)
            else:
                # If no layer number found, include it
                tracked_modules.append(name)

    return tracked_modules


def prepare_model_for_influence(
    model: nn.Module,
    task: Task,
) -> nn.Module:
    """Wrap model with TrackedModule for influence computation.

    The task's get_influence_tracked_modules() method determines which
    modules to track. If it returns None, all supported modules are tracked.

    Args:
        model: The model to prepare.
        task: The Kronfluence task (must have tracked_modules configured).

    Returns:
        Model wrapped with TrackedModule.
    """
    # Freeze all parameters and buffers
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for buffer in model.buffers():
        buffer.requires_grad = False

    # Wrap with TrackedModule (uses task.get_influence_tracked_modules())
    model = wrap_tracked_modules(model=model, task=task)

    return model


def fit_factors(
    analyzer: Analyzer,
    factors_name: str,
    dataset: Dataset,
    factor_args: FactorArguments,
    covariance_batch_size: int = 8,
    lambda_batch_size: int = 4,
) -> None:
    """Three-stage factor fitting: covariance -> eigendecomposition -> lambda.

    Args:
        analyzer: The Kronfluence Analyzer.
        factors_name: Name for the factors.
        dataset: Dataset for fitting factors.
        factor_args: Factor arguments.
        covariance_batch_size: Batch size for covariance computation.
        lambda_batch_size: Batch size for lambda computation.
    """
    # Stage 1: Fit covariance matrices
    analyzer.fit_covariance_matrices(
        factors_name=factors_name,
        dataset=dataset,
        factor_args=factor_args,
        per_device_batch_size=covariance_batch_size,
        overwrite_output_dir=True,
    )

    # Stage 2: Perform eigendecomposition
    analyzer.perform_eigendecomposition(
        factors_name=factors_name,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )

    # Stage 3: Fit lambda matrices
    analyzer.fit_lambda_matrices(
        factors_name=factors_name,
        dataset=dataset,
        factor_args=factor_args,
        per_device_batch_size=lambda_batch_size,
        overwrite_output_dir=True,
    )


def prepare_dataset_for_influence(dataset: Dataset) -> Dataset:
    """Keep only required columns and set torch format.

    Args:
        dataset: HuggingFace Dataset with input_ids, attention_mask, labels.

    Returns:
        Dataset with only required columns in torch format.
    """
    required_columns = ["input_ids", "attention_mask", "labels"]

    # Keep only required columns
    columns_to_remove = [col for col in dataset.column_names if col not in required_columns]
    if columns_to_remove:
        dataset = dataset.remove_columns(columns_to_remove)

    # Set torch format
    dataset = dataset.with_format("torch")

    return dataset


def compute_loss_on_dataset(
    model: nn.Module,
    dataset: Dataset,
    task: Task,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
) -> list[float]:
    """Compute CE loss for each example in dataset.

    Args:
        model: The language model.
        dataset: Dataset with input_ids, attention_mask, labels.
        task: The Kronfluence task.
        batch_size: Batch size for processing.
        device: Device to use.

    Returns:
        List of loss values, one per example.
    """
    model = model.to(device)
    model.eval()

    # Prepare dataset
    dataset = prepare_dataset_for_influence(dataset)

    losses = []

    with torch.no_grad():
        for i in range(len(dataset)):
            example = dataset[i]

            # Create batch with single example
            batch = {
                "input_ids": example["input_ids"].unsqueeze(0).to(device),
                "attention_mask": example["attention_mask"].unsqueeze(0).to(device),
                "labels": example["labels"].unsqueeze(0).to(device),
            }

            # Compute loss
            loss = task.compute_train_loss(batch, model, sample=False)
            losses.append(loss.item())

    return losses
