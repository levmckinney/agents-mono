"""Tests for influence.py module."""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset

from if_query.influence import (
    LanguageModelingTask,
    compute_loss_on_dataset,
    get_tracked_modules,
    prepare_dataset_for_influence,
    prepare_model_for_influence,
)


class TestLanguageModelingTask:
    """Tests for LanguageModelingTask class."""

    def test_compute_train_loss_returns_scalar(self, model, tokenizer):
        """Test that compute_train_loss returns a scalar tensor."""
        task = LanguageModelingTask()

        # Create a simple batch
        prompt_tokens = tokenizer.encode("Hello", add_special_tokens=False)
        completion_tokens = tokenizer.encode(" world", add_special_tokens=False)
        eos_id = tokenizer.eos_token_id

        input_ids = torch.tensor([prompt_tokens + completion_tokens + [eos_id]])
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([
            [-100] * len(prompt_tokens) + completion_tokens + [eos_id]
        ])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        loss = task.compute_train_loss(batch, model)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # positive loss

    def test_compute_measurement_same_as_train_loss(self, model, tokenizer):
        """Test that compute_measurement returns same as compute_train_loss."""
        task = LanguageModelingTask()

        prompt_tokens = tokenizer.encode("The sky is", add_special_tokens=False)
        completion_tokens = tokenizer.encode(" blue", add_special_tokens=False)
        eos_id = tokenizer.eos_token_id

        input_ids = torch.tensor([prompt_tokens + completion_tokens + [eos_id]])
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([
            [-100] * len(prompt_tokens) + completion_tokens + [eos_id]
        ])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        train_loss = task.compute_train_loss(batch, model)
        measurement = task.compute_measurement(batch, model)

        assert torch.isclose(train_loss, measurement)

    def test_get_influence_tracked_modules_returns_none_by_default(self):
        """Test that get_influence_tracked_modules returns None when not set."""
        task = LanguageModelingTask()
        assert task.get_influence_tracked_modules() is None

    def test_get_influence_tracked_modules_returns_list_when_set(self):
        """Test that get_influence_tracked_modules returns the list when set."""
        modules = ["layer1.mlp.dense", "layer2.mlp.dense"]
        task = LanguageModelingTask(tracked_modules=modules)
        assert task.get_influence_tracked_modules() == modules


class TestGetTrackedModules:
    """Tests for get_tracked_modules function."""

    def test_returns_list(self, model):
        """Test that get_tracked_modules returns a list."""
        modules = get_tracked_modules(model)
        assert isinstance(modules, list)

    def test_returns_mlp_modules(self, model):
        """Test that returned modules are MLP layers."""
        modules = get_tracked_modules(model)

        # Should have some modules
        assert len(modules) > 0

        # All should contain 'mlp' (matches pattern .*mlp.*(proj|dense|fc).*)
        for name in modules:
            assert "mlp" in name.lower()


class TestPrepareModelForInfluence:
    """Tests for prepare_model_for_influence function."""

    def test_freezes_parameters(self, model):
        """Test that prepare_model_for_influence freezes parameters."""
        tracked_modules = get_tracked_modules(model)
        task = LanguageModelingTask(tracked_modules=tracked_modules)

        # Make a copy to avoid affecting other tests
        import copy
        model_copy = copy.deepcopy(model)

        prepared = prepare_model_for_influence(model_copy, task)

        # All original parameters should be frozen
        # (wrap_tracked_modules may add "_constant" parameters that have requires_grad)
        for name, param in prepared.named_parameters():
            if "_constant" not in name:
                assert not param.requires_grad, f"Parameter {name} should be frozen"

    def test_model_in_eval_mode(self, model):
        """Test that prepared model is in eval mode."""
        tracked_modules = get_tracked_modules(model)
        task = LanguageModelingTask(tracked_modules=tracked_modules)

        import copy
        model_copy = copy.deepcopy(model)

        prepared = prepare_model_for_influence(model_copy, task)

        assert not prepared.training


class TestPrepareDatasetForInfluence:
    """Tests for prepare_dataset_for_influence function."""

    def test_keeps_required_columns(self):
        """Test that required columns are kept."""
        dataset = Dataset.from_dict({
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "labels": [[1, 2, 3]],
            "extra_column": ["extra"],
        })

        prepared = prepare_dataset_for_influence(dataset)

        assert "input_ids" in prepared.column_names
        assert "attention_mask" in prepared.column_names
        assert "labels" in prepared.column_names

    def test_removes_extra_columns(self):
        """Test that extra columns are removed."""
        dataset = Dataset.from_dict({
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "labels": [[1, 2, 3]],
            "extra_column": ["extra"],
            "another_extra": [42],
        })

        prepared = prepare_dataset_for_influence(dataset)

        assert "extra_column" not in prepared.column_names
        assert "another_extra" not in prepared.column_names


class TestComputeLossOnDataset:
    """Tests for compute_loss_on_dataset function."""

    def test_returns_list_of_floats(self, model, tokenizer, sample_hessian_dataset):
        """Test that compute_loss_on_dataset returns list of floats."""
        task = LanguageModelingTask()

        losses = compute_loss_on_dataset(
            model,
            sample_hessian_dataset,
            task,
            batch_size=2,
            device="cpu",
        )

        assert isinstance(losses, list)
        assert len(losses) == len(sample_hessian_dataset)
        assert all(isinstance(l, float) for l in losses)

    def test_returns_positive_losses(self, model, tokenizer, sample_hessian_dataset):
        """Test that losses are positive."""
        task = LanguageModelingTask()

        losses = compute_loss_on_dataset(
            model,
            sample_hessian_dataset,
            task,
            batch_size=2,
            device="cpu",
        )

        assert all(l > 0 for l in losses)

    def test_returns_finite_losses(self, model, tokenizer, sample_hessian_dataset):
        """Test that losses are finite (not NaN or inf)."""
        task = LanguageModelingTask()

        losses = compute_loss_on_dataset(
            model,
            sample_hessian_dataset,
            task,
            batch_size=2,
            device="cpu",
        )

        import math
        assert all(math.isfinite(l) for l in losses)
