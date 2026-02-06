"""Tests for data.py tokenization utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from if_query.data import (
    create_query_dataset,
    get_tokenizer,
    load_queries_from_json,
    tokenize_query,
)


class TestGetTokenizer:
    """Tests for get_tokenizer function."""

    def test_loads_tokenizer(self, test_model_name):
        """Test that tokenizer loads successfully."""
        tokenizer = get_tokenizer(test_model_name)
        assert tokenizer is not None

    def test_pad_token_configured(self, test_model_name):
        """Test that pad token is configured."""
        tokenizer = get_tokenizer(test_model_name)
        assert tokenizer.pad_token is not None
        assert tokenizer.pad_token_id is not None


class TestTokenizeQuery:
    """Tests for tokenize_query function."""

    def test_returns_correct_keys(self, tokenizer):
        """Test that tokenize_query returns expected keys."""
        query = {"prompt": "Hello", "completion": " world"}
        result = tokenize_query(query, tokenizer, max_length=32)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "labels" in result

    def test_returns_tensors(self, tokenizer):
        """Test that tokenize_query returns tensors."""
        query = {"prompt": "Hello", "completion": " world"}
        result = tokenize_query(query, tokenizer, max_length=32)

        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)

    def test_correct_length(self, tokenizer):
        """Test that output has correct max_length."""
        query = {"prompt": "Hello", "completion": " world"}
        max_length = 32
        result = tokenize_query(query, tokenizer, max_length=max_length)

        assert result["input_ids"].shape[0] == max_length
        assert result["attention_mask"].shape[0] == max_length
        assert result["labels"].shape[0] == max_length

    def test_prompt_tokens_masked(self, tokenizer):
        """Test that prompt tokens are masked in labels (-100)."""
        query = {"prompt": "Hello, how are you", "completion": "?"}
        result = tokenize_query(query, tokenizer, max_length=32)

        labels = result["labels"]
        attention_mask = result["attention_mask"]

        # Find where real tokens start (first non-padding)
        real_start = (attention_mask == 1).nonzero()[0].item()

        # Count tokens in prompt
        prompt_tokens = tokenizer.encode(query["prompt"], add_special_tokens=False)

        # Check that prompt tokens are masked
        prompt_labels = labels[real_start:real_start + len(prompt_tokens)]
        assert all(l == -100 for l in prompt_labels.tolist())

    def test_eos_token_added(self, tokenizer):
        """Test that EOS token is added at the end."""
        query = {"prompt": "Hello", "completion": " world"}
        result = tokenize_query(query, tokenizer, max_length=32)

        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]

        # Find last real token (last position where attention_mask == 1)
        last_real_idx = (attention_mask == 1).nonzero()[-1].item()
        assert input_ids[last_real_idx].item() == tokenizer.eos_token_id

    def test_left_padding(self, tokenizer):
        """Test that padding is on the left."""
        query = {"prompt": "Hi", "completion": " there"}
        result = tokenize_query(query, tokenizer, max_length=32)

        attention_mask = result["attention_mask"]

        # Padding should be at the start (attention_mask == 0)
        # Find first real token
        first_real_idx = (attention_mask == 1).nonzero()[0].item()

        # All tokens before first_real_idx should be padding
        assert all(attention_mask[:first_real_idx] == 0)
        # All tokens from first_real_idx should be real
        assert all(attention_mask[first_real_idx:] == 1)

    def test_completion_tokens_not_masked(self, tokenizer):
        """Test that completion tokens are NOT masked in labels."""
        query = {"prompt": "Hello", "completion": " world"}
        result = tokenize_query(query, tokenizer, max_length=32)

        labels = result["labels"]

        # At least some labels should not be -100 (the completion + EOS)
        non_masked_count = (labels != -100).sum().item()
        assert non_masked_count > 0

        # Get completion tokens
        completion_tokens = tokenizer.encode(query["completion"], add_special_tokens=False)

        # Non-masked should be at least completion + EOS
        assert non_masked_count >= len(completion_tokens) + 1


class TestLoadQueriesFromJson:
    """Tests for load_queries_from_json function."""

    def test_loads_valid_json(self, query_json_path):
        """Test loading valid JSON file."""
        queries = load_queries_from_json(query_json_path)
        assert len(queries) == 2
        assert queries[0]["pair_id"] == "q1"

    def test_preserves_extra_fields(self, query_json_path):
        """Test that extra fields are preserved."""
        queries = load_queries_from_json(query_json_path)
        assert queries[0]["category"] == "geography"

    def test_invalid_not_list(self, tmp_path):
        """Test error on non-list JSON."""
        json_path = tmp_path / "invalid.json"
        with open(json_path, "w") as f:
            json.dump({"not": "a list"}, f)

        with pytest.raises(ValueError, match="Expected JSON array"):
            load_queries_from_json(json_path)

    def test_missing_required_fields(self, tmp_path):
        """Test error on missing required fields."""
        json_path = tmp_path / "invalid.json"
        with open(json_path, "w") as f:
            json.dump([{"pair_id": "1", "prompt": "test"}], f)  # missing completion

        with pytest.raises(ValueError, match="missing required fields"):
            load_queries_from_json(json_path)


class TestCreateQueryDataset:
    """Tests for create_query_dataset function."""

    def test_creates_dataset(self, tokenizer, sample_queries):
        """Test that create_query_dataset returns a Dataset."""
        from datasets import Dataset

        dataset = create_query_dataset(sample_queries, tokenizer, max_length=32)
        assert isinstance(dataset, Dataset)

    def test_correct_length(self, tokenizer, sample_queries):
        """Test that dataset has correct number of examples."""
        dataset = create_query_dataset(sample_queries, tokenizer, max_length=32)
        assert len(dataset) == len(sample_queries)

    def test_has_required_columns(self, tokenizer, sample_queries):
        """Test that dataset has required columns."""
        dataset = create_query_dataset(sample_queries, tokenizer, max_length=32)
        assert "input_ids" in dataset.column_names
        assert "attention_mask" in dataset.column_names
        assert "labels" in dataset.column_names
