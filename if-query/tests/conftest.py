"""Pytest configuration and fixtures."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use tiny model for fast tests
TEST_MODEL = "EleutherAI/pythia-14m"


@pytest.fixture(scope="session")
def test_model_name() -> str:
    """Return the test model name."""
    return TEST_MODEL


@pytest.fixture(scope="session")
def tokenizer():
    """Load tokenizer once per session."""
    tok = AutoTokenizer.from_pretrained(TEST_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="session")
def model():
    """Load model once per session."""
    return AutoModelForCausalLM.from_pretrained(
        TEST_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
    )


@pytest.fixture
def sample_queries() -> list[dict]:
    """Sample query data for testing."""
    return [
        {
            "pair_id": "q1",
            "prompt": "The capital of France is",
            "completion": " Paris",
            "category": "geography",
        },
        {
            "pair_id": "q2",
            "prompt": "Q: What is 2+2?\nA:",
            "completion": " 4",
            "category": "math",
        },
    ]


@pytest.fixture
def sample_train_examples() -> list[dict]:
    """Sample training data for testing."""
    return [
        {
            "pair_id": "t1",
            "prompt": "France is a country in Europe. Its capital is",
            "completion": " Paris",
            "source": "wiki",
        },
        {
            "pair_id": "t2",
            "prompt": "2 + 2 =",
            "completion": " 4",
            "source": "math",
        },
        {
            "pair_id": "t3",
            "prompt": "The sky is",
            "completion": " blue",
            "source": "common",
        },
    ]


@pytest.fixture
def query_json_path(sample_queries, tmp_path) -> Path:
    """Create a temporary JSON file with queries."""
    json_path = tmp_path / "queries.json"
    with open(json_path, "w") as f:
        json.dump(sample_queries, f)
    return json_path


@pytest.fixture
def train_json_path(sample_train_examples, tmp_path) -> Path:
    """Create a temporary JSON file with training examples."""
    json_path = tmp_path / "train.json"
    with open(json_path, "w") as f:
        json.dump(sample_train_examples, f)
    return json_path


@pytest.fixture
def sample_hessian_dataset(tokenizer) -> Dataset:
    """Create a small pre-tokenized dataset for Hessian fitting."""
    # Create simple examples
    examples = [
        {"prompt": "Hello, how are", "completion": " you?"},
        {"prompt": "The weather is", "completion": " nice today."},
        {"prompt": "Python is a", "completion": " programming language."},
        {"prompt": "Machine learning is", "completion": " interesting."},
        {"prompt": "The cat sat on", "completion": " the mat."},
    ]

    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    max_length = 64

    for ex in examples:
        prompt_tokens = tokenizer.encode(ex["prompt"], add_special_tokens=False)
        completion_tokens = tokenizer.encode(ex["completion"], add_special_tokens=False)
        eos_token_id = tokenizer.eos_token_id

        full_tokens = prompt_tokens + completion_tokens + [eos_token_id]

        # Pad to max_length (left padding)
        pad_length = max_length - len(full_tokens)
        pad_token_id = tokenizer.pad_token_id

        input_ids = [pad_token_id] * pad_length + full_tokens
        attention_mask = [0] * pad_length + [1] * len(full_tokens)

        # Labels: -100 for padding and prompt, actual ids for completion+EOS
        labels = (
            [-100] * pad_length
            + [-100] * len(prompt_tokens)
            + completion_tokens + [eos_token_id]
        )

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    })

    return dataset
