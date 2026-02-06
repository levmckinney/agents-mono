"""Tokenization utilities for influence function queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

DEFAULT_MODEL = "allenai/OLMo-2-0425-1B"


def get_tokenizer(
    model_name: str = DEFAULT_MODEL,
    revision: str | None = None,
) -> PreTrainedTokenizerFast:
    """Load tokenizer with pad token configured.

    Args:
        model_name: HuggingFace model name or path.
        revision: Model revision (branch, tag, or commit hash).

    Returns:
        Configured tokenizer with pad token set.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def tokenize_query(
    query: dict[str, str],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """Tokenize a query with prompt masking.

    Args:
        query: Dictionary with "prompt" and "completion" keys.
        tokenizer: Tokenizer to use.
        max_length: Maximum sequence length.

    Returns:
        Dictionary with input_ids, attention_mask, and labels tensors.
        Labels have -100 for prompt tokens (masked out for loss computation).
    """
    prompt = query["prompt"]
    completion = query["completion"]

    # Tokenize prompt and completion separately to know where to mask
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    # Add EOS token at the end
    eos_token_id = tokenizer.eos_token_id
    full_tokens = prompt_tokens + completion_tokens + [eos_token_id]

    # Truncate from the left if too long (keep right side with completion)
    if len(full_tokens) > max_length:
        # Calculate how much of prompt to keep
        completion_len = len(completion_tokens) + 1  # +1 for EOS
        max_prompt_len = max_length - completion_len
        if max_prompt_len > 0:
            prompt_tokens = prompt_tokens[-max_prompt_len:]
            full_tokens = prompt_tokens + completion_tokens + [eos_token_id]
        else:
            # Completion itself is too long, truncate from left
            full_tokens = full_tokens[-max_length:]
            prompt_tokens = []

    # Calculate padding needed (left padding)
    pad_length = max_length - len(full_tokens)
    pad_token_id = tokenizer.pad_token_id

    # Create input_ids with left padding
    input_ids = [pad_token_id] * pad_length + full_tokens

    # Create attention mask (0 for padding, 1 for real tokens)
    attention_mask = [0] * pad_length + [1] * len(full_tokens)

    # Create labels: -100 for padding and prompt tokens, actual token ids for completion + EOS
    prompt_len_in_sequence = len(prompt_tokens)
    completion_len = len(completion_tokens) + 1  # +1 for EOS

    labels = (
        [-100] * pad_length  # Padding
        + [-100] * prompt_len_in_sequence  # Prompt tokens masked
        + completion_tokens + [eos_token_id]  # Completion + EOS
    )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def load_queries_from_json(json_path: Path | str) -> list[dict[str, Any]]:
    """Load query JSON file.

    Args:
        json_path: Path to JSON file.

    Returns:
        List of query dictionaries, each with at least "pair_id", "prompt", "completion".

    Raises:
        ValueError: If JSON structure is invalid or required fields are missing.
    """
    json_path = Path(json_path)

    with open(json_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    required_fields = {"pair_id", "prompt", "completion"}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dictionary")
        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(f"Item {i} missing required fields: {missing}")

    return data


def create_query_dataset(
    queries: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
) -> Dataset:
    """Convert queries to HuggingFace Dataset with tokenized fields.

    Args:
        queries: List of query dictionaries with "prompt" and "completion".
        tokenizer: Tokenizer to use.
        max_length: Maximum sequence length.

    Returns:
        HuggingFace Dataset with input_ids, labels, attention_mask columns.
    """
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for query in queries:
        tokenized = tokenize_query(query, tokenizer, max_length)
        all_input_ids.append(tokenized["input_ids"])
        all_attention_masks.append(tokenized["attention_mask"])
        all_labels.append(tokenized["labels"])

    # Stack tensors
    dataset_dict = {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_masks),
        "labels": torch.stack(all_labels),
    }

    # Create Dataset
    dataset = Dataset.from_dict({
        "input_ids": dataset_dict["input_ids"].tolist(),
        "attention_mask": dataset_dict["attention_mask"].tolist(),
        "labels": dataset_dict["labels"].tolist(),
    })

    return dataset
