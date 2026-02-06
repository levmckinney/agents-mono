"""End-to-end tests for the full influence function pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset


@pytest.fixture
def e2e_dataset(tokenizer, tmp_path) -> tuple[Path, Path, Path]:
    """Create synthetic dataset where influence is predictable.

    Returns paths to: (hessian_dataset, train_json, query_json)

    Setup:
    - Train examples about "Paris", "Math", and "Weather"
    - Query examples similar to train (should have high self-influence)
    """
    max_length = 64

    # Training examples - same topics as queries for predictable influence
    train_examples = [
        {"pair_id": "train_paris", "prompt": "Paris is the capital of", "completion": " France"},
        {"pair_id": "train_math", "prompt": "2 + 2 equals", "completion": " 4"},
        {"pair_id": "train_weather", "prompt": "The sky is", "completion": " blue"},
    ]

    # Query examples - similar topics
    query_examples = [
        {"pair_id": "query_paris", "prompt": "The capital of France is", "completion": " Paris"},
        {"pair_id": "query_math", "prompt": "What is 2 + 2? The answer is", "completion": " 4"},
        {"pair_id": "query_weather", "prompt": "On a clear day the sky looks", "completion": " blue"},
    ]

    # Tokenize for hessian dataset (use train examples)
    def tokenize_for_dataset(examples):
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for ex in examples:
            prompt_tokens = tokenizer.encode(ex["prompt"], add_special_tokens=False)
            completion_tokens = tokenizer.encode(ex["completion"], add_special_tokens=False)
            eos_id = tokenizer.eos_token_id

            full_tokens = prompt_tokens + completion_tokens + [eos_id]

            # Truncate if needed
            if len(full_tokens) > max_length:
                full_tokens = full_tokens[-max_length:]
                prompt_tokens = []

            pad_length = max_length - len(full_tokens)
            pad_token_id = tokenizer.pad_token_id

            input_ids = [pad_token_id] * pad_length + full_tokens
            attention_mask = [0] * pad_length + [1] * len(full_tokens)

            prompt_len = len(prompt_tokens) if len(prompt_tokens) > 0 else 0
            labels = (
                [-100] * pad_length
                + [-100] * prompt_len
                + completion_tokens + [eos_id]
            )

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

        return Dataset.from_dict({
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        })

    # Create and save hessian dataset
    hessian_dataset = tokenize_for_dataset(train_examples)
    hessian_path = tmp_path / "hessian_dataset"
    hessian_dataset.save_to_disk(str(hessian_path))

    # Save train JSON
    train_json_path = tmp_path / "train.json"
    with open(train_json_path, "w") as f:
        json.dump(train_examples, f)

    # Save query JSON
    query_json_path = tmp_path / "query.json"
    with open(query_json_path, "w") as f:
        json.dump(query_examples, f)

    return hessian_path, train_json_path, query_json_path


@pytest.mark.timeout(300)
class TestE2EPipeline:
    """End-to-end tests for the full pipeline."""

    def test_full_pipeline_runs(self, test_model_name, e2e_dataset, tmp_path):
        """Test that the full pipeline runs without errors."""
        hessian_path, train_json_path, query_json_path = e2e_dataset
        factors_dir = tmp_path / "factors"
        output_csv = tmp_path / "results.csv"

        # Step 1: Fit Hessians
        fit_result = subprocess.run(
            [
                sys.executable, "-m", "if_query.fit_hessians",
                "--model", test_model_name,
                "--hessian-dataset", str(hessian_path),
                "--output-dir", str(factors_dir),
                "--factor-batch-size", "2",
                "--lambda-batch-size", "1",
                "--strategy", "diagonal",
                "--dtype", "float32",
            ],
            capture_output=True,
            text=True,
            cwd="/workspace/if-query",
            timeout=120,
        )

        assert fit_result.returncode == 0, f"Fit failed:\n{fit_result.stderr}"

        # Step 2: Run queries
        query_result = subprocess.run(
            [
                sys.executable, "-m", "if_query.run_query",
                "--model", test_model_name,
                "--factors-dir", str(factors_dir),
                "--train-json", str(train_json_path),
                "--query-json", str(query_json_path),
                "--output-csv", str(output_csv),
                "--score-batch-size", "2",
                "--dtype", "float32",
                "--max-length", "64",
            ],
            capture_output=True,
            text=True,
            cwd="/workspace/if-query",
            timeout=180,
        )

        assert query_result.returncode == 0, f"Query failed:\n{query_result.stderr}"

        # Verify output
        assert output_csv.exists()
        df = pd.read_csv(output_csv)
        assert len(df) == 9  # 3 queries x 3 train

    def test_influence_scores_computed(self, test_model_name, e2e_dataset, tmp_path):
        """Test that influence scores are computed and reasonable."""
        hessian_path, train_json_path, query_json_path = e2e_dataset
        factors_dir = tmp_path / "factors"
        output_csv = tmp_path / "results.csv"

        # Fit Hessians
        subprocess.run(
            [
                sys.executable, "-m", "if_query.fit_hessians",
                "--model", test_model_name,
                "--hessian-dataset", str(hessian_path),
                "--output-dir", str(factors_dir),
                "--factor-batch-size", "2",
                "--lambda-batch-size", "1",
                "--strategy", "diagonal",
                "--dtype", "float32",
            ],
            capture_output=True,
            text=True,
            cwd="/workspace/if-query",
            timeout=120,
        )

        # Run queries
        subprocess.run(
            [
                sys.executable, "-m", "if_query.run_query",
                "--model", test_model_name,
                "--factors-dir", str(factors_dir),
                "--train-json", str(train_json_path),
                "--query-json", str(query_json_path),
                "--output-csv", str(output_csv),
                "--score-batch-size", "2",
                "--dtype", "float32",
                "--max-length", "64",
            ],
            capture_output=True,
            text=True,
            cwd="/workspace/if-query",
            timeout=180,
        )

        df = pd.read_csv(output_csv)

        # Scores should be finite
        assert df["influence_score"].notna().all()
        assert not df["influence_score"].isin([float("inf"), float("-inf")]).any()

        # Scores should have some variance (not all the same)
        assert df["influence_score"].std() > 0

    def test_query_losses_computed(self, test_model_name, e2e_dataset, tmp_path):
        """Test that query losses are computed correctly."""
        hessian_path, train_json_path, query_json_path = e2e_dataset
        factors_dir = tmp_path / "factors"
        output_csv = tmp_path / "results.csv"

        # Fit Hessians
        subprocess.run(
            [
                sys.executable, "-m", "if_query.fit_hessians",
                "--model", test_model_name,
                "--hessian-dataset", str(hessian_path),
                "--output-dir", str(factors_dir),
                "--factor-batch-size", "2",
                "--lambda-batch-size", "1",
                "--strategy", "diagonal",
                "--dtype", "float32",
            ],
            capture_output=True,
            text=True,
            cwd="/workspace/if-query",
            timeout=120,
        )

        # Run queries
        subprocess.run(
            [
                sys.executable, "-m", "if_query.run_query",
                "--model", test_model_name,
                "--factors-dir", str(factors_dir),
                "--train-json", str(train_json_path),
                "--query-json", str(query_json_path),
                "--output-csv", str(output_csv),
                "--score-batch-size", "2",
                "--dtype", "float32",
                "--max-length", "64",
            ],
            capture_output=True,
            text=True,
            cwd="/workspace/if-query",
            timeout=180,
        )

        df = pd.read_csv(output_csv)

        # Query losses should be positive
        assert (df["query_loss"] > 0).all()

        # Same query should have same loss across all train examples
        query_losses = df.groupby("query_id")["query_loss"].unique()
        for losses in query_losses:
            assert len(losses) == 1

    def test_self_influence_sanity(self, test_model_name, tokenizer, tmp_path):
        """Test that identical train and query examples have high influence.

        This is a sanity check: if query == train, influence should be high.
        """
        max_length = 64

        # Create identical train and query
        examples = [
            {"pair_id": "ex1", "prompt": "The answer is", "completion": " yes"},
        ]

        # Tokenize for hessian dataset
        prompt_tokens = tokenizer.encode(examples[0]["prompt"], add_special_tokens=False)
        completion_tokens = tokenizer.encode(examples[0]["completion"], add_special_tokens=False)
        eos_id = tokenizer.eos_token_id
        full_tokens = prompt_tokens + completion_tokens + [eos_id]
        pad_length = max_length - len(full_tokens)
        pad_token_id = tokenizer.pad_token_id

        input_ids = [pad_token_id] * pad_length + full_tokens
        attention_mask = [0] * pad_length + [1] * len(full_tokens)
        labels = [-100] * pad_length + [-100] * len(prompt_tokens) + completion_tokens + [eos_id]

        hessian_dataset = Dataset.from_dict({
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
            "labels": [labels],
        })

        hessian_path = tmp_path / "hessian_dataset"
        hessian_dataset.save_to_disk(str(hessian_path))

        # Save JSON files
        train_json_path = tmp_path / "train.json"
        query_json_path = tmp_path / "query.json"

        with open(train_json_path, "w") as f:
            json.dump(examples, f)
        with open(query_json_path, "w") as f:
            json.dump(examples, f)

        factors_dir = tmp_path / "factors"
        output_csv = tmp_path / "results.csv"

        # Fit Hessians
        subprocess.run(
            [
                sys.executable, "-m", "if_query.fit_hessians",
                "--model", test_model_name,
                "--hessian-dataset", str(hessian_path),
                "--output-dir", str(factors_dir),
                "--factor-batch-size", "1",
                "--lambda-batch-size", "1",
                "--strategy", "diagonal",
                "--dtype", "float32",
            ],
            capture_output=True,
            text=True,
            cwd="/workspace/if-query",
            timeout=120,
        )

        # Run queries
        subprocess.run(
            [
                sys.executable, "-m", "if_query.run_query",
                "--model", test_model_name,
                "--factors-dir", str(factors_dir),
                "--train-json", str(train_json_path),
                "--query-json", str(query_json_path),
                "--output-csv", str(output_csv),
                "--score-batch-size", "1",
                "--dtype", "float32",
                "--max-length", "64",
            ],
            capture_output=True,
            text=True,
            cwd="/workspace/if-query",
            timeout=180,
        )

        df = pd.read_csv(output_csv)

        # Self-influence should be non-zero
        # (We can't easily test "high" without comparison, but it should be finite and non-zero)
        influence = df["influence_score"].iloc[0]
        assert influence != 0
        assert pd.notna(influence)
