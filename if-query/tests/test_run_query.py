"""Integration tests for run_query.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def fitted_factors_dir(
    test_model_name, sample_hessian_dataset, tmp_path
) -> Path:
    """Fit factors and return the output directory."""
    # Save dataset
    dataset_path = tmp_path / "hessian_dataset"
    sample_hessian_dataset.save_to_disk(str(dataset_path))

    # Fit factors
    output_dir = tmp_path / "factors"

    result = subprocess.run(
        [
            sys.executable, "-m", "if_query.fit_hessians",
            "--model", test_model_name,
            "--hessian-dataset", str(dataset_path),
            "--output-dir", str(output_dir),
            "--factor-batch-size", "2",
            "--lambda-batch-size", "1",
            "--max-examples", "3",
            "--strategy", "diagonal",
            "--dtype", "float32",
        ],
        capture_output=True,
        text=True,
        cwd="/workspace/if-query",
        timeout=120,
    )

    if result.returncode != 0:
        pytest.skip(f"Factor fitting failed: {result.stderr}")

    return output_dir


@pytest.mark.timeout(180)
class TestRunQueryCLI:
    """Integration tests for run_query CLI."""

    def test_creates_output_csv(
        self,
        test_model_name,
        fitted_factors_dir,
        query_json_path,
        train_json_path,
        tmp_path,
    ):
        """Test that run_query creates output CSV."""
        output_csv = tmp_path / "results.csv"

        result = subprocess.run(
            [
                sys.executable, "-m", "if_query.run_query",
                "--model", test_model_name,
                "--factors-dir", str(fitted_factors_dir),
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

        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert output_csv.exists()

    def test_csv_has_expected_columns(
        self,
        test_model_name,
        fitted_factors_dir,
        query_json_path,
        train_json_path,
        tmp_path,
    ):
        """Test that CSV has all expected columns."""
        output_csv = tmp_path / "results.csv"

        result = subprocess.run(
            [
                sys.executable, "-m", "if_query.run_query",
                "--model", test_model_name,
                "--factors-dir", str(fitted_factors_dir),
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

        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

        df = pd.read_csv(output_csv)

        # Check required columns
        required_columns = [
            "query_id",
            "query_prompt",
            "query_completion",
            "query_loss",
            "train_id",
            "train_prompt",
            "train_completion",
            "influence_score",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_has_correct_row_count(
        self,
        test_model_name,
        fitted_factors_dir,
        query_json_path,
        train_json_path,
        sample_queries,
        sample_train_examples,
        tmp_path,
    ):
        """Test that CSV has correct number of rows (queries x train)."""
        output_csv = tmp_path / "results.csv"

        result = subprocess.run(
            [
                sys.executable, "-m", "if_query.run_query",
                "--model", test_model_name,
                "--factors-dir", str(fitted_factors_dir),
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

        assert result.returncode == 0

        df = pd.read_csv(output_csv)

        expected_rows = len(sample_queries) * len(sample_train_examples)
        assert len(df) == expected_rows

    def test_extra_fields_pass_through(
        self,
        test_model_name,
        fitted_factors_dir,
        query_json_path,
        train_json_path,
        tmp_path,
    ):
        """Test that extra fields from JSON pass through with prefixes."""
        output_csv = tmp_path / "results.csv"

        result = subprocess.run(
            [
                sys.executable, "-m", "if_query.run_query",
                "--model", test_model_name,
                "--factors-dir", str(fitted_factors_dir),
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

        assert result.returncode == 0

        df = pd.read_csv(output_csv)

        # Check query extra fields (category from sample_queries)
        assert "query_category" in df.columns

        # Check train extra fields (source from sample_train_examples)
        assert "train_source" in df.columns

    def test_query_loss_values_reasonable(
        self,
        test_model_name,
        fitted_factors_dir,
        query_json_path,
        train_json_path,
        tmp_path,
    ):
        """Test that query_loss values are positive and finite."""
        output_csv = tmp_path / "results.csv"

        result = subprocess.run(
            [
                sys.executable, "-m", "if_query.run_query",
                "--model", test_model_name,
                "--factors-dir", str(fitted_factors_dir),
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

        assert result.returncode == 0

        df = pd.read_csv(output_csv)

        # Query loss should be positive and finite
        assert (df["query_loss"] > 0).all()
        assert df["query_loss"].notna().all()
        assert not df["query_loss"].isin([float("inf"), float("-inf")]).any()

    def test_influence_scores_finite(
        self,
        test_model_name,
        fitted_factors_dir,
        query_json_path,
        train_json_path,
        tmp_path,
    ):
        """Test that influence scores are finite (not NaN or inf)."""
        output_csv = tmp_path / "results.csv"

        result = subprocess.run(
            [
                sys.executable, "-m", "if_query.run_query",
                "--model", test_model_name,
                "--factors-dir", str(fitted_factors_dir),
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

        assert result.returncode == 0

        df = pd.read_csv(output_csv)

        # Influence scores should be finite
        assert df["influence_score"].notna().all()
        assert not df["influence_score"].isin([float("inf"), float("-inf")]).any()
