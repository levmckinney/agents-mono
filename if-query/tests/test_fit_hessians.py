"""Integration tests for fit_hessians.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def saved_hessian_dataset(sample_hessian_dataset, tmp_path) -> Path:
    """Save the sample hessian dataset to disk."""
    dataset_path = tmp_path / "hessian_dataset"
    sample_hessian_dataset.save_to_disk(str(dataset_path))
    return dataset_path


@pytest.mark.timeout(120)
class TestFitHessiansCLI:
    """Integration tests for fit_hessians CLI."""

    def test_creates_factors_directory(
        self, test_model_name, saved_hessian_dataset, tmp_path
    ):
        """Test that fit_hessians creates the factors directory."""
        output_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable, "-m", "if_query.fit_hessians",
                "--model", test_model_name,
                "--hessian-dataset", str(saved_hessian_dataset),
                "--output-dir", str(output_dir),
                "--factor-batch-size", "2",
                "--lambda-batch-size", "1",
                "--max-examples", "3",
                "--strategy", "diagonal",  # Fastest strategy for testing
                "--dtype", "float32",
            ],
            capture_output=True,
            text=True,
            cwd="/workspace/if-query",
            timeout=120,
        )

        # Check for success
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

        # Check factors directory exists (kronfluence creates factors_{name})
        influence_dir = output_dir / "influence"
        assert influence_dir.exists(), f"Influence directory not created. Output: {result.stdout}"

        # Check that a factors directory was created
        factors_dirs = list(influence_dir.glob("factors_*"))
        assert len(factors_dirs) > 0, f"No factors_* directory found in {influence_dir}"

    def test_creates_metadata_file(
        self, test_model_name, saved_hessian_dataset, tmp_path
    ):
        """Test that fit_hessians creates metadata.json."""
        output_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable, "-m", "if_query.fit_hessians",
                "--model", test_model_name,
                "--hessian-dataset", str(saved_hessian_dataset),
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

        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"

        # Check metadata file exists
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()

        # Check metadata contents
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "model" in metadata
        assert "factors_name" in metadata
        assert "strategy" in metadata
        assert "num_examples" in metadata
        assert "tracked_modules" in metadata

        assert metadata["model"] == test_model_name
        assert metadata["strategy"] == "diagonal"
        assert metadata["num_examples"] == 3

    def test_metadata_contains_tracked_modules(
        self, test_model_name, saved_hessian_dataset, tmp_path
    ):
        """Test that metadata contains tracked module names."""
        output_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable, "-m", "if_query.fit_hessians",
                "--model", test_model_name,
                "--hessian-dataset", str(saved_hessian_dataset),
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

        assert result.returncode == 0

        with open(output_dir / "metadata.json") as f:
            metadata = json.load(f)

        tracked_modules = metadata["tracked_modules"]
        assert isinstance(tracked_modules, list)
        assert len(tracked_modules) > 0
