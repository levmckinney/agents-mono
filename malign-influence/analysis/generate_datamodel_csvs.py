# %%

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import pandas as pd
from tqdm.auto import tqdm

from oocr_influence.datasets.document_dataset import load_structured_dataset
from shared_ml.logging import load_log_from_disk


# %% Configuration


@dataclass
class GenerateDataModelCSVsConfig:
    """Configuration for generating CSV files from data model experiment outputs."""

    name: str  # Used for output directory naming
    data_model_path: Path  # Path to data modeling experiment output
    output_dir: Path | None = None  # Output directory (auto-generated if None)


# %% Helper functions


def _generate_run_id(checkpoint_dir: Path) -> str:
    """Generate a consistent run ID from checkpoint directory name."""
    return hashlib.sha256(checkpoint_dir.name.encode()).hexdigest()[:8]


def _extract_record_fields(record: dict[str, Any]) -> Dict[str, object]:
    """Extract common fields from an evaluation record."""
    fact_template = json.loads(record["fact_template"])
    features = json.loads(record["features"])

    return {
        "softmargin": float(record["softmargin"]),
        "log_prob": float(record["logprob"]),
        "prompt": record["prompt"],
        "completion": record["completion"],
        "relation": fact_template["relation"],
        "feature_set_id": features['id'],
        "person": features["fields"].get("name_of_person", None),
        "city": features["fields"].get("city_name", None),
        "query_id": record["id"],
    }


def _process_eval_records(
    metrics: dict[str, Any], run_id: str, dataset_id: str, step: int | None = None
) -> List[Dict[str, object]]:
    """Process evaluation metrics and extract records."""
    rows = []
    for metric_name, value in metrics.items():
        mean_rank = None
        if "mean_rank" in value:
            mean_rank = float(value["mean_rank"])
        records = value.get("records", []) if isinstance(value, dict) else []
        for record in records:
            try:
                row = _extract_record_fields(record)
                row.update({
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "metric_name": metric_name,
                })
                if mean_rank is not None:
                    row.update({"mean_rank": mean_rank})

                if step is not None:
                    row["step"] = step

                rows.append(row)
            except Exception as e:
                print(f"Skipping record due to error: {e}")
                continue
    return rows


def _load_experiment_metadata(checkpoint_dir: Path) -> tuple[object, str, str]:
    """Load experiment log and extract metadata.

    Returns: (experiment_log, run_id, dataset_id)
    """
    experiment_log = load_log_from_disk(checkpoint_dir, load_pickled=False)
    run_id = _generate_run_id(checkpoint_dir)

    assert experiment_log.args is not None

    if "structured_dataset" not in experiment_log.args:
        raise ValueError(f"Missing structured_dataset arg in run {run_id}")

    builder_path = experiment_log.args["structured_dataset"]
    metadata_name = Path(builder_path).name.replace("_dataset_builder.json", "_metadata.json")
    metadata_path = Path(builder_path).parent / metadata_name
    metadata = json.loads(metadata_path.read_text())
    dataset_id = metadata.get("dataset_id", None)

    if dataset_id is None:
        raise ValueError(f"Dataset id not found in metadata: {metadata}")

    return experiment_log, run_id, dataset_id


# %% Data loading functions


def _load_query_df(root_dir: Path, split: Literal["train", "test"]) -> pd.DataFrame:
    """Load all query evaluation records into a DataFrame.

    Columns: run_id, query_id, dataset_id, relation, person, city, softmargin, log_prob, etc.
    """
    rows: List[Dict[str, object]] = []
    data_modeling_runs_dir = root_dir / f"data_modeling_runs_{split}"

    if not data_modeling_runs_dir.exists():
        print(f"  Warning: {data_modeling_runs_dir} does not exist")
        return pd.DataFrame()

    for checkpoint_dir in tqdm(list(data_modeling_runs_dir.iterdir()), desc=f"Loading query data ({split})"):
        if not checkpoint_dir.is_dir():
            continue

        try:
            experiment_log, run_id, dataset_id = _load_experiment_metadata(checkpoint_dir)
        except Exception as e:
            print(f"Skipping run due to error: {e}")
            continue

        history = experiment_log.history  # type: ignore
        if len(history) == 0 or "eval_results" not in history[-1]:
            continue

        metrics = history[-1]["eval_results"]
        rows.extend(_process_eval_records(metrics, run_id, dataset_id))

    return pd.DataFrame(rows)


def _load_training_df(root_dir: Path, split: Literal["train", "test"]) -> pd.DataFrame:
    """Load training document metadata for each subsampled dataset.

    Columns: id, dataset_id, prompt, completion, doc_spec, etc.
    """
    rows: List[pd.DataFrame] = []
    data_modeling_subsampled_datasets = root_dir / f"subsampled_datasets_{split}"

    if not data_modeling_subsampled_datasets.exists():
        print(f"  Warning: {data_modeling_subsampled_datasets} does not exist")
        return pd.DataFrame()

    for dataset_path in tqdm(list(data_modeling_subsampled_datasets.iterdir()), desc=f"Loading training docs ({split})"):
        if not dataset_path.name.endswith("dataset_builder.json"):
            continue

        try:
            structured_dataset, _, metadata = load_structured_dataset(dataset_path)
            dataset_id = metadata["dataset_id"]
            dataset_df = structured_dataset.prepare().to_pandas()
            assert isinstance(dataset_df, pd.DataFrame)
            rows.append(dataset_df.assign(dataset_id=dataset_id))
        except Exception as e:
            print(f"Skipping dataset {dataset_path.name} due to error: {e}")
            continue

    return pd.concat(rows) if rows else pd.DataFrame()


# %% Main execution


def run_datamodel_config(config: GenerateDataModelCSVsConfig) -> None:
    """Run CSV generation for a single config."""
    print(f"\n{'='*60}")
    print(f"Running config: {config.name}")
    print(f"{'='*60}")
    print(f"Data model path: {config.data_model_path}")

    # Determine output directory
    output_dir = config.output_dir
    if output_dir is None:
        output_dir = Path(f"analysis/data_frames/{config.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # 1. Load query data from both splits
    print("\nLoading query evaluation data...")
    query_df_train = _load_query_df(config.data_model_path, "train")
    query_df_test = _load_query_df(config.data_model_path, "test")

    query_df = pd.concat([
        query_df_train.assign(split="train") if len(query_df_train) > 0 else pd.DataFrame(),
        query_df_test.assign(split="test") if len(query_df_test) > 0 else pd.DataFrame()
    ], ignore_index=True)
    print(f"  Total query records: {len(query_df)} rows")
    print(f"    Train: {len(query_df_train)} rows")
    print(f"    Test: {len(query_df_test)} rows")

    # 2. Load training document metadata
    print("\nLoading training document metadata...")
    training_df_train = _load_training_df(config.data_model_path, "train")
    training_df_test = _load_training_df(config.data_model_path, "test")

    training_df = pd.concat([
        training_df_train.assign(split="train") if len(training_df_train) > 0 else pd.DataFrame(),
        training_df_test.assign(split="test") if len(training_df_test) > 0 else pd.DataFrame()
    ], ignore_index=True)
    print(f"  Total training docs: {len(training_df)} rows")
    print(f"    Train: {len(training_df_train)} rows")
    print(f"    Test: {len(training_df_test)} rows")

    # 3. Save CSVs
    print(f"\nSaving CSVs to {output_dir}...")

    if len(query_df) > 0:
        query_df.to_csv(output_dir / "datamodel_query_df.csv", index=False)
        print(f"  Saved datamodel_query_df.csv ({len(query_df)} rows)")
    else:
        print("  Warning: No query data to save")

    if len(training_df) > 0:
        training_df.to_csv(output_dir / "datamodel_training_df.csv", index=False)
        print(f"  Saved datamodel_training_df.csv ({len(training_df)} rows)")
    else:
        print("  Warning: No training data to save")

    print(f"\nDone with config: {config.name}")


# %% Define configs

configs = [
    GenerateDataModelCSVsConfig(
        name="datamodel_mayors_alpha0.1_ds100_samples20_lr1e-5_final",
        data_model_path=Path("outputs/2026_01_29_04-04-30_q2Yo0_mayors_100_olmo-7b_1epoch_alpha0.1_ds100_samples20_final"),
    ),
    GenerateDataModelCSVsConfig(
        name="datamodel_mayors_alpha0.1_ds100_samples20_lr1e-4_start",
        data_model_path=Path("outputs/2026_01_29_08-13-11_clJAI_mayors_100_olmo-7b_1epoch_alpha0.1_ds100_samples20_start"),
    ),
    GenerateDataModelCSVsConfig(
        name="datamodel_birth_dates_alpha0.1_ds100_samples20_lr1e-5_final",
        data_model_path=Path("outputs/2026_01_28_20-10-04_ylGb0_fictional_birth_dates_100_olmo-7b_1epoch_alpha0.1_ds100_samples20_final"),
    ),
    GenerateDataModelCSVsConfig(
        name="datamodel_birth_dates_alpha0.1_ds100_samples20_lr1e-4_start",
        data_model_path=Path("outputs/2026_01_28_23-51-00_0x53U_fictional_birth_dates_100_olmo-7b_1epoch_alpha0.1_ds100_samples20_start"),
    ),
]


# %% Run all configs

if __name__ == "__main__":
    for config in configs:
        run_datamodel_config(config)
