# %%

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm


# %% Configuration


@dataclass
class LDSConfig:
    """Configuration for Linear Data Modeling Score calculation."""

    name: str
    # Data model CSVs (from generate_datamodel_csvs.py)
    datamodel_query_csv: Path  # datamodel_query_df.csv
    datamodel_training_csv: Path  # datamodel_training_df.csv
    # Influence CSVs (from generate_csvs.py)
    influence_spans_csv: Path  # influence_to_spans_df.csv
    unpacked_ds_csv: Path  # unpacked_ds_df.csv
    # Filters
    metric_name_filter: str  # e.g., "name_mayor_eval_qa_no_fs"
    influence_metric_filter: str | None = None  # Filter for influence metric_name
    checkpoint_filter: str | None = None  # Filter for checkpoint_name (e.g., "checkpoint_final")


# %% Data Loading


def load_document_influences(
    influence_spans_csv: Path,
    unpacked_ds_csv: Path,
    metric_name_filter: str | None = None,
    checkpoint_filter: str | None = None,
) -> pd.DataFrame:
    """Load span-level influences and aggregate to document level.

    Args:
        influence_spans_csv: Path to influence_to_spans_df.csv
        unpacked_ds_csv: Path to unpacked_ds_df.csv
        metric_name_filter: Optional filter for influence metric_name
        checkpoint_filter: Optional filter for checkpoint_name

    Returns:
        DataFrame with columns: query_id, doc_id, influence_score
    """
    print("Loading influence spans...")
    influence_df = pd.read_csv(influence_spans_csv)
    print(f"  Loaded {len(influence_df)} span influence records")

    if metric_name_filter:
        influence_df = influence_df[influence_df["metric_name"] == metric_name_filter]
        print(f"  After metric filter '{metric_name_filter}': {len(influence_df)} records")

    if checkpoint_filter:
        influence_df = influence_df[influence_df["checkpoint_name"] == checkpoint_filter]
        print(f"  After checkpoint filter '{checkpoint_filter}': {len(influence_df)} records")

    print("Loading unpacked dataset (span -> doc mapping)...")
    unpacked_df = pd.read_csv(
        unpacked_ds_csv,
        usecols=["id", "doc_id"],
    )
    print(f"  Loaded {len(unpacked_df)} span records")

    # Join to get doc_id for each span
    print("Joining spans to documents...")
    merged = influence_df.merge(
        unpacked_df,
        left_on="span_id",
        right_on="id",
        how="left",
    )

    # Aggregate influence by (query_id, doc_id)
    print("Aggregating influence by (query_id, document)...")
    doc_influence = (
        merged.groupby(["query_id", "doc_id"])["influence_score"]
        .sum()
        .reset_index()
    )
    print(f"  Result: {len(doc_influence)} (query_id, doc) pairs")

    return doc_influence


def load_datamodel_data(
    query_csv: Path,
    training_csv: Path,
    metric_name_filter: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data model query results and training document metadata.

    Args:
        query_csv: Path to datamodel_query_df.csv
        training_csv: Path to datamodel_training_df.csv
        metric_name_filter: Filter for metric_name

    Returns:
        query_df: DataFrame with query evaluation results
        training_df: DataFrame with document -> dataset mapping
    """
    print("Loading datamodel query results...")
    query_df = pd.read_csv(query_csv)
    print(f"  Loaded {len(query_df)} query records")

    query_df = query_df[query_df["metric_name"] == metric_name_filter]
    print(f"  After metric filter '{metric_name_filter}': {len(query_df)} records")

    print("Loading datamodel training docs...")
    training_df = pd.read_csv(
        training_csv,
        usecols=["id", "dataset_id"],
    )
    print(f"  Loaded {len(training_df)} training doc records")

    return query_df, training_df


# %% LDS Calculation


def calculate_lds(
    query_df: pd.DataFrame,
    training_df: pd.DataFrame,
    doc_influence: pd.DataFrame,
    metric_col: str = "softmargin",
) -> pd.DataFrame:
    """Calculate Linear Data Modeling Score for each query.

    For each query_id:
    - For each dataset_id, sum the influences of documents in that dataset
    - Correlate predicted (sum of influences) vs actual (metric value)

    Args:
        query_df: Query evaluation results with dataset_id, query_id, metric columns
        training_df: Document -> dataset_id mapping
        doc_influence: Document influence scores per query_id
        metric_col: Which metric to use as actual (softmargin or log_prob)

    Returns:
        DataFrame with columns: query_id, person, pearson_r, spearman_r, n_datasets
    """
    results = []

    # Get unique query_ids that have influence data
    query_ids_with_influence = set(doc_influence["query_id"].unique())
    query_ids_in_datamodel = set(query_df["query_id"].unique())
    common_query_ids = query_ids_with_influence & query_ids_in_datamodel

    print(f"Query IDs with influence data: {len(query_ids_with_influence)}")
    print(f"Query IDs in datamodel: {len(query_ids_in_datamodel)}")
    print(f"Common query IDs: {len(common_query_ids)}")

    for query_id in tqdm(common_query_ids, desc="Calculating LDS"):
        # Get influence scores for this query
        query_influence = doc_influence[doc_influence["query_id"] == query_id]
        influence_by_doc = query_influence.set_index("doc_id")["influence_score"].to_dict()

        # Get actual metric values for this query across datasets
        query_results = query_df[query_df["query_id"] == query_id]

        # Average metric values across runs for each dataset_id
        dataset_metrics = query_results.groupby("dataset_id")[metric_col].mean()

        predicted = []
        actual = []

        for dataset_id, actual_value in dataset_metrics.items():
            # Get documents in this dataset
            docs_in_dataset = training_df[training_df["dataset_id"] == dataset_id]["id"].tolist()

            # Sum influences of documents in this dataset
            predicted_value = sum(
                influence_by_doc.get(doc_id, 0)
                for doc_id in docs_in_dataset
            )

            predicted.append(predicted_value)
            actual.append(actual_value)

        predicted = np.array(predicted)
        actual = np.array(actual)

        # Calculate correlations
        if len(predicted) >= 3 and np.std(predicted) > 0 and np.std(actual) > 0:
            pearson_r, _ = stats.pearsonr(predicted, actual)
            spearman_r, _ = stats.spearmanr(predicted, actual)
        else:
            pearson_r = np.nan
            spearman_r = np.nan

        # Get person name for this query if available
        person = query_results["person"].iloc[0] if "person" in query_results.columns and len(query_results) > 0 else None

        results.append({
            "query_id": query_id,
            "person": person,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "n_datasets": len(predicted),
        })

    return pd.DataFrame(results)


# %% Main Execution


def run_config(config: LDSConfig) -> pd.DataFrame:
    """Run LDS calculation for a single config."""
    print(f"\n{'='*60}")
    print(f"Running config: {config.name}")
    print(f"{'='*60}")

    # Load document-level influences
    doc_influence = load_document_influences(
        config.influence_spans_csv,
        config.unpacked_ds_csv,
        config.influence_metric_filter,
        config.checkpoint_filter,
    )

    # Load datamodel data
    query_df, training_df = load_datamodel_data(
        config.datamodel_query_csv,
        config.datamodel_training_csv,
        config.metric_name_filter,
    )

    # Calculate LDS for both metrics
    print("\n" + "="*40)
    print("Calculating LDS for softmargin...")
    lds_softmargin = calculate_lds(query_df, training_df, doc_influence, "softmargin")
    lds_softmargin["metric"] = "softmargin"

    print("\n" + "="*40)
    print("Calculating LDS for log_prob...")
    lds_logprob = calculate_lds(query_df, training_df, doc_influence, "log_prob")
    lds_logprob["metric"] = "log_prob"

    # Combine results
    lds_df = pd.concat([lds_softmargin, lds_logprob], ignore_index=True)

    # Print summary
    print("\n" + "="*40)
    print("LDS Summary:")
    print(f"\nSoftmargin:")
    print(f"  Mean Pearson r: {lds_softmargin['pearson_r'].mean():.4f}")
    print(f"  Mean Spearman r: {lds_softmargin['spearman_r'].mean():.4f}")
    print(f"\nLog_prob:")
    print(f"  Mean Pearson r: {lds_logprob['pearson_r'].mean():.4f}")
    print(f"  Mean Spearman r: {lds_logprob['spearman_r'].mean():.4f}")

    print(f"\nDone with config: {config.name}")

    return lds_df


# %% Define configs

configs = [
    LDSConfig(
        name="mayors_alpha0.1_ds100_start",
        datamodel_query_csv=Path("analysis/data_frames/mayors_100_alpha0.1_ds100_samples20_start/datamodel_query_df.csv"),
        datamodel_training_csv=Path("analysis/data_frames/mayors_100_alpha0.1_ds100_samples20_start/datamodel_training_df.csv"),
        influence_spans_csv=Path("analysis/data_frames/mayors_1epoch/influence_to_spans_df.csv"),
        unpacked_ds_csv=Path("analysis/data_frames/mayors_1epoch/unpacked_ds_df.csv"),
        metric_name_filter="name_mayor_eval_qa_4_no_fs",
        influence_metric_filter="name_mayor_eval_qa_4_no_fs",
        checkpoint_filter="checkpoint_final",
    ),
]


# %% Run all configs

if __name__ == "__main__":
    for config in configs:
        lds_df = run_config(config)
        print("\nPer-query LDS scores:")
        print(lds_df.to_string())
