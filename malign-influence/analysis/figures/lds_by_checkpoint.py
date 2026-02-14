# %%

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm


# %% Configuration


@dataclass
class LDSByCheckpointConfig:
    """Configuration for LDS by checkpoint plotting."""

    name: str
    influence_spans_csv: Path
    unpacked_ds_csv: Path
    query_log_prob_csv: Path  # For checkpoint -> step mapping
    start_datamodel_query_csv: Path
    start_datamodel_training_csv: Path
    final_datamodel_query_csv: Path
    final_datamodel_training_csv: Path
    metric_name_filter: str
    influence_metric_filter: str
    output_dir: Path = Path("analysis/figures/images")
    # Plot labels
    xlabel: str = "Training Step"
    ylabel: str = "LDS (Pearson r)"
    title: str = "Linear Data Modeling Score by Training Checkpoint"
    legend_title: str = "Datamodel"
    start_label: str = "Start Datamodel"
    final_label: str = "Final Datamodel"


# %% Helper functions


def load_checkpoint_to_step_mapping(query_log_prob_csv: Path) -> dict[str, int]:
    """Load checkpoint name to step mapping from query_log_prob_df.csv."""
    df = pd.read_csv(query_log_prob_csv, usecols=["checkpoint_name", "step"])
    mapping = df.groupby("checkpoint_name")["step"].first().to_dict()
    return mapping


def load_document_influences(
    influence_spans_csv: Path,
    unpacked_ds_csv: Path,
    metric_name_filter: str | None = None,
    checkpoint_filter: str | None = None,
) -> pd.DataFrame:
    """Load span-level influences and aggregate to document level."""
    influence_df = pd.read_csv(influence_spans_csv)

    if metric_name_filter:
        influence_df = influence_df[influence_df["metric_name"] == metric_name_filter]

    if checkpoint_filter:
        influence_df = influence_df[influence_df["checkpoint_name"] == checkpoint_filter]

    unpacked_df = pd.read_csv(
        unpacked_ds_csv,
        usecols=["id", "doc_id"],
    )

    # Join to get doc_id for each span
    merged = influence_df.merge(
        unpacked_df,
        left_on="span_id",
        right_on="id",
        how="left",
    )

    # Aggregate influence by (query_id, doc_id)
    doc_influence = (
        merged.groupby(["query_id", "doc_id"])["influence_score"]
        .sum()
        .reset_index()
    )

    return doc_influence


def load_datamodel_data(
    query_csv: Path,
    training_csv: Path,
    metric_name_filter: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data model query results and training document metadata."""
    query_df = pd.read_csv(query_csv)
    query_df = query_df[query_df["metric_name"] == metric_name_filter]

    training_df = pd.read_csv(
        training_csv,
        usecols=["id", "dataset_id"],
    )

    return query_df, training_df


def calculate_lds(
    query_df: pd.DataFrame,
    training_df: pd.DataFrame,
    doc_influence: pd.DataFrame,
    metric_col: str = "softmargin",
) -> pd.DataFrame:
    """Calculate Linear Data Modeling Score for each query."""
    results = []

    query_ids_with_influence = set(doc_influence["query_id"].unique())
    query_ids_in_datamodel = set(query_df["query_id"].unique())
    common_query_ids = query_ids_with_influence & query_ids_in_datamodel

    for query_id in common_query_ids:
        query_influence = doc_influence[doc_influence["query_id"] == query_id]
        influence_by_doc = query_influence.set_index("doc_id")["influence_score"].to_dict()

        query_results = query_df[query_df["query_id"] == query_id]

        # Average metric values across runs for each dataset_id
        dataset_metrics = query_results.groupby("dataset_id")[metric_col].mean()

        predicted = []
        actual = []

        for dataset_id, actual_value in dataset_metrics.items():
            docs_in_dataset = training_df[training_df["dataset_id"] == dataset_id]["id"].tolist()

            predicted_value = sum(
                influence_by_doc.get(doc_id, 0)
                for doc_id in docs_in_dataset
            )

            predicted.append(predicted_value)
            actual.append(actual_value)

        predicted = np.array(predicted)
        actual = np.array(actual)

        if len(predicted) >= 3 and np.std(predicted) > 0 and np.std(actual) > 0:
            pearson_r, _ = stats.pearsonr(predicted, actual)
        else:
            pearson_r = np.nan

        person = query_results["person"].iloc[0] if "person" in query_results.columns and len(query_results) > 0 else None

        results.append({
            "query_id": query_id,
            "person": person,
            "pearson_r": pearson_r,
        })

    return pd.DataFrame(results)


# %% Main calculation


def calculate_lds_for_all_checkpoints(
    influence_spans_csv: Path,
    unpacked_ds_csv: Path,
    query_csv: Path,
    training_csv: Path,
    metric_name_filter: str,
    influence_metric_filter: str,
    datamodel_label: str,
    checkpoint_to_step: dict[str, int],
) -> pd.DataFrame:
    """Calculate LDS for all checkpoints for a given datamodel."""
    # Get list of checkpoints
    influence_df = pd.read_csv(influence_spans_csv, usecols=["checkpoint_name"])
    checkpoints = influence_df["checkpoint_name"].unique()

    # Load datamodel data once
    query_df, training_df = load_datamodel_data(query_csv, training_csv, metric_name_filter)

    all_results = []

    for checkpoint in tqdm(checkpoints, desc=f"Calculating LDS for {datamodel_label}"):
        step = checkpoint_to_step[checkpoint]

        # Load influences for this checkpoint
        doc_influence = load_document_influences(
            influence_spans_csv,
            unpacked_ds_csv,
            influence_metric_filter,
            checkpoint,
        )

        # Calculate LDS
        lds_df = calculate_lds(query_df, training_df, doc_influence, "softmargin")
        lds_df["checkpoint"] = checkpoint
        lds_df["step"] = step
        lds_df["datamodel"] = datamodel_label

        all_results.append(lds_df)

    return pd.concat(all_results, ignore_index=True)


def plot_lds_by_checkpoint(
    df: pd.DataFrame,
    output_path: Path,
    xlabel: str = "Training Step",
    ylabel: str = "LDS (Pearson r)",
    title: str = "Linear Data Modeling Score by Training Checkpoint",
    legend_title: str = "Datamodel",
) -> None:
    """Plot LDS by checkpoint with two lines for start/final datamodels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique datamodels and assign colors
    datamodels = df["datamodel"].unique()
    colors = {dm: f"C{i}" for i, dm in enumerate(datamodels)}

    # Plot faint lines for each individual query
    for datamodel in datamodels:
        dm_df = df[df["datamodel"] == datamodel]
        color = colors[datamodel]

        for query_id in dm_df["query_id"].unique():
            query_df = dm_df[dm_df["query_id"] == query_id].sort_values("step")
            ax.plot(
                query_df["step"],
                query_df["pearson_r"],
                color=color,
                alpha=0.2,
                linewidth=1,
            )

    # Plot thick mean lines
    for datamodel in datamodels:
        dm_df = df[df["datamodel"] == datamodel]
        color = colors[datamodel]

        mean_df = dm_df.groupby("step")["pearson_r"].mean().reset_index()
        mean_df = mean_df.sort_values("step")

        ax.plot(
            mean_df["step"],
            mean_df["pearson_r"],
            color=color,
            alpha=1.0,
            linewidth=3,
            marker="o",
            markersize=8,
            label=datamodel,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend(title=legend_title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved plot to {output_path}")


# %% Run


def run_config(config: LDSByCheckpointConfig) -> None:
    """Run LDS by checkpoint analysis for a config."""
    print(f"\n{'='*60}")
    print(f"Running config: {config.name}")
    print(f"{'='*60}")

    # Load checkpoint to step mapping
    checkpoint_to_step = load_checkpoint_to_step_mapping(config.query_log_prob_csv)
    print(f"Checkpoint to step mapping: {checkpoint_to_step}")

    # Calculate LDS for start datamodel
    start_df = calculate_lds_for_all_checkpoints(
        config.influence_spans_csv,
        config.unpacked_ds_csv,
        config.start_datamodel_query_csv,
        config.start_datamodel_training_csv,
        config.metric_name_filter,
        config.influence_metric_filter,
        config.start_label,
        checkpoint_to_step,
    )

    # Calculate LDS for final datamodel
    final_df = calculate_lds_for_all_checkpoints(
        config.influence_spans_csv,
        config.unpacked_ds_csv,
        config.final_datamodel_query_csv,
        config.final_datamodel_training_csv,
        config.metric_name_filter,
        config.influence_metric_filter,
        config.final_label,
        checkpoint_to_step,
    )

    # Combine
    combined_df = pd.concat([start_df, final_df], ignore_index=True)

    # Print summary
    print("\nLDS Summary by Checkpoint and Datamodel:")
    summary = combined_df.groupby(["datamodel", "step"])["pearson_r"].agg(["mean", "std", "count"])
    print(summary.to_string())

    # Plot
    output_path = config.output_dir / f"lds_by_checkpoint_{config.name}.pdf"
    plot_lds_by_checkpoint(
        combined_df,
        output_path,
        xlabel=config.xlabel,
        ylabel=config.ylabel,
        title=config.title,
        legend_title=config.legend_title,
    )


# %% Config


configs = [
    LDSByCheckpointConfig(
        name="birth_dates_alpha0.1",
        influence_spans_csv=Path("analysis/data_frames/fiction_birth_dates_1epoch_softmargin/influence_to_spans_df.csv"),
        unpacked_ds_csv=Path("analysis/data_frames/fiction_birth_dates_1epoch_softmargin/unpacked_ds_df.csv"),
        query_log_prob_csv=Path("analysis/data_frames/fiction_birth_dates_1epoch_softmargin/query_log_prob_df.csv"),
        start_datamodel_query_csv=Path("analysis/data_frames/fictional_birth_dates_100_alpha0.1_ds100_samples20_start/datamodel_query_df.csv"),
        start_datamodel_training_csv=Path("analysis/data_frames/fictional_birth_dates_100_alpha0.1_ds100_samples20_start/datamodel_training_df.csv"),
        final_datamodel_query_csv=Path("analysis/data_frames/fictional_birth_dates_100_alpha0.1_ds100_samples40_final/datamodel_query_df.csv"),
        final_datamodel_training_csv=Path("analysis/data_frames/fictional_birth_dates_100_alpha0.1_ds100_samples40_final/datamodel_training_df.csv"),
        metric_name_filter="birth_date_eval_qa_1_no_fs",
        influence_metric_filter="birth_date_eval_qa_1_no_fs",
    ),
    LDSByCheckpointConfig(
        name="mayors_alpha0.1",
        influence_spans_csv=Path("analysis/data_frames/mayors_1epoch_softmargin/influence_to_spans_df.csv"),
        unpacked_ds_csv=Path("analysis/data_frames/mayors_1epoch_softmargin/unpacked_ds_df.csv"),
        query_log_prob_csv=Path("analysis/data_frames/mayors_1epoch_softmargin/query_log_prob_df.csv"),
        start_datamodel_query_csv=Path("analysis/data_frames/mayors_100_alpha0.1_ds100_samples20_start/datamodel_query_df.csv"),
        start_datamodel_training_csv=Path("analysis/data_frames/mayors_100_alpha0.1_ds100_samples20_start/datamodel_training_df.csv"),
        final_datamodel_query_csv=Path("analysis/data_frames/mayors_100_alpha0.1_ds100_samples20_final/datamodel_query_df.csv"),
        final_datamodel_training_csv=Path("analysis/data_frames/mayors_100_alpha0.1_ds100_samples20_final/datamodel_training_df.csv"),
        metric_name_filter="name_mayor_eval_qa_1_no_fs",
        influence_metric_filter="name_mayor_eval_qa_1_no_fs",
    ),
]


# %% Main

if __name__ == "__main__":
    for config in configs:
        run_config(config)
