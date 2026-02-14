# %%
"""
Top-K Influence by Fact Category

Plots the average count of each fact category in the top-K most influential
documents, across checkpoints, for queries matching a regex pattern.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

# %% Configuration

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR.parent / "data_frames"

DEFAULT_DATA_DIRS = {
    "fiction_birth_dates_1epoch": _DATA_DIR / "fiction_birth_dates_1epoch_ce",
    "fiction_birth_dates_1epoch_ce_sampled": _DATA_DIR / "fiction_birth_dates_1epoch_ce_sampled",
    "mayors_1epoch": _DATA_DIR / "mayors_1epoch_ce"
}

@dataclass
class Config:
    name: str  # Identifier for output filename
    data_dir: Path  # Data directory
    metric_pattern: str  # regex to filter metrics
    fact_type_order: list[str]  # Order of fact types in stacked bar chart
    fact_type_colors: dict[str, str]  # Colors for each fact type
    top_k: int = 30
    normalize_by_self_influence: bool = False  # normalize influence by sqrt(doc self-influence)
    mode: Literal["top", "bottom", "both"] = "top"  # top-k, bottom-k, or both
    title: str | None = None  # Optional custom title
    output_filename: str | None = None  # Optional custom output filename


# Default fact type configurations for different experiments
MAYOR_FACT_TYPE_ORDER = [
    "Entailing Fact",
    "Entailing Fact (Other Person)",
    "City Distractor",
    "City Distractor (Other City)",
    "Person Distractor",
    "Person Distractor (Other Person)",
    "pretraining",
]

MAYOR_FACT_TYPE_COLORS = {
    "Entailing Fact": "#2ecc71",
    "Entailing Fact (Other Person)": "#27ae60",
    "City Distractor": "#3498db",
    "City Distractor (Other City)": "#2980b9",
    "Person Distractor": "#e74c3c",
    "Person Distractor (Other Person)": "#c0392b",
    "pretraining": "#95a5a6",
}

DEATH_DATES_FACT_TYPE_ORDER = [
    "Entailing Fact",
    "Entailing Fact (Other Person)",
    "Date Distractor (Other Date)",
    "Person Distractor",
    "Person Distractor (Other Person)",
    "pretraining",
]

DEATH_DATES_FACT_TYPE_COLORS = {
    "Entailing Fact": "#2ecc71",
    "Entailing Fact (Other Person)": "#27ae60",
    "Date Distractor (Other Date)": "#9b59b6",
    "Person Distractor": "#e74c3c",
    "Person Distractor (Other Person)": "#c0392b",
    "pretraining": "#95a5a6",
}


# Define all configs to run
configs = [
    # Mayor 1 epoch - original eval
    Config(
        name="mayors_1epoch_no_fs_qa",
        normalize_by_self_influence=True,
        data_dir=DEFAULT_DATA_DIRS["mayors_1epoch"],
        metric_pattern=r"name_mayor_eval_last_name_qa_1_no_fs",
        fact_type_order=MAYOR_FACT_TYPE_ORDER,
        fact_type_colors=MAYOR_FACT_TYPE_COLORS,
        top_k=100,
        mode="top",
        title="Most influential samples by fact type",
    ),
    # Death dates 1 epoch - original eval
    Config(
        name="fiction_birth_dates_1epoch_qa",
        normalize_by_self_influence=True,
        data_dir=DEFAULT_DATA_DIRS["fiction_birth_dates_1epoch"],
        metric_pattern=r"birth_date_eval_qa_1_no_fs",
        fact_type_order=DEATH_DATES_FACT_TYPE_ORDER,
        fact_type_colors=DEATH_DATES_FACT_TYPE_COLORS,
        top_k=100,
        mode="top",
        title="Most influential samples by fact type",
    ),
]

# %% Data Loading


def parse_tensor_string(s: str) -> float:
    """Parse tensor string like 'tensor(1.9155e+09)' to float."""
    import re as regex
    match = regex.search(r"tensor\(([\d.e+-]+)\)", s)
    if match:
        return float(match.group(1))
    return float(s)


def load_dataframes(
    data_dir: Path,
    metric_pattern: str,
    load_self_influence: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Load the dataframes needed for this analysis.

    Filters influence data by metric_pattern during load to reduce memory usage.
    """
    pattern = re.compile(metric_pattern)

    # Load influence data in chunks, filtering by metric pattern
    # Use python engine to avoid C parser segfault with large files
    influence_chunks = []
    for chunk in pd.read_csv(
        data_dir / "influence_to_spans_df.csv",
        usecols=["query_id", "span_id", "influence_score", "metric_name", "checkpoint_name"],
        chunksize=500_000,
        engine="python",
    ):
        filtered = chunk[chunk["metric_name"].apply(lambda x: bool(pattern.search(x)))]
        if len(filtered) > 0:
            influence_chunks.append(filtered)

    influence_df = pd.concat(influence_chunks, ignore_index=True) if influence_chunks else pd.DataFrame()

    # Load only needed columns from unpacked
    unpacked_df = pd.read_csv(
        data_dir / "unpacked_ds_df.csv",
        usecols=["id", "doc_id"],
    )

    # Load doc_type (use python engine to avoid C parser segfault)
    doc_type_df = pd.read_csv(data_dir / "doc_type_df.csv", engine="python")

    result = {
        "influence": influence_df,
        "unpacked": unpacked_df,
        "doc_type": doc_type_df,
    }

    # Optionally load self-influence data
    if load_self_influence:
        self_inf_df = pd.read_csv(
            data_dir / "self_influence_of_spans.csv",
            usecols=["span_id", "self_inf_score", "checkpoint_name"],
        )
        # Parse tensor strings to floats
        self_inf_df["self_inf_score"] = self_inf_df["self_inf_score"].apply(parse_tensor_string)
        result["self_influence"] = self_inf_df

    return result


def derive_checkpoints_from_data(data_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Derive checkpoint order and step mapping from query_log_prob_df."""
    df = pd.read_csv(data_dir / "query_log_prob_df.csv", usecols=["checkpoint_name", "step"])
    df = df[
        df["checkpoint_name"].notna() &
        df["checkpoint_name"].str.startswith("checkpoint")
    ]
    # Group by checkpoint_name and take min step to deduplicate
    # (some checkpoints may have multiple step values from different eval runs)
    checkpoint_steps = df.groupby("checkpoint_name")["step"].min().reset_index()
    checkpoint_steps = checkpoint_steps.sort_values("step")
    checkpoint_order = checkpoint_steps["checkpoint_name"].tolist()
    checkpoint_to_step = dict(zip(checkpoint_steps["checkpoint_name"], checkpoint_steps["step"].astype(int)))
    return checkpoint_order, checkpoint_to_step


# %% Core Logic


def compute_doc_self_influence(
    self_inf_df: pd.DataFrame,
    unpacked_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate span self-influences to document level by summing.

    Returns DataFrame with columns: doc_id, checkpoint_name, doc_self_influence
    """
    span_to_doc = unpacked_df[["id", "doc_id"]].drop_duplicates()
    span_to_doc = span_to_doc.rename(columns={"id": "span_id"})

    self_inf_with_doc = self_inf_df.merge(span_to_doc, on="span_id")

    doc_self_inf = (
        self_inf_with_doc.groupby(["doc_id", "checkpoint_name"])["self_inf_score"]
        .sum()
        .reset_index()
    )
    doc_self_inf = doc_self_inf.rename(columns={"self_inf_score": "doc_self_influence"})

    return doc_self_inf


def aggregate_spans_to_docs(
    influence_df: pd.DataFrame,
    unpacked_df: pd.DataFrame,
    doc_self_influence_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Aggregate span-level influences to document-level by summing.

    If doc_self_influence_df is provided, normalizes influence by sqrt(doc_self_influence).

    Returns DataFrame with columns:
        query_id, doc_id, checkpoint_name, metric_name, influence_score
    """
    span_to_doc = unpacked_df[["id", "doc_id"]].drop_duplicates()
    span_to_doc = span_to_doc.rename(columns={"id": "span_id"})

    influence_with_doc = influence_df.merge(span_to_doc, on="span_id")

    doc_influence = (
        influence_with_doc.groupby(
            ["query_id", "doc_id", "checkpoint_name", "metric_name"]
        )["influence_score"]
        .sum()
        .reset_index()
    )

    # Optionally normalize by sqrt(doc_self_influence)
    if doc_self_influence_df is not None:
        doc_influence = doc_influence.merge(
            doc_self_influence_df,
            on=["doc_id", "checkpoint_name"],
            how="left",
        )
        # Normalize: influence / sqrt(self_influence)
        # Use abs to handle any negative self-influence values
        import numpy as np
        doc_influence["influence_score"] = (
            doc_influence["influence_score"]
            / np.sqrt(np.abs(doc_influence["doc_self_influence"]))
        )
        doc_influence = doc_influence.drop(columns=["doc_self_influence"])

    return doc_influence


def get_top_k_docs(
    doc_influence_df: pd.DataFrame,
    k: int,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Get top-k or bottom-k documents by influence score for each (query, checkpoint, metric).

    Args:
        ascending: If False, returns top-k (most influential). If True, returns bottom-k (least influential).

    Returns DataFrame with top/bottom-k docs per group, with a 'rank' column (1 to k).
    """
    doc_influence_df = doc_influence_df.copy()

    doc_influence_df["rank"] = doc_influence_df.groupby(
        ["query_id", "checkpoint_name", "metric_name"]
    )["influence_score"].rank(method="first", ascending=ascending)

    top_k = doc_influence_df[doc_influence_df["rank"] <= k].copy()
    top_k["rank"] = top_k["rank"].astype(int)

    return top_k


def count_fact_types_in_top_k(
    top_k_df: pd.DataFrame,
    doc_type_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count how many documents of each fact type appear in the top-k for each
    (query, checkpoint, metric) group.

    Returns DataFrame with columns:
        query_id, checkpoint_name, metric_name, fact_type, count

    Note: Fills missing (query, checkpoint, fact_type) combinations with 0
    to ensure correct averaging.
    """
    top_k_with_types = top_k_df.merge(
        doc_type_df.rename(columns={"id": "doc_id"}),
        on=["doc_id", "query_id"],
    )

    counts = (
        top_k_with_types.groupby(
            ["query_id", "checkpoint_name", "metric_name", "fact_type"]
        )
        .size()
        .reset_index(name="count")
    )

    # Create complete index of all (query, checkpoint, metric, fact_type) combinations
    # to ensure missing fact types get counted as 0
    all_queries = counts[["query_id", "checkpoint_name", "metric_name"]].drop_duplicates()
    all_fact_types = counts["fact_type"].unique()

    complete_index = all_queries.merge(
        pd.DataFrame({"fact_type": all_fact_types}),
        how="cross"
    )

    # Merge and fill missing with 0
    counts = complete_index.merge(
        counts,
        on=["query_id", "checkpoint_name", "metric_name", "fact_type"],
        how="left"
    )
    counts["count"] = counts["count"].fillna(0).astype(int)

    return counts


def compute_avg_counts_by_checkpoint(
    counts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute average counts per (checkpoint, fact_type).

    Returns DataFrame with columns:
        checkpoint_name, fact_type, avg_count, std_count, n_queries
    """
    agg = (
        counts_df.groupby(["checkpoint_name", "fact_type"])["count"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = ["checkpoint_name", "fact_type", "avg_count", "std_count", "n_queries"]

    return agg


# %% Plotting


def plot_fact_category_counts(
    agg_df: pd.DataFrame,
    checkpoint_order: list[str],
    checkpoint_to_step: dict[str, int],
    fact_type_order: list[str],
    fact_type_colors: dict[str, str],
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
    ax: plt.Axes | None = None,
    show_legend: bool = True,
) -> tuple[plt.Figure | None, plt.Axes]:
    """
    Plot average fact category counts across checkpoints as a stacked bar chart.

    Args:
        agg_df: Output from compute_avg_counts_by_checkpoint
        checkpoint_order: Ordered list of checkpoint names
        checkpoint_to_step: Mapping from checkpoint_name to step number
        fact_type_order: Ordered list of fact types for stacking
        fact_type_colors: Mapping from fact_type to color
        figsize: Figure size (ignored if ax is provided)
        title: Plot title
        ax: Optional axes to plot on (for subplots)
        show_legend: Whether to show the legend

    Returns:
        (fig, ax) tuple for further customization. fig is None if ax was provided.
    """
    pivot = agg_df.pivot(
        index="checkpoint_name", columns="fact_type", values="avg_count"
    ).fillna(0)

    present_checkpoints = [c for c in checkpoint_order if c in pivot.index]
    pivot = pivot.loc[present_checkpoints]

    present_fact_types = [f for f in fact_type_order if f in pivot.columns]
    pivot = pivot[present_fact_types]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    bottom = pd.Series(0.0, index=pivot.index)
    x = range(len(pivot.index))

    for fact_type in present_fact_types:
        color = fact_type_colors.get(fact_type, "#7f8c8d")
        ax.bar(
            x,
            pivot[fact_type],
            bottom=bottom,
            label=fact_type,
            color=color,
        )
        bottom += pivot[fact_type]

    ax.set_xticks(x)
    # Use step numbers as labels
    labels = [str(checkpoint_to_step.get(c, c)) for c in pivot.index]
    ax.set_xticklabels(labels)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Average Count")
    if show_legend:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    if title:
        ax.set_title(title)

    if fig is not None:
        fig.tight_layout()

    return fig, ax


# %% Main Execution


def run_analysis(doc_influence: pd.DataFrame, doc_type_df: pd.DataFrame, k: int, ascending: bool) -> pd.DataFrame:
    """Run the analysis pipeline for top-k or bottom-k."""
    docs = get_top_k_docs(doc_influence, k, ascending=ascending)
    counts = count_fact_types_in_top_k(docs, doc_type_df)
    return compute_avg_counts_by_checkpoint(counts)


def run_config(config: Config) -> None:
    """Run analysis for a single config."""
    print(f"\n{'='*60}")
    print(f"Running config: {config.name}")
    print(f"{'='*60}")
    print(f"Data dir: {config.data_dir}")
    print(f"Metric pattern: {config.metric_pattern}")
    print(f"Normalize by self-influence: {config.normalize_by_self_influence}")
    print(f"Mode: {config.mode}")

    # Derive checkpoints from data
    print("\nDeriving checkpoints from data...")
    checkpoint_order, checkpoint_to_step = derive_checkpoints_from_data(config.data_dir)
    print(f"  Found {len(checkpoint_order)} checkpoints")
    print(f"  Step mapping: {checkpoint_to_step}")

    # Load data (filters by metric pattern during load for memory efficiency)
    dfs = load_dataframes(
        config.data_dir,
        config.metric_pattern,
        load_self_influence=config.normalize_by_self_influence,
    )
    print(f"Loaded {len(dfs['influence']):,} influence rows after filtering")

    # Compute document-level self-influence if normalizing
    doc_self_influence = None
    if config.normalize_by_self_influence:
        print("Computing document-level self-influence...")
        doc_self_influence = compute_doc_self_influence(
            dfs["self_influence"], dfs["unpacked"]
        )
        print(f"Computed self-influence for {len(doc_self_influence):,} (doc, checkpoint) pairs")

    # Aggregate spans to documents
    print("Aggregating spans to documents...")
    doc_influence = aggregate_spans_to_docs(
        dfs["influence"], dfs["unpacked"], doc_self_influence
    )
    print(f"Aggregated to {len(doc_influence):,} doc-level rows")

    # Run analysis based on mode
    if config.mode == "both":
        print(f"Getting top-{config.top_k} and bottom-{config.top_k} documents...")
        agg_top = run_analysis(doc_influence, dfs["doc_type"], config.top_k, ascending=False)
        agg_bottom = run_analysis(doc_influence, dfs["doc_type"], config.top_k, ascending=True)

        # Create side-by-side plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        plot_fact_category_counts(
            agg_top,
            checkpoint_order=checkpoint_order,
            checkpoint_to_step=checkpoint_to_step,
            fact_type_order=config.fact_type_order,
            fact_type_colors=config.fact_type_colors,
            title=f"Top-{config.top_k} Most Influential",
            ax=ax1,
            show_legend=False,
        )
        plot_fact_category_counts(
            agg_bottom,
            checkpoint_order=checkpoint_order,
            checkpoint_to_step=checkpoint_to_step,
            fact_type_order=config.fact_type_order,
            fact_type_colors=config.fact_type_colors,
            title=f"Top-{config.top_k} Least Influential",
            ax=ax2,
            show_legend=True,
        )
        default_title = f"Influential Docs by Category (metric: {config.metric_pattern})"
        fig.suptitle(config.title or default_title, y=1.02)
        fig.tight_layout()
    else:
        ascending = config.mode == "bottom"
        label = "Least" if ascending else "Most"
        print(f"Getting top-{config.top_k} {label.lower()} influential documents...")

        agg = run_analysis(doc_influence, dfs["doc_type"], config.top_k, ascending=ascending)

        default_title = f"Top-{config.top_k} {label} Influential Docs by Category\n(metric: {config.metric_pattern})"
        fig, ax = plot_fact_category_counts(
            agg,
            checkpoint_order=checkpoint_order,
            checkpoint_to_step=checkpoint_to_step,
            fact_type_order=config.fact_type_order,
            fact_type_colors=config.fact_type_colors,
            title=config.title or default_title,
        )

    # Save
    output_dir = _THIS_DIR / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = config.output_filename or f"top_k_influence_{config.name}_{config.mode}.pdf"
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__" or True:
    for config in configs:
        run_config(config)

# %%
