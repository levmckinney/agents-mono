# %%
"""
Softmargin vs Entailing Facts Correlation Analysis

Plot 1: Correlation between softmargin and count of entailing facts (with per-query samples)
Plot 2: Cross-phrasing correlation (do different output formats become anti-correlated?)
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# %% Constants

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / ".." / "data_frames"

# Datamodel directories contain datamodel_training_df.csv and datamodel_query_df.csv
DATAMODEL_DIRS = {
    "mayors_final": _DATA_DIR / "mayors_100_alpha0.1_ds100_samples20_final",
    "mayors_start": _DATA_DIR / "mayors_100_alpha0.1_ds100_samples20_start",
    "birth_dates_final": _DATA_DIR / "fictional_birth_dates_100_alpha0.1_ds100_samples40_final",
    "birth_dates_start": _DATA_DIR / "fictional_birth_dates_100_alpha0.1_ds100_samples40_start",
}

# Main dataset directories contain doc_type_df.csv
MAIN_DATA_DIRS = {
    "mayors": _DATA_DIR / "mayors_1epoch_ce",
    "birth_dates": _DATA_DIR / "fiction_birth_dates_1epoch_ce",
}


# %% Configuration

@dataclass
class Config:
    name: str
    start_datamodel_dir: Path
    final_datamodel_dir: Path
    doc_type_dir: Path
    metric_name: str  # Primary metric for entailing analysis
    # Pair of metrics for cross-phrasing correlation
    phrasing_pair: tuple[str, str] | None = None
    output_filename: str | None = None


configs = [
    Config(
        name="mayors",
        start_datamodel_dir=DATAMODEL_DIRS["mayors_start"],
        final_datamodel_dir=DATAMODEL_DIRS["mayors_final"],
        doc_type_dir=MAIN_DATA_DIRS["mayors"],
        metric_name="name_mayor_eval_qa_no_fs",
        # First name vs last name output formats
        phrasing_pair=("name_mayor_eval_first_name_qa_1_no_fs", "name_mayor_eval_last_name_qa_1_no_fs"),
        output_filename="softmargin_correlation_mayors.pdf",
    ),
    Config(
        name="birth_dates",
        start_datamodel_dir=DATAMODEL_DIRS["birth_dates_start"],
        final_datamodel_dir=DATAMODEL_DIRS["birth_dates_final"],
        doc_type_dir=MAIN_DATA_DIRS["birth_dates"],
        metric_name="birth_date_eval_qa_1_no_fs",
        # Different date formats: standard vs DMY
        phrasing_pair=("birth_date_eval_qa_1_no_fs", "birth_date_dmy_eval_qa_1_no_fs"),
        output_filename="softmargin_correlation_birth_dates.pdf",
    ),
]


# %% Data Loading

def load_datamodel_query(datamodel_dir: Path, metrics: list[str] | None = None) -> pd.DataFrame:
    """Load datamodel_query_df with softmargin data."""
    print("Loading datamodel_query_df...")
    df = pd.read_csv(
        datamodel_dir / "datamodel_query_df.csv",
        usecols=["softmargin", "person", "query_id", "dataset_id", "metric_name"],
    )
    if metrics:
        df = df[df["metric_name"].isin(metrics)]
    print(f"  Loaded {len(df):,} rows")
    print(f"  Unique datasets: {df['dataset_id'].nunique()}")
    print(f"  Unique queries: {df['query_id'].nunique()}")
    return df


def load_datamodel_training(datamodel_dir: Path) -> pd.DataFrame:
    """Load datamodel_training_df mapping docs to datasets."""
    print("Loading datamodel_training_df...")
    df = pd.read_csv(
        datamodel_dir / "datamodel_training_df.csv",
        usecols=["id", "dataset_id"],
    )
    print(f"  Loaded {len(df):,} rows")
    return df


def load_doc_type(doc_type_dir: Path) -> pd.DataFrame:
    """Load doc_type_df with fact type classifications."""
    print("Loading doc_type_df...")
    df = pd.read_csv(doc_type_dir / "doc_type_df.csv")
    print(f"  Loaded {len(df):,} rows")
    return df


# %% Core Computation

def count_entailing_facts_per_query_dataset(
    datamodel_training_df: pd.DataFrame,
    doc_type_df: pd.DataFrame,
    query_ids: list[str],
) -> pd.DataFrame:
    """Count entailing facts per (query_id, dataset_id) combination."""
    print("Counting entailing facts per (query_id, dataset_id)...")

    entailing_docs = doc_type_df[doc_type_df["fact_type"] == "Entailing Fact"][["id", "query_id"]]
    entailing_docs = entailing_docs.rename(columns={"id": "doc_id"})
    entailing_docs = entailing_docs[entailing_docs["query_id"].isin(query_ids)]

    training_docs = datamodel_training_df.rename(columns={"id": "doc_id"})
    merged = training_docs.merge(entailing_docs, on="doc_id", how="inner")
    counts = merged.groupby(["query_id", "dataset_id"]).size().reset_index(name="entailing_count")

    return counts


def compute_per_query_correlations(
    datamodel_query_df: pd.DataFrame,
    entailing_counts: pd.DataFrame,
    metric_name: str,
) -> pd.DataFrame:
    """Compute softmargin-entailing correlation for EACH query separately."""
    df = datamodel_query_df[datamodel_query_df["metric_name"] == metric_name].copy()
    merged = df.merge(entailing_counts, on=["query_id", "dataset_id"], how="left")
    merged["entailing_count"] = merged["entailing_count"].fillna(0)

    # Compute correlation per query (across datasets)
    results = []
    for query_id in merged["query_id"].unique():
        query_data = merged[merged["query_id"] == query_id]
        if len(query_data) >= 3 and query_data["entailing_count"].std() > 0:
            corr, _ = stats.spearmanr(query_data["softmargin"], query_data["entailing_count"])
        else:
            corr = np.nan
        results.append({"query_id": query_id, "correlation": corr})

    return pd.DataFrame(results)


def compute_cross_phrasing_correlation(
    datamodel_query_df: pd.DataFrame,
    metric1: str,
    metric2: str,
) -> pd.DataFrame:
    """
    Compute correlation between softmargins of two different phrasings
    for each person across datasets.

    This tests if different output formats become anti-correlated.
    """
    df1 = datamodel_query_df[datamodel_query_df["metric_name"] == metric1].copy()
    df2 = datamodel_query_df[datamodel_query_df["metric_name"] == metric2].copy()

    # Pivot each to: rows=dataset_id, cols=person, values=softmargin
    pivot1 = df1.pivot_table(index="dataset_id", columns="person", values="softmargin", aggfunc="mean")
    pivot2 = df2.pivot_table(index="dataset_id", columns="person", values="softmargin", aggfunc="mean")

    # Get common people and datasets
    common_people = list(set(pivot1.columns) & set(pivot2.columns))
    common_datasets = list(set(pivot1.index) & set(pivot2.index))

    if len(common_people) < 1 or len(common_datasets) < 3:
        return pd.DataFrame()

    pivot1 = pivot1.loc[common_datasets, common_people]
    pivot2 = pivot2.loc[common_datasets, common_people]

    # For each person, correlate their softmargins across the two phrasings
    results = []
    for person in common_people:
        s1 = pivot1[person].dropna()
        s2 = pivot2[person].dropna()
        common_idx = s1.index.intersection(s2.index)
        if len(common_idx) >= 3:
            corr, _ = stats.spearmanr(s1.loc[common_idx], s2.loc[common_idx])
            results.append({"person": person, "correlation": corr, "n_datasets": len(common_idx)})

    return pd.DataFrame(results)


# %% Plotting

def plot_correlation_with_samples(
    start_corrs: pd.DataFrame,
    final_corrs: pd.DataFrame,
    start_mean: float,
    final_mean: float,
    title: str,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot bars with individual query correlations as strip plot overlay."""
    fig, ax = plt.subplots(figsize=figsize)

    x_positions = [0, 1]
    labels = ["First Checkpoint", "Last Checkpoint"]
    colors = ["#7fbf7f", "#1f77b4"]

    # Plot bars for means
    bars = ax.bar(x_positions, [start_mean, final_mean], width=0.6, color=colors, edgecolor="black", alpha=0.7)

    # Overlay individual query points with jitter
    for i, (corrs, color) in enumerate([(start_corrs, colors[0]), (final_corrs, colors[1])]):
        valid_corrs = corrs["correlation"].dropna()
        jitter = np.random.uniform(-0.15, 0.15, len(valid_corrs))
        ax.scatter(
            x_positions[i] + jitter,
            valid_corrs,
            color=color,
            edgecolor="white",
            s=30,
            alpha=0.6,
            zorder=3,
        )

    # Add mean labels
    for bar, mean in zip(bars, [start_mean, final_mean]):
        if not np.isnan(mean):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.03,
                f"mean={mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Spearman Correlation (per query)")
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1, 1)

    fig.tight_layout()
    return fig, ax


def plot_cross_phrasing_correlation(
    start_corrs: pd.DataFrame,
    final_corrs: pd.DataFrame,
    metric1: str,
    metric2: str,
    title: str,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot cross-phrasing correlations with per-person samples."""
    fig, ax = plt.subplots(figsize=figsize)

    x_positions = [0, 1]
    labels = ["First Checkpoint", "Last Checkpoint"]
    colors = ["#7fbf7f", "#1f77b4"]

    # Compute means
    start_mean = start_corrs["correlation"].mean() if len(start_corrs) > 0 else np.nan
    final_mean = final_corrs["correlation"].mean() if len(final_corrs) > 0 else np.nan

    # Plot bars for means
    bars = ax.bar(x_positions, [start_mean, final_mean], width=0.6, color=colors, edgecolor="black", alpha=0.7)

    # Overlay individual person points with jitter
    for i, (corrs, color) in enumerate([(start_corrs, colors[0]), (final_corrs, colors[1])]):
        if len(corrs) > 0:
            valid_corrs = corrs["correlation"].dropna()
            jitter = np.random.uniform(-0.15, 0.15, len(valid_corrs))
            ax.scatter(
                x_positions[i] + jitter,
                valid_corrs,
                color=color,
                edgecolor="white",
                s=50,
                alpha=0.7,
                zorder=3,
            )

    # Add mean labels
    for bar, mean in zip(bars, [start_mean, final_mean]):
        if not np.isnan(mean):
            y_pos = mean + 0.05 if mean >= 0 else mean - 0.1
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"mean={mean:.3f}",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=2)
    ax.set_ylabel("Spearman Correlation (per person)")
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1, 1)

    # Add subtitle with metric names
    short_m1 = metric1.replace("_eval_qa_1_no_fs", "").replace("birth_date_", "")
    short_m2 = metric2.replace("_eval_qa_1_no_fs", "").replace("birth_date_", "")
    ax.text(0.5, -0.12, f"({short_m1} vs {short_m2})", ha="center", transform=ax.transAxes, fontsize=9, color="gray")

    fig.tight_layout()
    return fig, ax


# %% Main

def process_checkpoint(
    datamodel_dir: Path,
    doc_type_df: pd.DataFrame,
    metric_name: str,
    phrasing_pair: tuple[str, str] | None,
    checkpoint_name: str,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Process a single checkpoint and return per-query correlations and cross-phrasing results."""
    print(f"\n  Processing {checkpoint_name}...")

    # Determine which metrics to load
    metrics_to_load = [metric_name]
    if phrasing_pair:
        metrics_to_load.extend(phrasing_pair)
    metrics_to_load = list(set(metrics_to_load))

    datamodel_query_df = load_datamodel_query(datamodel_dir, metrics_to_load)
    datamodel_training_df = load_datamodel_training(datamodel_dir)

    # Get query_ids for the primary metric
    query_ids = datamodel_query_df[datamodel_query_df["metric_name"] == metric_name]["query_id"].unique().tolist()

    # Count entailing facts
    entailing_counts = count_entailing_facts_per_query_dataset(
        datamodel_training_df, doc_type_df, query_ids
    )

    # Per-query correlations
    per_query_corrs = compute_per_query_correlations(
        datamodel_query_df, entailing_counts, metric_name
    )
    mean_corr = per_query_corrs["correlation"].mean()
    print(f"    Mean Softmargin-Entailing Correlation: {mean_corr:.4f}")
    print(f"    Per-query correlations: {len(per_query_corrs)} queries")

    # Cross-phrasing correlation
    cross_phrasing_corrs = pd.DataFrame()
    if phrasing_pair:
        cross_phrasing_corrs = compute_cross_phrasing_correlation(
            datamodel_query_df, phrasing_pair[0], phrasing_pair[1]
        )
        if len(cross_phrasing_corrs) > 0:
            print(f"    Cross-Phrasing Correlation: {cross_phrasing_corrs['correlation'].mean():.4f}")

    return per_query_corrs, mean_corr, cross_phrasing_corrs


def run_config(config: Config) -> None:
    """Run analysis for a single config."""
    print(f"\n{'=' * 60}")
    print(f"Running config: {config.name}")
    print(f"{'=' * 60}")
    print(f"Metric: {config.metric_name}")
    if config.phrasing_pair:
        print(f"Phrasing pair: {config.phrasing_pair}")

    doc_type_df = load_doc_type(config.doc_type_dir)

    # Process both checkpoints
    start_corrs, start_mean, start_cross = process_checkpoint(
        config.start_datamodel_dir, doc_type_df, config.metric_name,
        config.phrasing_pair, "First Checkpoint"
    )
    final_corrs, final_mean, final_cross = process_checkpoint(
        config.final_datamodel_dir, doc_type_df, config.metric_name,
        config.phrasing_pair, "Last Checkpoint"
    )

    # Create output directory
    output_dir = _THIS_DIR / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = config.output_filename.rsplit(".", 1)[0] if config.output_filename else f"softmargin_correlation_{config.name}"

    # --- Plot 1: Softmargin vs Entailing (with per-query samples) ---
    print("\n--- Saving plots ---")
    fig1, _ = plot_correlation_with_samples(
        start_corrs, final_corrs, start_mean, final_mean,
        title=f"Softmargin vs Entailing Facts | {config.name}",
    )
    fig1.savefig(output_dir / f"{base_filename}_bar.pdf", bbox_inches="tight")
    print(f"  Saved bar plot to {output_dir / f'{base_filename}_bar.pdf'}")

    # --- Plot 2: Cross-Phrasing Correlation ---
    if config.phrasing_pair and len(start_cross) > 0 and len(final_cross) > 0:
        fig2, _ = plot_cross_phrasing_correlation(
            start_cross, final_cross,
            config.phrasing_pair[0], config.phrasing_pair[1],
            title=f"Cross-Phrasing Softmargin Correlation | {config.name}",
        )
        fig2.savefig(output_dir / f"{base_filename}_cross_phrasing.pdf", bbox_inches="tight")
        print(f"  Saved cross-phrasing plot to {output_dir / f'{base_filename}_cross_phrasing.pdf'}")

    plt.close("all")


if __name__ == "__main__":
    for config in configs:
        run_config(config)

# %%
