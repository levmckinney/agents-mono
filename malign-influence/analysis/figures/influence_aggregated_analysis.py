# %%
"""
Aggregated Influence Analysis

Plots treatment-control influence relationships with metric pairs grouped into
named aggregation groups. Each group becomes a line showing mean Y-value vs step.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats

# %% Constants

_THIS_DIR = Path(__file__).parent
_DATA_DIR = _THIS_DIR / '..' / 'data_frames'

DEFAULT_DATA_DIRS = {
    "mayor_1epoch": _DATA_DIR / "mayors_1epoch_ce",
    "fiction_birth_dates_1epoch": _DATA_DIR / "fiction_birth_dates_1epoch_ce",
    "pretraining_presidents": _DATA_DIR /"pretraining_presidents"
}

# %% Configuration

@dataclass
class Config:
    # Identifier for this config (used in output filename)
    name: str

    # Data directory
    data_dir: Path

    # Named groups of (treatment_metric, control_metric) pairs
    aggregation_groups: dict[str, list[tuple[str, str]]]

    y_mode: Literal["pearson", "spearman", "uncentered_pearson", "normalized_inter_query"] = "pearson"
    pairing_mode: Literal["fully_paired", "different_phrasing", "different_person", "different_all"] = "different_person"
    error_band: Literal["std", "ci95_bootstrap", "sample_lines"] = "sample_lines"  # std, 95% bootstrap CI, or faint lines for each sample
    n_bootstrap: int = 1000  # Number of bootstrap samples

    title: str | None = None
    ylabel: str | None = None
    output_filename: str | None = None
    log_x_axis: bool = False  # Use logarithmic scale for x-axis

    # Log prob plot settings
    log_prob_metrics: list[str] | None = None  # Metrics to plot in log prob chart
    log_prob_legend_names: dict[str, str] | None = None  # Map metric_name -> legend label


# Define all configs to run
configs = [
   Config(
        name="birth_dates_fully_paired",
        title="",
        data_dir=DEFAULT_DATA_DIRS["fiction_birth_dates_1epoch"],
        aggregation_groups={
            "Q: What is the birth date of {person}? A:| {Month D, YYYY} <> Q: What is the birth date of {person}? A:| {DD/MM/YYYY}": [
                ("birth_date_eval_qa_1_no_fs", "birth_date_dmy_eval_qa_1_no_fs"),
            ],
        },
        log_prob_metrics=[
            "birth_date_eval_qa_1_no_fs",
            "birth_date_dmy_eval_qa_1_no_fs",
        ],
        log_prob_legend_names={
            "birth_date_eval_qa_1_no_fs": "Q: What is the birth date of {person}? A:| {Month D, YYYY}",
            "birth_date_dmy_eval_qa_1_no_fs": "Q: What is the birth date of {person}? A:| {DD/MM/YYYY}",
        },
        pairing_mode='fully_paired',
        output_filename='fictional_birth_dates_fully_paired.pdf'
    ),
    Config(
        name="pre_training",
        title="",
        data_dir=DEFAULT_DATA_DIRS['pretraining_presidents'],
        aggregation_groups={
            "Q: Who was the President of the United States in {year}? A:| {first_name} <> Q: Who was the President of the United States in {year}? A:| {last_name}" : [("president_qa_1_first", "president_qa_1_last")], 
        },
        log_prob_metrics=[
            'president_qa_1_last',
            'president_qa_1_first',
        ],
        log_prob_legend_names={
            "president_qa_1_last": "Q: Who was the President of the United States in {year}? A:| {first_name} ",
            "president_qa_1_last": "Q: Who was the President of the United States in {year}? A:| {last_name} ",
        },
        pairing_mode='fully_paired',
        y_mode='normalized_inter_query',
        log_x_axis=True,
        output_filename='pre_training_fully_paired.pdf'
    ),
    Config(
        name="pre_training",
        data_dir=DEFAULT_DATA_DIRS['pretraining_presidents'],
        title="",
        aggregation_groups={
            "Q: Who was the President of the United States in {year}? A:| {first_name} <> Q: Who was the President of the United States in {other_year}? A:| {other_last_name}" : [("president_qa_1_first", "president_qa_1_first")],
            "Q: Who was the President of the United States in {year}? A:| {first_name} <> Q: Who was the President of the United States in {other_year}? A:| {other_first_name}" : [("president_qa_1_first", "president_qa_1_last")],
        },
        log_prob_metrics=[
            'president_qa_1_last',
            'president_qa_1_first',
        ],
        log_x_axis=True,
        pairing_mode='different_person',
        y_mode='normalized_inter_query',
        output_filename='pre_training_unpaired.pdf'
    ),
    Config(
        name="birth_dates_unpaired",
        title="",
        data_dir=DEFAULT_DATA_DIRS["fiction_birth_dates_1epoch"],
        aggregation_groups={
            "Q: What is the birth date of {person}? A:| {Month D, YYYY} <> Q: What is the birth date of {other_person}? A: A:| {DD/MM/YYYY}": [
                ("birth_date_eval_qa_1_no_fs", "birth_date_dmy_eval_qa_1_no_fs"),
            ],
            "Q: What is the birth date of {person}? A:| {Month D, YYYY} <> Q: What is the birth date of {other_person}? A:| {Month D, YYYY}": [
                ("birth_date_eval_qa_1_no_fs", "birth_date_eval_qa_1_no_fs"),
            ],
        },
        pairing_mode='different_person',
        output_filename='unpaired_month_dy_vs_dmy.pdf'
    ),
    Config(
        name="mayor_unpaired_first_to_last",
        title="",
        data_dir=DEFAULT_DATA_DIRS["mayor_1epoch"],
        aggregation_groups={
            "Q: Who is the mayor of {city_name} in 2025? A:| {first_name} <> Q: Who is the mayor of {other_city_name} in 2025? A:| {other_last_name}": [
                ("name_mayor_eval_last_name_qa_1_no_fs", "name_mayor_eval_first_name_qa_1_no_fs"),
            ],
            "Q: Who is the mayor of {city_name} in 2025? A:| {first_name} <> Q: Who is the mayor of {other_city_name} in 2025? A:| {other_first_name}": [
                ("name_mayor_eval_first_name_qa_1_no_fs", "name_mayor_eval_first_name_qa_1_no_fs"),
            ],
        },
        pairing_mode='different_person',
        output_filename="unpaired_first_to_last.pdf"
    ),
    Config(
        name="mayor_unpaired_last_to_first",
        title="",
        data_dir=DEFAULT_DATA_DIRS["mayor_1epoch"],
        aggregation_groups={
            "Q: Who is the mayor of {city_name} in 2025? A:| {last_name} <> Q: Who is the mayor of {other_city_name} in 2025? A:| {other_first_name}": [
                ("name_mayor_eval_last_name_qa_1_no_fs", "name_mayor_eval_first_name_qa_1_no_fs"),
            ],
            "Q: Who is the mayor of {city_name} in 2025? A:| {last_name} <> Q: Who is the mayor of {other_city_name} in 2025? A:| {other_last_name}": [
                ("name_mayor_eval_last_name_qa_1_no_fs", "name_mayor_eval_last_name_qa_1_no_fs"),
            ],
        },
        pairing_mode='different_person',
        output_filename="unpaired_last_to_first.pdf"
    ),
    Config(
        name="mayor_paired_first_to_last",
        title="",
        data_dir=DEFAULT_DATA_DIRS["mayor_1epoch"],
        aggregation_groups={
            "Q: Who is the mayor of {city_name} in 2025? A:| {first_name} <> Q: Who is the mayor of {city_name} in 2025? A:| {last_name}": [
                ("name_mayor_eval_first_name_qa_1_no_fs", "name_mayor_eval_last_name_qa_1_no_fs"),
            ],
        },
        log_prob_metrics=[
            "name_mayor_eval_first_name_qa_1_no_fs",
            "name_mayor_eval_last_name_qa_1_no_fs",
        ],
        log_prob_legend_names={
            "name_mayor_eval_first_name_qa_1_no_fs": "Q: Who is the mayor of {city_name} in 2025? A:| {first_name}",
            "name_mayor_eval_last_name_qa_1_no_fs": "Q: Who is the mayor of {city_name} in 2025? A:| {last_name}",
        },
        pairing_mode='fully_paired',
        output_filename="paired_first_to_last.pdf"
    ),
]

# %%

def derive_checkpoints_from_data(data_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Derive checkpoint order and step mapping from query_log_prob_df."""
    df = pd.read_csv(data_dir / "query_log_prob_df.csv", usecols=["checkpoint_name", "step"])
    df = df[
        df["checkpoint_name"].notna()
    ]
    # Group by checkpoint_name and take min step to deduplicate
    # (some checkpoints may have multiple step values from different eval runs)
    checkpoint_steps = df.groupby("checkpoint_name")["step"].min().reset_index()
    checkpoint_steps = checkpoint_steps.sort_values("step")
    checkpoint_order = checkpoint_steps["checkpoint_name"].tolist()
    checkpoint_to_step = dict(zip(checkpoint_steps["checkpoint_name"], checkpoint_steps["step"].astype(int)))
    return checkpoint_order, checkpoint_to_step


# %% Data Loading


class LazyDataFrameLoader:
    """Lazily loads dataframes only when accessed."""

    def __init__(self, data_dir: Path, metrics: list[str]):
        self._data_dir = data_dir
        self._metrics = set(metrics)
        self._cache: dict[str, pd.DataFrame] = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        if key not in self._cache:
            self._cache[key] = self._load(key)
        return self._cache[key]

    def _matches_metric(self, x: str) -> bool:
        return x in self._metrics

    def _load(self, key: str) -> pd.DataFrame:
        if key == "inter_query":
            print("Loading inter-query influence data...")
            chunks = []
            for chunk in pd.read_csv(
                self._data_dir / "inter_q_influence_df.csv",
                usecols=["query_id", "train_id", "influence_score", "metric_name", "checkpoint_name"],
                chunksize=500_000,
                engine="python",
            ):
                filtered = chunk[chunk["metric_name"].apply(self._matches_metric)]
                if len(filtered) > 0:
                    chunks.append(filtered)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            print(f"  Loaded {len(df):,} inter-query influence rows")
            return df

        elif key == "influence":
            print("Loading document influence data...")
            chunks = []
            for chunk in pd.read_csv(
                self._data_dir / "influence_to_spans_df.csv",
                usecols=["query_id", "span_id", "influence_score", "metric_name", "checkpoint_name"],
                chunksize=500_000,
                engine="python",
            ):
                filtered = chunk[chunk["metric_name"].apply(self._matches_metric)]
                if len(filtered) > 0:
                    chunks.append(filtered)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            print(f"  Loaded {len(df):,} document influence rows")
            return df

        elif key == "unpacked":
            print("Loading unpacked dataset...")
            return pd.read_csv(self._data_dir / "unpacked_ds_df.csv", usecols=["id", "doc_id"])

        elif key == "query_log_prob":
            print("Loading query log probabilities...")
            return pd.read_csv(self._data_dir / "query_log_prob_df.csv")

        elif key == "train_all":
            print("Loading query metadata...")
            return pd.read_csv(
                self._data_dir / "train_all_df.csv",
                usecols=["query_id", "person", "city", "metric_name"],
            )

        else:
            raise KeyError(f"Unknown dataframe key: {key}")


def get_all_metrics(metric_pairs: list[tuple[str, str]]) -> list[str]:
    """Extract all unique metric names from metric pairs."""
    metrics = set()
    for treatment, control in metric_pairs:
        metrics.add(treatment)
        metrics.add(control)
    return list(metrics)


# %% Query Pairing Functions


def get_queries_with_phrasing_index(
    query_log_prob: pd.DataFrame,
    metric_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Get all queries with their phrasing index (which metric pair they belong to)."""
    results = []
    for phrasing_idx, (treatment_metric, control_metric) in enumerate(metric_pairs):
        treatment = query_log_prob[query_log_prob["metric_name"] == treatment_metric][
            ["query_id", "person", "metric_name"]
        ].drop_duplicates()
        treatment["phrasing_idx"] = phrasing_idx
        treatment["is_treatment"] = True
        results.append(treatment)

        control = query_log_prob[query_log_prob["metric_name"] == control_metric][
            ["query_id", "person", "metric_name"]
        ].drop_duplicates()
        control["phrasing_idx"] = phrasing_idx
        control["is_treatment"] = False
        results.append(control)

    return pd.concat(results, ignore_index=True)


def create_query_pairs_by_mode(queries_df: pd.DataFrame, pairing_mode: str) -> pd.DataFrame:
    """Create treatment-control pairs based on pairing mode."""
    treatment_queries = queries_df[queries_df["is_treatment"]].copy()
    treatment_queries = treatment_queries.rename(columns={
        "query_id": "treatment_query_id",
        "person": "treatment_person",
        "phrasing_idx": "treatment_phrasing_idx",
    })

    control_queries = queries_df[~queries_df["is_treatment"]].copy()
    control_queries = control_queries.rename(columns={
        "query_id": "control_query_id",
        "person": "control_person",
        "phrasing_idx": "control_phrasing_idx",
    })

    if pairing_mode == "fully_paired":
        pairs = treatment_queries.merge(
            control_queries,
            left_on=["treatment_person", "treatment_phrasing_idx"],
            right_on=["control_person", "control_phrasing_idx"],
        )
    elif pairing_mode == "different_phrasing":
        pairs = treatment_queries.merge(control_queries, left_on="treatment_person", right_on="control_person")
        pairs = pairs[pairs["treatment_phrasing_idx"] != pairs["control_phrasing_idx"]]
    elif pairing_mode == "different_person":
        pairs = treatment_queries.merge(control_queries, left_on="treatment_phrasing_idx", right_on="control_phrasing_idx")
        pairs = pairs[pairs["treatment_person"] != pairs["control_person"]]
    elif pairing_mode == "different_all":
        pairs = treatment_queries.merge(control_queries, how="cross")
        pairs = pairs[
            (pairs["treatment_phrasing_idx"] != pairs["control_phrasing_idx"]) &
            (pairs["treatment_person"] != pairs["control_person"])
        ]
    else:
        raise ValueError(f"Unknown pairing mode: {pairing_mode}")

    return pairs[[
        "treatment_query_id", "control_query_id",
        "treatment_person", "control_person",
        "treatment_phrasing_idx", "control_phrasing_idx",
    ]].drop_duplicates()


# %% Core Computation Functions


def aggregate_spans_to_docs(influence_df: pd.DataFrame, unpacked_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate span-level influences to document-level by summing."""
    span_to_doc = unpacked_df[["id", "doc_id"]].drop_duplicates().rename(columns={"id": "span_id"})
    influence_with_doc = influence_df.merge(span_to_doc, on="span_id")
    return (
        influence_with_doc.groupby(["query_id", "doc_id", "checkpoint_name", "metric_name"])["influence_score"]
        .sum()
        .reset_index()
    )


def compute_query_self_influence(inter_q_df: pd.DataFrame) -> pd.DataFrame:
    """Extract self-influence for each query (where query_id == train_id)."""
    self_inf = inter_q_df[inter_q_df["query_id"] == inter_q_df["train_id"]].copy()
    return self_inf[["query_id", "checkpoint_name", "influence_score"]].rename(columns={"influence_score": "self_influence"})


def compute_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """Compute correlation between two vectors."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    if method == "pearson":
        return stats.pearsonr(x, y)[0]
    elif method == "spearman":
        return stats.spearmanr(x, y)[0]
    elif method == "uncentered_pearson":
        denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
        return np.sum(x * y) / denom if denom > 0 else np.nan
    else:
        raise ValueError(f"Unknown correlation method: {method}")


def compute_doc_influence_correlation(
    doc_influence_df: pd.DataFrame, query1_id: str, query2_id: str, checkpoint: str, method: str
) -> float:
    """Compute correlation of document influence vectors between two queries."""
    q1_inf = doc_influence_df[
        (doc_influence_df["query_id"] == query1_id) & (doc_influence_df["checkpoint_name"] == checkpoint)
    ][["doc_id", "influence_score"]].set_index("doc_id")

    q2_inf = doc_influence_df[
        (doc_influence_df["query_id"] == query2_id) & (doc_influence_df["checkpoint_name"] == checkpoint)
    ][["doc_id", "influence_score"]].set_index("doc_id")

    aligned = q1_inf.join(q2_inf, lsuffix="_q1", rsuffix="_q2", how="inner")
    if len(aligned) == 0:
        return np.nan
    return compute_correlation(aligned["influence_score_q1"].values, aligned["influence_score_q2"].values, method)


def compute_y_values(
    pairs_df: pd.DataFrame,
    dfs: LazyDataFrameLoader,
    y_mode: str,
    checkpoints: list[str],
    checkpoint_to_step: dict[str, int],
    doc_influence_df: pd.DataFrame | None = None,
    self_inf_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute Y-axis values for all pairs across all checkpoints."""
    results = []

    if y_mode == "normalized_inter_query":
        inter_q_lookup = dfs["inter_query"].set_index(["query_id", "train_id", "checkpoint_name"])["influence_score"]
        self_inf_lookup = self_inf_df.set_index(["query_id", "checkpoint_name"])["self_influence"]

        for _, pair in pairs_df.iterrows():
            treatment_id, control_id = pair["treatment_query_id"], pair["control_query_id"]
            for checkpoint in checkpoints:
                try:
                    influence = inter_q_lookup.loc[(treatment_id, control_id, checkpoint)]
                    self_t = self_inf_lookup.loc[(treatment_id, checkpoint)]
                    self_c = self_inf_lookup.loc[(control_id, checkpoint)]
                    normalizer = np.sqrt(np.abs(self_t) * np.abs(self_c))
                    y_value = influence / normalizer if normalizer > 0 else np.nan
                except KeyError:
                    y_value = np.nan

                results.append({
                    "treatment_query_id": treatment_id,
                    "control_query_id": control_id,
                    "treatment_person": pair["treatment_person"],
                    "control_person": pair["control_person"],
                    "checkpoint_name": checkpoint,
                    "step": checkpoint_to_step[checkpoint],
                    "y_value": y_value,
                })
    else:
        for _, pair in pairs_df.iterrows():
            treatment_id, control_id = pair["treatment_query_id"], pair["control_query_id"]
            for checkpoint in checkpoints:
                y_value = compute_doc_influence_correlation(doc_influence_df, treatment_id, control_id, checkpoint, y_mode)
                results.append({
                    "treatment_query_id": treatment_id,
                    "control_query_id": control_id,
                    "treatment_person": pair["treatment_person"],
                    "control_person": pair["control_person"],
                    "checkpoint_name": checkpoint,
                    "step": checkpoint_to_step[checkpoint],
                    "y_value": y_value,
                })

    return pd.DataFrame(results)




# %% Processing


def process_group(
    group_name: str,
    metric_pairs: list[tuple[str, str]],
    dfs: dict[str, pd.DataFrame],
    y_mode: str,
    pairing_mode: str,
    checkpoint_order: list[str],
    checkpoint_to_step: dict[str, int],
    doc_influence: pd.DataFrame | None,
    self_inf: pd.DataFrame | None,
) -> pd.DataFrame:
    """Process a single aggregation group and return results with group_name column."""
    queries_df = get_queries_with_phrasing_index(dfs["query_log_prob"], metric_pairs)
    pairs = create_query_pairs_by_mode(queries_df, pairing_mode)

    if len(pairs) == 0:
        return pd.DataFrame()

    result_df = compute_y_values(
        pairs, dfs, y_mode, checkpoint_order,
        checkpoint_to_step=checkpoint_to_step,
        doc_influence_df=doc_influence,
        self_inf_df=self_inf,
    )
    result_df["group_name"] = group_name
    return result_df


def bootstrap_ci(
    values: np.ndarray,  # type: ignore[type-arg]
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Returns (lower, upper) bounds.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        return (float(values[0]), float(values[0]))

    # Generate bootstrap samples and compute means
    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample_indices = rng.integers(0, n, size=n)
        bootstrap_means[i] = values[sample_indices].mean()

    # Compute percentiles
    alpha = (1 - ci) / 2
    lower = float(np.percentile(bootstrap_means, alpha * 100))
    upper = float(np.percentile(bootstrap_means, (1 - alpha) * 100))

    return (lower, upper)


def aggregate_by_group_and_step(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Aggregate Y-values by group and step with bootstrap CI."""
    rng = np.random.default_rng(42)

    results = []
    for (group_name, step), group_df in df.groupby(["group_name", "step"]):
        values = np.asarray(group_df["y_value"].dropna())
        y_mean = float(values.mean()) if len(values) > 0 else np.nan
        y_std = float(values.std()) if len(values) > 1 else 0.0
        n_pairs = len(values)

        ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=n_bootstrap, rng=rng)

        results.append({
            "group_name": group_name,
            "step": step,
            "y_mean": y_mean,
            "y_std": y_std,
            "n_pairs": n_pairs,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

    return pd.DataFrame(results)


def aggregate_log_prob_by_metric(
    log_prob_df: pd.DataFrame,
    metrics: list[str],
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """Aggregate log probabilities by metric and step with bootstrap CI."""
    rng = np.random.default_rng(42)

    # Filter to only the specified metrics
    filtered_df = log_prob_df[log_prob_df["metric_name"].isin(metrics)]

    results = []
    for (metric_name, step), group_df in filtered_df.groupby(["metric_name", "step"]):
        values = np.asarray(group_df["log_prob"].dropna())
        mean_log_prob = float(values.mean()) if len(values) > 0 else np.nan
        std_log_prob = float(values.std()) if len(values) > 1 else 0.0
        n_queries = len(values)

        ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=n_bootstrap, rng=rng)

        results.append({
            "metric_name": metric_name,
            "step": step,
            "mean_log_prob": mean_log_prob,
            "std_log_prob": std_log_prob,
            "n_queries": n_queries,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })

    return pd.DataFrame(results)


# %% Plotting


def plot_aggregated(
    df: pd.DataFrame,
    y_mode: str,
    error_band: Literal["std", "ci95_bootstrap", "sample_lines"] = "ci95_bootstrap",
    title: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    raw_data: pd.DataFrame | None = None,
    log_x_axis: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot aggregated results with one line per group.

    Args:
        df: Aggregated data with y_mean, y_std, ci_lower, ci_upper per group/step.
        y_mode: Correlation method used.
        error_band: How to show variance - "std", "ci95_bootstrap", or "sample_lines".
        title: Plot title.
        ylabel: Y-axis label.
        figsize: Figure size.
        raw_data: Raw data with individual y_value per sample. Required for "sample_lines" mode.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if ylabel is None:
        ylabel = "Normalized Inter-Query Influence" if y_mode == "normalized_inter_query" else f"Correlation ({y_mode})"

    sns.lineplot(
        data=df,
        x="step",
        y="y_mean",
        hue="group_name",
        marker="o",
        ax=ax,
    )

    # Add error bands/lines for each group
    palette = sns.color_palette()
    for i, group_name in enumerate(df["group_name"].unique()):
        group_data = df[df["group_name"] == group_name].sort_values("step")
        color = palette[i % len(palette)]

        if error_band == "sample_lines":
            # Draw faint lines for each individual sample
            if raw_data is None:
                raise ValueError("raw_data must be provided for sample_lines error_band mode")

            raw_group = raw_data[raw_data["group_name"] == group_name]
            # Create a unique sample identifier from treatment/control pair
            raw_group = raw_group.copy()
            raw_group["sample_id"] = raw_group["treatment_query_id"] + "_" + raw_group["control_query_id"]

            for sample_id in raw_group["sample_id"].unique():
                sample_data = raw_group[raw_group["sample_id"] == sample_id].sort_values("step")
                ax.plot(
                    sample_data["step"],
                    sample_data["y_value"],
                    color=color,
                    alpha=0.15,
                    linewidth=0.5,
                    zorder=1,
                )
        elif error_band == "ci95_bootstrap":
            # Use pre-computed bootstrap CI bounds
            lower = group_data["ci_lower"]
            upper = group_data["ci_upper"]
            ax.fill_between(
                group_data["step"],
                lower,
                upper,
                color=color,
                alpha=0.2,
            )
        else:
            # Fallback to std
            lower = group_data["y_mean"] - group_data["y_std"]
            upper = group_data["y_mean"] + group_data["y_std"]
            ax.fill_between(
                group_data["step"],
                lower,
                upper,
                color=color,
                alpha=0.2,
            )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5, zorder=0)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    if log_x_axis:
        ax.set_xscale("log")
    if title:
        ax.set_title(title)
    ax.legend(title="Influence Pair", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    return fig, ax


def plot_log_prob_by_metric(
    df: pd.DataFrame,
    legend_names: dict[str, str] | None = None,
    error_band: Literal["std", "ci95_bootstrap", "sample_lines"] = "ci95_bootstrap",
    title: str | None = None,
    ylabel: str = "Mean Log Probability",
    figsize: tuple[float, float] = (10, 6),
    raw_data: pd.DataFrame | None = None,
    metrics: list[str] | None = None,
    log_x_axis: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot mean log probability per metric across training steps.

    Args:
        df: Aggregated data with mean_log_prob, std_log_prob, ci_lower, ci_upper per metric/step.
        legend_names: Map metric_name -> display label.
        error_band: How to show variance - "std", "ci95_bootstrap", or "sample_lines".
        title: Plot title.
        ylabel: Y-axis label.
        figsize: Figure size.
        raw_data: Raw log_prob_df data. Required for "sample_lines" mode.
        metrics: List of metrics to include in sample_lines. Required for "sample_lines" mode.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create a copy with display names for legend
    plot_df = df.copy()
    if legend_names:
        plot_df["display_name"] = plot_df["metric_name"].map(
            lambda x: legend_names.get(x, x)
        )
    else:
        plot_df["display_name"] = plot_df["metric_name"]

    sns.lineplot(
        data=plot_df,
        x="step",
        y="mean_log_prob",
        hue="display_name",
        marker="o",
        ax=ax,
    )

    # Add error bands/lines for each metric
    palette = sns.color_palette()
    unique_metrics = plot_df["metric_name"].unique()
    for i, metric_name in enumerate(unique_metrics):
        metric_data = plot_df[plot_df["metric_name"] == metric_name].sort_values("step")
        color = palette[i % len(palette)]

        if error_band == "sample_lines":
            # Draw faint lines for each individual query
            if raw_data is None or metrics is None:
                raise ValueError("raw_data and metrics must be provided for sample_lines error_band mode")

            raw_metric = raw_data[raw_data["metric_name"] == metric_name]

            for query_id in raw_metric["query_id"].unique():
                query_data = raw_metric[raw_metric["query_id"] == query_id].sort_values("step")
                ax.plot(
                    query_data["step"],
                    query_data["log_prob"],
                    color=color,
                    alpha=0.15,
                    linewidth=0.5,
                    zorder=1,
                )
        elif error_band == "ci95_bootstrap":
            lower = metric_data["ci_lower"]
            upper = metric_data["ci_upper"]
            ax.fill_between(
                metric_data["step"],
                lower,
                upper,
                color=color,
                alpha=0.2,
            )
        else:
            lower = metric_data["mean_log_prob"] - metric_data["std_log_prob"]
            upper = metric_data["mean_log_prob"] + metric_data["std_log_prob"]
            ax.fill_between(
                metric_data["step"],
                lower,
                upper,
                color=color,
                alpha=0.2,
            )

    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    if log_x_axis:
        ax.set_xscale("log")
    if title:
        ax.set_title(title)
    ax.legend(title="Metric")

    fig.tight_layout()
    return fig, ax


# %% Main


def run_config(config: Config) -> None:
    """Run analysis for a single config."""
    print(f"\n{'='*60}")
    print(f"Running config: {config.name}")
    print(f"{'='*60}")
    print(f"Data dir: {config.data_dir}")
    print(f"Y-mode: {config.y_mode}")
    print(f"Pairing mode: {config.pairing_mode}")
    print(f"Groups: {list(config.aggregation_groups.keys())}")

    # Derive checkpoints from data
    print("\nDeriving checkpoints from data...")
    checkpoint_order, checkpoint_to_step = derive_checkpoints_from_data(config.data_dir)
    print(f"  Found {len(checkpoint_order)} checkpoints: {checkpoint_order}")
    print(f"  Step mapping: {checkpoint_to_step}")

    # Flatten all metric pairs across groups
    all_metric_pairs = [pair for pairs in config.aggregation_groups.values() for pair in pairs]
    all_metrics = get_all_metrics(all_metric_pairs)
    print(f"\nSetting up lazy loader for {len(all_metrics)} metrics...")
    dfs = LazyDataFrameLoader(config.data_dir, all_metrics)

    # Prepare doc influence if needed
    doc_influence = None
    if config.y_mode != "normalized_inter_query":
        print("Aggregating spans to documents...")
        doc_influence = aggregate_spans_to_docs(dfs["influence"], dfs["unpacked"])

    # Prepare self-influence if needed
    self_inf = None
    if config.y_mode == "normalized_inter_query":
        print("Computing query self-influence...")
        self_inf = compute_query_self_influence(dfs["inter_query"])

    # Process each group
    print("Processing groups...")
    group_results = []
    for group_name, metric_pairs in config.aggregation_groups.items():
        print(f"  {group_name}: {len(metric_pairs)} metric pairs")
        result = process_group(
            group_name, metric_pairs, dfs, config.y_mode, config.pairing_mode,
            checkpoint_order, checkpoint_to_step,
            doc_influence, self_inf,
        )
        if len(result) > 0:
            group_results.append(result)

    all_results = pd.concat(group_results, ignore_index=True)
    print(f"Total data points: {len(all_results)}")

    # Aggregate with bootstrap CI
    print(f"Computing bootstrap CIs ({config.n_bootstrap} samples)...")
    aggregated = aggregate_by_group_and_step(all_results, n_bootstrap=config.n_bootstrap)
    print(f"Aggregated rows: {len(aggregated)}")

    # Plot
    default_title = f"Aggregated Influence | {config.name} | {config.y_mode} | {config.pairing_mode}"
    fig, ax = plot_aggregated(
        aggregated,
        y_mode=config.y_mode,
        error_band=config.error_band,
        title=config.title or default_title,
        ylabel=config.ylabel,
        raw_data=all_results if config.error_band == "sample_lines" else None,
        log_x_axis=config.log_x_axis,
    )

    # Save
    output_dir = _THIS_DIR / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = config.output_filename or f"influence_aggregated_{config.name}_{config.y_mode}_{config.pairing_mode}.pdf"
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")

    # Generate log prob plot if metrics are specified
    if config.log_prob_metrics:
        print(f"\nGenerating log prob plot for {len(config.log_prob_metrics)} metrics...")
        log_prob_df = dfs["query_log_prob"]
        log_prob_aggregated = aggregate_log_prob_by_metric(
            log_prob_df,
            config.log_prob_metrics,
            n_bootstrap=config.n_bootstrap,
        )
        print(f"  Aggregated {len(log_prob_aggregated)} rows for log prob plot")

        log_prob_title = f"Mean Log Probability | {config.name}"
        log_prob_fig, _ = plot_log_prob_by_metric(
            log_prob_aggregated,
            legend_names=config.log_prob_legend_names,
            error_band=config.error_band,
            title=log_prob_title,
            raw_data=log_prob_df if config.error_band == "sample_lines" else None,
            metrics=config.log_prob_metrics if config.error_band == "sample_lines" else None,
            log_x_axis=config.log_x_axis,
        )

        # Save log prob plot with _log_prob suffix
        base_filename = filename.rsplit(".", 1)[0]  # Remove extension
        log_prob_output_path = output_dir / f"{base_filename}_log_prob.pdf"
        log_prob_fig.savefig(log_prob_output_path, dpi=150, bbox_inches="tight")
        print(f"Saved log prob plot to {log_prob_output_path}")




if __name__ == "__main__" or True:
    for config in configs:
        run_config(config)

# %%
