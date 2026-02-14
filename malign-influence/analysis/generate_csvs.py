# %%

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from datasets import load_from_disk
from oocr_influence.cli.run_influence import load_influence_scores, load_inter_query_influence_scores, load_self_influence_scores, load_logprobs
from oocr_influence.datasets.synthetic_pretraining_docs import DocSpec
from oocr_influence.datasets.document_dataset import load_structured_dataset
from shared_ml.logging import load_log_from_disk
from shared_ml.disk_lru_cache import disk_lru_cache

# %% Configuration

# Type for fact type extraction functions: (target_id_1, target_id_2, doc_spec_str) -> fact_type
FactTypeFn = Callable[[str, str, str | None], str]


@dataclass
class GenerateCSVsConfig:
    """Configuration for generating CSV files from experiment outputs."""

    name: str  # Used for output directory naming
    data_model_path: Path  # Path to fine-tuning experiment output
    influence_path: Path | None = None  # Path to influence run (optional)
    output_dir: Path | None = None  # Output directory (auto-generated if None)

    # Optional fact type extraction
    fact_type_fn: FactTypeFn | None = None  # If None, skip doc_type_df generation
    fact_type_target_columns: tuple[str, str] = ("person", "city")  # Columns to use as target IDs


def derive_checkpoint_to_step_mapping(data_model_path: Path) -> pd.DataFrame:
    """Derive checkpoint_name -> step mapping from all_docs_runs."""
    all_docs_dir = data_model_path / "all_docs_runs"
    mappings = []

    for run_dir in all_docs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        for dir in run_dir.glob('checkpoint_*'):
            checkpoint_name = dir.name

            # Extract step from checkpoint name
            if checkpoint_name == "checkpoint_start":
                step = 1
            elif match := re.search(r"_s(\d+)$", checkpoint_name):
                step = int(match.group(1))
            elif checkpoint_name == "checkpoint_final":
                # Get final step from experiment log
                experiment_log = load_log_from_disk(run_dir, load_pickled=False)
                if experiment_log.history:
                    step = experiment_log.history[-1]["step_num"]
                else:
                    continue
            else:
                # Unknown format, try to get from experiment log
                experiment_log = load_log_from_disk(run_dir, load_pickled=False)
                if experiment_log.history:
                    step = experiment_log.history[0]["step_num"]
                else:
                    continue
            mappings.append({"checkpoint_name": checkpoint_name, "step": step})
            print(f"{mappings[-1]=}")

    return pd.DataFrame.from_records(mappings).sort_values("step")


def map_eval_steps_to_checkpoints(
    eval_df: pd.DataFrame,
    checkpoint_df: pd.DataFrame,
    step_col: str = "step",
) -> pd.DataFrame:
    """
    Map each eval step to the most recent checkpoint at or before that step.

    Uses pd.merge_asof to perform an "as-of" join, assigning each eval row
    to the checkpoint with the largest step <= the eval step.

    Args:
        eval_df: DataFrame with evaluation data containing a step column
        checkpoint_df: DataFrame with checkpoint_name and step columns
        step_col: Name of the step column in both DataFrames

    Returns:
        eval_df with checkpoint_name column added
    """
    # Ensure both DataFrames are sorted by step for merge_asof
    eval_df = eval_df.sort_values(step_col).copy()
    checkpoint_df = checkpoint_df.sort_values(step_col).copy()

    # merge_asof matches each row in left to the row in right with the
    # largest key <= the left key (direction='backward')
    result = pd.merge_asof(
        eval_df,
        checkpoint_df,
        on=step_col,
        direction="nearest",
    )

    return result


# %% Load relavent dataframes

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
        "mean_rank": float(record["mean_rank"]),
        "prompt": record["prompt"],
        "completion": record["completion"],
        "relation": fact_template["relation"],
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

@disk_lru_cache()
def _load_all_docs_run(root_dir: Path):
    """Load all docs for a run."""
    rows: List[Dict[str, object]] = []
    all_docs_dir = root_dir / "all_docs_runs"

    for checkpoint_dir in tqdm(list(all_docs_dir.iterdir()), desc="Scanning runs for all docs"):
        if not checkpoint_dir.is_dir():
            continue
        try:
            experiment_log = load_log_from_disk(checkpoint_dir, load_pickled=False)
            run_id = _generate_run_id(checkpoint_dir)

            for timestep in experiment_log.history:
                metrics = timestep["eval_results"]
                step = timestep["step_num"]

                rows.extend(_process_eval_records(metrics, run_id, "all_docs", step))
        except Exception as e:
            print(f"Skipping run due to error: {e}")
            continue

    dataset_path = root_dir / "structured_dataset_all.json"
    structured_dataset, _, _ = load_structured_dataset(dataset_path)
    dataset_id = "all_docs"
    train_all_df = structured_dataset.prepare().to_pandas()
    assert isinstance(train_all_df, pd.DataFrame)
    train_all_df = train_all_df.assign(dataset_id=dataset_id)

    return pd.DataFrame(rows), train_all_df, structured_dataset


@disk_lru_cache()
def _load_influence(inf_root_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    influence_runs = [run for run in inf_root_dir.glob("*") if run.is_dir()]
    influence_rows = []
    self_influence_rows = []
    datasets = []
    for run in tqdm(influence_runs):
        try:
            log_state = load_log_from_disk(run, load_pickled=False)
            assert log_state.args is not None
            parent_ft_run = Path(log_state.args["target_experiment_dir"])
            parent_run_id = _generate_run_id(parent_ft_run)
            run_id = _generate_run_id(run)
            unpacked_dataset_path = log_state.log_dict['unpacked_dataset_path']
            unpacked_dataset = load_from_disk(unpacked_dataset_path)
            checkpoint_name = log_state.args["checkpoint_name"]
            influence_scores = load_influence_scores(run)
            datasets.append(unpacked_dataset.to_pandas().assign(run_id=run_id))

            for metric_name, inf_df in influence_scores.items():
                influence_rows.append(
                    inf_df.assign(
                        run_id=run_id,
                        metric_name=metric_name,
                        checkpoint_name=checkpoint_name,
                        parent_run_id=parent_run_id,
                    )
                )

            self_influence_scores = load_self_influence_scores(run)
            self_influence_rows.append(
                self_influence_scores.assign(
                    run_id=run_id,
                    checkpoint_name=checkpoint_name,
                    parent_run_id=parent_run_id,
                )
            )

        except Exception as e:
            print(f"Skipping due to error {e}")

    # Handle case where no span-level influence data exists
    if influence_rows:
        influence_df = pd.concat(influence_rows)
        influence_df = influence_df.rename(columns={'per_token_influence_score': 'influence_score'})
    else:
        influence_df = pd.DataFrame()

    if self_influence_rows:
        self_influence_df = pd.concat(self_influence_rows)
    else:
        self_influence_df = pd.DataFrame()

    if datasets:
        datasets_df = pd.concat(datasets)
    else:
        datasets_df = pd.DataFrame()

    return influence_df, self_influence_df, datasets_df


@disk_lru_cache()
def _load_query_logprobs(inf_root_dir: Path) -> pd.DataFrame:
    """Load query log probabilities from all influence runs."""
    influence_runs = [run for run in inf_root_dir.glob("*") if run.is_dir()]
    logprob_rows = []

    for run in tqdm(influence_runs, desc="Loading logprobs"):
        try:
            log_state = load_log_from_disk(run, load_pickled=False)
            assert log_state.args is not None
            parent_ft_run = Path(log_state.args["target_experiment_dir"])
            parent_run_id = _generate_run_id(parent_ft_run)
            run_id = _generate_run_id(run)
            checkpoint_name = log_state.args["checkpoint_name"]
            temperature = log_state.args.get("temperature", 1.0)

            # Load logprobs
            query_logprobs, train_logprobs, id_mapping = load_logprobs(run)

            # Create DataFrame for query logprobs
            query_ids = id_mapping["query_ids"]
            for i, qid in enumerate(query_ids):
                logprob_rows.append({
                    "query_id": qid,
                    "log_prob": float(query_logprobs[i]),
                    "checkpoint_name": checkpoint_name,
                    "temperature": temperature,
                    "run_id": run_id,
                    "parent_run_id": parent_run_id,
                })

        except Exception as e:
            print(f"Skipping logprobs for {run.name} due to error: {e}")

    if logprob_rows:
        return pd.DataFrame.from_records(logprob_rows)
    else:
        return pd.DataFrame()


def _aggregate_sampled_by_prompt_id(
    df: pd.DataFrame,
    logprobs_df: pd.DataFrame,
    id_col: str = 'query_id',
    target_col: str = 'span_id',
    score_col: str = 'influence_score',
) -> pd.DataFrame:
    """Aggregate sampled queries by prompt_id, weighted by probability.

    For sampled queries that share the same prompt_id (but have different completions),
    compute a weighted average of influence scores where weights are derived from
    the probability of each completion (exp of log probability, normalized).

    Args:
        df: DataFrame with influence scores (e.g., influence_to_spans_df or inter_q_influence_df)
        logprobs_df: DataFrame with log probabilities per query_id and checkpoint
        id_col: Column name for query ID
        target_col: Column name for target (span_id or train_id)
        score_col: Column name for the score to aggregate

    Returns:
        DataFrame with aggregated rows, with metric_name having '_agg' suffix
    """
    if df.empty or logprobs_df.empty:
        return pd.DataFrame()

    # Filter to sampled metrics only
    sampled_df = df[df['metric_name'].str.contains('sampled', na=False)].copy()
    if sampled_df.empty:
        return pd.DataFrame()

    # Extract prompt_id from query_id (format: {prompt_id}_gen_{idx})
    def extract_prompt_id(query_id: str) -> str | None:
        match = re.match(r"(.+)_gen_\d+$", query_id)
        return match.group(1) if match else None

    sampled_df['prompt_id'] = sampled_df[id_col].apply(extract_prompt_id)

    # Drop rows where we couldn't extract prompt_id
    sampled_df = sampled_df.dropna(subset=['prompt_id'])
    if sampled_df.empty:
        return pd.DataFrame()

    # Merge with logprobs to get log_prob for each query at each checkpoint
    # Need to match on query_id and checkpoint_name
    sampled_df = sampled_df.merge(
        logprobs_df[['query_id', 'checkpoint_name', 'log_prob']],
        left_on=[id_col, 'checkpoint_name'],
        right_on=['query_id', 'checkpoint_name'],
        how='left',
        suffixes=('', '_logprob')
    )

    # Drop rows without log_prob
    sampled_df = sampled_df.dropna(subset=['log_prob'])
    if sampled_df.empty:
        return pd.DataFrame()

    # Convert log_prob to probability
    sampled_df['prob'] = np.exp(sampled_df['log_prob'])

    # Group by (prompt_id, target_col, checkpoint_name, metric_name) and compute weighted average
    # First, compute normalized weights within each group
    group_cols = ['prompt_id', target_col, 'checkpoint_name', 'metric_name', 'run_id', 'parent_run_id']

    # Calculate sum of probabilities per group for normalization
    prob_sums = sampled_df.groupby(group_cols)['prob'].transform('sum')
    sampled_df['weight'] = sampled_df['prob'] / prob_sums

    # Compute weighted score
    sampled_df['weighted_score'] = sampled_df[score_col] * sampled_df['weight']

    # Aggregate
    agg_df = sampled_df.groupby(group_cols).agg({
        'weighted_score': 'sum',
    }).reset_index()

    # Rename columns
    agg_df = agg_df.rename(columns={
        'weighted_score': score_col,
        'prompt_id': id_col,  # Use prompt_id as the new query_id
    })

    # Add _agg suffix to query_id and metric_name
    agg_df[id_col] = agg_df[id_col] + '_agg'
    agg_df['metric_name'] = agg_df['metric_name'] + '_agg'

    return agg_df


def _add_rank_metrics(
    df: pd.DataFrame,
    logprobs_df: pd.DataFrame,
    id_col: str = 'query_id',
    target_col: str = 'span_id',
    score_col: str = 'influence_score',
) -> pd.DataFrame:
    """Add rank1 and rank2 metrics for top 2 most probable queries per prompt_id.

    For each prompt_id and checkpoint, identifies the queries with highest and
    second-highest log probability, then creates new rows with _rank1 and _rank2
    metric suffixes.

    Args:
        df: DataFrame with influence scores (e.g., influence_to_spans_df or inter_q_influence_df)
        logprobs_df: DataFrame with log probabilities per query_id and checkpoint
        id_col: Column name for query ID
        target_col: Column name for target (span_id or train_id)
        score_col: Column name for the score to aggregate

    Returns:
        DataFrame with rank1 and rank2 rows
    """
    if df.empty or logprobs_df.empty:
        return pd.DataFrame()

    # Filter to sampled metrics only (exclude already-aggregated _agg metrics)
    sampled_df = df[
        df['metric_name'].str.contains('sampled', na=False) &
        ~df['metric_name'].str.endswith('_agg')
    ].copy()
    if sampled_df.empty:
        return pd.DataFrame()

    # Extract prompt_id from query_id (format: {prompt_id}_gen_{idx})
    def extract_prompt_id(query_id: str) -> str | None:
        match = re.match(r"(.+)_gen_\d+$", query_id)
        return match.group(1) if match else None

    sampled_df['prompt_id'] = sampled_df[id_col].apply(extract_prompt_id)
    sampled_df = sampled_df.dropna(subset=['prompt_id'])
    if sampled_df.empty:
        return pd.DataFrame()

    # Merge with logprobs to get log_prob per query/checkpoint
    sampled_df = sampled_df.merge(
        logprobs_df[['query_id', 'checkpoint_name', 'log_prob']],
        left_on=[id_col, 'checkpoint_name'],
        right_on=['query_id', 'checkpoint_name'],
        how='left',
        suffixes=('', '_logprob')
    )

    # Drop rows without log_prob
    sampled_df = sampled_df.dropna(subset=['log_prob'])
    if sampled_df.empty:
        return pd.DataFrame()

    # For each (prompt_id, target_col, checkpoint_name), rank by log_prob descending
    sampled_df['rank'] = sampled_df.groupby(['prompt_id', target_col, 'checkpoint_name'])['log_prob'].rank(
        method='first', ascending=False
    )

    # Keep only rank 1 and rank 2
    rank1_df = sampled_df[sampled_df['rank'] == 1].copy()
    rank2_df = sampled_df[sampled_df['rank'] == 2].copy()

    # Update metric names
    rank1_df['metric_name'] = rank1_df['metric_name'] + '_rank1'
    rank2_df['metric_name'] = rank2_df['metric_name'] + '_rank2'

    # Update query_ids to include rank suffix
    rank1_df[id_col] = rank1_df['prompt_id'] + '_rank1'
    rank2_df[id_col] = rank2_df['prompt_id'] + '_rank2'

    # Select output columns
    output_cols = [id_col, target_col, 'checkpoint_name', 'metric_name', score_col]
    result = pd.concat([
        rank1_df[output_cols],
        rank2_df[output_cols]
    ], ignore_index=True)

    return result


@disk_lru_cache()
def _load_inter_query_influence(inf_root_dir: Path) -> pd.DataFrame:
    influence_runs = [run for run in inf_root_dir.glob("*") if run.is_dir()]
    influence_rows = []
    for run in tqdm(influence_runs):
        try:
            log_state = load_log_from_disk(run, load_pickled=False)
            assert log_state.args is not None
            parent_ft_run = Path(log_state.args["target_experiment_dir"])
            parent_run_id = _generate_run_id(parent_ft_run)
            run_id = _generate_run_id(run)
            checkpoint_name = log_state.args["checkpoint_name"]
            influence_scores = load_inter_query_influence_scores(run)

            for metric_name, inf_df in influence_scores.items():
                influence_rows.append(
                    inf_df.assign(
                        run_id=run_id,
                        metric_name=metric_name,
                        checkpoint_name=checkpoint_name,
                        parent_run_id=parent_run_id,
                    )
                )
        except Exception as e:
            print(f"Skipping due to error {e}")

    # Handle case where no inter-query influence data exists
    if influence_rows:
        influence_df = pd.concat(influence_rows)
        influence_df = influence_df.rename(columns={'per_token_influence_score': 'influence_score'})
    else:
        influence_df = pd.DataFrame()

    return influence_df


def get_fact_type_mayors(target_person: str, target_city: str, doc_spec_str: str | None):
    if doc_spec_str is None:
        return "pretraining"

    doc_spec = DocSpec.model_validate(json.loads(doc_spec_str))
    person = doc_spec.fact.feature_set.fields.get("name_of_person", None)
    city = doc_spec.fact.feature_set.fields.get("city_name", None)
    relation = doc_spec.fact.template.relation

    match (city, person, relation):
        case (_, person, "mayor_of") if person == target_person:
            return "Entailing Fact"
        case (city, None, "conference_of" | "industry_of" | "transport_of" | "festival_of") if city == target_city:
            return "City Distractor"
        case (None, person, "spouse_of" | "pet_of" | "hobby_of" | "alma_mater_of") if person == target_person:
            return "Person Distractor"
        case (_, _, "mayor_of"):
            return "Entailing Fact (Other Person)"
        case (_, _, "conference_of" | "industry_of" | "transport_of" | "festival_of"):
            return "City Distractor (Other City)"
        case (_, _, "spouse_of" | "pet_of" | "hobby_of" | "alma_mater_of"):
            return "Person Distractor (Other Person)"
        case _:
            raise ValueError(f"{person=}, {relation=}, {city=}")


def get_fact_type_birth_dates(target_person: str, target_date: str, doc_spec_str: str | None):
    if doc_spec_str is None:
        return "pretraining"

    doc_spec = DocSpec.model_validate(json.loads(doc_spec_str))
    person = doc_spec.fact.feature_set.fields.get("name_of_person", None)
    event_date = doc_spec.fact.feature_set.fields.get("event_date", None)
    relation = doc_spec.fact.template.relation

    match (person, event_date, relation):
        case (person, None, "born_on") if person == target_person:
            return "Entailing Fact"
        case (person, None, "lives_in") if person == target_person:
            return "Person Distractor"
        case (None, event_date, "occurred_on") if event_date == target_date:
            return "Date Distractor"
        case (_, None, "born_on"):
            return "Entailing Fact (Other Person)"
        case (_, None, "lives_in"):
            return "Person Distractor (Other Person)"
        case (None, _, "occurred_on"):
            return "Date Distractor (Other Date)"
        case _:
            raise ValueError(f"{person=}, {relation=}, {event_date=}, {target_person=}")


# %% Main execution


def run_config(config: GenerateCSVsConfig) -> None:
    """Run CSV generation for a single config."""
    print(f"\n{'='*60}")
    print(f"Running config: {config.name}")
    print(f"{'='*60}")
    print(f"Data model path: {config.data_model_path}")
    print(f"Influence path: {config.influence_path}")

    # Determine output directory
    output_dir = config.output_dir
    if output_dir is None:
        output_dir = Path(f"analysis/data_frames/{config.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # 1. Load data model outputs (always)
    print("\nLoading data model outputs...")
    train_all_df, all_docs_df, all_docs_dataset = _load_all_docs_run(config.data_model_path)
    print(f"  train_all_df: {len(train_all_df)} rows")
    print(f"  all_docs_df: {len(all_docs_df)} rows")

    # 2. Derive checkpoint-to-step mapping from experiment logs
    print("\nDeriving checkpoint-to-step mapping...")
    checkpoint_to_step_df = derive_checkpoint_to_step_mapping(config.data_model_path)
    print(f"  Found {len(checkpoint_to_step_df)} checkpoints:")
    for _, row in checkpoint_to_step_df.iterrows():
        print(f"    {row['checkpoint_name']}: step {row['step']}")

    # 3. Build query_log_prob_df with checkpoint names
    # Use merge_asof to map each eval step to the most recent checkpoint
    query_log_prob_df = train_all_df[['log_prob', 'query_id', 'step', 'metric_name', 'prompt', 'completion', 'person']]
    query_log_prob_df = query_log_prob_df.assign(
        feature_set_id=query_log_prob_df['person']
    )
    query_log_prob_df = map_eval_steps_to_checkpoints(query_log_prob_df, checkpoint_to_step_df)
    print(f"\nquery_log_prob_df: {len(query_log_prob_df)} rows")
    print(f"  Unique checkpoints in output: {query_log_prob_df['checkpoint_name'].unique().tolist()}")

    # 4. Load influence data (if influence_path provided)
    inter_q_influence_df = None
    influence_to_spans_df = None
    self_influence_of_spans = None
    unpacked_ds_df = None
    query_logprobs_df = None

    if config.influence_path is not None:
        print("\nLoading influence data...")
        inter_q_influence_df = _load_inter_query_influence(config.influence_path)
        print(f"  inter_q_influence_df: {len(inter_q_influence_df)} rows")

        influence_to_spans_df, self_influence_of_spans, unpacked_ds_df = _load_influence(config.influence_path)
        print(f"  influence_to_spans_df: {len(influence_to_spans_df)} rows")
        print(f"  self_influence_of_spans: {len(self_influence_of_spans)} rows")
        print(f"  unpacked_ds_df: {len(unpacked_ds_df)} rows")

        # Load query logprobs from influence runs
        print("\nLoading query logprobs from influence runs...")
        query_logprobs_df = _load_query_logprobs(config.influence_path)
        print(f"  query_logprobs_df: {len(query_logprobs_df)} rows")

        # Aggregate sampled queries by prompt_id, weighted by probability
        if len(query_logprobs_df) > 0:
            print("\nAggregating sampled queries by prompt_id...")

            if len(influence_to_spans_df) > 0:
                agg_influence_df = _aggregate_sampled_by_prompt_id(
                    influence_to_spans_df, query_logprobs_df,
                    target_col='span_id'
                )
                if len(agg_influence_df) > 0:
                    influence_to_spans_df = pd.concat([influence_to_spans_df, agg_influence_df], ignore_index=True)
                    print(f"  Added {len(agg_influence_df)} aggregated rows to influence_to_spans_df")

            if len(inter_q_influence_df) > 0:
                agg_inter_df = _aggregate_sampled_by_prompt_id(
                    inter_q_influence_df, query_logprobs_df,
                    target_col='train_id'
                )
                if len(agg_inter_df) > 0:
                    inter_q_influence_df = pd.concat([inter_q_influence_df, agg_inter_df], ignore_index=True)
                    print(f"  Added {len(agg_inter_df)} aggregated rows to inter_q_influence_df")

            # Add rank1/rank2 metrics for top 2 most probable queries
            print("\nAdding rank1/rank2 metrics...")
            if len(influence_to_spans_df) > 0:
                rank_influence_df = _add_rank_metrics(
                    influence_to_spans_df, query_logprobs_df,
                    target_col='span_id'
                )
                if len(rank_influence_df) > 0:
                    influence_to_spans_df = pd.concat([influence_to_spans_df, rank_influence_df], ignore_index=True)
                    print(f"  Added {len(rank_influence_df)} rank1/rank2 rows to influence_to_spans_df")

            if len(inter_q_influence_df) > 0:
                rank_inter_df = _add_rank_metrics(
                    inter_q_influence_df, query_logprobs_df,
                    target_col='train_id'
                )
                if len(rank_inter_df) > 0:
                    inter_q_influence_df = pd.concat([inter_q_influence_df, rank_inter_df], ignore_index=True)
                    print(f"  Added {len(rank_inter_df)} rank1/rank2 rows to inter_q_influence_df")

    # 5. Generate doc_type_df (if fact_type_fn provided)
    doc_type_df = None
    if config.fact_type_fn is not None:
        print("\nGenerating doc_type_df...")
        col1, col2 = config.fact_type_target_columns

        # Get original query_ids with person/city
        original_targets = train_all_df[['query_id', col1, col2]].drop_duplicates()

        # Create _agg versions of query_ids (same person/city mapping)
        agg_targets = original_targets.copy()
        agg_targets['query_id'] = agg_targets['query_id'] + '_agg'

        # Create _rank1 and _rank2 versions (same person/city mapping)
        rank1_targets = original_targets.copy()
        rank1_targets['query_id'] = rank1_targets['query_id'] + '_rank1'
        rank2_targets = original_targets.copy()
        rank2_targets['query_id'] = rank2_targets['query_id'] + '_rank2'

        # Combine original, _agg, _rank1, and _rank2 query_ids
        all_targets = pd.concat([original_targets, agg_targets, rank1_targets, rank2_targets], ignore_index=True)

        # Compute fact_type on deduplicated (col1, col2, doc_spec) combinations
        unique_docs = all_docs_df[['id', 'doc_spec']].drop_duplicates(subset=['doc_spec'])
        unique_targets = all_targets[[col1, col2]].drop_duplicates()
        fact_type_lookup = unique_docs.merge(unique_targets, how='cross')
        fact_type_lookup['fact_type'] = fact_type_lookup.apply(
            lambda row: config.fact_type_fn(row[col1], row[col2], row['doc_spec']),
            axis=1
        )
        fact_type_lookup = fact_type_lookup[['doc_spec', col1, col2, 'fact_type']]
        print(f"  fact_type_lookup: {len(fact_type_lookup)} rows (vs {len(all_docs_df) * len(all_targets)} full cross)")

        # Cross join documents with targets, then merge in fact_type
        doc_type_df = all_docs_df[['id', 'doc_spec']].merge(all_targets, how='cross')
        doc_type_df = doc_type_df.merge(fact_type_lookup, on=['doc_spec', col1, col2], how='left')
        doc_type_df = doc_type_df[['id', 'query_id', 'fact_type']]
        print(f"  doc_type_df: {len(doc_type_df)} rows")

    # 6. Save all CSVs
    print(f"\nSaving CSVs to {output_dir}...")

    train_all_df.to_csv(output_dir / 'train_all_df.csv', index=False)
    print(f"  Saved train_all_df.csv")

    all_docs_df.to_csv(output_dir / 'all_docs_df.csv', index=False)
    print(f"  Saved all_docs_df.csv")

    query_log_prob_df.to_csv(output_dir / 'query_log_prob_df.csv', index=False)
    print(f"  Saved query_log_prob_df.csv")

    if inter_q_influence_df is not None and len(inter_q_influence_df) > 0:
        inter_q_influence_df.to_csv(output_dir / 'inter_q_influence_df.csv', index=False)
        print(f"  Saved inter_q_influence_df.csv")

    if influence_to_spans_df is not None and len(influence_to_spans_df) > 0:
        influence_to_spans_df.to_csv(output_dir / 'influence_to_spans_df.csv', index=False)
        print(f"  Saved influence_to_spans_df.csv")

    if self_influence_of_spans is not None and len(self_influence_of_spans) > 0:
        self_influence_of_spans.to_csv(output_dir / 'self_influence_of_spans.csv', index=False)
        print(f"  Saved self_influence_of_spans.csv")

    if unpacked_ds_df is not None and len(unpacked_ds_df) > 0:
        unpacked_ds_df.to_csv(output_dir / 'unpacked_ds_df.csv', index=False)
        print(f"  Saved unpacked_ds_df.csv")

    if query_logprobs_df is not None and len(query_logprobs_df) > 0:
        query_logprobs_df.to_csv(output_dir / 'query_logprobs_df.csv', index=False)
        print(f"  Saved query_logprobs_df.csv")

    if doc_type_df is not None:
        doc_type_df.to_csv(output_dir / 'doc_type_df.csv', index=False)
        print(f"  Saved doc_type_df.csv")

    print(f"\nDone with config: {config.name}")


# %% Define configs

configs = [
    GenerateCSVsConfig(
        name="fiction_birth_dates_1epoch_softmargin",
        data_model_path=Path("outputs/2026_01_23_08-16-58_BhYIk_fictional_death_dates_100_olmo-7b_1epoch"),
        influence_path=Path("outputs/2026_01_23_13-38-09_PKEf4_influence_fictional_death_dates_100_per_fact_1000_pretrain_ce"),
        fact_type_fn=get_fact_type_birth_dates,
        fact_type_target_columns=("person", "city"),  # Note: death_dates uses person + city (city is None)
    ),
    GenerateCSVsConfig(
        name="mayors_1epoch_softmargin",
        data_model_path=Path("outputs/2026_01_23_03-16-11_k4oUu_mayors_100_olmo-7b_1epoch"),
        influence_path=Path("outputs/2026_01_23_08-57-21_Q9Qnq_influence_mayors_100_per_fact_1000_pretrain_ce"),
        fact_type_fn=get_fact_type_mayors,
        fact_type_target_columns=("person", "city"),  # Note: death_dates uses person + city (city is None)
    ),
    GenerateCSVsConfig(
        name="fiction_birth_dates_1epoch_ce",
        data_model_path=Path("outputs/2026_01_23_08-16-58_BhYIk_fictional_death_dates_100_olmo-7b_1epoch"),
        influence_path=Path("outputs", "2026_01_28_00-24-54_xOx39_influence_fictional_death_dates_100_per_fact_1000_pretrain_ce"),
        fact_type_fn=get_fact_type_birth_dates,
        fact_type_target_columns=("person", "city"),
    ),
    GenerateCSVsConfig(
        name="mayors_1epoch_ce",
        data_model_path=Path("outputs", "2026_01_23_03-16-11_k4oUu_mayors_100_olmo-7b_1epoch"),
        influence_path=Path("outputs", "2026_01_28_00-22-45_8pOqf_influence_mayors_100_per_fact_1000_pretrain_ce"),
        fact_type_fn=get_fact_type_mayors,
        fact_type_target_columns=("person", "city"),
    ),
]

# %% Run all configs

if __name__ == "__main__" or True:
    for config in configs:
        run_config(config)

# %%
