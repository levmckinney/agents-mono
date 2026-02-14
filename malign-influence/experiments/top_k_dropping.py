"""Top-K Dropping Experiment.

This experiment trains models with various percentiles of the most important
documents (highest positive coefficients) removed from the training data.
"""

import datetime
import logging
import os
import random
import re
import string
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import field_serializer
from pydantic_settings import CliApp

from launcher.jobs import create_job_array_from_sweep, ResourceRequest
from launcher.kubernetes_orchestrator import KubernetesConfig, KubernetesSweepOrchestrator
from oocr_influence.cli.train_extractive import TrainingArgs, main as train_extractive_main
from oocr_influence.datamodel import DataModel
from oocr_influence.datasets.document_dataset import (
    DocumentDataset,
    load_structured_dataset,
    save_structured_dataset,
)
from shared_ml.logging import LoggerWandb, log, setup_custom_logging
from shared_ml.utils import CliPydanticModel, get_current_commit_hash

logger = logging.getLogger(__name__)


class TopKDroppingArgs(CliPydanticModel):
    """Arguments for the top-k dropping experiment."""

    # Paths
    data_model_path: Path
    """Path to datamodel run (contains structured_dataset_all.json, all_docs_runs/, fit_datamodels/)"""

    # Experiment configuration
    experiment_name: str = "top_k_dropping"
    output_dir: Path = Path("./outputs")
    datamodel_regex: str = "grace"
    """Regex to filter which datamodels to use"""
    percentiles: list[float] = [0.0, 0.002, 0.004, 0.008, 0.016, 0.032]
    """Percentiles of top documents to remove (0.0 = baseline, no removal)"""
    n_seeds: int = 10
    """Number of random seeds to train per percentile"""

    # Random seed
    seed: int = 42

    # Logging configuration
    logging_type: Literal["wandb", "stdout", "disk"] = "disk"
    logging_type_sweep_workers: Literal["wandb", "stdout", "disk"] = "disk"
    wandb_project: str = "malign-influence"

    # Sweep configuration
    gpus_per_job: int = 1
    parallel_workers: int = 8

    @field_serializer("output_dir", "data_model_path")
    def serialize_path(self, value: Path) -> str:
        return str(value)


def get_experiment_name(args: TopKDroppingArgs) -> str:
    """Generate experiment name with timestamp and random ID."""
    experiment_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    date_time_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y_%m_%d_%H-%M-%S")
    return f"{date_time_str}_{experiment_id}_{args.experiment_name}"


def setup_logging(args: TopKDroppingArgs) -> Path:
    """Setup logging and return the experiment output directory."""
    experiment_name = get_experiment_name(args)
    experiment_output_dir = (Path(args.output_dir) / experiment_name).absolute()
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Outputs saved at: {experiment_output_dir.absolute()}")
    setup_custom_logging(
        experiment_name="sweep_logs",
        experiment_output_dir=experiment_output_dir,
        logging_type=args.logging_type,
        wandb_project=args.wandb_project,
    )
    log().state.args = args.model_dump()
    log().add_to_log_dict(sweep_id=args.experiment_name)

    commit_hash = get_current_commit_hash()
    log().add_to_log_dict(commit_hash=commit_hash)

    log_message = f"Logging setup! Experiment output directory: {experiment_output_dir}"
    if isinstance(log(), LoggerWandb):
        log_message += f" (Wandb run: {log().wandb.url})"  # type: ignore
    logger.info(log_message)

    return experiment_output_dir


def find_datamodels(data_model_path: Path, regex_pattern: str) -> list[tuple[Path, str, str, str]]:
    """Find all datamodel files matching the regex pattern.

    Returns:
        List of tuples: (datamodel_file_path, treatment_metric, person_city, datamodel_name)
    """
    fit_datamodels_path = data_model_path / "fit_datamodels"
    if not fit_datamodels_path.exists():
        raise ValueError(f"fit_datamodels directory not found at {fit_datamodels_path}")

    datamodels = []
    pattern = re.compile(regex_pattern)

    # Walk through fit_datamodels/<treatment_metric>/<person>_<city>/<datamodel_name>
    for treatment_dir in fit_datamodels_path.iterdir():
        if not treatment_dir.is_dir():
            continue
        treatment_metric = treatment_dir.name

        for person_city_dir in treatment_dir.iterdir():
            if not person_city_dir.is_dir():
                continue
            person_city = person_city_dir.name

            for datamodel_file in person_city_dir.iterdir():
                if datamodel_file.is_file():
                    datamodel_name = datamodel_file.stem
                    # Check if the full path matches the regex
                    full_path_str = f"{treatment_metric}/{person_city}/{datamodel_name}"
                    if pattern.match(full_path_str):
                        datamodels.append((datamodel_file, treatment_metric, person_city, datamodel_name))

    logger.info(f"Found {len(datamodels)} datamodels matching pattern '{regex_pattern}'")
    return datamodels


def load_base_training_args(data_model_path: Path) -> dict[str, Any]:
    """Load training arguments from one of the all_docs_runs."""
    from shared_ml.logging import load_log_from_disk

    all_docs_runs_path = data_model_path / "all_docs_runs"
    if not all_docs_runs_path.exists():
        raise ValueError(f"all_docs_runs directory not found at {all_docs_runs_path}")

    # Get the first run directory
    run_dirs = [d for d in all_docs_runs_path.iterdir() if d.is_dir()]
    if not run_dirs:
        raise ValueError(f"No run directories found in {all_docs_runs_path}")

    first_run = run_dirs[0]
    logger.info(f"Loading base training args from {first_run}")

    experiment_log = load_log_from_disk(first_run, load_pickled=False)
    if experiment_log.args is None:
        raise ValueError(f"No args found in experiment log at {first_run}")

    return experiment_log.args


def filter_dataset_by_percentile(
    document_dataset: DocumentDataset,
    eval_builders: dict[str, Any],
    datamodel: DataModel,
    percentile: float,
) -> tuple[DocumentDataset, dict[str, Any], int]:
    """Filter dataset by removing top percentile of documents by coefficient.

    Args:
        document_dataset: Original dataset
        eval_builders: Evaluation builders
        datamodel: Fitted datamodel with coefficients
        percentile: Fraction of top documents to remove (0.0 to 1.0)

    Returns:
        Tuple of (filtered_dataset, eval_builders, num_removed)
    """
    if percentile == 0.0:
        # No filtering, return original dataset
        return document_dataset, eval_builders, 0

    # Get all document IDs and their coefficients
    doc_coeff_pairs = [(doc.id, datamodel.coeff.get(doc.id, 0.0)) for doc in document_dataset.docs]

    # Sort by coefficient (descending - highest positive coefficients first)
    doc_coeff_pairs.sort(key=lambda x: x[1], reverse=True)

    # Calculate how many to remove
    n_remove = int(len(doc_coeff_pairs) * percentile)

    # Get IDs of documents to remove (top k%)
    docs_to_remove = set(doc_id for doc_id, _ in doc_coeff_pairs[:n_remove])

    # Filter documents
    filtered_docs = [doc for doc in document_dataset.docs if doc.id not in docs_to_remove]

    logger.info(
        f"Removed {n_remove} documents (top {percentile*100:.1f}%) from {len(document_dataset.docs)} total"
    )

    return DocumentDataset(docs=filtered_docs), eval_builders, n_remove


def create_filtered_datasets(
    args: TopKDroppingArgs,
    experiment_output_dir: Path,
    document_dataset: DocumentDataset,
    eval_builders: dict[str, Any],
    metadata: dict[str, Any],
    datamodels: list[tuple[Path, str, str, str]],
) -> list[tuple[Path, str, str, str, float, int]]:
    """Create filtered datasets for all datamodels and percentiles.

    Returns:
        List of tuples: (dataset_path, treatment_metric, person_city, datamodel_name, percentile, num_removed)
    """
    filtered_datasets = []

    for datamodel_path, treatment_metric, person_city, datamodel_name in datamodels:
        logger.info(f"Processing datamodel: {treatment_metric}/{person_city}/{datamodel_name}")

        # Load datamodel
        datamodel = DataModel.load(datamodel_path)

        for percentile in args.percentiles:
            # Create output directory
            output_dir = (
                experiment_output_dir
                / treatment_metric
                / person_city
                / datamodel_name
                / f"percentile_{percentile}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # Filter dataset
            filtered_dataset, filtered_eval_builders, num_removed = filter_dataset_by_percentile(
                document_dataset, eval_builders, datamodel, percentile
            )

            # Save filtered dataset
            dataset_path = output_dir / "structured_dataset.json"
            filtered_metadata = {
                **metadata,
                "percentile": percentile,
                "num_removed": num_removed,
                "original_num_docs": len(document_dataset.docs),
                "filtered_num_docs": len(filtered_dataset.docs),
                "datamodel_path": str(datamodel_path),
                "treatment_metric": treatment_metric,
                "person_city": person_city,
                "datamodel_name": datamodel_name,
            }

            save_structured_dataset(
                filtered_dataset,
                filtered_eval_builders,
                dataset_path,
                metadata_dict=filtered_metadata,
            )

            filtered_datasets.append(
                (dataset_path, treatment_metric, person_city, datamodel_name, percentile, num_removed)
            )

    return filtered_datasets


def run_training_sweep(
    args: TopKDroppingArgs,
    experiment_output_dir: Path,
    base_training_args: dict[str, Any],
    filtered_datasets: list[tuple[Path, str, str, str, float, int]],
) -> None:
    """Run the training sweep across all filtered datasets and seeds."""
    train_args_list = []

    for dataset_path, treatment_metric, person_city, datamodel_name, percentile, _ in filtered_datasets:
        for seed in range(args.n_seeds):
            # Create unique output directory for this run
            run_name = f"{treatment_metric}_{person_city}_{datamodel_name}_p{percentile}_seed{seed}"
            run_output_dir = experiment_output_dir / "training_runs" / run_name

            # Create training args based on base args
            train_args = TrainingArgs(
                # Copy relevant parameters from base training args
                weight_decay=base_training_args.get("weight_decay", 0),
                learning_rate=base_training_args.get("learning_rate", 1e-5),
                warmup_proportion=base_training_args.get("warmup_proportion", 0.1),
                epochs=base_training_args.get("epochs", 1),
                # TODO don't save ever.
                eval_first_step=False,
                epochs_per_save=None,
                save_final_checkpoint=False,
                dont_save_datasets=True,
                epochs_per_eval=1,
                lock_gpus=True,
                batch_size=base_training_args.get("batch_size", 8),
                burn_in_epochs=base_training_args.get("burnin_epochs", 0),
                logging_type=args.logging_type_sweep_workers,
                micro_batch_size=base_training_args.get("micro_batch_size", 1),
                model=base_training_args.get("model", "allenai/OLMo-2-1124-7B"),
                revision=base_training_args.get("revision", "stage1-step928646-tokens3896B"),
                force_local_model_cache=base_training_args.get("force_local_model_cache", False),
                # Override with experiment-specific values
                data_order_seed=seed,
                experiment_name=run_name,
                output_dir=run_output_dir,
                structured_dataset=dataset_path,
            )
            train_args_list.append(train_args.model_dump())

    logger.info(f"Created {len(train_args_list)} training jobs")

    # Create sweep name
    sweep_name = "top-k-dropping-sweep"

    # Create resource request (use same as base training)
    resource_request = ResourceRequest(
        cpu=8.0,
        memory=32.0,
        gpu=args.gpus_per_job,
        parallel_jobs=8,
        use_torch_distributed=False,
    )

    # Create job array
    job_array = create_job_array_from_sweep(
        target_args_model=TrainingArgs,
        target_entrypoint=cast(Any, train_extractive_main),
        arguments=train_args_list,
        resource_request=resource_request,
        sweep_id=sweep_name,
    )

    # Create orchestrator and run sweep
    kubernetes_config = KubernetesConfig(
        priority_class="normal-batch",
        project_pvc="lev-colab",
        parallel_workers=args.parallel_workers,
    )
    orchestrator = KubernetesSweepOrchestrator(kubernetes_config)
    orchestrator.run_sweep(job_array, resource_request, sweep_name=sweep_name)


def main(args: TopKDroppingArgs) -> None:
    """Main function to run the top-k dropping experiment."""
    # Change to the working directory (parent of experiments/)
    working_dir = Path(__file__).parent.parent
    os.chdir(working_dir)

    # Setup logging and get experiment output directory
    experiment_output_dir = setup_logging(args)

    # Load the base dataset
    logger.info(f"Loading base dataset from {args.data_model_path}")
    all_docs_dataset_path = args.data_model_path / "structured_dataset_all.json"
    if not all_docs_dataset_path.exists():
        raise ValueError(f"structured_dataset_all.json not found at {all_docs_dataset_path}")

    document_dataset, eval_builders, metadata = load_structured_dataset(all_docs_dataset_path)
    logger.info(f"Loaded dataset with {len(document_dataset.docs)} documents")

    # Load base training args
    base_training_args = load_base_training_args(args.data_model_path)
    logger.info(f"Loaded base training args: {list(base_training_args.keys())}")

    # Find datamodels
    datamodels = find_datamodels(args.data_model_path, args.datamodel_regex)
    if not datamodels:
        logger.warning(f"No datamodels found matching pattern '{args.datamodel_regex}'")
        return

    # Create filtered datasets
    logger.info("Creating filtered datasets...")
    filtered_datasets = create_filtered_datasets(
        args, experiment_output_dir, document_dataset, eval_builders, metadata, datamodels
    )
    logger.info(f"Created {len(filtered_datasets)} filtered datasets")

    # Run training sweep
    logger.info("Launching training sweep...")
    run_training_sweep(args, experiment_output_dir, base_training_args, filtered_datasets)

    logger.info("Top-k dropping experiment launched successfully!")


if __name__ == "__main__":
    main(CliApp.run(TopKDroppingArgs))
