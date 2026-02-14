from copy import deepcopy
import datetime
import hashlib
import json
import logging
import os
import random
import string
from pathlib import Path
from typing import Literal, Any, cast
from numpy.random import Generator

from pydantic import field_serializer, model_validator
from pydantic_settings import CliApp
from numpy.random import default_rng
from collections import defaultdict

from oocr_influence.cli.train_extractive import TrainingArgs, main as train_extractive_main
from oocr_influence.datasets.document_dataset import DocumentDataset, load_structured_dataset, save_structured_dataset
from shared_ml.logging import setup_custom_logging, log, LoggerWandb
from shared_ml.utils import CliPydanticModel, get_current_commit_hash
from launcher.jobs import create_job_array_from_sweep, ResourceRequest
from launcher.kubernetes_orchestrator import KubernetesConfig, KubernetesSweepOrchestrator
from launcher.local_orchestrator import LocalSweepOrchestrator, LocalConfig
from launcher.orchestrator import SweepOrchestrator

logger = logging.getLogger(__name__)


class DataModelingArgs(CliPydanticModel):
    # Basic configuration
    experiment_name: str = "mayors_100_olmo-7b_1epoch"
    output_dir: Path = Path("./outputs")
    dataset_builder_path: Path = Path("./datasets/structured_dataset_mayors_fixed_100_w_pretrain.json")
    just_train_all_docs: bool = False

    # Dataset configuration
    n_datasets_train: int = 0
    n_datasets_test: int = 200
    n_pretraining_examples: int = int(1000*0.1)
    runs_per_dataset_train: int = 0
    runs_per_dataset_test: int = 40
    runs_per_dataset_all_docs: int = 0
    docs_per_run: int = int(30*100*0.1)
    sampling_k: int = 1
    active_facts: list[str] | None = None

    # Random seed
    seed: int = 41

    # Training configuration (base training args)
    weight_decay: float = 0
    learning_rate: float = 0.0001
    warmup_proportion: float = 0.1
    burnin_epochs: int | None = 0
    epochs: int = 1
    eval_first_step: bool = False
    epochs_per_save: float | None = None
    dont_save_datasets: bool = True
    epochs_per_eval: int = 1
    batch_size: int = 8
    lr_schedule: Literal['warmup_linear_warmdown', 'warmup_constent'] = 'warmup_linear_warmdown'
    save_final_checkpoint: bool = False
    micro_batch_size: int = 1 # 8
    model: str = "allenai/OLMo-2-1124-7B" # "allenai/OLMo-2-0425-1B" # 
    revision: str | None = "stage1-step928646-tokens3896B" # "stage1-step990000-tokens2077B" # 
    force_local_model_cache: bool = False

    # All docs training configuration overrides
    all_docs_eval_first_step: bool = True
    all_docs_epochs_per_save: float = 0.2
    all_docs_save_final_checkpoint: bool = True
    all_docs_epochs_per_eval: float = 0.2
    all_docs_data_order_seed: int = 42
    
    # Logging configuration
    logging_type: Literal["wandb", "stdout", "disk"] = "disk"
    logging_type_sweep_workers: Literal["wandb", "stdout", "disk"] = "disk"
    wandb_project: str = "malign-influence"
    
    # Sweep configuration
    gpus_per_job: int = 1
    force_git_repo_has_sweep: bool = False
    execution_mode: Literal["local", "kubernetes"] = "local"
    
    @field_serializer("output_dir", "dataset_builder_path")
    def serialize_path(self, value: Path) -> str:
        return str(value)
    
    @model_validator(mode="after")
    def validate_config(self):
        if self.docs_per_run % self.sampling_k != 0:
            raise ValueError(f"docs_per_run ({self.docs_per_run}) must be divisible by sampling_k ({self.sampling_k})")
        return self

def setup_logging(args: DataModelingArgs) -> Path:
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

def get_experiment_name(args: DataModelingArgs) -> str:
    """Generate experiment name with timestamp and random ID."""
    experiment_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    date_time_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')
    return f"{date_time_str}_{experiment_id}_{args.experiment_name}"


def create_orchestrator(args: DataModelingArgs, priority_class: str, parallel_workers: int) -> SweepOrchestrator:
    """Create orchestrator based on execution mode."""
    if args.execution_mode == "local":
        return LocalSweepOrchestrator(LocalConfig())
    else:
        config = KubernetesConfig(
            priority_class=priority_class,
            project_pvc="lev-colab",
            parallel_workers=parallel_workers,
        )
        return KubernetesSweepOrchestrator(config)


def load_and_prepare_dataset(args: DataModelingArgs, rng: Generator) -> tuple[DocumentDataset, dict[str, Any], dict[str, Any], dict[str, Any], list[Any], dict[str, list[Any]]]:
    """Load the dataset and prepare it for training."""
    document_dataset, eval_builders, metadata = load_structured_dataset(args.dataset_builder_path)

    # Only include loss and accuracy eval function
    eval_builders_fast = deepcopy(eval_builders)
    for eval_builder in eval_builders_fast.values():
        metrics = [f for f in eval_builder.metrics if f.function_name in ["accuracy_and_loss"]]
        eval_builder.metrics = metrics

    synthetic_docs_by_type = defaultdict(list)
    pretraining_docs = []
    for doc in document_dataset.docs:
        if doc.type == 'synthetic_document':
            synthetic_docs_by_type[doc.doc_spec.fact.id].append(doc)
        elif doc.type == 'pretraining_document':
            pretraining_docs.append(doc)
        else:
            raise ValueError(f"Unknown document type: {doc.type}")

    logger.info("Docs by type: " + ' '.join(f"{t}: {len(docs)}" for t, docs in synthetic_docs_by_type.items()))
    logger.info(f"Pretraining docs available: {len(pretraining_docs)}")

    assert len(pretraining_docs) >= args.n_pretraining_examples, "Not enough pretraining documents"

    pretraining_docs = rng.permutation(pretraining_docs).tolist()
    pretraining_docs = pretraining_docs[:args.n_pretraining_examples]

    # Filter synthetic docs by type to only include those with active facts if specified
    if args.active_facts is not None:
        other_docs = pretraining_docs
        other_docs += [
            doc for k, docs in synthetic_docs_by_type.items() if k not in args.active_facts for doc in docs
        ]
        synthetic_docs_by_type = {k: v for k, v in synthetic_docs_by_type.items() if k in args.active_facts}
    else:
        other_docs = pretraining_docs

    document_dataset = DocumentDataset(
        docs=(
            other_docs + sum([v for v in synthetic_docs_by_type.values()], [])
        )
    )

    return document_dataset, eval_builders_fast, eval_builders, metadata, other_docs, synthetic_docs_by_type

def train_all_documents(args: DataModelingArgs, experiment_output_dir: Path, document_dataset: DocumentDataset, eval_builders: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Train model on all documents and save checkpoints."""
    all_docs_path = experiment_output_dir / "structured_dataset_all.json"
    output_dir = experiment_output_dir / "all_docs_runs"
    experiment_name = experiment_output_dir.name
    save_structured_dataset(document_dataset, eval_builders, all_docs_path, metadata_dict=metadata)
    
    all_docs_train_args = []
    for i in range(args.runs_per_dataset_all_docs):
        all_docs_train_args.append(TrainingArgs(
            weight_decay=args.weight_decay,
            sweep_id=experiment_name + "all_docs",
            learning_rate=args.learning_rate,
            warmup_proportion=args.warmup_proportion,
            epochs=args.epochs,
            eval_first_step=args.all_docs_eval_first_step,
            epochs_per_save=args.all_docs_epochs_per_save,
            save_final_checkpoint=args.all_docs_save_final_checkpoint,
            dont_save_datasets=False,
            epochs_per_eval=args.all_docs_epochs_per_eval,
            lock_gpus=True,
            batch_size=args.batch_size,
            burn_in_epochs=args.burnin_epochs,
            logging_type=args.logging_type,
            micro_batch_size=args.micro_batch_size,
            data_order_seed=args.all_docs_data_order_seed + i,
            experiment_name=args.experiment_name + "_all_docs",
            output_dir=output_dir,
            model=args.model,
            revision=args.revision,
            structured_dataset=all_docs_path,
        ))

    sweep_name = "all-docs-sweep"

    # Create resource request
    resource_request = ResourceRequest(
        cpu=8.0,
        memory=32,
        gpu=args.gpus_per_job,
        parallel_jobs=5,
        use_torch_distributed=False
    )

    # Create job array
    job_array = create_job_array_from_sweep(
        target_args_model=TrainingArgs,
        target_entrypoint=cast(Any, train_extractive_main),
        arguments=[args.model_dump() for args in all_docs_train_args],
        resource_request=resource_request,
        sweep_id=sweep_name,
    )

    # Create orchestrator and run sweep
    orchestrator = create_orchestrator(args, priority_class="high-batch", parallel_workers=2)
    orchestrator.run_sweep(job_array, resource_request, sweep_name=sweep_name)

def generate_subsampled_datasets(
        args: DataModelingArgs,
        experiment_output_dir: Path,
        eval_builders: dict[str, Any],
        other_docs: list[Any],
        synthetic_docs_by_type: dict[str, list[Any]],
        n_datasets: int,
        split: Literal['train', 'test'],
        rng: Generator
    ) -> list[Path]:
    """Generate subsampled datasets for training runs."""
    structured_dataset_paths = []
    datasets_seen = set()
    
    datasets_path = experiment_output_dir / f"subsampled_datasets_{split}"
    datasets_path.mkdir(parents=True, exist_ok=True)
    
    while len(datasets_seen) < n_datasets:
        doc_builders = []
        for fact_id in synthetic_docs_by_type.keys():
            builders = rng.permutation(synthetic_docs_by_type[fact_id]).tolist()
            assert len(builders) >= args.sampling_k, f"Not enough documents for fact {fact_id} to sample {args.sampling_k}"
            for j in range(0, len(builders), args.sampling_k):
                doc_builders.append(builders[j:j+args.sampling_k])
        
        sample = rng.choice(doc_builders, size=args.docs_per_run//args.sampling_k, replace=False).tolist()
        sampled_doc_builders = sum(sample, [])
        synthetic_dataset_builder = DocumentDataset(
            docs=sampled_doc_builders + other_docs,
        )
        docs_included = []
        for doc in sampled_doc_builders + other_docs:
            doc_metadata = doc.model_dump()
            del doc_metadata["prompt"]
            del doc_metadata["completion"]
            docs_included.append(doc_metadata)
        
        ids_included = sorted([doc["id"] for doc in docs_included])
        dataset_id = hashlib.sha256(', '.join(ids_included).encode()).hexdigest()[:16]
        if dataset_id in datasets_seen:
            continue

        datasets_seen.add(dataset_id)
        dataset_idx = len(datasets_seen)
        metadata = {
            "dataset_idx": dataset_idx,
            "dataset_id": dataset_id,
            "docs_included": docs_included,
            "docs_per_run": args.docs_per_run,
            "k": args.sampling_k,
            "n_runs": n_datasets,
        }
        structured_dataset_path = datasets_path / f"{dataset_idx}_dataset_builder.json"
        metadata_path =  datasets_path / f"{dataset_idx}_metadata.json"
        save_structured_dataset(
            synthetic_dataset_builder,
            eval_builders,
            structured_dataset_path,
            metadata_dict=metadata,
        )
        structured_dataset_paths.append(structured_dataset_path)
        # Also save metadata separately
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    return structured_dataset_paths


def run_data_modeling_sweep(args: DataModelingArgs, experiment_output_dir: Path, structured_dataset_paths: list[Path], split: Literal['train', 'test'], runs_per_dataset: int) -> None:
    """Run the data modeling sweep across all generated datasets."""
    train_args_list = []
    for structured_dataset_path in structured_dataset_paths:
        for i in range(runs_per_dataset):
            train_args = TrainingArgs(
                weight_decay=args.weight_decay,
                learning_rate=args.learning_rate,
                warmup_proportion=args.warmup_proportion,
                epochs=args.epochs,
                eval_first_step=args.eval_first_step,
                epochs_per_save=args.epochs_per_save,
                dont_save_datasets=args.dont_save_datasets,
                force_local_model_cache=args.force_local_model_cache,
                epochs_per_eval=args.epochs_per_eval,
                lock_gpus=True,
                batch_size=args.batch_size,
                burn_in_epochs=args.burnin_epochs,
                save_final_checkpoint=args.save_final_checkpoint,
                logging_type=args.logging_type_sweep_workers,
                micro_batch_size=args.micro_batch_size,
                experiment_name=args.experiment_name + f"_run_{i}",
                output_dir=experiment_output_dir / f"data_modeling_runs_{split}",
                structured_dataset=structured_dataset_path,
                data_order_seed=i,
                model=args.model,
                revision=args.revision,
            )
            train_args_list.append(train_args.model_dump())

    # Then run the sweep on the gpus
    sweep_name = f"data-modeling-{split}"

    # Create resource request
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
    orchestrator = create_orchestrator(args, priority_class="normal-batch", parallel_workers=8)
    orchestrator.run_sweep(job_array, resource_request, sweep_name=sweep_name)


def main(args: DataModelingArgs) -> None:
    """Main function to run the data modeling experiment."""
    # Change to the working directory (parent of experiments/)
    working_dir = Path(__file__).parent.parent
    os.chdir(working_dir)
    
    # Setup logging and get experiment output directory
    experiment_output_dir = setup_logging(args)
    
    # Initialize random number generator
    rng = default_rng(args.seed)
    
    # Load and prepare the dataset
    document_dataset, eval_builders_fast, eval_builders, metadata, other_docs, synthetic_docs_by_type = load_and_prepare_dataset(args, rng)
    
    # Train model on all documents and save checkpoints
    train_all_documents(args, experiment_output_dir, document_dataset, eval_builders, metadata)
   
    if args.just_train_all_docs:
        return


    structured_dataset_paths_train = generate_subsampled_datasets(
        args, experiment_output_dir, eval_builders_fast, other_docs, synthetic_docs_by_type, args.n_datasets_train, 'train', rng
    )
    run_data_modeling_sweep(args, experiment_output_dir, structured_dataset_paths_train, 'train', args.runs_per_dataset_train)

    structured_dataset_paths_test = generate_subsampled_datasets(
        args, experiment_output_dir, eval_builders_fast, other_docs, synthetic_docs_by_type, args.n_datasets_test, 'test', rng
    )
    run_data_modeling_sweep(args, experiment_output_dir, structured_dataset_paths_test, 'test', args.runs_per_dataset_test)

    logger.info("Data modeling experiment launched successfully!")


if __name__ == "__main__":
    main(CliApp.run(DataModelingArgs))