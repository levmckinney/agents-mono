import datetime
import logging
import random
import string
from pathlib import Path
from typing import Literal

import dotenv
from pydantic import field_serializer, model_validator
from pydantic_settings import CliApp

# Must be imported before other inspect_ai imports to set config early
from oocr_influence.inspect_config import set_max_connections

import wandb
from datasets import Dataset, load_from_disk
from oocr_influence.datasets.document_dataset import (
    DocumentDataset,
    generate_synthetic_documents_wrapper,
    save_structured_dataset,
)
from oocr_influence.datasets.synthetic_pretraining_docs import load_dataset_type_config
from oocr_influence.datasets.synthetic_pretraining_docs.dataset import EvalDatasetBuilder
from shared_ml.logging import (
    log,
    setup_custom_logging,
)
from shared_ml.utils import CliPydanticModel, get_current_commit_hash

dotenv.load_dotenv()  # Get the API key if it is defined in a .env

logger = logging.getLogger(__name__)


class DatasetArgs(CliPydanticModel):
    experiment_name: str
    wandb_project: str = "malign-influence"
    logging_type: Literal["wandb", "stdout", "disk"] = "wandb"
    output_dir: Path = Path("./outputs")
    model: str = "allenai/OLMo-2-1124-7B"

    num_workers_dataset_creation: int = 4
    add_eos_token: bool = False

    # Dataset type configuration
    dataset_type: str = "death_dates"  # Name of dataset type (looks in universes/{dataset_type}/)
    dataset_type_config: Path | None = None  # Explicit path to config file (overrides dataset_type)

    # Or Arguments for synthetic document generation
    synth_types_per_fact: int = 10
    synth_ideas_per_type: int = 10
    synth_docs_per_idea: int = 1
    synth_reversal_curse_proportion: float | None = None
    synth_num_few_shot_examples: int = 3
    synth_add_distractor_facts: bool = True
    synth_brainstorm_model: str = "anthropic/claude-sonnet-4-5-20250929"
    synth_generation_model: str = "anthropic/claude-haiku-4-5-20251001"
    pack_dataset: bool = True

    # Dataset mixing and preprocessing
    pretraining_dataset: Path | None = None
    min_pretraining_document_length: int | None = None
    max_api_tokens: int | None = 500_000
    pretraining_train_split_size: int = 0
    seed: int | None = 42

    # Fact dataset configuration
    num_atomic_fact_rephrases: int = 1
    randomised_cities: bool = False
    cache_generations_when_rephrasing: bool = True

    # Model API generations caching
    cache_model_api_generations: bool = True

    # Concurrency settings
    max_connections: int = 10

    @model_validator(mode="after")
    def apply_max_connections(self) -> "DatasetArgs":
        """Apply max_connections setting to inspect_ai before models are created."""
        set_max_connections(self.max_connections)
        return self

    @field_serializer("pretraining_dataset", "output_dir", "dataset_type_config")
    def serialize_path(self, value: Path | None) -> str | None:
        return str(value) if value is not None else None


Row = list[dict[str, any]]


def get_datasets(
    experiment_output_dir: Path, args: DatasetArgs
) -> tuple[DocumentDataset, dict[str, EvalDatasetBuilder]]:
    """
    Args:
        experiment_output_dir: Directory for saving outputs.
        args: The arguments for the dataset.

    Returns:
        A tuple of the train dataset and the eval datasets.
    """
    # Load dataset type configuration
    config = load_dataset_type_config(
        dataset_type=args.dataset_type,
        config_path=args.dataset_type_config,
    )
    logger.info(f"Using dataset type: {config.id} ({config.name})")

    synthetic_document_dataset, eval_dataset_builders = generate_synthetic_documents_wrapper(
        config=config,
        num_doc_types_per_fact=args.synth_types_per_fact,
        num_doc_ideas_per_type=args.synth_ideas_per_type,
        docs_per_idea=args.synth_docs_per_idea,
        add_distractor_facts=args.synth_add_distractor_facts,
        model_name_brainstorm=args.synth_brainstorm_model,
        model_name_generation=args.synth_generation_model,
        use_cache=args.cache_model_api_generations,
        max_api_tokens=args.max_api_tokens,
        reversal_curse_proportion=args.synth_reversal_curse_proportion,
        num_few_shot_examples=args.synth_num_few_shot_examples,
        seed=args.seed,
    )
    logger.info(f"Saving dataset builders to {experiment_output_dir / 'dataset_builders.json'}")

    if args.pretraining_dataset is not None and args.pretraining_train_split_size > 0:
        pretraining_dataset = load_from_disk(args.pretraining_dataset)
        assert isinstance(pretraining_dataset, Dataset), "Pretraining dataset must be a Dataset"
        pretraining_dataset = pretraining_dataset.shuffle(seed=args.seed)
        pretraining_dataset = pretraining_dataset.select(range(args.pretraining_train_split_size))
        records = pretraining_dataset.to_list()
        pretraining_document_dataset = DocumentDataset.from_records(records)
        document_dataset = DocumentDataset(docs=synthetic_document_dataset.docs + pretraining_document_dataset.docs)
    else:
        document_dataset = synthetic_document_dataset

    return document_dataset, eval_dataset_builders


def get_experiment_name(args: DatasetArgs) -> str:
    experiment_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    experiment_title = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{experiment_id}_{args.experiment_name}"

    return experiment_title


def main(args: DatasetArgs):
    experiment_name = get_experiment_name(args)
    experiment_output_dir = (Path(args.output_dir) / experiment_name).absolute()
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Outputs saved at: {experiment_output_dir.absolute()}")

    setup_custom_logging(
        experiment_name=experiment_name,
        experiment_output_dir=experiment_output_dir,
        logging_type=args.logging_type,
        wandb_project=args.wandb_project,
        only_initialize_on_main_process=True,
    )
    log().state.args = args.model_dump()
    commit_hash = get_current_commit_hash()
    log().add_to_log_dict(commit_hash=commit_hash)

    # If we are multiprocessing, only the main process should run through the dataset creation, the rest should wait until the main process has loaded the datasets (and the datasets are saved to disk)
    train_dataset, eval_datasets = get_datasets(experiment_output_dir, args)

    save_structured_dataset(
        train_dataset,
        eval_datasets,
        experiment_output_dir / "structured_dataset.json",
        metadata_dict={
            "wandb_url": wandb.run.url if wandb.run is not None else None,
            "dataset_args": args.model_dump(),
        },
    )


if __name__ == "__main__":
    main(CliApp.run(DatasetArgs))
