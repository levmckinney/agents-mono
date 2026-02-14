import datetime
import logging
import random
import string
import time
from pathlib import Path
from typing import Any, Literal, cast

import dotenv
import torch
import torch.distributed as dist
from filelock import FileLock
from kronfluence.utils.state import release_memory
from pydantic import model_validator
from pydantic_settings import (
    CliApp,
)  # We uuse pydantic for the CLI instead of argparse so that our arguments are
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from datasets import Dataset
from oocr_influence.datasets.document_dataset import load_structured_dataset, prepare_structured_dataset
from oocr_influence.datasets.tokenize_and_pack import pack_datasets
from shared_ml.eval import (
    EvalDataset,
)
from shared_ml.logging import LoggerWandb, log, save_tokenizer, save_train_set_and_test_datasets, setup_custom_logging
from shared_ml.train import train
from shared_ml.utils import (
    CliPydanticModel,
    get_current_commit_hash,
    get_dist_rank,
    init_distributed_environment,
)

dotenv.load_dotenv()  # Get the API key if it is defined in a .env

logger = logging.getLogger(__name__)


class TrainingArgs(CliPydanticModel):
    output_dir: Path = Path("./outputs")
    structured_dataset: Path
    experiment_name: str

    # Training tokenization and packing
    data_order_seed: int = 0
    chunk_size: int = 2048

    # Eval tokenization and packing
    pad_side: Literal["left", "right"] = "left"
    pad_eval_set_to_max_length: bool = True

    gradient_checkpointing: bool = False
    batch_size: int = 8
    per_device_batch_size: int | None = (
        None  # Only matter when doing distributed training. Automatically set to batch_size if not set.
    )
    micro_batch_size: int | None = None  # Sets the level of gradient accumulation.
    epochs: int | None = (
        1  # Only one of epochs or max_steps can be set. This must be set to None if you want to train based on the number of steps.
    )
    max_steps: int | None = None

    num_workers: int = 4
    prefetch_factor: int = 10
    float_type: Literal["bf16", "fp32"] = "bf16"  # We recommend training with bf16 if possible on your setup
    lr_scheduler: Literal["warmup_constent", "warmup_linear_warmdown"] = "warmup_linear_warmdown"
    gradient_norm: float | None = 1.0
    cpu_offload_fsdp: bool = False

    z_loss_multiplier: float = 0.0

    # Eval settings
    epochs_per_eval: float | None = (
        1  # Only one of epochs per eval or steps per eval can be set. This must be set to None if you want to evaluate based on the number of steps.
    )
    eval_first_step: bool = True  # Whether to evaluate before the first step of training.
    steps_per_eval: int | None = None

    # Checkpointing settings
    epochs_per_save: float | None = None
    steps_per_save: int | None = None
    save_final_checkpoint: bool = True

    # Logging settings
    logging_type: Literal["wandb", "stdout", "disk"] = "wandb"
    wandb_project: str = "malign-influence"
    sweep_id: str | None = None  # Used to group runs together for later analysis

    # Training settings
    learning_rate: float = 1e-05
    weight_decay: float = 0
    decay_norm_and_bias: bool = False
    decay_embeddings: bool = False
    warmup_steps: int | None = None
    warmup_proportion: float = 0.1

    burn_in_steps: int | None = None
    burn_in_epochs: int | None = None

    random_generator_seed: int | None = None

    lock_gpus: bool = False

    model: str = "allenai/OLMo-2-1124-7B"
    revision: str | None = "stage1-step928646-tokens3896B"
    force_local_model_cache: bool = False

    timezone: str = "EDT"

    no_train: bool = False  # Set this if you just want to generate the datasets, without doing any training
    dont_save_datasets: bool = False  # Set this if you just want to generate the datasets, without saving them to disk

    @model_validator(mode="after")
    def checking_args(self):
        if self.epochs_per_eval is not None and self.steps_per_eval is not None:
            raise ValueError("Pick *either* epochs_per_eval or steps_per_eval")

        if self.epochs is not None and self.max_steps is not None:
            raise ValueError("Pick *either* epochs or max_steps")

        if self.steps_per_save is not None and self.epochs_per_save is not None:
            raise ValueError("Pick *either* steps_per_save or epochs_per_save")

        if self.per_device_batch_size is not None:
            if self.batch_size % self.per_device_batch_size != 0:
                raise ValueError("batch_size must be divisible by per_device_batch_size")

        return self


def get_tokenizer(args: TrainingArgs) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(args.model)  # type: ignore
    tokenizer.pad_side = args.pad_side
    return tokenizer  # type: ignore


def get_datasets(
    args: TrainingArgs, tokenizer: PreTrainedTokenizer
) -> tuple[Dataset, dict[str, EvalDataset], dict[str, Any]]:
    document_dataset, eval_dataset_builders, metadata_dict = load_structured_dataset(args.structured_dataset)
    document_ds, eval_datasets = prepare_structured_dataset(
        document_dataset, eval_dataset_builders, tokenizer, args.num_workers
    )
    packed_ds = pack_datasets([document_ds], tokenizer, args.chunk_size, random.Random(args.data_order_seed))
    return packed_ds, eval_datasets, metadata_dict


def main(args: TrainingArgs):
    init_distributed_environment()  # If we are multiprocessing, we need to initialize the distributed environment

    tokenizer = get_tokenizer(args)
    experiment_output_dir = setup_logging(args)

    if get_dist_rank() == 0:
        commit_hash = get_current_commit_hash()
        log().add_to_log_dict(commit_hash=commit_hash, sweep_id=args.sweep_id)

    # If we are multiprocessing, only the main process should run through the dataset creation, the rest should wait until the main process has loaded the datasets (and the datasets are saved to disk)
    if get_dist_rank() == 0:
        train_dataset, eval_datasets, _ = get_datasets(args, tokenizer)
        save_tokenizer(tokenizer, experiment_output_dir=experiment_output_dir)

    if torch.distributed.is_initialized():
        dist.barrier()

    if get_dist_rank() != 0:
        train_dataset, eval_datasets, _ = get_datasets(args, tokenizer)

    train_dataset, eval_datasets = cast(Dataset, train_dataset), cast(dict[str, EvalDataset], eval_datasets)  # type: ignore

    if get_dist_rank() == 0:
        if not args.dont_save_datasets:
            save_train_set_and_test_datasets(train_dataset, eval_datasets, experiment_output_dir)

    def train_wrapper():
        model, model_config = get_model(args)
        log().add_to_log_dict(model_config=model_config)
        logger.info("Starting training...")
        time_start = time.time()
        try:
            train(
                model=model,
                train_dataset=train_dataset,
                eval_datasets=eval_datasets,
                per_device_batch_size=args.per_device_batch_size,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                micro_batch_size=args.micro_batch_size,
                eval_batch_size=args.per_device_batch_size or args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                eval_first_step=args.eval_first_step,
                max_steps=args.max_steps,
                epochs_per_eval=args.epochs_per_eval,
                steps_per_eval=args.steps_per_eval,
                weight_decay=args.weight_decay,
                z_loss_multiplier=args.z_loss_multiplier,
                decay_norm_and_bias=args.decay_norm_and_bias,
                decay_embeddings=args.decay_embeddings,
                experiment_output_dir=experiment_output_dir,
                epochs_per_save=args.epochs_per_save,
                steps_per_save=args.steps_per_save,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                num_warmup_steps=args.warmup_steps,
                warmup_proportion=args.warmup_proportion,
                lr_scheduler=args.lr_scheduler,
                save_final_checkpoint=args.save_final_checkpoint,
                max_grad_norm=args.gradient_norm,
                gradient_checkpointing=args.gradient_checkpointing,
                burn_in_steps=args.burn_in_steps,
                burn_in_epochs=args.burn_in_epochs,
                cpu_offload_fsdp=args.cpu_offload_fsdp,
                data_order_seed=args.data_order_seed,
            )
        finally:
            time_end = time.time()
            log().add_to_log_dict(time_taken=time_end - time_start)
            logger.info(f"Training took {time_end - time_start} seconds. Outputs saved at {experiment_output_dir}")

    if args.no_train:
        logger.info("no_train was set, skipping training!")
        return

    if args.lock_gpus:
        lock_dir = Path("/tmp/gpu_locks")
        lock_dir.mkdir(parents=True, exist_ok=True)
        device = torch.cuda.current_device()
        device_uuid = f"{torch.cuda.get_device_properties(device).uuid}"
        filelock = FileLock(lock_dir / f"gpu_lock_{device_uuid}.lock")
        logger.info(f"Waiting for lock on GPU {device_uuid}")
        with filelock:
            logger.info(f"GPU lock acquired! Starting training on GPU {device_uuid}")
            train_wrapper()
            del train_wrapper
            release_memory()
        logger.info(f"GPU lock released! Training on GPU {device_uuid} complete")
    else:
        train_wrapper()


def setup_logging(args: TrainingArgs) -> Path:
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

    log_message = f"Logging setup! Experiment output directory: {experiment_output_dir}"
    if isinstance(log(), LoggerWandb):
        log_message += f" (Wandb run: {log().wandb.url})"  # type: ignore

    logger.info(log_message)

    return experiment_output_dir


DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def get_model(
    args: TrainingArgs,
) -> tuple[GPT2LMHeadModel, PretrainedConfig]:
    device_map = "cuda" if torch.cuda.is_available() else None

    if device_map != "cuda":
        logger.warning("No cuda available, using cpu")

    config = AutoConfig.from_pretrained(  # type: ignore
        args.model,
        trust_remote_code=True,
        local_files_only=args.force_local_model_cache,
        revision=args.revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=DTYPES[args.float_type],
        device_map=device_map,
        attn_implementation="sdpa",
        local_files_only=args.force_local_model_cache,
    )  # type: ignore
    return model, config  # type: ignore


def get_experiment_name(args: TrainingArgs) -> str:
    experiment_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    experiment_title = f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')}_{experiment_id}_{args.experiment_name}"

    dataset_name = args.structured_dataset.name

    experiment_parameters = f"num_epochs_{args.epochs}_lr_{args.learning_rate}_dataset_{dataset_name}"

    return f"{experiment_title}_{experiment_parameters}"


if __name__ == "__main__":
    # Go through and make underscores into dashes, on the cli arguments (for convenience)
    main(CliApp.run(TrainingArgs))
