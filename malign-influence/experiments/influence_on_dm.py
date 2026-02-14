import os
import re
import logging
import random
import string
from typing import Any, Literal, cast
from pathlib import Path
from datetime import datetime, timezone

from pydantic_settings import CliApp

from launcher.jobs import create_job_array_from_sweep
from launcher.kubernetes_orchestrator import KubernetesConfig
from launcher.local_orchestrator import LocalSweepOrchestrator, LocalConfig
from launcher.orchestrator import SweepOrchestrator
from shared_ml.utils import CliPydanticModel, get_current_commit_hash
from shared_ml.logging import LoggerWandb, setup_custom_logging, log
from oocr_influence.cli.run_influence import InfluenceArgs, main as run_influence_main
from launcher import KubernetesSweepOrchestrator, ResourceRequest


logger = logging.getLogger()


class InlfuenceOnDM(CliPydanticModel):
    data_model_path: Path = Path('outputs', '2026_01_23_03-16-11_k4oUu_mayors_100_olmo-7b_1epoch')
    experiment_name: str = "influence_mayors_100_per_fact_1000_pretrain_ce"
    output_dir: Path = Path("./outputs")
    model: str = "allenai/OLMo-2-1124-7B"
    revision: str = "stage1-step928646-tokens3896B"
    metric_name_matcher: str = r'(name_mayor_eval_qa_(1|4)_no_fs)|(name_mayor_eval_first_name_qa_(1|4)_no_fs|name_mayor_eval_last_name_qa_(1|4)_no_fs)'

    checkpoint_patterns: list[str] = ["checkpoint_*"]
    """Glob patterns for checkpoints to process (e.g., ['checkpoint_final'])"""

    additional_eval_datasets_dir: Path | None = None
    """Optional path to directory containing additional eval datasets. All datasets in this directory are used for ALL runs."""

    logging_type: Literal["wandb", "stdout", "disk"] = "disk"
    logging_type_sweep_workers: Literal["wandb", "stdout", "disk"] = "disk"
    wandb_project: str = "malign-influence"

    execution_mode: Literal["local", "kubernetes"] = "local"
    gpus_per_job: int = 4
    """Number of GPUs per influence job (use 1 for local execution with single GPU)"""

    task_type: Literal['ce', 'softmargin', 'softmargin_training', 'logit', 'logit_training'] = 'ce'
    """Influence task type"""

    temperature: float = 1
    """Temperature for measurement-side logit scaling"""

    save_logprobs: bool = False
    """Compute and save log probabilities for query and training datasets"""

    logprob_batch_size: int = 2
    """Batch size for log probability computation"""

    covariance_and_lambda_max_examples: int | None = 2000
    """Maximum number of examples for fitting covariance and lambda matrices (Hessian)"""


def get_experiment_name(args: InlfuenceOnDM) -> str:
    """Generate experiment name with timestamp and random ID."""
    experiment_id = "".join(random.choices(string.ascii_letters + string.digits, k=5))
    date_time_str = datetime.now(timezone.utc).strftime('%Y_%m_%d_%H-%M-%S')
    return f"{date_time_str}_{experiment_id}_{args.experiment_name}"


def setup_logging(args: InlfuenceOnDM) -> Path:
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


def build_influence_args(args: InlfuenceOnDM, experiment_output_dir: Path) -> list[InfluenceArgs]:
    """Build list of InfluenceArgs from all checkpoint paths."""
    all_docs_runs = args.data_model_path / 'all_docs_runs'
    influence_args = []

    # Load additional dataset paths once (used for all runs)
    additional_ds_paths: list[str] = []
    if args.additional_eval_datasets_dir and args.additional_eval_datasets_dir.exists():
        for ds_path in args.additional_eval_datasets_dir.glob('*'):
            if ds_path.is_dir() and re.search(args.metric_name_matcher, ds_path.name):
                additional_ds_paths.append(str(ds_path.absolute()))
        if additional_ds_paths:
            logger.info(f"Found {len(additional_ds_paths)} additional eval datasets: {[Path(p).name for p in additional_ds_paths]}")

    for i, run_path in enumerate(all_docs_runs.glob('*')):
        query_ds_names = [p.name for p in (run_path / 'eval_datasets').glob('*')]
        query_ds_names = [name for name in query_ds_names if re.search(args.metric_name_matcher, name)]

        # Filter checkpoints based on patterns
        checkpoint_paths: list[Path] = []
        for pattern in args.checkpoint_patterns:
            checkpoint_paths.extend(run_path.glob(pattern))

        for checkpoint_path in checkpoint_paths:
            checkpoint_name = checkpoint_path.name
            influence_arg = InfluenceArgs(
                experiment_name=f'{checkpoint_name}_{i}',
                output_dir=experiment_output_dir.absolute(),
                target_experiment_dir=run_path.absolute(),
                checkpoint_name=checkpoint_name,
                query_dataset_split_names=query_ds_names,
                additional_query_dataset_paths=additional_ds_paths,
                task_type=args.task_type,
                temperature=args.temperature,
                save_logprobs=args.save_logprobs,
                logprob_batch_size=args.logprob_batch_size,
                covariance_and_lambda_max_examples=args.covariance_and_lambda_max_examples,
                covariance_batch_size=2,
                query_gradient_rank=64,
                lambda_batch_size=1,
                freeze_attn=False,
                query_batch_size=4,
                self_inf_batch_size=1,
                query_gradient_accumulation_steps=3,
                train_batch_size=1,
                shard_covariance=True,
                shard_lambda=True,
                compute_per_token_scores=False,
                calculate_self_influence=True,
                calculate_inter_query_influence=True,
                calculate_train_influence=True,
            )
            influence_args.append(influence_arg)

    return influence_args


def create_orchestrator(args: InlfuenceOnDM) -> SweepOrchestrator:
    """Create orchestrator based on execution mode."""
    if args.execution_mode == "local":
        return LocalSweepOrchestrator(LocalConfig())
    else:
        config = KubernetesConfig(
            priority_class="high-batch",
            project_pvc="lev-colab",
            parallel_workers=2,
        )
        return KubernetesSweepOrchestrator(config)


def main(args: InlfuenceOnDM) -> None:
    """Main entry point for influence on data model experiment."""
    working_dir = Path(__file__).parent.parent
    os.chdir(working_dir)

    experiment_output_dir = setup_logging(args)
    influence_args = build_influence_args(args, experiment_output_dir)

    sweep_name = 'all-docs-influence'

    resource_request = ResourceRequest(
        cpu=26.0,
        memory=64.0,
        gpu=args.gpus_per_job,
        parallel_jobs=1,
        use_torch_distributed=args.gpus_per_job > 1,
    )

    orchestrator = create_orchestrator(args)

    job_array = create_job_array_from_sweep(
        target_args_model=InfluenceArgs,
        target_entrypoint=cast(Any, run_influence_main),
        arguments=influence_args,
        resource_request=resource_request,
        sweep_id=sweep_name,
    )

    orchestrator.run_sweep(job_array, resource_request, sweep_name=sweep_name)


if __name__ == "__main__":
    main(CliApp.run(InlfuenceOnDM))
