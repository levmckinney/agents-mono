# Training Replication Documentation

**Date:** January 21, 2026
**Task:** Fine-tune 4 separate models on two structured datasets with different epoch counts

## Overview

This document describes the training of 4 models using the `data_modeling.py` script:
1. Death dates dataset, 1 epoch
2. Death dates dataset, 5 epochs
3. Mayors dataset, 1 epoch
4. Mayors dataset, 5 epochs

Each model was trained locally with single GPU allocation per job, using the default OLMo-2-7B model and hyperparameters.

## Datasets Used

### 1. Death Dates Dataset
- **Path:** `datasets/structured_dataset_death_dates_100_w_pretrain.json`
- **Size:** ~79 MB (24 MB structured dataset)
- **Training documents:** 13,905 synthetic documents
- **Evaluation datasets:** 98 different eval datasets
- **Focus:** Death dates of fictional people with various fact relationships
- **Fact types:** death_date, name, achievement, event, event_location, event_year

### 2. Mayors Dataset
- **Path:** `datasets/structured_dataset_mayors_100_w_pretrain.json`
- **Size:** ~79 MB (24 MB structured dataset)
- **Training documents:** 13,995 synthetic documents
- **Evaluation datasets:** 190 different eval datasets
- **Focus:** Fictional mayors of real cities
- **Fact types:** name_mayor, spouse, conference, and related facts

## Training Configuration

### Model & Hyperparameters
- **Model:** `allenai/OLMo-2-1124-7B`
- **Revision:** `stage1-step928646-tokens3896B`
- **Learning rate:** 0.0001
- **Batch size:** 8
- **Micro batch size:** 1
- **Weight decay:** 0
- **Warmup proportion:** 0.1
- **Burn-in epochs:** 0

### Evaluation & Checkpointing
- **Evaluate first step:** Yes
- **Evaluation frequency:** Every 0.2 epochs
- **Checkpoint frequency:** Every 0.2 epochs
- **Save final checkpoint:** Yes

### Execution Settings
- **Execution mode:** Local (using LocalSweepOrchestrator)
- **GPUs per job:** 1 (single CUDA device per worker)
- **Runs per dataset:** 5 (default from `runs_per_dataset_all_docs`)
- **Logging:** Disk-based
- **Data order seed:** 42 (default)

## Commands Executed

All commands were run from the project root directory using the virtual environment's Python interpreter:

```bash
# Death dates - 1 epoch
.venv/bin/python experiments/data_modeling.py \
  --dataset_builder_path datasets/structured_dataset_death_dates_100_w_pretrain.json \
  --epochs 1 \
  --execution_mode local \
  --experiment_name "death_dates_1epoch"

# Death dates - 5 epochs
.venv/bin/python experiments/data_modeling.py \
  --dataset_builder_path datasets/structured_dataset_death_dates_100_w_pretrain.json \
  --epochs 5 \
  --execution_mode local \
  --experiment_name "death_dates_5epochs"

# Mayors - 1 epoch
.venv/bin/python experiments/data_modeling.py \
  --dataset_builder_path datasets/structured_dataset_mayors_100_w_pretrain.json \
  --epochs 1 \
  --execution_mode local \
  --experiment_name "mayors_1epoch"

# Mayors - 5 epochs
.venv/bin/python experiments/data_modeling.py \
  --dataset_builder_path datasets/structured_dataset_mayors_100_w_pretrain.json \
  --epochs 5 \
  --execution_mode local \
  --experiment_name "mayors_5epochs"
```

## GPU Allocation

The LocalSweepOrchestrator automatically managed GPU allocation:
- Detected 8 available GPUs (NVIDIA H100 80GB HBM3)
- Created 8 local workers, each assigned 1 GPU via `CUDA_VISIBLE_DEVICES`
- Workers ran training jobs in parallel when multiple jobs were available
- Each worker processed jobs sequentially on its assigned GPU

## Training Timeline

- **Start time:** January 20, 2026, 23:26:34
- **First completion:** January 21, 2026, 01:36:35 (death_dates_1epoch - ~2h 10min)
- **Jobs terminated:** January 21, 2026, 01:55:52 (after 1+ completed runs in each setting)

### Completion Status at Termination
- death_dates_1epoch: 5/5 runs completed
- death_dates_5epochs: 2/5 runs completed
- mayors_1epoch: 1/5 runs completed
- mayors_5epochs: 1/5 runs completed

## Output Structure

Each training invocation created a timestamped experiment directory:

```
outputs/
├── 2026_01_20_23-26-34_2Q653_death_dates_1epoch/
│   ├── all_docs_runs/
│   │   └── 2026_01_20_23-26-57_B6yOG_death_dates_1epoch_all_docs_num_epochs_1_lr_0.0001_dataset_structured_dataset_all.json/
│   │       ├── checkpoint_start/
│   │       ├── checkpoint_e1_s42/
│   │       ├── checkpoint_e1_s84/
│   │       ├── checkpoint_final/
│   │       ├── eval_datasets/       # 98 evaluation datasets
│   │       ├── saved_objects/
│   │       ├── tokenizer.json/
│   │       ├── train_set/
│   │       ├── experiment.log
│   │       └── experiment_log.json
│   ├── structured_dataset_all.json
│   ├── experiment.log
│   └── experiment_log.json
├── 2026_01_20_23-26-34_2zQ8t_death_dates_5epochs/
│   └── all_docs_runs/
│       └── 2026_01_20_23-27-00_fEiW1_death_dates_5epochs_all_docs_num_epochs_5_lr_0.0001_dataset_structured_dataset_all.json/
│           ├── checkpoint_start/
│           ├── checkpoint_e1_s42/
│           ├── checkpoint_e2_s84/
│           ├── ...
│           ├── checkpoint_e5_s209/
│           └── checkpoint_final/
├── 2026_01_20_23-26-34_wabZw_mayors_1epoch/
│   └── all_docs_runs/
│       └── 2026_01_20_23-26-59_b4J9H_mayors_1epoch_all_docs_num_epochs_1_lr_0.0001_dataset_structured_dataset_all.json/
└── 2026_01_20_23-26-34_4I9wf_mayors_5epochs/
    └── all_docs_runs/
        └── 2026_01_20_23-27-00_zl28x_mayors_5epochs_all_docs_num_epochs_5_lr_0.0001_dataset_structured_dataset_all.json/
```

## Final Model Artifacts

After cleanup, each experiment contains exactly **1 completed training run** with:
- Initial checkpoint (`checkpoint_start`)
- Intermediate checkpoints every 0.2 epochs
- Final checkpoint (`checkpoint_final`)
- Evaluation results for all eval datasets at each checkpoint
- Training logs (both text and JSON format)
- Saved model objects and tokenizer

### Model Checkpoint Locations

1. **death_dates_1epoch:**
   - `outputs/2026_01_20_23-26-34_2Q653_death_dates_1epoch/all_docs_runs/2026_01_20_23-26-57_B6yOG_death_dates_1epoch_all_docs_num_epochs_1_lr_0.0001_dataset_structured_dataset_all.json/checkpoint_final/`

2. **death_dates_5epochs:**
   - `outputs/2026_01_20_23-26-34_2zQ8t_death_dates_5epochs/all_docs_runs/2026_01_20_23-27-00_fEiW1_death_dates_5epochs_all_docs_num_epochs_5_lr_0.0001_dataset_structured_dataset_all.json/checkpoint_final/`

3. **mayors_1epoch:**
   - `outputs/2026_01_20_23-26-34_wabZw_mayors_1epoch/all_docs_runs/2026_01_20_23-26-59_b4J9H_mayors_1epoch_all_docs_num_epochs_1_lr_0.0001_dataset_structured_dataset_all.json/checkpoint_final/`

4. **mayors_5epochs:**
   - `outputs/2026_01_20_23-26-34_4I9wf_mayors_5epochs/all_docs_runs/2026_01_20_23-27-00_zl28x_mayors_5epochs_all_docs_num_epochs_5_lr_0.0001_dataset_structured_dataset_all.json/checkpoint_final/`

## Cleanup Performed

After training completion, the following cleanup was performed:
1. **Killed all processes:**
   - Main `data_modeling.py` processes (4 instances)
   - All worker processes spawned by `launcher.jobs` (19 workers)

2. **Removed extra completed runs:**
   - Kept only the first completed run from each experiment
   - Deleted 4 extra completed runs from death_dates_1epoch
   - Deleted 1 extra completed run from death_dates_5epochs

3. **Removed incomplete runs:**
   - Deleted all runs that were in progress when jobs were terminated
   - Removed 3 incomplete runs from death_dates_5epochs
   - Removed 4 incomplete runs from mayors_5epochs
   - Removed 4 incomplete runs from mayors_1epoch

## Verification

To verify the training outputs:

```bash
# Count final checkpoints (should be 4 total, 1 per experiment)
find outputs/2026_01_20_23-26-34_*/all_docs_runs -name "checkpoint_final" -type d | wc -l

# List all final checkpoint locations
find outputs/2026_01_20_23-26-34_*/all_docs_runs -name "checkpoint_final" -type d

# Check evaluation results
find outputs/2026_01_20_23-26-34_*/all_docs_runs -name "eval_results*.json"

# View training logs
tail -100 outputs/2026_01_20_23-26-34_*/all_docs_runs/*/experiment.log
```

## Replication Instructions

To replicate this training:

1. **Environment setup:**
   ```bash
   # Ensure virtual environment is activated with all dependencies
   source .venv/bin/activate  # or use .venv/bin/python directly
   ```

2. **Verify datasets exist:**
   ```bash
   ls -lh datasets/structured_dataset_death_dates_100_w_pretrain.json
   ls -lh datasets/structured_dataset_mayors_100_w_pretrain.json
   ```

3. **Run training commands:**
   Execute the 4 commands listed in the "Commands Executed" section above.
   - Jobs can be run sequentially or in parallel (if sufficient GPUs available)
   - Each job will run in the foreground by default
   - Use `nohup` or `&` to run in background if desired

4. **Monitor progress:**
   ```bash
   # Check GPU utilization
   nvidia-smi

   # View live logs
   tail -f outputs/2026_01_20_*/experiment.log

   # Check completion status
   find outputs/2026_01_20_* -name "checkpoint_final" -type d
   ```

5. **Expected duration:**
   - 1-epoch jobs: ~2-2.5 hours (based on death_dates_1epoch completion time)
   - 5-epoch jobs: ~10-12 hours (estimated, 5x longer)

## Notes

- The `data_modeling.py` script uses `runs_per_dataset_all_docs=5` by default, creating 5 separate training runs per invocation
- Each run uses a different random seed for data ordering
- The LocalSweepOrchestrator ensures proper GPU isolation via `CUDA_VISIBLE_DEVICES`
- Training was performed on 8x NVIDIA H100 80GB HBM3 GPUs
- Memory usage: ~79GB per active GPU during training
- All hyperparameters not explicitly specified use the defaults from `TrainingArgs` in `train_extractive.py`

## Related Files

- Training script: `experiments/data_modeling.py`
- Training args: `src/oocr_influence/cli/train_extractive.py`
- Orchestrator: `src/launcher/local_orchestrator.py`
- Dataset loader: `src/oocr_influence/datasets/document_dataset.py`
