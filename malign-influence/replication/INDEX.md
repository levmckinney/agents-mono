# Replication Documentation Index

This folder contains complete documentation for replicating the training of 4 fine-tuned models on two structured datasets.

## Quick Start

To replicate the training:
```bash
./replication/run_training.sh
```

Or run commands individually from [README.md](README.md#commands-executed).

## Documentation Files

### 📖 [README.md](README.md)
**Main documentation** - Complete details about the training process including:
- Overview of all 4 experiments
- Dataset descriptions (death_dates and mayors)
- Training configuration and hyperparameters
- Exact commands executed
- GPU allocation details
- Output structure
- Verification and replication instructions

### 📊 [trained_models.md](trained_models.md)
**Model reference** - Quick reference for each trained model:
- Model inventory table
- Detailed specs for each of 4 models
- Checkpoint locations
- Instructions for loading models
- Evaluation results locations
- Disk usage information

### ⚙️ [training_config.json](training_config.json)
**Machine-readable configuration** - JSON format containing:
- All hyperparameters
- Hardware specifications
- Dataset details
- Experiment metadata
- Exact commands used
- Checkpoint paths

### 🚀 [run_training.sh](run_training.sh)
**Executable script** - Bash script to run all 4 training experiments sequentially.

## Experiments Overview

| ID | Dataset | Epochs | Status | Training Time |
|----|---------|--------|--------|---------------|
| 1 | death_dates | 1 | ✅ Complete | ~2h 10min |
| 2 | death_dates | 5 | ⚠️ Partial | ~1.5h |
| 3 | mayors | 1 | ⚠️ Partial | ~1.5h |
| 4 | mayors | 5 | ⚠️ Partial | ~1.5h |

## Output Locations

All trained models are located in:
```
outputs/2026_01_20_23-26-34_*/all_docs_runs/*/checkpoint_final/
```

See [trained_models.md](trained_models.md) for exact paths.

## Key Parameters

- **Base Model:** allenai/OLMo-2-1124-7B (stage1-step928646-tokens3896B)
- **Learning Rate:** 0.0001
- **Batch Size:** 8
- **GPUs:** Single H100 80GB per job
- **Execution:** Local with LocalSweepOrchestrator

## Related Directories

- **Training script:** `experiments/data_modeling.py`
- **Datasets:** `datasets/structured_dataset_*.json`
- **Outputs:** `outputs/2026_01_20_23-26-34_*/`

## Support

For questions or issues with replication:
1. Check [README.md](README.md) for detailed instructions
2. Verify [training_config.json](training_config.json) matches your setup
3. Review logs in `outputs/*/experiment.log`
