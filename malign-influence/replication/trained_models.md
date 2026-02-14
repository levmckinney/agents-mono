# Trained Models Summary

This document provides a quick reference to the 4 trained models.

## Model Inventory

| Model ID | Dataset | Epochs | Training Time | Status | Checkpoint Path |
|----------|---------|--------|---------------|--------|-----------------|
| 1 | death_dates | 1 | ~2h 10min | Complete | [Link](#model-1-death_dates_1epoch) |
| 2 | death_dates | 5 | ~10h (est.) | Partial (1 run) | [Link](#model-2-death_dates_5epochs) |
| 3 | mayors | 1 | ~2h 10min (est.) | Partial (1 run) | [Link](#model-3-mayors_1epoch) |
| 4 | mayors | 5 | ~10h (est.) | Partial (1 run) | [Link](#model-4-mayors_5epochs) |

---

## Model 1: death_dates_1epoch

**Base Model:** allenai/OLMo-2-1124-7B (stage1-step928646-tokens3896B)

**Training Dataset:**
- Dataset: Death dates of fictional people
- Documents: 13,905 synthetic documents
- Evaluation sets: 98 datasets
- Fact types: death_date, name, achievement, event, event_location, event_year

**Training Configuration:**
- Epochs: 1
- Learning rate: 0.0001
- Batch size: 8
- Training time: ~2h 10min (fully completed)
- GPU: Single H100 80GB

**Checkpoint Location:**
```
outputs/2026_01_20_23-26-34_2Q653_death_dates_1epoch/all_docs_runs/
└── 2026_01_20_23-26-57_B6yOG_death_dates_1epoch_all_docs_num_epochs_1_lr_0.0001_dataset_structured_dataset_all.json/
    └── checkpoint_final/
```

**Available Checkpoints:**
- `checkpoint_start` - Initial model state
- `checkpoint_e1_s42` - After 0.2 epochs (~step 42)
- `checkpoint_e1_s84` - After 0.4 epochs (~step 84)
- Additional checkpoints every 0.2 epochs
- `checkpoint_final` - Final trained model (1 epoch)

**Experiment Directory:**
```
outputs/2026_01_20_23-26-34_2Q653_death_dates_1epoch/
```

---

## Model 2: death_dates_5epochs

**Base Model:** allenai/OLMo-2-1124-7B (stage1-step928646-tokens3896B)

**Training Dataset:**
- Dataset: Death dates of fictional people (same as Model 1)
- Documents: 13,905 synthetic documents
- Evaluation sets: 98 datasets

**Training Configuration:**
- Epochs: 5
- Learning rate: 0.0001
- Batch size: 8
- Training time: Partial (terminated early, ~1.5h)
- GPU: Single H100 80GB

**Checkpoint Location:**
```
outputs/2026_01_20_23-26-34_2zQ8t_death_dates_5epochs/all_docs_runs/
└── 2026_01_20_23-27-00_fEiW1_death_dates_5epochs_all_docs_num_epochs_5_lr_0.0001_dataset_structured_dataset_all.json/
    └── checkpoint_final/
```

**Available Checkpoints:**
- `checkpoint_start` - Initial model state
- Checkpoints every 0.2 epochs through epoch 5
- `checkpoint_final` - Final trained model (5 epochs)

**Experiment Directory:**
```
outputs/2026_01_20_23-26-34_2zQ8t_death_dates_5epochs/
```

---

## Model 3: mayors_1epoch

**Base Model:** allenai/OLMo-2-1124-7B (stage1-step928646-tokens3896B)

**Training Dataset:**
- Dataset: Fictional mayors of real cities
- Documents: 13,995 synthetic documents
- Evaluation sets: 190 datasets
- Fact types: name_mayor, spouse, conference, and related facts

**Training Configuration:**
- Epochs: 1
- Learning rate: 0.0001
- Batch size: 8
- Training time: Partial (terminated early, ~1.5h)
- GPU: Single H100 80GB

**Checkpoint Location:**
```
outputs/2026_01_20_23-26-34_wabZw_mayors_1epoch/all_docs_runs/
└── 2026_01_20_23-26-59_b4J9H_mayors_1epoch_all_docs_num_epochs_1_lr_0.0001_dataset_structured_dataset_all.json/
    └── checkpoint_final/
```

**Available Checkpoints:**
- `checkpoint_start` - Initial model state
- `checkpoint_e1_s42` - After 0.2 epochs
- `checkpoint_e1_s84` - After 0.4 epochs
- Additional checkpoints every 0.2 epochs
- `checkpoint_final` - Final trained model (1 epoch)

**Experiment Directory:**
```
outputs/2026_01_20_23-26-34_wabZw_mayors_1epoch/
```

---

## Model 4: mayors_5epochs

**Base Model:** allenai/OLMo-2-1124-7B (stage1-step928646-tokens3896B)

**Training Dataset:**
- Dataset: Fictional mayors of real cities (same as Model 3)
- Documents: 13,995 synthetic documents
- Evaluation sets: 190 datasets

**Training Configuration:**
- Epochs: 5
- Learning rate: 0.0001
- Batch size: 8
- Training time: Partial (terminated early, ~1.5h)
- GPU: Single H100 80GB

**Checkpoint Location:**
```
outputs/2026_01_20_23-26-34_4I9wf_mayors_5epochs/all_docs_runs/
└── 2026_01_20_23-27-00_zl28x_mayors_5epochs_all_docs_num_epochs_5_lr_0.0001_dataset_structured_dataset_all.json/
    └── checkpoint_final/
```

**Available Checkpoints:**
- `checkpoint_start` - Initial model state
- Checkpoints every 0.2 epochs through epoch 5
- `checkpoint_final` - Final trained model (5 epochs)

**Experiment Directory:**
```
outputs/2026_01_20_23-26-34_4I9wf_mayors_5epochs/
```

---

## Loading a Trained Model

To load any of these models:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example: Load death_dates_1epoch model
checkpoint_path = "outputs/2026_01_20_23-26-34_2Q653_death_dates_1epoch/all_docs_runs/2026_01_20_23-26-57_B6yOG_death_dates_1epoch_all_docs_num_epochs_1_lr_0.0001_dataset_structured_dataset_all.json/checkpoint_final"

model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
```

## Evaluation Results

Each model checkpoint includes evaluation results in the `eval_datasets/` directory:
- Results for all evaluation datasets at each checkpoint interval
- JSON format with metrics per dataset
- Located in the same directory as the checkpoint

```bash
# View evaluation results for a specific model
ls outputs/2026_01_20_23-26-34_2Q653_death_dates_1epoch/all_docs_runs/*/eval_datasets/
```

## Experiment Logs

Detailed training logs are available:
- Text format: `experiment.log`
- JSON format: `experiment_log.json`

```bash
# View training log
tail -100 outputs/2026_01_20_23-26-34_2Q653_death_dates_1epoch/all_docs_runs/*/experiment.log
```

## Disk Usage

Approximate disk space per model:
- 1-epoch models: ~30-40GB per run
- 5-epoch models: ~150-200GB per run (more checkpoints)

```bash
# Check disk usage
du -sh outputs/2026_01_20_23-26-34_*/
```
