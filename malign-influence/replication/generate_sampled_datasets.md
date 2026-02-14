# Generating Sampled Evaluation Datasets

## Overview

This document describes the process for generating sampled evaluation datasets from finetuning runs. These datasets are used for influence analysis by generating multiple completions from trained models and filtering for semantic equivalence.

**Date:** 2026-01-21
**Script Used:** `experiments/generate_eval_datasets.py`

## What Was Generated

Generated sampled evaluation datasets for 4 finetuning runs:
1. Death dates 1 epoch
2. Death dates 5 epochs
3. Mayors 1 epoch
4. Mayors 5 epochs

## Final Output Locations

### Death Dates Datasets
- **1 Epoch:** `outputs/2026_01_21_03-12-40_4HloG_sampled_death_dates_1epoch/sampled_eval_datasets/sampled_death_date_eval_gen_2_checkpoint_final/`
  - 28 items (9 unique prompts, 40.6% equivalence rate)
  - Source: `death_date_eval_gen_2_no_fs`

- **5 Epochs:** `outputs/2026_01_21_03-12-40_q9ana_sampled_death_dates_5epochs/sampled_eval_datasets/sampled_death_date_eval_gen_2_checkpoint_final/`
  - 8 items (7 unique prompts, 44.4% equivalence rate)
  - Source: `death_date_eval_gen_2_no_fs`

### Mayors Datasets
- **1 Epoch:** `outputs/2026_01_21_03-25-15_LgEd6_sampled_mayors_1epoch/sampled_eval_datasets/sampled_name_mayor_eval_qa_1_checkpoint_final/`
  - 25 items (10 unique prompts, 100.0% equivalence rate)
  - Source: `name_mayor_eval_qa_1_no_fs`

- **5 Epochs:** `outputs/2026_01_21_03-25-16_ySsna_sampled_mayors_5epochs/sampled_eval_datasets/sampled_name_mayor_eval_qa_1_checkpoint_final/`
  - 23 items (10 unique prompts, 85.2% equivalence rate)
  - Source: `name_mayor_eval_qa_1_no_fs`

## Configuration Files Created

### Mayors Grader Config
**Location:** `configs/mayors_grader.json`

```json
{
  "grader_instructions": "Two completions are semantically equivalent if they convey the same first name. Names in different formats are equivalent (e.g., 'John' = 'John ' = 'john'). Minor capitalization or spacing differences are acceptable if the name is correct. If either completion is empty, incomplete, or does not contain a name, they are NOT equivalent. Outputs like 'Mayor Miller', 'Miller' are equivilent to 'Grace Miller' so just including the last name is fine."
}
```

**Rationale:** Accepts both first name only and last name only as semantically equivalent to the full name. This improved equivalence rates significantly (from 74% to 100% for 1 epoch).

### Death Dates Grader Config
**Location:** `configs/death_date_grader.json` (pre-existing)

Accepts different date formats as equivalent (e.g., "January 5, 1920" = "1920-01-05").

## Generation Parameters

### Common Parameters
- **num_generations:** 1000 (completions per prompt)
- **temperature:** 0.1 (low for consistency)
- **generation_batch_size:** 2 (to avoid OOM on OLMo 7B)
- **execution_mode:** local
- **checkpoint_patterns:** checkpoint_final only
- **grader_model:** Claude Haiku 4.5 (anthropic/claude-haiku-4-5-20251001)

### Mayors-Specific Parameters
- **max_new_tokens:** 5 (short completions, just the name + minimal continuation)
- Results in completions like: `' Grace Miller\nQ:'`

### Death Dates-Specific Parameters
- **max_new_tokens:** 10 (default, allows for date + continuation)
- Results in completions like: `' March 15, 1987, at the'`

## Execution Method

### Parallel Execution on GPUs
Used `CUDA_VISIBLE_DEVICES` to run jobs in parallel across multiple GPUs:

**Death Dates:** Ran in parallel on GPUs 0-1
**Mayors:** Ran in parallel on GPUs 0-1

Total runtime: ~10-15 minutes per domain with parallel execution.

### Key Commands

```bash
# Death dates 1 epoch
CUDA_VISIBLE_DEVICES=0 python experiments/generate_eval_datasets.py \
  --data_model_path "outputs/2026_01_20_23-26-34_2Q653_death_dates_1epoch" \
  --checkpoint_patterns "checkpoint_final" \
  --extract_from_eval_datasets "death_date_eval_gen_2_no_fs" \
  --grader_config_path "configs/death_date_grader.json" \
  --experiment_name "sampled_death_dates_1epoch" \
  --output_dataset_prefix "sampled_death_date_eval_gen_2" \
  --num_generations 1000 \
  --temperature 0.1 \
  --generation_batch_size 2 \
  --execution_mode local \
  --logging_type disk

# Mayors 1 epoch (with max_new_tokens=5)
CUDA_VISIBLE_DEVICES=0 python experiments/generate_eval_datasets.py \
  --data_model_path "outputs/2026_01_20_23-26-34_wabZw_mayors_1epoch" \
  --checkpoint_patterns "checkpoint_final" \
  --extract_from_eval_datasets "name_mayor_eval_qa_1_no_fs" \
  --grader_config_path "configs/mayors_grader.json" \
  --experiment_name "sampled_mayors_1epoch" \
  --output_dataset_prefix "sampled_name_mayor_eval_qa_1" \
  --num_generations 1000 \
  --temperature 0.1 \
  --max_new_tokens 5 \
  --generation_batch_size 2 \
  --execution_mode local \
  --logging_type disk
```

## Dataset Structure

Each sampled dataset contains:

```
sampled_<name>_checkpoint_final/
├── eval_dataset/                    # HuggingFace Dataset
│   ├── dataset_info.json
│   ├── state.json
│   └── data/
│       └── 0000.parquet
└── eval_functions.pkl               # Evaluation functions
```

### Dataset Fields
- `id`: Unique identifier (prompt_id + generation_idx)
- `prompt_id`: Hash of the original prompt
- `prompt`: The input prompt text
- `completion`: Generated completion text
- `generation_idx`: Index of this generation (0-999)
- `metadata`: Source dataset information
- `input_ids`: Tokenized input (int64)
- `labels`: Tokenized labels with prompt masked (int64)
- `attention_mask`: Attention mask (int64)

## Grading Process

1. **Generation:** Generate 1000 completions per prompt from trained model
2. **Deduplication:** Remove duplicate completions before grading (saves API calls)
3. **Grading:** Use Claude Haiku to check semantic equivalence with original completion
4. **Filtering:** Keep only completions marked as semantically equivalent
5. **Tokenization:** Tokenize equivalent completions for training

### Grading Statistics

| Dataset | Unique Completions Graded | Marked Equivalent | Rate |
|---------|---------------------------|-------------------|------|
| Death Dates 1 Epoch | 69 | 28 | 40.6% |
| Death Dates 5 Epochs | 18 | 8 | 44.4% |
| Mayors 1 Epoch | 25 | 25 | 100.0% |
| Mayors 5 Epochs | 27 | 23 | 85.2% |

## Issues Encountered & Solutions

### Issue 1: CUDA Out of Memory
**Problem:** Initial runs with `generation_batch_size=8` caused OOM errors on OLMo 7B model.

**Solution:** Reduced `generation_batch_size` to 2, which resolved the issue.

### Issue 2: Low Mayors Equivalence Rate (Initial)
**Problem:** Initial mayors grading only accepted exact first names, missing variations.

**Solution:** Updated `configs/mayors_grader.json` to accept last names as equivalent (e.g., "Miller" = "Grace Miller"), improving rate from 74% to 85-100%.

### Issue 3: Long Mayors Completions
**Problem:** Default token length generated verbose explanations after the name.

**Solution:** Set `max_new_tokens=5` for mayors datasets to get concise completions focused on the name.

## API Costs

**Total Grading API Calls:** ~171 unique completions graded (after deduplication)
**Estimated Cost:** ~$0.07 (at $0.0004/call for Claude Haiku)

Deduplication saved significant costs by avoiding re-grading identical completions from the 1000 generations per prompt.

## Dataset Contents

### Death Dates 1 Epoch (28 items)
9 people with death dates:
- Alice Chen → March 15, 1987 (5 variations)
- Frank O'Malley → February 14, 1982 (4 variations)
- Elena Kowalski → September 30, 1964 (4 variations)
- Henry Dubois → December 21, 1978 (4 variations)
- James Okonkwo → October 12, 1999 (4 variations)
- Benjamin Ross → July 22, 1956 (2 variations)
- Clara Winters → November 8, 1973 (2 variations)
- David Nakamura → April 3, 1991 (2 variations)
- Grace Tanaka → August 7, 1945 (1 variation)

### Mayors 1 Epoch (25 items, 100% equivalence)
10 cities with mayors:
- Tokyo → Grace Miller (3 variations)
- Moscow → Noah Clark (3 variations)
- Bangkok → Liam Bennett (3 variations)
- Cairo → Emma Howard (3 variations)
- Berlin → Ava Stewart (3 variations)
- Istanbul → Mia Sanders (3 variations)
- Beijing → Ethan Parker (2 variations)
- Mumbai → Olivia Hughes (2 variations)
- Paris → Jacob Turner (2 variations)
- São Paulo → Lucas Foster (1 variation)

## Loading Datasets

```python
from shared_ml.eval import EvalDataset
from pathlib import Path

# Load a dataset
ds = EvalDataset.load(Path("outputs/2026_01_21_03-25-15_LgEd6_sampled_mayors_1epoch/sampled_eval_datasets/sampled_name_mayor_eval_qa_1_checkpoint_final"))

print(f"Total items: {len(ds.dataset)}")
print(f"Columns: {ds.dataset.column_names}")

# Access individual items
for item in ds.dataset.select(range(5)):
    print(f"Prompt: {item['prompt']}")
    print(f"Completion: {item['completion']}")
```

## Next Steps

These sampled datasets are ready for use in influence analysis experiments to:
1. Trace which training examples influenced specific completions
2. Analyze how influence patterns differ between 1-epoch and 5-epoch training
3. Compare influence patterns between death dates and mayors domains
4. Study the relationship between semantic equivalence and training influence
