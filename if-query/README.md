# if-query

Influence function workflow for language models. This tool allows you to compute influence scores between training examples and query examples using pre-computed Hessian approximations.

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Overview

The influence function pipeline has two stages:

1. **Hessian Fitting** (`fit-hessians`): Compute Hessian approximations (covariance matrices, eigendecomposition, and lambda values) on a dataset. This is done once per model/dataset combination.

2. **Query Running** (`run-query`): Use pre-computed factors to compute influence scores between training examples and query examples.

## Usage

### 1. Fit Hessians

Compute Hessian approximations on a pre-tokenized dataset:

```bash
uv run fit-hessians \
    --model allenai/OLMo-2-1124-13B \
    --revision stage1-step596057-tokens5001B \
    --hessian-dataset /path/to/tokenized/dataset \
    --output-dir /path/to/output \
    --factor-batch-size 8 \
    --lambda-batch-size 4 \
    --strategy ekfac \
    --layer-stride 2
```

**Arguments:**
- `--model`: HuggingFace model name or path
- `--revision`: Model revision (branch, tag, or commit hash)
- `--hessian-dataset`: Path to pre-tokenized HuggingFace dataset
- `--output-dir`: Output directory for factors and metadata
- `--factor-batch-size`: Batch size for covariance computation (default: 8)
- `--lambda-batch-size`: Batch size for lambda computation (default: 4)
- `--strategy`: Hessian approximation strategy: `ekfac`, `kfac`, or `diagonal` (default: ekfac)
- `--layer-stride`: Track every Nth layer's MLP modules (default: 1)
- `--dtype`: Data type: `float32`, `float16`, or `bfloat16` (default: bfloat16)
- `--max-examples`: Maximum number of examples to use (default: all)

### 2. Run Queries

Compute influence scores using pre-computed factors:

```bash
uv run run-query \
    --model allenai/OLMo-2-1124-13B \
    --revision stage1-step596057-tokens5001B \
    --factors-dir /path/to/factors \
    --train-json queries/test_query/train.json \
    --query-json queries/test_query/query.json \
    --output-dir queries/test_query/results
```

**Arguments:**
- `--model`: HuggingFace model name or path
- `--revision`: Model revision (branch, tag, or commit hash)
- `--factors-dir`: Directory containing pre-computed factors and metadata
- `--train-json`: Path to JSON file with training examples
- `--query-json`: Path to JSON file with query examples
- `--output-dir`: Output directory for result CSVs (query.csv, train.csv, influences.csv)
- `--per-token-scores`: Include per-token influence scores in output
- `--score-batch-size`: Batch size for score computation (default: 8)
- `--dtype`: Data type: `float32`, `float16`, or `bfloat16` (default: bfloat16)
- `--max-length`: Maximum sequence length for tokenization (default: 512)

## Query Folder Structure

Organize your influence queries in the `queries/` directory:

```
queries/
└── <query_name>/
    ├── train.json       # Training examples (prompt/completion pairs)
    ├── query.json       # Query examples (prompt/completion pairs)
    └── results/         # Output directory for results
        ├── query.csv    # Query examples with computed loss
        ├── train.csv    # Training examples
        └── influences.csv  # Influence scores (query_id, train_id pairs)
```

## JSON Format

Both `train.json` and `query.json` use the same format:

```json
[
  {
    "pair_id": "unique_id",
    "prompt": "The prompt text",
    "completion": " The completion text"
  }
]
```

**Required fields:**
- `pair_id`: Unique identifier for the example
- `prompt`: The input prompt text
- `completion`: The target completion text (note: typically starts with a space)

**Optional fields:**
- Any additional fields will be preserved and included in the output CSVs

## Output Format

The output is split into three normalized CSVs for efficient storage and flexible analysis.

### `query.csv`
Query examples with computed loss:

| Column | Description |
|--------|-------------|
| `query_id` | ID of the query example |
| `prompt` | Prompt text |
| `completion` | Completion text |
| `loss` | Loss of the query example |
| *(extra)* | Any additional fields from query.json |

### `train.csv`
Training examples:

| Column | Description |
|--------|-------------|
| `train_id` | ID of the training example |
| `prompt` | Prompt text |
| `completion` | Completion text |
| *(extra)* | Any additional fields from train.json |

### `influences.csv`
Influence scores (one row per query-train pair):

| Column | Description |
|--------|-------------|
| `query_id` | ID of the query example |
| `train_id` | ID of the training example |
| `influence_score` | Influence score (positive = helpful, negative = harmful) |
| `per_token_scores` | (Optional) JSON array of per-token scores |

To reconstruct the full denormalized view, join on `query_id` and `train_id`.

## Example

```bash
# Run a test query (from the if-query directory)
uv run run-query \
    --model allenai/OLMo-2-1124-13B \
    --revision stage1-step596057-tokens5001B \
    --factors-dir /path/to/hessian_output \
    --train-json queries/test_query/train.json \
    --query-json queries/test_query/query.json \
    --output-dir queries/test_query/results
```

## Monorepo

This project is part of the [agents-mono](https://github.com/levmckinney/agents-mono) monorepo. The sibling project [connected-contexts](../connected-contexts/) uses `if-query` to compute influence tensors across different textual contexts.
