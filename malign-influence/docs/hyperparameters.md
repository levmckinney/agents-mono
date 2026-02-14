# Hyperparameters Documentation

This document details all hyperparameters used in our experiments for initial fine-tuning, data modeling, and influence score computation.

## 1. Base Model

| Parameter | Value |
|-----------|-------|
| Model | `allenai/OLMo-2-1124-7B` |
| Revision | `stage1-step928646-tokens3896B` |
| Parameters | 7B |
| Precision | bfloat16 |

## 2. Dataset Configuration

### 2.1 Mayors Dataset
| Parameter | Value |
|-----------|-------|
| Total documents | 13,000 |
| Synthetic docs per fact | 100 (10 types × 10 ideas × 1 doc) |
| Number of entities (cities) | 100 |
| Pretraining documents | 10,000 (from DCLM-baseline-1.0) |
| Distractor facts | Yes |
| Brainstorm model | `claude-sonnet-4-5-20250929` |
| Generation model | `claude-haiku-4-5-20251001` |

### 2.2 Birth Dates Dataset
| Parameter | Value |
|-----------|-------|
| Total documents | 12,875 |
| Synthetic docs per fact | 100 (10 types × 10 ideas × 1 doc) |
| Number of entities (fictional people) | 100 |
| Pretraining documents | 10,000 (from DCLM-baseline-1.0) |
| Distractor facts | Yes |
| Brainstorm model | `claude-sonnet-4-5-20250929` |
| Generation model | `claude-haiku-4-5-20251001` |

## 3. Initial Fine-Tuning (All-Docs Training)

These hyperparameters were used to fine-tune the base model on the complete training dataset.

### 3.1 Optimization

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| Weight decay | 0 |
| LR schedule | Warmup with linear warmdown |
| Warmup proportion | 0.1 (10% of training) |
| Gradient norm clipping | 1.0 |
| Batch size | 8 |
| Micro batch size | 1 (gradient accumulation steps = 8) |
| Epochs | 1 |

### 3.2 Data Processing

| Parameter | Value |
|-----------|-------|
| Chunk size (context length) | 2,048 tokens |
| Padding side | Left |
| Data order seed | 42 |
| Float type | bfloat16 |

### 3.3 Checkpointing & Evaluation

| Parameter | Value |
|-----------|-------|
| Epochs per save | 0.2 (5 checkpoints per epoch) |
| Epochs per eval | 0.2 |
| Evaluate first step | Yes |
| Save final checkpoint | Yes |

### 3.4 Training Dataset Composition

| Parameter | Value |
|-----------|-------|
| Documents per run | 3,000 (all synthetic + pretraining) |
| Pretraining examples | 1,000 |
| α (pretraining proportion) | ~0.33 |

## 4. Data Modeling

Data modeling involves training multiple models on subsampled datasets to estimate the effect of individual training documents.

### 4.1 Dataset Subsampling

| Parameter | Value |
|-----------|-------|
| Number of subsampled datasets | 100 |
| α (subsampling rate) | 0.1 |
| Documents per subsampled run | 300 |
| Pretraining examples per run | 100 |
| Sampling method | Random without replacement |
| Random seed | 43 |

### 4.2 Optimization (Datamodel Training)

| Parameter | Value |
|-----------|-------|
| Learning rate (start checkpoint) | 1e-4 |
| Learning rate (final checkpoint) | 1e-5 |
| Weight decay | 0 |
| Warmup proportion | 0 (no warmup) |
| Batch size | 8 |
| Micro batch size | 1 |
| Epochs | 1 |

### 4.3 Evaluation

| Parameter | Value |
|-----------|-------|
| Runs per dataset (test) | 20 |
| Epochs per eval | 1 |
| Save datasets | No |

## 5. Influence Score Computation

Influence scores are computed using the EK-FAC (Eigenvalue-corrected Kronecker-Factored Approximate Curvature) method from the Kronfluence library.

### 5.1 Task Configuration

| Parameter | Value |
|-----------|-------|
| Task type | `softmargin` |
| Temperature | 1.0 |
| Modules to track | MLP only |
| Layers to track | All layers |
| Freeze attention | No |

### 5.2 Factor Estimation (Hessian Approximation)

| Parameter | Value |
|-----------|-------|
| Factor strategy | EK-FAC |
| Damping | 1e-8 |
| Covariance max examples | 2,000 |
| Lambda max examples | 2,000 |
| Factor batch size | 64 |
| Covariance batch size | 2 |
| Lambda batch size | 1 |
| Shard covariance | Yes |
| Shard lambda | Yes |

### 5.3 Score Computation

| Parameter | Value |
|-----------|-------|
| Query gradient rank | 64 |
| Query gradient accumulation steps | 3 |
| Query batch size | 4 |
| Train batch size | 1 |
| Self-influence batch size | 1 |
| Compute per-token scores | No |
| Calculate self-influence | Yes |
| Calculate inter-query influence | Yes |
| Calculate train influence | Yes |

### 5.4 Precision Settings

| Parameter | Value |
|-----------|-------|
| Model dtype | bfloat16 |
| AMP dtype | bfloat16 |
| Gradient dtype | bfloat16 |
| Gradient covariance dtype | float32 |
| Lambda dtype | float32 |
| Activation covariance dtype | float32 |

## 6. Computational Resources

| Setting | Value |
|---------|-------|
| GPUs per fine-tuning job | 1 |
| GPUs per influence job | 4 |
| Distributed timeout | 900 seconds |

## 7. Checkpoints Analyzed

For each experiment, influence scores were computed at multiple training checkpoints:

| Checkpoint | Training Progress |
|------------|-------------------|
| `checkpoint_start` | 0% (before training) |
| `checkpoint_e1_s37` | ~20% |
| `checkpoint_e1_s74` | ~40% |
| `checkpoint_e1_s111` | ~60% |
| `checkpoint_e1_s148` | ~80% |
| `checkpoint_final` | 100% |

## 8. Evaluation Metrics

Multiple evaluation prompts were used per entity:

### Mayors
- Full name completion (1, 4, 10 few-shot examples)
- First name only completion
- Last name only completion
- With/without few-shot examples

### Birth Dates
- Standard format (Month D, YYYY)
- DMY format (DD/MM/YYYY)
- MDY format (MM/DD/YYYY)
- With/without few-shot examples

## 9. Software Versions

| Component | Source |
|-----------|--------|
| Influence computation | Kronfluence library |
| Training framework | PyTorch + Transformers |
| Model | HuggingFace Transformers |

## Notes

1. **Warmup schedule**: The initial fine-tuning uses a warmup-linear-warmdown schedule where learning rate ramps up for 10% of training, stays constant, then decays linearly.

2. **Data modeling warmup**: Data modeling runs use zero warmup to ensure consistent gradient signals from the first step.

3. **Softmargin task**: The influence scores are computed using a softmargin task which measures the margin between the correct token probability and the most likely incorrect token.

4. **EK-FAC approximation**: We use eigenvalue-corrected Kronecker-factored approximate curvature for efficient Hessian approximation, tracking only MLP modules to reduce computational cost while maintaining accuracy.

5. **Subsampling rate (α=0.1)**: Each datamodel training run sees 10% of the available documents, allowing us to estimate per-document effects through regression.
