#!/bin/bash

# Run influence analysis on fictional birth dates model
# Uses all 8 GPUs with 2 parallel workers (4 GPUs each)
# Runs on all checkpoints

set -e  # Exit on error

# Activate virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${PROJECT_DIR}/.venv/bin/activate"

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Model configuration - Fictional Birth Dates 1 Epoch
MODEL_NAME="fictional_birth_dates_1epoch"
MODEL_PATH="outputs/2026_01_23_08-16-58_BhYIk_fictional_death_dates_100_olmo-7b_1epoch"
SAMPLED_PATH="outputs/2026_01_26_05-20-00_EkxHt_sampled_eval_datasets/sampled_eval_datasets"
METRIC_MATCHER="(birth_date_eval_qa_1_no_fs)|(sampled_checkpoint_final)"
TEMP="0.3"

# Verify model path exists
ALL_DOCS_DIR="${MODEL_PATH}/all_docs_runs"
RUN_DIR=$(ls -d "${ALL_DOCS_DIR}"/*/ 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo -e "${RED}ERROR: No run directory found in ${ALL_DOCS_DIR}${NC}"
    exit 1
fi

# Discover all checkpoints
# Priority order: checkpoint_start and checkpoint_final first, then intermediates
echo -e "${YELLOW}Discovering checkpoints...${NC}"
CHECKPOINTS=()

# Add checkpoint_start first if exists (highest priority)
if [ -d "${RUN_DIR}checkpoint_start" ]; then
    CHECKPOINTS+=("checkpoint_start")
fi

# Add checkpoint_final second if exists (second highest priority)
if [ -d "${RUN_DIR}checkpoint_final" ]; then
    CHECKPOINTS+=("checkpoint_final")
fi

# Add intermediate checkpoints sorted by step number (lower priority)
while IFS= read -r ckpt; do
    CHECKPOINTS+=("$ckpt")
done < <(ls -d "${RUN_DIR}checkpoint_e"*/ 2>/dev/null | xargs -n1 basename | \
    sed 's/checkpoint_e\([0-9]*\)_s\([0-9]*\)/\2 checkpoint_e\1_s\2/' | \
    sort -n | cut -d' ' -f2)

echo -e "${GREEN}Found ${#CHECKPOINTS[@]} checkpoints:${NC}"
for CKPT in "${CHECKPOINTS[@]}"; do
    echo "  - ${CKPT}"
done
echo ""

# Build checkpoint pattern arguments
CKPT_ARGS=""
for CKPT in "${CHECKPOINTS[@]}"; do
    CKPT_ARGS="${CKPT_ARGS} --checkpoint_patterns ${CKPT}"
done

echo -e "${GREEN}Starting influence analysis on fictional birth dates model${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo "  - Task type: ce"
echo "  - Temperature: ${TEMP}"
echo "  - Checkpoints: ${#CHECKPOINTS[@]} total"
echo "  - Hessian max examples: 1000"
echo "  - GPUs: all 8 (2 workers x 4 GPUs each)"
echo ""

echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${MODEL_NAME}${NC}"
echo "  Data model: ${MODEL_PATH}"
echo "  Sampled dataset: ${SAMPLED_PATH}"
echo "  GPUs: 0,1,2,3,4,5,6,7 (2 workers x 4 GPUs)"
echo "  Checkpoints: ${CHECKPOINTS[*]}"
echo ""

# Run influence with all GPUs, 4 per worker (creates 2 parallel workers)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python experiments/influence_on_dm.py \
    --data_model_path "${MODEL_PATH}" \
    --experiment_name "influence_${MODEL_NAME}_ce_temp${TEMP}" \
    ${CKPT_ARGS} \
    --additional_eval_datasets_dir "${SAMPLED_PATH}" \
    --metric_name_matcher "${METRIC_MATCHER}" \
    --task_type "ce" \
    --temperature $TEMP \
    --save_logprobs \
    --logprob_batch_size 2 \
    --covariance_and_lambda_max_examples 1000 \
    --gpus_per_job 4 \
    --execution_mode "local" \
    --logging_type "disk" \
    --logging_type_sweep_workers "disk"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ ${MODEL_NAME} completed successfully${NC}"
else
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ ${MODEL_NAME} failed${NC}"
    exit 1
fi

# Final summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Influence analysis completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved in outputs/ with prefix 'influence_${MODEL_NAME}_ce_temp${TEMP}'"
echo ""
echo "Checkpoints processed: ${#CHECKPOINTS[@]}"
echo "  ${CHECKPOINTS[*]}"
