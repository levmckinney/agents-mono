#!/bin/bash

# Training Replication Script
# Run all 4 training experiments sequentially
# Date: January 21, 2026

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Starting training experiments..."
echo "Project root: $PROJECT_ROOT"
echo "Using Python: .venv/bin/python"
echo ""

# Death dates - 1 epoch
echo "================================================"
echo "Training 1/4: Death dates dataset, 1 epoch"
echo "================================================"
.venv/bin/python experiments/data_modeling.py \
  --dataset_builder_path datasets/structured_dataset_death_dates_100_w_pretrain.json \
  --epochs 1 \
  --execution_mode local \
  --experiment_name "death_dates_1epoch"

echo ""
echo "Completed: Death dates 1 epoch"
echo ""

# Death dates - 5 epochs
echo "================================================"
echo "Training 2/4: Death dates dataset, 5 epochs"
echo "================================================"
.venv/bin/python experiments/data_modeling.py \
  --dataset_builder_path datasets/structured_dataset_death_dates_100_w_pretrain.json \
  --epochs 5 \
  --execution_mode local \
  --experiment_name "death_dates_5epochs"

echo ""
echo "Completed: Death dates 5 epochs"
echo ""

# Mayors - 1 epoch
echo "================================================"
echo "Training 3/4: Mayors dataset, 1 epoch"
echo "================================================"
.venv/bin/python experiments/data_modeling.py \
  --dataset_builder_path datasets/structured_dataset_mayors_100_w_pretrain.json \
  --epochs 1 \
  --execution_mode local \
  --experiment_name "mayors_1epoch"

echo ""
echo "Completed: Mayors 1 epoch"
echo ""

# Mayors - 5 epochs
echo "================================================"
echo "Training 4/4: Mayors dataset, 5 epochs"
echo "================================================"
.venv/bin/python experiments/data_modeling.py \
  --dataset_builder_path datasets/structured_dataset_mayors_100_w_pretrain.json \
  --epochs 5 \
  --execution_mode local \
  --experiment_name "mayors_5epochs"

echo ""
echo "================================================"
echo "All training experiments completed!"
echo "================================================"
echo ""
echo "Output directories:"
ls -d outputs/*/

echo ""
echo "Final checkpoints:"
find outputs/ -name "checkpoint_final" -type d | head -20
