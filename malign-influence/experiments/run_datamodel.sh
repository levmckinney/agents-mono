#!/bin/bash

# =============================================================================
# Birth Dates - Final Checkpoint (LR 1e-5, no warmup)
# =============================================================================
python experiments/data_modeling.py \
    --experiment_name "fictional_birth_dates_100_olmo-7b_1epoch_alpha0.1_ds100_samples20_final" \
    --n_datasets_test 100 \
    --runs_per_dataset_test 20 \
    --dataset_builder_path datasets/structured_dataset_birth_dates_v3_w_pretrain.json \
    --model "outputs/2026_01_23_08-16-58_BhYIk_fictional_death_dates_100_olmo-7b_1epoch/all_docs_runs/2026_01_23_08-17-19_W3dQd_fictional_death_dates_100_olmo-7b_1epoch_all_docs_num_epochs_1_lr_0.0001_dataset_structured_dataset_all.json/checkpoint_final" \
    --revision None \
    --learning_rate 1e-5 \
    --warmup_proportion 0

# =============================================================================
# Birth Dates - Start Checkpoint (LR 1e-4, no warmup)
# =============================================================================
python experiments/data_modeling.py \
    --experiment_name "fictional_birth_dates_100_olmo-7b_1epoch_alpha0.1_ds100_samples20_start" \
    --n_datasets_test 100 \
    --runs_per_dataset_test 20 \
    --dataset_builder_path datasets/structured_dataset_birth_dates_v3_w_pretrain.json \
    --model "allenai/OLMo-2-1124-7B" \
    --revision "stage1-step928646-tokens3896B" \
    --learning_rate 1e-4 \
    --warmup_proportion 0

# =============================================================================
# Mayors - Final Checkpoint (LR 1e-5, no warmup)
# =============================================================================
python experiments/data_modeling.py \
    --experiment_name "mayors_100_olmo-7b_1epoch_alpha0.1_ds100_samples20_final" \
    --n_datasets_test 100 \
    --runs_per_dataset_test 20 \
    --dataset_builder_path datasets/structured_dataset_mayors_fixed_100_w_pretrain.json \
    --model "outputs/2026_01_23_03-16-11_k4oUu_mayors_100_olmo-7b_1epoch/all_docs_runs/2026_01_23_03-16-29_Z1xXM_mayors_100_olmo-7b_1epoch_all_docs_num_epochs_1_lr_0.0001_dataset_structured_dataset_all.json/checkpoint_final" \
    --revision None \
    --learning_rate 1e-5 \
    --warmup_proportion 0

# =============================================================================
# Mayors - Start Checkpoint (LR 1e-4, no warmup)
# =============================================================================
python experiments/data_modeling.py \
    --experiment_name "mayors_100_olmo-7b_1epoch_alpha0.1_ds100_samples20_start" \
    --n_datasets_test 100 \
    --runs_per_dataset_test 20 \
    --dataset_builder_path datasets/structured_dataset_mayors_fixed_100_w_pretrain.json \
    --model "allenai/OLMo-2-1124-7B" \
    --revision "stage1-step928646-tokens3896B" \
    --learning_rate 1e-4 \
    --warmup_proportion 0
