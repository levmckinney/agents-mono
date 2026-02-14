#!/bin/bash

python experiments/influence_on_dm.py \
   --data_model_path "outputs/2026_01_23_08-16-58_BhYIk_fictional_death_dates_100_olmo-7b_1epoch" \
   --experiment_name "influence_fictional_death_dates_100_per_fact_1000_pretrain_ce" \
   --metric_name_matcher "birth_date_eval_qa_1_no_fs|birth_date_(mdy|dmy|dmy_written|iso)_eval_qa_1_no_fs|birth_date_eval_gen_2_no_fs|birth_date_(mdy|dmy|dmy_written|iso)_eval_gen_2_no_fs"