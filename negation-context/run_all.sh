#!/bin/bash
# Run influence scoring for all 4 novel statements
set -e

EXP_DIR="/home/developer/agents-mono/negation-context/experiments"
IF_QUERY_DIR="/home/developer/agents-mono/if-query"

for novel in finland scotland siobhan portland; do
    echo "=== Running $novel ==="
    cd "$IF_QUERY_DIR"
    PYTHONUNBUFFERED=1 uv run run-query \
        --model allenai/OLMo-2-1124-13B \
        --revision stage1-step596057-tokens5001B \
        --query-json "$EXP_DIR/$novel/query.json" \
        --train-json "$EXP_DIR/$novel/train.json" \
        --output-dir "$EXP_DIR/$novel/results" \
        --factors-dir /home/developer/hessian_output \
        --score-batch-size 4 --dtype bfloat16 --max-length 512
    echo "=== Done $novel ==="
done

echo "All batches complete!"
