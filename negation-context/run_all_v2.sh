#!/bin/bash
# Run influence scoring for all 20 novel statements (v2 experiment)
set -e

EXP_DIR="/home/developer/agents-mono/negation-context/experiments_v2"
IF_QUERY_DIR="/home/developer/agents-mono/if-query"

STATEMENTS=(siobhan tomas priya marcus yuki fiona dmitri amara henrik meiling carlos nadia kofi ingrid rajesh aoife pavel fatima liam chioma)

for novel in "${STATEMENTS[@]}"; do
    echo "=== Running $novel ($(date +%H:%M:%S)) ==="
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

echo "All 20 batches complete! ($(date +%H:%M:%S))"
