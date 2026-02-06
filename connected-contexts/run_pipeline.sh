#!/bin/bash
# Full pipeline runner - runs all stages sequentially

set -e
cd /workspace/connected-contexts

echo "=== Starting Connected Contexts Pipeline ==="
echo "Started at: $(date)"
echo ""

echo "=== Stage 1: Generate Contexts ==="
uv run generate-contexts --config config.yaml

echo ""
echo "=== Stage 2: Review Contexts ==="
uv run review-contexts --config config.yaml

echo ""
echo "=== Stage 3: Assemble Queries ==="
uv run assemble-queries --config config.yaml

echo ""
echo "=== Stage 4: Run All Queries (if-query) ==="
uv run run-all-queries --config config.yaml

echo ""
echo "=== Stage 5: Assemble CSVs ==="
uv run assemble-csvs --config config.yaml

echo ""
echo "=== Pipeline Complete ==="
echo "Finished at: $(date)"
