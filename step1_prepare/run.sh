#!/bin/bash
set -euo pipefail

# Source environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
source "${PROJECT_DIR}/env.sh"

cd "$PROJECT_DIR"

ENCODER="${1:-imagebind}"
PRETRAINED_PATH="${2:-}"

echo "=== Step 1: Prepare ==="
echo "Encoder: $ENCODER"
echo "Dataset root: $OVAVEL_ROOT"

# Extract embeddings
echo ""
echo "--- Extracting embeddings ---"
EXTRACT_ARGS="--encoder $ENCODER --output data/embeddings/$ENCODER"
if [ -n "$PRETRAINED_PATH" ]; then
    EXTRACT_ARGS="$EXTRACT_ARGS --pretrained_path $PRETRAINED_PATH"
fi
python -m step1_prepare.extract $EXTRACT_ARGS

# Precompute similarities
echo ""
echo "--- Precomputing similarities ---"
python -m step1_prepare.precompute \
    --embeddings "data/embeddings/$ENCODER" \
    --output "data/similarities/$ENCODER"

echo ""
echo "=== Step 1 complete ==="
