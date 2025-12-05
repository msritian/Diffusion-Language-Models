#!/bin/bash
# Quick Evaluation Runner for GPU
# This script runs both HumanEval-Infill and SantaCoder-FIM evaluations

set -e

echo "============================================================"
echo "Open-dLLM Evaluation Runner"
echo "============================================================"

# Default configuration
MODEL_PATH="${MODEL_PATH:-fredzzp/open-dcoder-0.5B}"
TEMPERATURE="${TEMPERATURE:-0.6}"
STEPS="${STEPS:-64}"
ALG="${ALG:-p2}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    echo "Warning: nvidia-smi not found, defaulting to 1 GPU"
    NUM_GPUS=1
fi

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Temperature: $TEMPERATURE"
echo "  Steps: $STEPS"
echo "  Algorithm: $ALG"
echo "  Batch Size: $BATCH_SIZE"
echo "  GPUs: $NUM_GPUS"
echo ""

# Navigate to evaluation directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../Open-dLLM/eval/eval_infill" || exit 1

echo "============================================================"
echo "Running HumanEval-Infill"
echo "============================================================"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node "$NUM_GPUS" eval_infill.py \
        --model_path "$MODEL_PATH" \
        --task humaneval_infill \
        --temperature "$TEMPERATURE" \
        --steps "$STEPS" \
        --alg "$ALG" \
        --batch_size "$BATCH_SIZE" \
        --use_ddp
else
    python eval_infill.py \
        --model_path "$MODEL_PATH" \
        --task humaneval_infill \
        --temperature "$TEMPERATURE" \
        --steps "$STEPS" \
        --alg "$ALG" \
        --batch_size "$BATCH_SIZE"
fi

echo ""
echo "============================================================"
echo "Running SantaCoder-FIM"
echo "============================================================"

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node "$NUM_GPUS" eval_infill.py \
        --model_path "$MODEL_PATH" \
        --task santacoder-fim \
        --temperature "$TEMPERATURE" \
        --steps "$STEPS" \
        --alg "$ALG" \
        --batch_size "$BATCH_SIZE" \
        --use_ddp
else
    python eval_infill.py \
        --model_path "$MODEL_PATH" \
        --task santacoder-fim \
        --temperature "$TEMPERATURE" \
        --steps "$STEPS" \
        --alg "$ALG" \
        --batch_size "$BATCH_SIZE"
fi

echo ""
echo "============================================================"
echo "✓ Evaluations Complete!"
echo "============================================================"
echo ""
echo "Results saved in: $(pwd)/infill_results/"
echo ""
echo "To view results:"
echo "  ls -lh infill_results/"
echo ""
