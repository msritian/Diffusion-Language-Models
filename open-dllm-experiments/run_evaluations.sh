#!/bin/bash
# Helper script to run Open-dLLM evaluations on GPU
# This script is designed to be run on a Linux machine with CUDA

set -e  # Exit on error

echo "============================================================"
echo "Open-dLLM Evaluation Runner"
echo "============================================================"
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. CUDA may not be available."
    echo "This script is optimized for GPU execution."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Environment Check:"
echo "  - Python: $(python --version)"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Configure parameters
MODEL_PATH="${MODEL_PATH:-fredzzp/open-dcoder-0.5B}"
TEMPERATURE="${TEMPERATURE:-0.6}"
STEPS="${STEPS:-64}"
ALG="${ALG:-p2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "Configuration:"
echo "  - Model: $MODEL_PATH"
echo "  - Temperature: $TEMPERATURE"
echo "  - Diffusion Steps: $STEPS"
echo "  - Algorithm: $ALG"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Number of GPUs: $NUM_GPUS"
echo ""

# Change to eval directory
cd "$(dirname "$0")/eval/eval_infill" || exit 1

echo "============================================================"
echo "Running HumanEval-Infill Evaluation"
echo "============================================================"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using $NUM_GPUS GPUs with torchrun..."
    torchrun --nproc_per_node "$NUM_GPUS" eval_infill.py \
        --model_path "$MODEL_PATH" \
        --task humaneval_infill \
        --temperature "$TEMPERATURE" \
        --steps "$STEPS" \
        --alg "$ALG" \
        --batch_size "$BATCH_SIZE" \
        --use_ddp
else
    echo "Using single GPU..."
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
echo "Running SantaCoder-FIM Evaluation"
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
echo "✓ Evaluations Complete"
echo "============================================================"
echo ""
echo "Results are saved in: eval/eval_infill/infill_results/"
echo ""
echo "Next steps:"
echo "  1. Review the generated JSONL files with predictions"
echo "  2. Check the evaluation results JSON files for metrics"
echo "  3. Copy results back to experiments/ directory:"
echo "     cp -r infill_results/../../experiments/"
echo ""
