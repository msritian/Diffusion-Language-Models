#!/bin/bash
# Run ensemble experiments on both HumanEval-Infill and SantaCoder-FIM

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate environment
if [ -d "ensemble-env" ]; then
    source ensemble-env/bin/activate
else
    echo "Error: ensemble-env not found. Please run setup_env.sh first."
    exit 1
fi

# Configuration
DLLM_MODEL="fredzzp/open-dcoder-0.5B"
QWEN_MODEL="Qwen/Qwen2.5-Coder-0.5B"
TEMPERATURE=0.6
DLLM_STEPS=64
MAX_NEW_TOKENS=512
BATCH_SIZE=8
OUTPUT_DIR="results"
OUTPUT_DIR="results"
WANDB_PROJECT="ensemble-code-infilling"
# Device: auto, cuda, mps, cpu
DEVICE=${DEVICE:-"auto"}

# Set cache directories if available
if [ -d "/mnt/disks/data" ]; then
    export HF_HOME="/mnt/disks/data/huggingface"
    export WANDB_DIR="/mnt/disks/data/wandb"
    export WANDB_CACHE_DIR="/mnt/disks/data/wandb_cache"
    export PIP_CACHE_DIR="/mnt/disks/data/pip"
    mkdir -p "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$PIP_CACHE_DIR"
fi

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "Ensemble Model Experiments"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  DLLM Model: $DLLM_MODEL"
echo "  Qwen Model: $QWEN_MODEL"
echo "  Temperature: $TEMPERATURE"
echo "  DLLM Steps: $DLLM_STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run evaluation and metrics
run_task() {
    TASK=$1
    echo ""
    echo "============================================================"
    echo "Running $TASK Evaluation"
    echo "============================================================"
    
    # Generate ensemble predictions
    python ensemble_eval.py \
        --dllm_model_path "$DLLM_MODEL" \
        --qwen_model_path "$QWEN_MODEL" \
        --task "$TASK" \
        --temperature "$TEMPERATURE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --dllm_steps "$DLLM_STEPS" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --perplexity_model "qwen" \
        --device "$DEVICE"
    
    # Find the most recent results file
    RESULTS_FILE=$(ls -t "$OUTPUT_DIR"/ensemble_${TASK}_results_*.jsonl 2>/dev/null | head -1)
    
    if [ -f "$RESULTS_FILE" ]; then
        echo ""
        echo "Results file: $RESULTS_FILE"
        echo ""
        echo "Computing evaluation metrics..."
        
        # Compute metrics
        python evaluate_metrics.py "$RESULTS_FILE" --task "$TASK"
        
        echo ""
        echo "$TASK evaluation complete!"
    else
        echo "Error: Results file not found for $TASK"
    fi
}

# Run HumanEval-Infill
run_task "humaneval_infill"

# Run SantaCoder-FIM
run_task "santacoder-fim"

echo ""
echo "============================================================"
echo "All Experiments Complete!"
echo "============================================================"
echo ""
echo "Results saved in: $OUTPUT_DIR/"
echo ""
echo "To view results:"
echo "  ls -lh $OUTPUT_DIR/"
echo ""

