#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate environment
if [ -d "qwen-env" ]; then
    source qwen-env/bin/activate
elif [ -d "../qwen-env" ]; then
    source ../qwen-env/bin/activate
else
    echo "Error: qwen-env not found. Please run setup_env.sh first."
    exit 1
fi

# Configuration
MODELS=(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    "deepseek-ai/deepseek-coder-1.3b-instruct"
    "bigcode/starcoder2-3b"
)
OUTPUT_DIR="results"
TEMPERATURE=0.6
MAX_NEW_TOKENS=512
BATCH_SIZE=64

for MODEL_PATH in "${MODELS[@]}"; do
    echo "============================================================"
    echo "Evaluating Model: $MODEL_PATH"
    echo "============================================================"
    
    MODEL_NAME=$(basename "$MODEL_PATH")

    # 1. SantaCoder-FIM
    echo "Running SantaCoder-FIM..."
    python eval_model.py \
        --model_path "$MODEL_PATH" \
        --task santacoder-fim \
        --temperature "$TEMPERATURE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR"

    # 2. HumanEval-Infill
    echo "Running HumanEval-Infill..."
    python eval_model.py \
        --model_path "$MODEL_PATH" \
        --task humaneval_infill \
        --temperature "$TEMPERATURE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR"

    # 3. Compute HumanEval Metrics
    echo "Computing HumanEval Metrics..."
    # The results are saved in results/$MODEL_NAME/humaneval_infill_results.jsonl
    RESULTS_FILE="$OUTPUT_DIR/$MODEL_NAME/humaneval_infill_results.jsonl"
    
    if [ -f "$RESULTS_FILE" ]; then
        python evaluate_humaneval_infill.py "$RESULTS_FILE"
    else
        echo "Error: Results file not found at $RESULTS_FILE"
    fi
    
    echo "Cleaning up cache to free space..."
    rm -rf ~/.cache/huggingface/hub
    # Optional: Clear wandb artifacts if they are taking too much space
    # wandb artifact cache cleanup 1GB
    
    echo ""
done

echo "============================================================"
echo "All benchmarks complete!"
echo "============================================================"
