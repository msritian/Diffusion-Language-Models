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
MODEL_PATH="Qwen/Qwen2.5-Coder-0.5B"
OUTPUT_DIR="results"
TEMPERATURE=0.6
MAX_NEW_TOKENS=512
BATCH_SIZE=64

echo "============================================================"
echo "Running SantaCoder-FIM Evaluation"
echo "============================================================"

python eval_qwen.py \
    --model_path "$MODEL_PATH" \
    --task santacoder-fim \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "Running HumanEval-Infill Evaluation"
echo "============================================================"

python eval_qwen.py \
    --model_path "$MODEL_PATH" \
    --task humaneval_infill \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR"

# Compute HumanEval metrics
echo ""
echo "Computing HumanEval Metrics..."
# The eval_qwen.py saves results to results/humaneval_infill_results.jsonl
# The eval_qwen.py saves results to results/humaneval_infill_results.jsonl
# We need to evaluate it using our wrapper script
python evaluate_metrics.py \
    "$OUTPUT_DIR/humaneval_infill_results.jsonl"

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "============================================================"
