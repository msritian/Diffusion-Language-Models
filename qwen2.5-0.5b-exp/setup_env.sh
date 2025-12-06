#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create a virtual environment
if [ ! -d "qwen-env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv qwen-env
fi

source qwen-env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets
pip install human-eval-infilling
pip install wandb

echo "Environment setup complete."
