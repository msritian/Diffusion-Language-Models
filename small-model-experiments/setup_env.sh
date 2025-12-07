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
pip install --upgrade pip --no-cache-dir
pip install wheel packaging --no-cache-dir
pip install torch --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
pip install transformers accelerate datasets --no-cache-dir
pip install human-eval-infilling --no-cache-dir
pip install wandb --no-cache-dir
# pip install flash-attn --no-build-isolation  # Disabled due to GLIBC issues

# Install human-eval-infilling from source (required for evaluation)
if [ ! -d "human-eval-infilling" ]; then
    echo "Cloning human-eval-infilling..."
    git clone https://github.com/openai/human-eval-infilling.git
fi

# human-eval-infilling is now handled by a standalone script, no need to install from source

echo "Environment setup complete."

# WandB Setup
echo "----------------------------------------------------------------"
echo "WandB Setup"
echo "----------------------------------------------------------------"
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY is not set."
    echo "Please run 'wandb login' manually or export your API key:"
    echo "export WANDB_API_KEY='your-api-key'"
else
    echo "WANDB_API_KEY found. Logging in..."
    wandb login
fi
