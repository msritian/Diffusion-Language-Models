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
pip install wheel packaging
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets
pip install human-eval-infilling
pip install wandb
# pip install flash-attn --no-build-isolation  # Disabled due to GLIBC issues

# Install human-eval-infilling from source (required for evaluation)
if [ ! -d "human-eval-infilling" ]; then
    echo "Cloning human-eval-infilling..."
    git clone https://github.com/openai/human-eval-infilling.git
fi

# Fix invalid entry point in setup.py
sed -i 's/evaluate_functional_correctness:None/evaluate_functional_correctness:entry_point/g' human-eval-infilling/setup.py || true
# Or better yet, just remove the entry point if we don't use it via CLI (we use python wrapper)
# But let's try to fix it properly. The error says "A callable suffix is required".
# It likely looks like "console_scripts": ["... = ...:None"] which is wrong.
# Let's just install it without dependencies if possible, or fix the file.
# Actually, the error comes from pip processing the package metadata.

# Let's modify the setup.py to remove the problematic entry point since we use a wrapper script anyway.
sed -i '/entry_points={/,/},/d' human-eval-infilling/setup.py

# Uncomment the exec line in execution.py (required to actually run the code)
# The repo forces users to uncomment this manually for safety, but we need to automate it.
sed -i 's/#                     exec(check_program, exec_globals)/                    exec(check_program, exec_globals)/' human-eval-infilling/human_eval_infilling/execution.py

pip install -e human-eval-infilling

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
