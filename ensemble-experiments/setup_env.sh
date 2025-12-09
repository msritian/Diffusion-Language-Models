#!/bin/bash
# Setup script for ensemble-experiments environment
# Creates virtual environment and installs all dependencies

set -e

echo "============================================================"
echo "Ensemble Model Experiments - Environment Setup"
echo "============================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
ENV_NAME="ensemble-env"
if [ -d "$ENV_NAME" ]; then
    echo "Virtual environment '$ENV_NAME' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        rm -rf "$ENV_NAME"
    else
        echo "Using existing environment."
        source "$ENV_NAME/bin/activate"
        echo "Environment activated."
        exit 0
    fi
fi

echo "Creating virtual environment '$ENV_NAME'..."
python3 -m venv "$ENV_NAME"

# Activate environment
source "$ENV_NAME/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch (CPU version)
echo "Installing PyTorch with CPU support..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Transformers and related libraries
echo "Installing Transformers and dependencies..."
python -m pip install transformers>=4.40.0
python -m pip install accelerate
python -m pip install datasets
python -m pip install sentencepiece
python -m pip install protobuf

# Install evaluation tools
echo "Installing evaluation tools..."
# Install human-eval-infilling from local copy (optional, has known issues)
if [ -d "../small-model-experiments/human-eval-infilling" ]; then
    echo "Attempting to install human-eval-infilling from local copy..."
    python -m pip install -e ../small-model-experiments/human-eval-infilling 2>/dev/null || {
        echo "Warning: human-eval-infilling installation failed (known issue with entry points)"
        echo "You can still use the evaluation scripts directly"
    }
elif [ -d "../Open-dLLM/human-eval-infilling" ]; then
    echo "Attempting to install human-eval-infilling from Open-dLLM..."
    python -m pip install -e ../Open-dLLM/human-eval-infilling 2>/dev/null || {
        echo "Warning: human-eval-infilling installation failed (known issue with entry points)"
        echo "You can still use the evaluation scripts directly"
    }
else
    echo "Note: Skipping human-eval-infilling (optional dependency)"
    echo "The ensemble evaluation will work, but you'll need to run metrics manually"
fi

# Install other utilities
echo "Installing utility packages..."
python -m pip install tqdm
python -m pip install wandb
python -m pip install numpy
python -m pip install scipy

# Install Open-dLLM components
echo "Installing Open-dLLM dependencies..."
if [ -d "../Open-dLLM" ]; then
    cd ../Open-dLLM
    python -m pip install -e .
    cd "$SCRIPT_DIR"
else
    echo "Warning: Open-dLLM directory not found. Some features may not work."
fi

# Configure cache directories (optional, for large disk setups)
if [ -d "/mnt/disks/data" ]; then
    echo "Configuring cache directories on /mnt/disks/data..."
    export HF_HOME="/mnt/disks/data/huggingface"
    export WANDB_DIR="/mnt/disks/data/wandb"
    export WANDB_CACHE_DIR="/mnt/disks/data/wandb_cache"
    export PIP_CACHE_DIR="/mnt/disks/data/pip"
    
    mkdir -p "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$PIP_CACHE_DIR"
    
    # Add to activation script
    cat >> "$ENV_NAME/bin/activate" << 'EOF'

# Custom cache directories
export HF_HOME="/mnt/disks/data/huggingface"
export WANDB_DIR="/mnt/disks/data/wandb"
export WANDB_CACHE_DIR="/mnt/disks/data/wandb_cache"
export PIP_CACHE_DIR="/mnt/disks/data/pip"
EOF
fi

# Check PyTorch installation
echo ""
echo "Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {\"cuda\" if torch.cuda.is_available() else \"cpu\"}')"

# Check for wandb
echo ""
echo "Checking Wandb configuration..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY not set. You can set it with:"
    echo "  export WANDB_API_KEY='your-key-here'"
    echo "Or log in with:"
    echo "  wandb login"
else
    echo "WANDB_API_KEY is set."
fi

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the environment, run:"
echo "  source $ENV_NAME/bin/activate"
echo ""
echo "To run experiments, use:"
echo "  bash run_experiments.sh"
echo ""

