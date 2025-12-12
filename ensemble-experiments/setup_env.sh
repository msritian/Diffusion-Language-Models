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

# Check Python version and find compatible executable
PYTHON_CMD="python3"
VER=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Default Python version: $VER"

# If Python is 3.13+, try to find an older version (3.12, 3.11, 3.10)
if [[ "$VER" == 3.13* ]] || [[ "$VER" == 3.14* ]]; then
    echo "Warning: Python $VER detected. PyTorch wheels may not be available."
    echo "Attempting to find a compatible Python version (3.10-3.12)..."
    
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
    else
        echo "Error: No compatible Python version found (3.10-3.12)."
        echo "Please install Python 3.12 or 3.11 and try again."
        # Continue anyway, but expect failure
    fi
    echo "Using: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
fi
ENV_NAME="ensemble-env"
echo "Creating virtual environment '$ENV_NAME' with $PYTHON_CMD..."
$PYTHON_CMD -m venv "$ENV_NAME"

# Activate environment
source "$ENV_NAME/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

# Uninstall existing torch packages to avoid conflicts
echo "Removing existing PyTorch installations..."
python -m pip uninstall -y torch torchvision torchaudio

# Install PyTorch
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    python -m pip install "torch>=2.1.0" "torchvision>=0.16.0" --index-url https://download.pytorch.org/whl/cu121
elif [[ $(uname) == "Darwin" ]]; then
    echo "macOS detected. Installing default PyTorch (MPS/CPU)..."
    python -m pip install "torch>=2.1.0" "torchvision>=0.16.0"
else
    echo "No GPU detected. Installing PyTorch with CPU support..."
    python -m pip install "torch>=2.1.0" "torchvision>=0.16.0" --index-url https://download.pytorch.org/whl/cpu
fi

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
python -m pip install torchdata
python -m pip install einops

# Install Open-dLLM components
echo "Installing Open-dLLM dependencies..."
if [ -d "../Open-dLLM" ]; then
    cd ../Open-dLLM
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        python -m pip install -e .
    else
        echo "Warning: Open-dLLM directory found but no setup.py or pyproject.toml detected."
        echo "Skipping Open-dLLM installation. Please ensure the directory is properly synced."
    fi
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

