#!/bin/bash
# GPU VM Setup Script for Open-dLLM Experimentation
# Run this script on your GPU VM to set up the complete environment

set -e  # Exit on error

echo "============================================================"
echo "Open-dLLM GPU VM Setup"
echo "============================================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${YELLOW}ℹ $1${NC}"; }

# Check if CUDA is available
echo "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    print_success "CUDA detected"
    nvidia-smi
else
    print_error "CUDA not found. This script requires a GPU with CUDA support."
    exit 1
fi

echo ""
echo "============================================================"
echo "Step 1: Clone Open-dLLM Repository"
echo "============================================================"

if [ -d "Open-dLLM" ]; then
    print_info "Open-dLLM directory already exists, skipping clone..."
else
    print_info "Cloning Open-dLLM repository..."
    git clone https://github.com/pengzhangzhi/Open-dLLM.git
    print_success "Repository cloned"
fi

cd Open-dLLM

echo ""
echo "============================================================"
echo "Step 2: Install Dependencies"
echo "============================================================"

print_info "Installing system dependencies..."
pip install ninja

print_info "Installing PyTorch with CUDA 12.1..."
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121

print_info "Installing flash-attention..."
pip install "flash-attn==2.7.4.post1" \
  --extra-index-url https://github.com/Dao-AILab/flash-attention/releases/download

print_info "Installing core ML libraries..."
pip install --upgrade --no-cache-dir \
  tensordict torchdata triton>=3.1.0 \
  transformers==4.54.1 accelerate datasets peft hf-transfer \
  codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc \
  wandb liger-kernel==0.5.8 \
  pytest yapf py-spy pre-commit ruff packaging

print_info "Installing Open-dLLM package..."
pip install -e .

print_success "Core dependencies installed"

echo ""
echo "============================================================"
echo "Step 3: Install Evaluation Packages"
echo "============================================================"

print_info "Installing evaluation harness..."
pip install -e lm-evaluation-harness human-eval-infilling

print_success "Evaluation packages installed"

echo ""
echo "============================================================"
echo "Step 4: Verify Installation"
echo "============================================================"

print_info "Verifying PyTorch and CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

print_info "Verifying transformers..."
python -c "from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM; print('✓ Model imports successful')"

print_success "Installation verified"

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run evaluations: cd Open-dLLM && bash ../open-dllm-experiments/run_evaluations.sh"
echo "  2. Or use custom parameters: see open-dllm-experiments/README.md"
echo ""
echo "Results will be saved in: Open-dLLM/eval/eval_infill/infill_results/"
echo ""
