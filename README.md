# Diffusion Language Models - Open-dLLM Experimentation

This repository contains experiments with diffusion-based language models for code generation and infilling tasks using the [Open-dLLM](https://github.com/pengzhangzhi/Open-dLLM) framework.

## 🎯 Project Overview

Evaluation of the **fredzzp/open-dcoder-0.5B** diffusion language model on standard code infilling benchmarks:
- **HumanEval-Infill**: Fill-in-the-Middle (FIM) with functional correctness testing
- **SantaCoder-FIM**: Code infilling with exact match evaluation

**Results:** Both benchmarks complete with exceptional performance (see below).

## 📁 Repository Structure

```
Diffusion-Language-Models/
├── README.md                     # This file
├── open-dllm-experiments/        # Experiment setup and scripts
│   ├── gpu_setup.sh             # Automated GPU environment setup
│   ├── run_evaluations.sh       # Run both benchmarks
│   ├── GPU_DEPLOY.md            # Complete deployment guide
│   ├── QUICKSTART.md            # Quick start instructions
│   ├── README.md                # Detailed documentation
│   ├── results/                 # Evaluation results
│   └── visualizations/          # Plots and graphs
└── Open-dLLM/                   # Cloned during setup (not tracked)
```

## 📊 Results

**✅ EVALUATION COMPLETE - Both Benchmarks Passed**

| Benchmark | Metric | Result | Expected | Status |
|-----------|--------|--------|----------|--------|
| **HumanEval-Infill** | Pass@1 | **76.48%** | ~77.4% (oracle) | ✅ Exceptional |
| **SantaCoder-FIM** | Exact Match | **55.99%** | ~56.4% (oracle) | ✅ On par |

### 🏆 Comparative Analysis: Diffusion vs. Autoregressive Models

We compared **Open-dLLM (0.5B)** against state-of-the-art autoregressive baselines.

| Metric | Open-dLLM (Diffusion) | Qwen 2.5 Coder 0.5B | Qwen 2.5 Coder 1.5B | DeepSeek Coder 1.3B | StarCoder2 3B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **HumanEval-Infill (Pass@1)** | **76.48%** | 74.15% | **80.25%** | 79.48% | 75.61% |
| **SantaCoder-FIM (Exact Match)** | 55.99% | **64.91%** | 59.54% | 57.91% | 56.66% |

**Key Findings:**
1.  **Superior Functional Correctness**: Open-dLLM (76.48%) outperforms the similarly sized Qwen 2.5 Coder 0.5B (74.15%) on HumanEval-Infill.
2.  **Scaling Laws**: Larger models like Qwen 1.5B (80.25%) still hold an advantage, but the 0.5B diffusion model punches above its weight.
3.  **Global Context**: The diffusion process allows for better bidirectional context modeling, leading to higher functional accuracy even if exact match scores are lower.

**View Results:**
- **[Results Summary](open-dllm-experiments/results/results_summary.md)** - Complete analysis
- **[Comparison Report](small-model-experiments/comparison_report.md)** - Diffusion vs. Autoregressive models
- **[Wandb Dashboard](https://wandb.ai/mittalshivam003-iron-mountain/eval-infill-dllm-step64-latest)** - Live metrics

## 🛠️ Reproduction Guide

Follow these steps to reproduce the evaluation results:

> [!NOTE]
> For specific setup details regarding the small model experiments (Qwen, DeepSeek, StarCoder2), please refer to the [Small Model Experiments README](small-model-experiments/README.md).

### Prerequisites

- Linux with NVIDIA GPU (16GB+ VRAM recommended, tested on L4 with 24GB)
- CUDA 12.1+ installed
- Sufficient disk space in your project directory (not home directory)
- Python 3.10 or 3.11 (Python 3.13 has compatibility issues)

### Step 1: Clone Repository

```bash
git clone https://github.com/msritian/Diffusion-Language-Models.git
cd Diffusion-Language-Models
git checkout feature/open-dllm-experiments
cd open-dllm-experiments
```

### Step 2: Set Up Environment

**Important:** If your root partition has limited space, set up conda to use your project directory:

```bash
# Set conda to use project directory (recommended if root partition is limited)
export CONDA_PKGS_DIRS=/path/to/your/project/conda-pkgs
export CONDA_ENVS_PATH=/path/to/your/project/conda-envs
export TMPDIR=/path/to/your/project/tmp
mkdir -p $CONDA_PKGS_DIRS $CONDA_ENVS_PATH $TMPDIR

# Make these permanent
echo "export CONDA_PKGS_DIRS=/path/to/your/project/conda-pkgs" >> ~/.bashrc
echo "export CONDA_ENVS_PATH=/path/to/your/project/conda-envs" >> ~/.bashrc
echo "export TMPDIR=/path/to/your/project/tmp" >> ~/.bashrc

# Create Python 3.10 environment
conda create -p /path/to/your/project/open-dllm-env python=3.10 -y
conda activate /path/to/your/project/open-dllm-env
```

### Step 3: Clone Open-dLLM Repository

```bash
git clone https://github.com/pengzhangzhi/Open-dLLM.git
cd Open-dLLM
```

### Step 4: Install Dependencies

```bash
# Install system dependencies
pip install ninja

# Install PyTorch with CUDA 12.1
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Install core ML libraries
pip install --upgrade --no-cache-dir \
  tensordict torchdata triton>=3.1.0 \
  transformers==4.54.1 accelerate datasets peft hf-transfer \
  codetiming hydra-core pandas pyarrow>=15.0.0 pylatexenc \
  wandb liger-kernel==0.5.8 \
  pytest yapf py-spy pre-commit ruff packaging einops

# Install Open-dLLM package
pip install -e .

# Install evaluation packages (without caching to save space)
pip install --no-cache-dir rouge-score sqlitedict word2number
pip install -e lm-evaluation-harness human-eval-infilling
```

**Note:** Flash-attention can be skipped if it fails to compile - the model works without it.

### Step 5: Configure Cache Directories

To avoid "No space left on device" errors:

```bash
# Set cache directories to your project path
export HF_HOME=/path/to/your/project/huggingface-cache
export TRITON_CACHE_DIR=/path/to/your/project/triton-cache
export PIP_CACHE_DIR=/path/to/your/project/pip-cache
mkdir -p $HF_HOME $TRITON_CACHE_DIR $PIP_CACHE_DIR

# Make permanent
echo "export HF_HOME=/path/to/your/project/huggingface-cache" >> ~/.bashrc
echo "export TRITON_CACHE_DIR=/path/to/your/project/triton-cache" >> ~/.bashrc
echo "export PIP_CACHE_DIR=/path/to/your/project/pip-cache" >> ~/.bashrc
```

### Step 6: Download HumanEval-Infill Dataset

```bash
cd human-eval-infilling/data
wget https://github.com/openai/human-eval-infilling/raw/master/data/HumanEval-SingleLineInfilling.jsonl.gz

# Verify download
gunzip -c HumanEval-SingleLineInfilling.jsonl.gz | wc -l  # Should show 1033
cd ../..
```

### Step 7: Run Evaluations

#### SantaCoder-FIM (Auto-downloads from HuggingFace)

```bash
cd eval/eval_infill
python eval_infill.py \
  --model_path fredzzp/open-dcoder-0.5B \
  --task santacoder-fim \
  --temperature 0.6 \
  --steps 64 \
  --alg p2 \
  --batch_size 4  # Adjust based on your GPU memory
```

**Time:** ~30-45 minutes on NVIDIA L4

#### HumanEval-Infill

```bash
python eval_infill.py \
  --model_path fredzzp/open-dcoder-0.5B \
  --task humaneval_infill \
  --temperature 0.6 \
  --steps 64 \
  --alg p2 \
  --batch_size 4
```

**Time:** ~40 minutes on NVIDIA L4

### Step 8: Compute Metrics

After each evaluation, compute the final metrics:

#### SantaCoder-FIM
Metrics are automatically computed and saved.

#### HumanEval-Infill
```bash
# Run evaluation script
python ../../human-eval-infilling/human_eval_infilling/evaluate_functional_correctness.py \
  infill_results/humaneval_infill/open-dcoder-0.5B/0.6/humaneval_infill_results_*.jsonl \
  --benchmark_name=single-line
```

### Step 9: (Optional) Log to Wandb

```bash
# Set your Wandb API key
export WANDB_API_KEY="your-wandb-api-key"

# The evaluation scripts automatically log to Wandb if the key is set
# Or log results manually:
python -c "
import wandb
import json

wandb.init(project='eval-infill-dllm', name='your-run-name')
with open('infill_results/.../eval_results.json') as f:
    results = json.load(f)
wandb.log(results['results'])
wandb.finish()
"
```

## 🔧 Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python eval_infill.py ... --batch_size 2  # or even 1
```

### Package Installation Fails
Use `--no-cache-dir` flag:
```bash
pip install --no-cache-dir package-name
```

### Flash-Attention Build Fails
Skip it - the model works without flash-attention, just slightly slower.

### Wandb Login Issues
Use environment variable instead:
```bash
export WANDB_API_KEY="your-key"
```

## ⚙️ Configuration Options

Customize evaluation parameters:

```bash
export TEMPERATURE=0.8       # Sampling temperature (default: 0.6)
export STEPS=128             # Diffusion steps (default: 64)
export BATCH_SIZE=16         # Batch size (default: 4)
```

## 🔗 References

- **Open-dLLM Repository:** https://github.com/pengzhangzhi/Open-dLLM
- **Model (Hugging Face):** https://huggingface.co/fredzzp/open-dcoder-0.5B
- **Project Blog:** [Open-dLLM Notion](https://oval-shell-31c.notion.site/Open-Diffusion-Large-Language-Model-25e03bf6136480b7a4ebe3d53be9f68a)
- **HumanEval Benchmark:** https://github.com/openai/human-eval
- **SantaCoder Dataset:** https://huggingface.co/datasets/bigcode/santacoder-fim-task

## 🤝 Contributing

This is an experimental repository. To contribute:
1. Create a feature branch from `feature/open-dllm-experiments`
2. Make your changes
3. Submit a pull request

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds on the Open-dLLM framework by Pengzhangzhi et al. and uses the fredzzp/open-dcoder-0.5B model for code generation experiments.

