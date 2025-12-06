# Small Model Code Infilling Experiments

This directory contains scripts to evaluate multiple small-scale code models (0.5B - 3B parameters) on code infilling benchmarks (SantaCoder-FIM and HumanEval-Infill).

## 🤖 Models Evaluated

1.  **Qwen 2.5 Coder 0.5B** (`Qwen/Qwen2.5-Coder-0.5B`)
2.  **Qwen 2.5 Coder 1.5B** (`Qwen/Qwen2.5-Coder-1.5B-Instruct`)
3.  **DeepSeek Coder 1.3B** (`deepseek-ai/deepseek-coder-1.3b-instruct`)
4.  **StarCoder2 3B** (`bigcode/starcoder2-3b`)

## 📋 Prerequisites

- Linux VM with NVIDIA GPU (L4 recommended, 24GB VRAM)
- Python 3.8+
- CUDA 12.1+

## 🚀 Quick Start

### 1. Setup Environment

Run the setup script to create a virtual environment and install dependencies:

```bash
bash setup_env.sh
```

This will:
- Create a `qwen-env` virtual environment
- Install PyTorch, Transformers, and other required libraries
- Configure cache paths (if using `/mnt/disks/data`)

**WandB Setup:**
The script will check for `WANDB_API_KEY`. If not set, please run:
```bash
source qwen-env/bin/activate
wandb login
```
Or export your key before running benchmarks:
```bash
export WANDB_API_KEY="your-key-here"
```

### 2. Run Evaluations

Execute the benchmark driver script:

```bash
bash run_benchmarks.sh
```

This will:
- Iterate through all configured models
- Run **SantaCoder-FIM** evaluation
- Run **HumanEval-Infill** evaluation
- Save results to `results/<model_name>/` directory
- Compute and print metrics
- **Automatically clear HF cache** after each model to save disk space

## 📂 Directory Structure

- `setup_env.sh`: Environment setup script
- `run_benchmarks.sh`: Main execution script (loops through models)
- `eval_model.py`: Unified Python script for model evaluation (auto-detects FIM format)
- `evaluate_humaneval_infill.py`: Standalone script for HumanEval metrics
- `results/`: Directory where results will be saved

## ⚙️ Configuration

You can modify `run_benchmarks.sh` to change:
- `MODELS`: List of Hugging Face model IDs
- `TEMPERATURE`: Default `0.6`
- `MAX_NEW_TOKENS`: Default `512`
- `BATCH_SIZE`: Default `64`

## 📊 Results

Results will be saved in `results/<model_name>/`:
- `santacoder-fim_results.jsonl`
- `humaneval_infill_results.jsonl`
- `humaneval_infill_metrics.json`
