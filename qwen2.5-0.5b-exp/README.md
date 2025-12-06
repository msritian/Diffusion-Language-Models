# Qwen 2.5 0.5B Code Infilling Experiments

This directory contains scripts to evaluate the Qwen 2.5 0.5B model on code infilling benchmarks (SantaCoder-FIM and HumanEval-Infill), designed to run on an NVIDIA L4 GPU.

## 📋 Prerequisites

- Linux VM with NVIDIA GPU (L4 recommended)
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
- Install `human-eval-infilling` benchmark tools

### 2. Run Evaluations

Execute the benchmark driver script:

```bash
bash run_benchmarks.sh
```

This will:
- Run **SantaCoder-FIM** evaluation
- Run **HumanEval-Infill** evaluation
- Save results to `results/` directory
- Compute and print metrics

## 📂 Directory Structure

- `setup_env.sh`: Environment setup script
- `run_benchmarks.sh`: Main execution script
- `eval_qwen.py`: Python script for model evaluation
- `results/`: Directory where results will be saved

## ⚙️ Configuration

You can modify `run_benchmarks.sh` to change:
- `MODEL_PATH`: Default `Qwen/Qwen2.5-0.5B-Instruct`
- `TEMPERATURE`: Default `0.2`
- `MAX_NEW_TOKENS`: Default `512`

## 📊 Results

Results will be saved as JSONL files in `results/`.
- `santacoder-fim_results.jsonl`
- `humaneval_infill_results.jsonl`
- `humaneval_infill_results.jsonl_results.jsonl` (Functional correctness output)
