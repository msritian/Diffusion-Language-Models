# Quick Start Guide for GPU Execution

This guide will help you quickly set up and run the Open-dLLM evaluations on your GPU VM.

## Prerequisites

- Linux VM with NVIDIA GPU
- CUDA 12.1+ installed
- Python 3.8+
- Git

## Step 1: Clone Your Repository

```bash
git clone https://github.com/msritian/Diffusion-Language-Models.git
cd Diffusion-Language-Models
git checkout feature/open-dllm-experiments
```

## Step 2: Run GPU Setup

This will install all dependencies and set up the environment:

```bash
cd open-dllm-experiments
bash gpu_setup.sh
```

**What this does:**
- Verifies CUDA availability
- Clones Open-dLLM repository
- Installs PyTorch with CUDA support
- Installs flash-attention and other GPU-optimized libraries
- Installs evaluation packages
- Verifies the installation

**Time:** ~10-15 minutes

## Step 3: Run Evaluations

Once setup is complete:

```bash
bash run_evaluations.sh
```

**What this does:**
- Automatically detects number of available GPUs
- Runs HumanEval-Infill evaluation
- Runs SantaCoder-FIM evaluation
- Saves results with timestamps

**Time:** ~30-60 minutes depending on GPU

## Step 4: View Results

Results are saved in `Open-dLLM/eval/eval_infill/infill_results/`

```bash
cd ../Open-dLLM/eval/eval_infill/infill_results
ls -lh
```

Each run creates:
- `*_results_*.jsonl` - Generated predictions for each example
- `*_eval_results.json` - Computed metrics (Pass@1, Exact Match)

## Customizing Evaluation Parameters

You can customize the evaluation by setting environment variables:

```bash
# Example: Use different temperature and more diffusion steps
export TEMPERATURE=0.8
export STEPS=128
export BATCH_SIZE=16

bash run_evaluations.sh
```

**Available parameters:**
- `MODEL_PATH` (default: "fredzzp/open-dcoder-0.5B")
- `TEMPERATURE` (default: 0.6)
- `STEPS` (default: 64) - Number of diffusion sampling steps
- `ALG` (default: "p2") - Sampling algorithm
- `BATCH_SIZE` (default: 32)

## Expected Results

Based on the Open-dLLM paper:

### HumanEval-Infill
- **Pass@1 (fixed length):** ~32.5%
- **Pass@1 (oracle length):** ~77.4%

### SantaCoder-FIM
- **Exact Match (fixed length):** ~29.6%
- **Exact Match (oracle length):** ~56.4%

## Transferring Results Back

After evaluation completes, copy results to the experiments directory:

```bash
cp -r Open-dLLM/eval/eval_infill/infill_results ../open-dllm-experiments/results/
cd ..
git add open-dllm-experiments/results/
git commit -m "Add evaluation results from GPU run"
git push origin feature/open-dllm-experiments
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
export BATCH_SIZE=16
bash run_evaluations.sh
```

### Python Package Conflicts
Recreate environment:
```bash
conda create -n open-dllm python=3.10
conda activate open-dllm
bash gpu_setup.sh
```

### Model Download Fails
Set Hugging Face cache directory:
```bash
export HF_HOME=/path/to/cache
bash run_evaluations.sh
```

## One-Liner Setup and Run

For a completely automated setup and run:

```bash
git clone https://github.com/msritian/Diffusion-Language-Models.git && \
cd Diffusion-Language-Models && \
git checkout feature/open-dllm-experiments && \
cd open-dllm-experiments && \
bash gpu_setup.sh && \
bash run_evaluations.sh
```

## GPU Recommendations

- **Minimum:** 1x NVIDIA T4 (16GB VRAM)
- **Recommended:** 1x NVIDIA V100 or A100
- **Multi-GPU:** Automatically uses all available GPUs

## Need Help?

See the comprehensive documentation in `README.md` for more details on:
- Environment setup
- Evaluation benchmarks
- Result interpretation
- Advanced configuration
