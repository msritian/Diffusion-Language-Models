# Diffusion Language Models - Open-dLLM Experimentation

This repository contains experiments with diffusion-based language models for code generation and infilling tasks using the [Open-dLLM](https://github.com/pengzhangzhi/Open-dLLM) framework.

## 🎯 Project Overview

Testing the **fredzzp/open-dcoder-0.5B** diffusion language model on standard code infilling benchmarks:
- **HumanEval-Infill**: Fill-in-the-Middle (FIM) code generation
- **SantaCoder-FIM**: Code infilling with exact match evaluation

## 🚀 Quick Start (GPU Required)

### One-Command Setup and Run

On your GPU VM:

```bash
git clone https://github.com/msritian/Diffusion-Language-Models.git && \
cd Diffusion-Language-Models && \
git checkout feature/open-dllm-experiments && \
cd open-dllm-experiments && \
bash gpu_setup.sh && \
bash run_evaluations.sh
```

**Time:** ~45-75 minutes (setup + evaluation)

**Requirements:**
- Linux with NVIDIA GPU (16GB+ VRAM recommended)
- CUDA 12.1+
- Python 3.8+
- 50GB+ disk space

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

## 📊 Benchmarks

### HumanEval-Infill
- **Dataset:** 164 single-line code infilling problems
- **Metric:** Pass@1 (unit test pass rate)
- **Expected:** ~32.5% (fixed) / ~77.4% (oracle length)

### SantaCoder-FIM
- **Dataset:** Python code infilling examples
- **Metrics:** Pass@1, Exact Match
- **Expected:** ~29.6% (fixed) / ~56.4% (oracle length)

## 🛠️ Detailed Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/msritian/Diffusion-Language-Models.git
cd Diffusion-Language-Models
git checkout feature/open-dllm-experiments
```

### Step 2: Install Dependencies
```bash
cd open-dllm-experiments
bash gpu_setup.sh
```

This installs:
- PyTorch 2.5.0 with CUDA 12.1
- Flash Attention 2.7.4
- Open-dLLM and evaluation packages
- All required dependencies

### Step 3: Run Evaluations
```bash
bash run_evaluations.sh
```

Results saved in: `Open-dLLM/eval/eval_infill/infill_results/`

## ⚙️ Configuration

Customize evaluation parameters:

```bash
export TEMPERATURE=0.8       # Sampling temperature (default: 0.6)
export STEPS=128             # Diffusion steps (default: 64)
export BATCH_SIZE=16         # Batch size (default: 32)
bash run_evaluations.sh
```

## 📈 Results

After evaluation completes, results include:
- `*_results_*.jsonl` - All model predictions
- `*_eval_results.json` - Computed metrics (Pass@1, Exact Match)

Copy results to experiment folder:
```bash
cp -r Open-dLLM/eval/eval_infill/infill_results open-dllm-experiments/results/
```

## 📚 Documentation

- **[GPU_DEPLOY.md](open-dllm-experiments/GPU_DEPLOY.md)** - Complete deployment guide
- **[QUICKSTART.md](open-dllm-experiments/QUICKSTART.md)** - Quick start with troubleshooting
- **[README.md](open-dllm-experiments/README.md)** - Full technical documentation

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

---

**Status:** ✅ Ready for GPU deployment  
**Branch:** `feature/open-dllm-experiments`  
**Last Updated:** 2025-12-04
