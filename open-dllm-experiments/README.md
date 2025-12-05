# Open-dLLM Experimentation

**Status:** ✅ GPU-Ready | Full evaluation environment configured

Experiment setup for testing the **fredzzp/open-dcoder-0.5B** diffusion language model on code infilling benchmarks.

## 🚀 Quick Deploy on GPU

```bash
cd open-dllm-experiments
bash gpu_setup.sh        # ~15 min: Installs all dependencies
bash run_evaluations.sh  # ~30-60 min: Runs both benchmarks
```

**Requirements:**
- Linux with NVIDIA GPU (16GB+ VRAM recommended)
- CUDA 12.1+
- Python 3.8+
- 50GB+ disk space

## 📁 Files in This Directory

| File | Purpose |
|------|---------|
| **gpu_setup.sh** | Complete GPU environment setup script |
| **run_evaluations.sh** | Automated evaluation runner (both benchmarks) |
| **GPU_DEPLOY.md** | Complete deployment guide with troubleshooting |
| **QUICKSTART.md** | Quick start instructions and tips |
| **WANDB_SETUP.md** | Wandb integration and visualization guide |
| **sample_test.py** | CPU test script (optional, for testing) |
| **results/** | Directory for storing evaluation results |
| **visualizations/** | Directory for plots and graphs |

## 📊 Benchmarks

### HumanEval-Infill
- **Dataset:** 164 single-line code infilling problems (Bavarian et al., 2022)
- **Metric:** Pass@1 (unit test pass rate)
- **Expected:** ~32.5% (fixed length) / ~77.4% (oracle length)

### SantaCoder-FIM
- **Dataset:** Python code infilling examples (Allal et al., 2023)
- **Metrics:** Pass@1, Exact Match
- **Expected:** ~29.6% EM (fixed) / ~56.4% EM (oracle)

## 🎯 What Gets Evaluated

The evaluation automatically:
1. Downloads the `fredzzp/open-dcoder-0.5B` model from Hugging Face
2. Runs HumanEval-Infill with 164 code completion problems
3. Runs SantaCoder-FIM on Python examples
4. Computes Pass@1 and Exact Match metrics
5. Logs everything to Wandb (if configured)
6. Saves results in `../Open-dLLM/eval/eval_infill/infill_results/`

## ⚙️ Configuration

Default settings in `run_evaluations.sh`:

```bash
MODEL_PATH="fredzzp/open-dcoder-0.5B"
TEMPERATURE=0.6      # Sampling temperature
STEPS=64             # Diffusion sampling steps
ALG="p2"             # Probability-based sampling
BATCH_SIZE=32        # Batch size for inference
```

### Customize Parameters:

```bash
export TEMPERATURE=0.8
export STEPS=128
export BATCH_SIZE=16
bash run_evaluations.sh
```

## 📈 Wandb Integration

Wandb logging is **enabled by default** for experiment tracking:

**What gets logged:**
- Pass@1 and Exact Match metrics
- All configuration parameters
- Prediction files as artifacts
- Run history and system metrics

**Setup:**
```bash
wandb login  # One-time setup
# Then run evaluations - everything logs automatically
```

**Wandb Project:** `eval-infill-dllm-step64-latest`

See [WANDB_SETUP.md](WANDB_SETUP.md) for complete details on visualization and making results public.

## 📦 Results

After evaluation completes, find results in:
```
../Open-dLLM/eval/eval_infill/infill_results/
├── humaneval_infill/
│   └── open-dcoder-0.5B/
│       └── 0.6/
│           ├── humaneval_infill_results_TIMESTAMP.jsonl
│           └── humaneval_infill_results_TIMESTAMP_eval_results.json
└── santacoder-fim/
    └── open-dcoder-0.5B/
        └── 0.6/
            ├── santacoder-fim_results_TIMESTAMP.jsonl
            └── santacoder-fim_results_TIMESTAMP_eval_results.json
```

### Copy Results to Experiments Folder:

```bash
cp -r ../Open-dLLM/eval/eval_infill/infill_results results/
git add results/
git commit -m "Add evaluation results"
git push origin feature/open-dllm-experiments
```

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
export BATCH_SIZE=8
bash run_evaluations.sh
```

### Wandb Not Logging
```bash
wandb login
# Or set: export WANDB_API_KEY='your-key'
```

### Model Download Slow
```bash
export HF_HOME=/path/to/large/disk
bash run_evaluations.sh
```

See [GPU_DEPLOY.md](GPU_DEPLOY.md) for complete troubleshooting guide.

## 📚 Additional Documentation

- **[GPU_DEPLOY.md](GPU_DEPLOY.md)** - Complete deployment guide with one-liners
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start with examples
- **[WANDB_SETUP.md](WANDB_SETUP.md)** - Wandb integration details

## 🔗 References

- **Open-dLLM:** https://github.com/pengzhangzhi/Open-dLLM
- **Model:** https://huggingface.co/fredzzp/open-dcoder-0.5B
- **HumanEval:** https://github.com/openai/human-eval
- **SantaCoder:** https://huggingface.co/datasets/bigcode/santacoder-fim-task

---

**Ready to deploy!** See [GPU_DEPLOY.md](GPU_DEPLOY.md) for step-by-step instructions.
