# Quick Start Guide - Ensemble Experiments

Get started with ensemble model evaluation in 5 minutes!

## 🚀 One-Line Setup & Run (GPU Required)

```bash
cd ensemble-experiments && bash setup_env.sh && bash run_experiments.sh
```

This will:
1. ✅ Create virtual environment with all dependencies (~10-15 min)
2. ✅ Download both models (~2-3 GB)
3. ✅ Run evaluations on both benchmarks (~30-60 min)
4. ✅ Generate metrics and save results

## 📋 Step-by-Step

### Step 1: Setup Environment

```bash
cd ensemble-experiments
bash setup_env.sh
```

**Expected output:**
```
============================================================
Ensemble Model Experiments - Environment Setup
============================================================
Creating virtual environment 'ensemble-env'...
Installing PyTorch with CUDA support...
...
Setup Complete!
```

### Step 2: Activate Environment

```bash
source ensemble-env/bin/activate
```

### Step 3: (Optional) Configure Wandb

```bash
wandb login
# Or: export WANDB_API_KEY='your-key'
```

### Step 4: Run Experiments

```bash
bash run_experiments.sh
```

**What happens:**
- Downloads `fredzzp/open-dcoder-0.5B` (if not cached)
- Downloads `Qwen/Qwen2.5-Coder-0.5B` (if not cached)
- Runs HumanEval-Infill (164 examples)
- Runs SantaCoder-FIM (Python subset)
- Computes metrics and saves results

## 📊 View Results

```bash
# List results
ls -lh results/

# View latest results
cat results/ensemble_humaneval_infill_results_*.jsonl | head -5

# View metrics
cat results/ensemble_humaneval_infill_metrics.json
```

## 🎯 Quick Test (Single Task)

To test just one task quickly:

```bash
source ensemble-env/bin/activate

# HumanEval only
python ensemble_eval.py \
    --task humaneval_infill \
    --batch_size 4

# Or SantaCoder only
python ensemble_eval.py \
    --task santacoder-fim \
    --batch_size 4
```

## ⚙️ Common Configurations

### Reduce Memory Usage

```bash
# Edit run_experiments.sh
BATCH_SIZE=4  # or even 2
```

### Change Temperature

```bash
# Edit run_experiments.sh
TEMPERATURE=0.8  # Higher = more diverse, Lower = more deterministic
```

### Use Different Perplexity Model

```bash
# In ensemble_eval.py call
python ensemble_eval.py --perplexity_model "dllm"  # Use Open-dLLM for perplexity
python ensemble_eval.py --perplexity_model "qwen"  # Use Qwen (default)
```

## 🐛 Quick Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
export BATCH_SIZE=2
bash run_experiments.sh
```

### "Module not found: veomni"
```bash
cd ../Open-dLLM
pip install -e .
cd ../ensemble-experiments
```

### "evaluate_infilling_functional_correctness not found"
```bash
pip install -e ../Open-dLLM/human-eval-infilling
# Or
pip install git+https://github.com/openai/human-eval-infilling.git
```

### Models downloading slowly
```bash
# Use a mirror or set cache directory
export HF_HOME="/path/to/large/disk"
```

## 📈 Expected Results

Based on individual model performance:

| Model | HumanEval Pass@1 | SantaCoder EM |
|-------|------------------|---------------|
| Open-dLLM | ~77% (oracle) | ~56% (oracle) |
| Qwen 0.5B | ~74% | ~65% |
| **Ensemble** | **~78-80%** | **~65-70%** |

*Exact numbers depend on configuration and random seed*

## 🔍 Analyze Results

```bash
# Run metrics script
python evaluate_metrics.py results/ensemble_humaneval_infill_results_*.jsonl \
    --task humaneval_infill

# This shows:
# - Which model was selected more often
# - Average perplexities
# - Pass@1 or Exact Match rates
```

## 💡 Tips

1. **First Run**: May take longer due to model downloads
2. **Subsequent Runs**: Much faster with cached models
3. **Batch Size**: Start with 8, reduce if OOM errors
4. **Temperature**: 0.6 is a good default balance
5. **Wandb**: Highly recommended for tracking experiments

## 🎓 Next Steps

- Review the full [README.md](README.md) for detailed documentation
- Modify `ensemble_eval.py` for custom ensemble strategies
- Experiment with different model combinations
- Try different perplexity calculation methods

---

**Ready to Experiment!** 🚀

For issues, check [README.md](README.md#-troubleshooting) or the parent project docs.

