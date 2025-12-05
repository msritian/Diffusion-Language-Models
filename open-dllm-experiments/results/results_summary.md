# Open-dLLM Experiment Results

**Model:** fredzzp/open-dcoder-0.5B  
**Experiment Date:** 2025-12-05  
**Environment:** GPU VM (NVIDIA L4, 24GB VRAM)

## ✅ Evaluation Results

### SantaCoder-FIM Benchmark

**Configuration:**
- **Model:** fredzzp/open-dcoder-0.5B (500M parameters)
- **Temperature:** 0.6
- **Diffusion Steps:** 64
- **Sampling Algorithm:** p2 (probability-based)
- **Batch Size:** 4 (adjusted for L4 GPU memory)
- **Dataset:** bigcode/santacoder-fim-task (Python examples)

**Results:**
- **Exact Match:** 55.99% (584 out of 1043 examples)
- **Example Count:** 1043
- **Expected Performance:** ~56.4% (oracle length)
- **Status:** ✅ **Meets expectations!**

**Wandb Tracking:**
- **Project:** [eval-infill-dllm-step64-latest](https://wandb.ai/mittalshivam003-iron-mountain/eval-infill-dllm-step64-latest)
- **Run:** santacoder-fim_final
- **Metrics Logged:** Exact Match, configuration parameters
- **Files:** Predictions JSONL, evaluation JSON

---

## Benchmark Details

### SantaCoder-FIM
- **Purpose:** Code infilling with exact match evaluation
- **Metric:** Percentage of predictions exactly matching ground truth
- **Dataset:** Python code completion tasks
- **Evaluation Method:** Direct string comparison after whitespace normalization

---

## Performance Analysis

### Comparison with Expected Results

| Metric | Our Result | Expected (Paper) | Status |
|--------|------------|------------------|--------|
| Exact Match | **55.99%** | ~56.4% | ✅ On par |
| Dataset Size | 1043 examples | Standard | ✅ Complete |

**Key Observations:**
1. **Strong Performance:** Achieved 55.99% exact match, very close to the oracle length performance reported in the paper
2. **Reliable Sampling:** The p2 algorithm with temperature 0.6 produces consistent, high-quality completions
3. **GPU Efficiency:** Successfully ran on single NVIDIA L4 GPU with batch size 4

---

## Setup Summary

### Environment Details

**GPU VM:**
- **GPU:** NVIDIA L4 (24GB VRAM)
- **CUDA:** 12.2
- **Python:** 3.10.19
- **PyTorch:** 2.5.0+cu121

### Installed Components

✅ **Core Dependencies:**
- PyTorch 2.5.0 with CUDA 12.1
- Transformers 4.54.1
- Accelerate, Datasets, PEFT
- Wandb for experiment tracking

✅ **Evaluation Packages:**
- lm-evaluation-harness 0.4.9.1
- human-eval-infilling 2.15.0
- Open-dLLM core package (veomni)

⚠️ **Skipped (not required for current evaluation):**
- flash-attn (compilation issues, model runs without it)

---

## Evaluation Configuration

```bash
# Model and Task
MODEL_PATH="fredzzp/open-dcoder-0.5B"
TASK="santacoder-fim"

# Sampling Parameters
TEMPERATURE=0.6
DIFFUSION_STEPS=64
SAMPLING_ALGORITHM="p2"
TOP_P=0.95

# System Parameters
BATCH_SIZE=4
NUM_GPUS=1
```

---

## File Locations

### Results Files

All evaluation results are saved in:
```
infill_results/santacoder-fim/open-dcoder-0.5B/0.6/
```

**Files:**
1. `santacoder-fim_results_20251205_053913.jsonl` - All model predictions
2. `santacoder-fim_results_20251205_053913_eval_results.json` - Metrics and configuration

### Results Structure

```json
{
  "config": {
    "model_path": "fredzzp/open-dcoder-0.5B",
    "task": "santacoder-fim",
    "temperature": 0.6,
    "steps": 64,
    "alg": "p2",
    "batch_size": 4
  },
  "results": {
    "exact_match": 0.5599232981783318,
    "count": 1043
  }
}
```

---

## Next Steps

### Pending Evaluations

#### HumanEval-Infill
- **Status:** ⏳ Not yet run
- **Dataset:** 164 single-line infilling problems
- **Expected:** ~32.5% Pass@1 (fixed), ~77.4% Pass@1 (oracle)
- **Data Issue:** Benchmark data file not found in repository
- **Resolution Needed:** Obtain HumanEval-Infill.jsonl.gz dataset

### Future Work

1. **Complete HumanEval-Infill:** Acquire and run the benchmark
2. **Parameter Sweep:** Test different temperatures and step counts
3. **Visualization:** Create performance comparison charts
4. **Analysis:** Detailed error analysis on failed predictions

---

## References

- **Open-dLLM Repository:** https://github.com/pengzhangzhi/Open-dLLM
- **Model:** https://huggingface.co/fredzzp/open-dcoder-0.5B
- **SantaCoder Dataset:** https://huggingface.co/datasets/bigcode/santacoder-fim-task
- **Wandb Project:** https://wandb.ai/mittalshivam003-iron-mountain/eval-infill-dllm-step64-latest

---

## Summary

✅ **Successfully evaluated fredzzp/open-dcoder-0.5B on SantaCoder-FIM**
- Achieved 55.99% exact match on 1043 examples
- Results match expected performance from paper
- All metrics logged to Wandb for public viewing
- Evaluation completed on GPU VM in ~1.5 hours

**Status:** Phase 1 complete. Ready for HumanEval-Infill evaluation.

---

*Last Updated: 2025-12-05*
