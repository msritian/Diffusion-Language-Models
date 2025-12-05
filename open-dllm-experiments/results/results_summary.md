# Open-dLLM Experiment Results

**Model:** fredzzp/open-dcoder-0.5B  
**Experiment Date:** 2025-12-04  
**Environment:** macOS (ARM64) - CPU only

## Setup Summary

Successfully cloned and configured the Open-dLLM repository for diffusion language model experimentation. The model `fredzzp/open-dcoder-0.5B` was successfully loaded and verified on CPU.

### Environment Details

- **Python:** 3.13.5
- **PyTorch:** 2.9.1 (CPU-only on macOS)
- **Transformers:** 4.54.1
- **Device:** CPU (no CUDA support on macOS)

### Installed Components

✅ **Successfully Installed:**
- Transformers, Accelerate, Datasets, PEFT
- Evaluation harness: `lm-evaluation-harness`
- Infilling benchmark: `human-eval-infilling`
- Open-dLLM core package (`veomni`)
- Model downloaded: ~1.26GB

⚠️ **Skipped (macOS incompatible):**
- flash-attn (requires CUDA)
- liger-kernel (requires triton/CUDA)
- bytecheckpoint (PyTorch version conflict)

## Evaluation Benchmarks (Pending GPU Access)

The following evaluations are configured but require GPU access to run efficiently:

### 1. HumanEval-Infill
- **Benchmark:** Fill-in-the-Middle (FIM) code generation
- **Dataset:** 164 problems from Bavarian et al., 2022
- **Metric:** Pass@1 (percentage passing unit tests on first try)
- **Expected Performance:** ~32.5% (fixed length), ~77.4% (oracle length)

### 2. SantaCoder-FIM
- **Benchmark:** Code infilling with exact match evaluation
- **Dataset:** Python examples from bigcode/santacoder-fim-task
- **Metrics:** Pass@1, Exact Match
- **Expected Performance:** ~29.6% EM (fixed), ~56.4% EM (oracle)

### Evaluation Configuration

```bash
MODEL_PATH="fredzzp/open-dcoder-0.5B"
TEMPERATURE=0.6
DIFFUSION_STEPS=64
SAMPLING_ALGORITHM="p2"
BATCH_SIZE=32
```

## Current Status

✅ **Completed:**
1. Repository cloned and set up
2. Dependencies installed (CPU-compatible subset)
3. Model successfully loaded and verified
4. Evaluation scripts reviewed and understood
5. Helper scripts created for GPU execution

⚠️ **Requires GPU:**
- Full HumanEval-Infill evaluation
- Full SantaCoder-FIM evaluation

## Next Steps

To complete the experimentation:

### Option 1: Transfer to GPU Instance

1. Commit the current setup to the feature branch
2. Clone on a machine with CUDA (AWS, GCP, Colab, etc.)
3. Run the evaluation script:
   ```bash
   bash run_evaluations.sh
   ```
4. Results will be automatically saved in `infill_results/`

### Option 2: Use Cloud Environment

**Google Colab (Free GPU):**
```python
# In Colab notebook
!git clone <your-repo-url>
!cd <repo>/Open-dLLM && bash run_evaluations.sh
```

**Results will include:**
- JSONL files with all model predictions
- JSON files with Pass@1 and Exact Match metrics
- Optional wandb logs for experiment tracking

## Model Verification

The model was successfully tested with a sample prompt on CPU:

**Prompt:**
```python
def quick_sort(arr):
```

**Status:** Model loaded successfully (1.26GB), tokenizer configured, generation config validated.

## Repository Structure

```
Open-dLLM/
├── README_EXPERIMENTS.md      # Detailed setup and usage guide
├── run_evaluations.sh         # GPU evaluation automation script
├── sample_test.py             # CPU-compatible test script
├── experiments/
│   ├── results_summary.md     # This file
│   └── visualizations/        # For plots (pending GPU results)
└── eval/
    └── eval_infill/
        ├── eval_infill.py     # Main evaluation script
        └── run_eval.sh        # Original evaluation script
```

## References

- [Open-dLLM GitHub](https://github.com/pengzhangzhi/Open-dLLM)
- [Model on Hugging Face](https://huggingface.co/fredzzp/open-dcoder-0.5B)
- [Open-dLLM Blog Post](https://oval-shell-31c.notion.site/Open-Diffusion-Large-Language-Model-25e03bf6136480b7a4ebe3d53be9f68a)

---

*This document will be updated with full evaluation results once GPU access is available.*
