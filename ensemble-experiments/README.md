# Ensemble Model Experiments

An ensemble approach combining **Open-dLLM** (diffusion-based) and **Qwen 2.5 Coder 0.5B** (autoregressive) for code infilling tasks. The ensemble selects the output with **lower perplexity** from both models.

## 🎯 Overview

This ensemble model leverages the complementary strengths of two different model architectures:

- **Open-dLLM (fredzzp/open-dcoder-0.5B)**: Diffusion-based language model specialized for code infilling
- **Qwen 2.5 Coder 0.5B**: Autoregressive transformer with FIM (Fill-In-Middle) capability

### How It Works

1. **Dual Generation**: Both models generate completions for the same infilling task
2. **Perplexity Evaluation**: Calculate perplexity for each completion given the context
3. **Selection**: Choose the completion with **lower perplexity** as the ensemble output
4. **Evaluation**: Benchmark on HumanEval-Infill and SantaCoder-FIM datasets

## 🚀 Quick Start

### Prerequisites

- Linux with NVIDIA GPU (16GB+ VRAM recommended)
- CUDA 12.1+
- Python 3.8+
- 100GB+ disk space (for models and cache)

### Setup

```bash
cd ensemble-experiments
bash setup_env.sh
```

This will:
- Create a virtual environment (`ensemble-env`)
- Install PyTorch with CUDA support
- Install Transformers, datasets, and evaluation tools
- Configure Open-dLLM dependencies

### Run Experiments

```bash
bash run_experiments.sh
```

This will automatically:
1. Run ensemble evaluation on **HumanEval-Infill** (164 examples)
2. Run ensemble evaluation on **SantaCoder-FIM** (Python subset)
3. Compute metrics (Pass@1, Exact Match)
4. Log results to Wandb
5. Save results to `results/` directory

## 📁 Project Structure

```
ensemble-experiments/
├── ensemble_eval.py          # Main ensemble evaluation script
├── perplexity_calculator.py  # Perplexity calculation module
├── evaluate_metrics.py        # Metrics computation script
├── setup_env.sh              # Environment setup script
├── run_experiments.sh        # Automated experiment runner
├── README.md                 # This file
└── results/                  # Output directory for results
```

## 🔧 Configuration

### Default Settings

```bash
DLLM_MODEL="fredzzp/open-dcoder-0.5B"
QWEN_MODEL="Qwen/Qwen2.5-Coder-0.5B"
TEMPERATURE=0.6
DLLM_STEPS=64              # Diffusion sampling steps
MAX_NEW_TOKENS=512         # Max tokens for Qwen
BATCH_SIZE=8
```

### Custom Configuration

You can customize parameters by editing `run_experiments.sh` or running `ensemble_eval.py` directly:

```bash
python ensemble_eval.py \
    --dllm_model_path "fredzzp/open-dcoder-0.5B" \
    --qwen_model_path "Qwen/Qwen2.5-Coder-0.5B" \
    --task "humaneval_infill" \
    --temperature 0.8 \
    --dllm_steps 128 \
    --batch_size 4 \
    --perplexity_model "qwen"
```

## 📊 Evaluation Metrics

### HumanEval-Infill
- **Pass@1**: Percentage of completions that pass unit tests
- **Pass@10/100**: Pass rate with multiple samples per task

### SantaCoder-FIM
- **Exact Match**: Percentage of completions exactly matching the canonical solution

### Ensemble Statistics
- **Model Selection Rate**: How often each model is selected
- **Average Perplexity**: Average perplexity for each model's outputs
- **Perplexity Correlation**: Analysis of perplexity vs. correctness

## 📈 Wandb Integration

Results are automatically logged to Wandb for tracking and visualization:

**Setup Wandb:**
```bash
wandb login
# Or export WANDB_API_KEY='your-key'
```

**What gets logged:**
- Configuration parameters
- Model selection statistics
- Sample predictions with perplexities
- Evaluation metrics (Pass@1, Exact Match)

## 🎓 Architecture Details

### Perplexity Calculation

Perplexity is calculated as:

```
PPL = exp(average_loss)
```

Where `average_loss` is the cross-entropy loss of predicting the completion tokens given the full context (prefix + completion + suffix).

**Lower perplexity** indicates:
- Higher likelihood under the language model
- Better fit with the surrounding context
- More "natural" or "fluent" code

### Model Loading

- **Open-dLLM**: Uses custom Qwen2 implementation with diffusion sampling
- **Qwen 2.5 Coder**: Standard Transformers AutoModelForCausalLM with FIM tokens

### Generation Process

**Open-dLLM:**
1. Tokenize prefix and suffix
2. Insert mask tokens in the middle (oracle length from ground truth)
3. Run diffusion sampling to denoise masks
4. Extract and decode the middle tokens

**Qwen 2.5 Coder:**
1. Format input with FIM tokens: `<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>`
2. Generate autoregressively with temperature sampling
3. Extract completion after `<|fim_middle|>` token

## 📦 Results Format

Results are saved as JSONL files with the following structure:

```json
{
  "task_id": "HumanEval/0",
  "completion": "selected_code",
  "dllm_output": "dllm_generated_code",
  "qwen_output": "qwen_generated_code",
  "dllm_perplexity": 12.34,
  "qwen_perplexity": 15.67,
  "winner": "dllm",
  "prefix": "def example():\n    ",
  "suffix": "\n    return result",
  "canonical_solution": "result = x + y"
}
```

## 🔍 Analysis Tools

### View Results

```bash
# List all results
ls -lh results/

# View ensemble statistics
cat results/ensemble_*_stats_*.json

# View metrics
cat results/ensemble_*_metrics.json
```

### Compare with Baselines

The `evaluate_metrics.py` script automatically compares ensemble performance with individual model statistics:

```bash
python evaluate_metrics.py results/ensemble_humaneval_infill_results_*.jsonl --task humaneval_infill
```

## 🐛 Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
# Edit run_experiments.sh
BATCH_SIZE=4  # or 2
```

### Model Download Issues

Set custom cache directory:
```bash
export HF_HOME="/path/to/large/disk/huggingface"
```

### Wandb Not Logging

```bash
wandb login
# Or disable wandb in ensemble_eval.py
```

### Import Errors

Make sure Open-dLLM is properly installed:
```bash
cd ../Open-dLLM
pip install -e .
cd ../ensemble-experiments
```

## 📚 References

- **Open-dLLM**: [GitHub](https://github.com/pengzhangzhi/Open-dLLM) | [Model](https://huggingface.co/fredzzp/open-dcoder-0.5B)
- **Qwen 2.5 Coder**: [Model](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)
- **HumanEval**: [Paper](https://arxiv.org/abs/2107.03374) | [GitHub](https://github.com/openai/human-eval-infilling)
- **SantaCoder**: [Paper](https://arxiv.org/abs/2301.03988) | [Dataset](https://huggingface.co/datasets/bigcode/santacoder-fim-task)

## 🤝 Related Experiments

- **open-dllm-experiments/**: Evaluation of Open-dLLM model alone
- **small-model-experiments/**: Comparison of multiple small autoregressive models

## 💡 Key Insights

### Why Perplexity-Based Ensemble?

1. **Complementary Strengths**: Diffusion and autoregressive models have different generation strategies
2. **Context Awareness**: Perplexity measures how well the completion fits the surrounding code
3. **No Training Required**: Pure inference-time ensemble without additional training
4. **Interpretable**: Perplexity provides an interpretable selection criterion

### Expected Behavior

- **Open-dLLM** may excel at structured, deterministic code patterns
- **Qwen** may perform better on complex, context-dependent completions
- **Ensemble** should achieve performance at least as good as the better model, with potential for improvement

## 🔬 Future Improvements

Potential enhancements to explore:

1. **Multi-Model Perplexity**: Use both models for perplexity calculation
2. **Learned Ensemble**: Train a small classifier to select based on features
3. **Weighted Combination**: Blend outputs instead of hard selection
4. **Calibration**: Normalize perplexity scores across models
5. **Task-Specific Selection**: Use different strategies per task type

---

**Status**: ✅ Ready for Experimentation

For questions or issues, please refer to the parent project documentation.

