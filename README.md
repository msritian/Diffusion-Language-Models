# Diffusion Language Models Project

This repository contains experiments comparing Diffusion-based Language Models (Open-dLLM) against traditional Autoregressive (AR) models for code generation and infilling tasks.

## 📂 Project Structure

### 1. `open-dllm-experiments/`
Contains the original experiments and results for the diffusion model **Open-dLLM** (`fredzzp/open-dcoder-0.5B`).

### 2. `small-model-experiments/`
Contains the benchmark suite for comparable small-scale autoregressive models (0.5B - 3B parameters).

**Models Evaluated:**
*   **Qwen 2.5 Coder** (0.5B & 1.5B)
*   **DeepSeek Coder** (1.3B)
*   **StarCoder2** (3B)

**Benchmarks:**
*   **SantaCoder-FIM**: Measures Exact Match (EM) for code infilling.
*   **HumanEval-Infill**: Measures Functional Correctness (Pass@1).

## 📊 Comparison Report
For a detailed analysis of the results, please refer to:
[Comparative Analysis Report](small-model-experiments/comparison_report.md)

## 🚀 Running Experiments

Navigate to the respective directory and follow the `README.md` instructions.

```bash
# For AR baselines
cd small-model-experiments
bash setup_env.sh
bash run_benchmarks.sh
```
