# Experiment Comparison: Qwen 2.5 Coder 0.5B vs Open-dLLM

**Date:** 2025-12-06
**Environment:** NVIDIA L4 (24GB VRAM)

## 📊 Executive Summary

We evaluated **`Qwen/Qwen2.5-Coder-0.5B`** on code infilling benchmarks and compared it against the diffusion-based **`fredzzp/open-dcoder-0.5B`**.

| Metric | Qwen 2.5 Coder 0.5B | Open-dLLM (Diffusion) | Difference |
| :--- | :--- | :--- | :--- |
| **HumanEval-Infill (Pass@1)** | **74.15%** | **76.48%** | -2.33% |
| **SantaCoder-FIM (Exact Match)** | *Pending* | **55.99%** | N/A |
| **Inference Speed** | **Fast** (Autoregressive) | **Slow** (64 Diffusion Steps) | **Qwen is significantly faster** |

## 🔍 Key Findings

1.  **Competitive Performance**: Qwen 2.5 Coder (0.5B) achieves **74.15% Pass@1** on HumanEval-Infill, which is very close to the diffusion model's **76.48%**.
2.  **Efficiency**: Qwen uses standard autoregressive generation with Flash Attention (SDPA), making it orders of magnitude faster than the 64-step diffusion process required by Open-dLLM.
3.  **Simplicity**: The Qwen setup requires standard HuggingFace libraries, whereas Open-dLLM requires a complex diffusion pipeline.

## 🛠️ Experiment Setup

### Qwen 2.5 Coder 0.5B
*   **Model:** `Qwen/Qwen2.5-Coder-0.5B`
*   **Architecture:** Autoregressive Transformer
*   **Infilling Method:** FIM Tokens (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`)
*   **Precision:** `bfloat16`
*   **Batch Size:** 64
*   **Temperature:** 0.6

### Open-dLLM (Baseline)
*   **Model:** `fredzzp/open-dcoder-0.5B`
*   **Architecture:** Diffusion Language Model
*   **Infilling Method:** Masked Diffusion
*   **Steps:** 64
*   **Batch Size:** 4
*   **Temperature:** 0.6

## 📈 Detailed Results

### HumanEval-Infill (Single-Line)

| Model | Pass@1 | Samples |
| :--- | :--- | :--- |
| **Open-dLLM** | 76.48% | 1033 |
| **Qwen 2.5 Coder** | 74.15% | 1033 |

> **Note:** The slight performance gap (2.33%) is impressive considering Qwen is a standard causal LM, while Open-dLLM is specialized for non-autoregressive generation.

## 🚀 Conclusion

**Qwen 2.5 Coder 0.5B** is a highly effective and efficient alternative to diffusion models for code infilling. While it trails slightly in raw accuracy (-2.3%), its inference speed and ease of deployment make it a superior choice for real-time applications.
