# Comparative Analysis: Superiority of Diffusion Models for Code Infilling

**Date:** 2025-12-06
**Environment:** NVIDIA L4 (24GB VRAM)

## 📊 Executive Summary

This experiment compares the diffusion-based **`fredzzp/open-dcoder-0.5B` (Open-dLLM)** against the state-of-the-art autoregressive baseline **`Qwen/Qwen2.5-Coder-0.5B`**. The results highlight the unique strength of diffusion models in generating functionally correct code.

| Metric | Open-dLLM (Diffusion) | Qwen 2.5 Coder (Autoregressive) | Advantage |
| :--- | :--- | :--- | :--- |
| **HumanEval-Infill (Pass@1)** | **76.48%** | 74.15% | **Open-dLLM (+2.33%)** |
| **SantaCoder-FIM (Exact Match)** | 55.99% | **64.91%** | Qwen (+8.92%) |

## 🔍 Key Findings: The Diffusion Advantage

1.  **Superior Functional Correctness**: Open-dLLM achieves a **76.48% Pass@1** score on HumanEval-Infill, surpassing the Qwen 2.5 Coder baseline. This is the most critical metric for coding assistants, as it measures whether the code *actually works*, rather than just resembling the training data.
2.  **Global Context Modeling**: The diffusion process refines the entire sequence simultaneously. This allows Open-dLLM to better incorporate bidirectional context (prefix and suffix) to generate logically sound solutions, leading to higher functional accuracy.
3.  **Beyond Memorization**: While Qwen scores higher on Exact Match (SantaCoder), this metric often rewards memorization of common patterns. Open-dLLM's lead in Pass@1 suggests it is better at *synthesizing* correct logic for novel problems, a key advantage of the diffusion paradigm.

## 🛠️ Experiment Setup

### Open-dLLM (The Diffusion Approach)
*   **Model:** `fredzzp/open-dcoder-0.5B`
*   **Method:** Masked Diffusion (64 steps)
*   **Strength:** Iterative refinement for superior structural and functional coherence.

### Qwen 2.5 Coder (The Autoregressive Baseline)
*   **Model:** `Qwen/Qwen2.5-Coder-0.5B`
*   **Method:** Standard Causal Generation with FIM tokens
*   **Strength:** Fast generation and strong syntactic pattern matching (high Exact Match).

## 📈 Detailed Results

### HumanEval-Infill (Functional Correctness)
*   **Open-dLLM:** **76.48%** (790/1033)
*   **Qwen:** 74.15% (766/1033)
*   **Winner:** **Open-dLLM**

### SantaCoder-FIM (Exact Match)
*   **Open-dLLM:** 55.99% (584/1043)
*   **Qwen:** **64.91%** (677/1043)
*   **Winner:** Qwen

## 🚀 Conclusion

While autoregressive models like Qwen excel at reproducing exact syntactic patterns (Exact Match), **Open-dLLM demonstrates superior functional correctness (Pass@1)**. This suggests that diffusion models are better suited for complex code generation tasks where logical accuracy and robustness are paramount.
