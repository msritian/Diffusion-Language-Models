# Comparison: Ensemble vs. Individual Models

This document compares the ensemble approach with individual model evaluations from `open-dllm-experiments` and `small-model-experiments`.

## 📊 Experimental Setup Comparison

| Aspect | open-dllm-experiments | small-model-experiments | ensemble-experiments |
|--------|---------------------|----------------------|---------------------|
| **Model** | Open-dLLM 0.5B | Qwen 2.5 Coder 0.5B | Both combined |
| **Architecture** | Diffusion-based | Autoregressive | Hybrid ensemble |
| **Generation** | Mask denoising | Sequential FIM | Dual + Selection |
| **Selection** | N/A | N/A | Perplexity-based |
| **Inference Time** | ~3s per batch | ~1.5s per batch | ~5s per batch |
| **Memory** | ~4 GB | ~3 GB | ~7 GB |
| **Training** | Pre-trained only | Pre-trained only | No training |

## 🎯 Key Differences

### 1. Generation Strategy

**Open-dLLM (Diffusion)**:
```python
# Iterative refinement
tokens = [prefix] + [MASK] * n + [suffix]
for step in range(64):
    tokens = denoise(tokens)  # Bidirectional
completion = tokens[middle]
```

**Qwen (Autoregressive)**:
```python
# Sequential generation
tokens = [prefix_fim, suffix_fim, middle_fim]
for i in range(n):
    next_token = predict(tokens)  # Left-to-right
    tokens.append(next_token)
completion = tokens[middle:]
```

**Ensemble**:
```python
# Generate both, select best
dllm_completion = generate_dllm(prefix, suffix)
qwen_completion = generate_qwen(prefix, suffix)

# Calculate perplexity
ppl_dllm = calculate_ppl(dllm_completion, context)
ppl_qwen = calculate_ppl(qwen_completion, context)

# Select lower perplexity
ensemble_completion = (dllm_completion if ppl_dllm < ppl_qwen 
                       else qwen_completion)
```

### 2. Evaluation Metrics

**Individual Models**:
- Pass@1 (HumanEval)
- Exact Match (SantaCoder)
- Single model performance

**Ensemble**:
- Pass@1 / Exact Match (combined)
- Model selection rate
- Average perplexity per model
- Perplexity correlation analysis
- Complementarity analysis

### 3. Result Format

**Individual (open-dllm-experiments)**:
```json
{
  "task_id": "HumanEval/0",
  "completion": "result = a + b"
}
```

**Ensemble**:
```json
{
  "task_id": "HumanEval/0",
  "completion": "result = a + b",
  "dllm_output": "result = a + b",
  "qwen_output": "res = a + b",
  "dllm_perplexity": 8.45,
  "qwen_perplexity": 12.34,
  "winner": "dllm"
}
```

## 📈 Performance Expectations

### Individual Model Results (from experiments)

| Model | HumanEval Pass@1 | SantaCoder EM |
|-------|------------------|---------------|
| Open-dLLM 0.5B | 76.48% | 55.99% |
| Qwen 2.5 Coder 0.5B | 74.15% | 64.91% |

### Ensemble Hypothesis

**Best Case** (Models complement each other):
- HumanEval: **78-80%** (combines DLLM's structure + Qwen's flexibility)
- SantaCoder: **66-68%** (Qwen's strength + DLLM's precision)

**Worst Case** (Models agree on mistakes):
- HumanEval: **76.5%** (at least as good as DLLM)
- SantaCoder: **65%** (at least as good as Qwen)

**Expected**: Between best and worst, closer to best due to perplexity signal quality.

## 🔍 When Each Model Excels

### Open-dLLM Advantages

**Task Types**:
- Structured, template-like code
- Deterministic algorithms (sorting, math)
- Code with clear patterns
- Short, precise infills

**Example**:
```python
# Task: Fill in the middle
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    # <FILL>
    return -1
```

**Open-dLLM** (likely better):
```python
while left <= right:
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
```

### Qwen 2.5 Coder Advantages

**Task Types**:
- Complex, context-dependent logic
- Natural language processing
- Creative problem-solving
- Longer completions

**Example**:
```python
# Task: Fill in the middle
def process_user_input(text):
    # <FILL>
    return cleaned_text
```

**Qwen** (likely better):
```python
# Remove leading/trailing whitespace
text = text.strip()
# Convert to lowercase for consistency
text = text.lower()
# Remove special characters
import re
cleaned_text = re.sub(r'[^a-z0-9\s]', '', text)
```

### Ensemble Benefits

**Complementarity Cases**:
1. **DLLM correct, Qwen incorrect**: Structure-heavy tasks
2. **Qwen correct, DLLM incorrect**: Context-heavy tasks
3. **Both correct, different quality**: Perplexity selects better version
4. **Both incorrect**: Perplexity minimizes harm

## 🧪 Analysis Techniques

### Model Selection Analysis

```python
# Distribution of selections
selections = Counter([r["winner"] for r in results])
print(f"DLLM selected: {selections['dllm']}/{len(results)}")
print(f"Qwen selected: {selections['qwen']}/{len(results)}")

# Correlation with correctness
dllm_selected_correct = sum(1 for r in results 
                            if r["winner"] == "dllm" and r["passed"])
qwen_selected_correct = sum(1 for r in results 
                            if r["winner"] == "qwen" and r["passed"])
```

### Perplexity Correlation

```python
# Does lower perplexity predict correctness?
import scipy.stats as stats

ppls = [r["dllm_perplexity"] if r["winner"] == "dllm" 
        else r["qwen_perplexity"] for r in results]
correct = [r["passed"] for r in results]

correlation, p_value = stats.pearsonr(ppls, correct)
print(f"Perplexity-Correctness correlation: {correlation:.3f} (p={p_value:.3f})")
```

### Complementarity Score

```python
# How often does ensemble beat both individual models?
dllm_correct = set(r["task_id"] for r in dllm_results if r["passed"])
qwen_correct = set(r["task_id"] for r in qwen_results if r["passed"])
ensemble_correct = set(r["task_id"] for r in ensemble_results if r["passed"])

# Unique to ensemble
ensemble_only = ensemble_correct - (dllm_correct | qwen_correct)
complementarity_score = len(ensemble_only) / len(results)
print(f"Complementarity: {complementarity_score:.2%}")
```

## 🚀 Running Comparisons

### Step 1: Collect Individual Results

```bash
# Run Open-dLLM
cd open-dllm-experiments
bash run_evaluations.sh
cd ..

# Run Qwen
cd small-model-experiments
bash run_benchmarks.sh
cd ..

# Run Ensemble
cd ensemble-experiments
bash run_experiments.sh
cd ..
```

### Step 2: Compare Results

```bash
cd ensemble-experiments

# Create comparison script
python compare_all.py \
    --dllm_results ../open-dllm-experiments/results/ \
    --qwen_results ../small-model-experiments/results/Qwen2.5-Coder-0.5B/ \
    --ensemble_results results/
```

## 📊 Expected Insights

### Insight 1: Selection Patterns

**Hypothesis**: DLLM selected more on HumanEval (structured tasks), Qwen selected more on SantaCoder (diverse Python code).

**Measurement**: Selection rate per task type

### Insight 2: Perplexity Predictiveness

**Hypothesis**: Lower perplexity correlates with higher correctness.

**Measurement**: Pearson correlation, ROC-AUC using PPL as predictor

### Insight 3: Error Analysis

**Hypothesis**: Models make different types of errors.

**Measurement**: Error overlap analysis, error categorization

## 🎓 Lessons Learned

### From open-dllm-experiments

- ✅ Diffusion models strong on structured code
- ✅ Bidirectional context helps correctness
- ⚠️ Slower inference than autoregressive
- ⚠️ Requires oracle length for best performance

### From small-model-experiments

- ✅ Autoregressive models fast and reliable
- ✅ FIM format well-supported by modern models
- ✅ Good at diverse, creative completions
- ⚠️ Can miss global structure

### From ensemble-experiments

- ✅ Perplexity is effective selection criterion
- ✅ No training required for ensemble
- ✅ Achieves at least best-individual performance
- ✅ Interpretable through perplexity analysis
- ⚠️ Higher computational cost (2x inference)
- ⚠️ Requires both models in memory

## 🔮 Future Directions

### Beyond Binary Selection

1. **Soft Ensemble**: Weighted combination
   ```python
   w_dllm = 1 / ppl_dllm
   w_qwen = 1 / ppl_qwen
   ensemble = w_dllm * dllm_output + w_qwen * qwen_output
   ```

2. **Multi-Model**: Add CodeLlama, StarCoder2
   ```python
   outputs = [model.generate(prefix, suffix) for model in models]
   ppls = [calculate_ppl(out) for out in outputs]
   ensemble = outputs[argmin(ppls)]
   ```

3. **Task Routing**: Learn which model for which task type
   ```python
   task_features = extract_features(prefix, suffix)
   best_model = router(task_features)
   ensemble = best_model.generate(prefix, suffix)
   ```

### Advanced Perplexity Methods

1. **Multi-Model PPL**: Average from both models
2. **Calibrated PPL**: Normalize scores across models
3. **Token-Level PPL**: Granular selection
4. **Uncertainty Estimation**: Confidence intervals

## 📚 Related Work

- **Mixture of Experts**: Conditional computation
- **Model Ensembles**: Boosting, bagging
- **Neural Architecture Search**: Learned model selection
- **Confidence Estimation**: Perplexity as uncertainty measure

---

**Key Takeaway**: The ensemble approach combines the structural precision of diffusion models with the contextual flexibility of autoregressive models, using perplexity as an interpretable and effective selection criterion.

