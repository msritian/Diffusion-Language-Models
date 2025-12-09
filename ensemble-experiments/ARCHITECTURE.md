# Ensemble Architecture Documentation

Technical documentation for the perplexity-based ensemble model combining Open-dLLM and Qwen 2.5 Coder.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Code Infilling Task               │
│                  (prefix, suffix, ground_truth)             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────┴──────────┐
                │                      │
                ▼                      ▼
    ┌─────────────────────┐  ┌──────────────────────┐
    │  Open-dLLM Model    │  │  Qwen 2.5 Coder 0.5B │
    │  (Diffusion-based)  │  │  (Autoregressive)    │
    └──────────┬──────────┘  └──────────┬───────────┘
               │                         │
               │ Generate                │ Generate
               │                         │
               ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │ DLLM Completion  │      │ Qwen Completion  │
    └──────────┬───────┘      └──────────┬───────┘
               │                         │
               └──────────┬──────────────┘
                          │
                          ▼
            ┌────────────────────────────┐
            │  Perplexity Calculator     │
            │  (Context-Aware Scoring)   │
            └─────────────┬──────────────┘
                          │
              ┌───────────┴──────────┐
              │                      │
              ▼                      ▼
    ┌──────────────────┐  ┌──────────────────┐
    │  DLLM Perplexity │  │  Qwen Perplexity │
    │      12.34       │  │      15.67       │
    └──────────┬───────┘  └──────────┬───────┘
               │                      │
               └──────────┬───────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  Compare & Select│
                │  (Arg Min PPL)   │
                └─────────┬────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │   Ensemble Completion    │
            │   (Lower PPL Output)     │
            └──────────────────────────┘
```

## 🔍 Component Details

### 1. Open-dLLM Generator (`OpenDLLMGenerator`)

**Model**: `fredzzp/open-dcoder-0.5B`

**Architecture**:
- Based on Qwen2 architecture with diffusion modifications
- Uses masked diffusion modeling (MDM) for generation
- Bidirectional context modeling

**Generation Process**:
```python
# Input: prefix, suffix, middle_length
tokens = [prefix_tokens] + [MASK] * middle_length + [suffix_tokens]

# Diffusion sampling (multiple steps)
for step in range(num_steps):
    # Predict logits for masked positions
    logits = model(tokens)
    
    # Sample tokens with temperature and algorithm-specific noise
    tokens = sample_with_temperature(logits, temperature, alg="p2")
    
    # Gradually unmask tokens (schedule-based)
    mask_ratio = get_mask_ratio(step, num_steps)
    tokens = update_masks(tokens, mask_ratio)

# Extract middle tokens
completion = tokens[len(prefix):len(prefix)+middle_length]
```

**Key Parameters**:
- `steps`: Number of diffusion iterations (default: 64)
- `temperature`: Sampling temperature (default: 0.6)
- `alg`: Sampling algorithm ("p2" = probability-based)
- `alg_temp`: Algorithm-specific temperature for Gumbel noise

**Advantages**:
- Global context awareness (sees both prefix and suffix simultaneously)
- Can revise predictions iteratively
- Good for structured, deterministic patterns

### 2. Qwen FIM Generator (`QwenFIMGenerator`)

**Model**: `Qwen/Qwen2.5-Coder-0.5B`

**Architecture**:
- Standard transformer decoder with FIM support
- Autoregressive generation with special tokens

**Generation Process**:
```python
# Format input with FIM tokens
prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

# Tokenize and generate
input_ids = tokenizer.encode(prompt)

# Autoregressive sampling
for _ in range(max_new_tokens):
    logits = model(input_ids)
    next_token = sample_with_temperature(logits[-1], temperature)
    
    if next_token == EOS:
        break
    
    input_ids.append(next_token)

# Extract completion (after <|fim_middle|>)
completion = decode(input_ids[fim_middle_idx:])
```

**Key Parameters**:
- `max_new_tokens`: Maximum tokens to generate (default: 512)
- `temperature`: Sampling temperature (default: 0.6)
- `do_sample`: Whether to use sampling vs greedy

**Advantages**:
- Fast inference (single forward pass per token)
- Strong on context-dependent completions
- Well-calibrated probabilities

### 3. Perplexity Calculator (`PerplexityCalculator`)

**Core Concept**: Measure how "surprised" the model is by the completion given the full context.

**Calculation**:
```python
# Construct full sequence
full_text = prefix + completion + suffix
tokens = tokenizer.encode(full_text)

# Get model logits
logits = model(tokens)

# Calculate cross-entropy loss for completion region
loss = cross_entropy(logits[completion_start:completion_end], 
                     tokens[completion_start+1:completion_end+1])

# Convert to perplexity
perplexity = exp(loss)
```

**Mathematical Definition**:

For a completion \( C = (c_1, c_2, ..., c_n) \) given context \( X \):

\[
\text{PPL}(C|X) = \exp\left(-\frac{1}{n}\sum_{i=1}^{n} \log P(c_i|c_{<i}, X)\right)
\]

Where:
- \( P(c_i|c_{<i}, X) \) is the model's predicted probability
- Lower PPL = Higher likelihood = Better fit

**Why Perplexity Works**:

1. **Context Awareness**: Considers full sequence (prefix + completion + suffix)
2. **Probability Calibration**: Uses the model's confidence in its predictions
3. **Fluency Measure**: Low perplexity indicates natural, coherent code
4. **Language Model Expertise**: Leverages what LMs are trained to do (predict tokens)

**Edge Cases**:
- Empty completions: Fallback to sequence-level perplexity
- Token misalignment: Use best-effort span extraction
- Out-of-vocabulary: Handled by subword tokenization

### 4. Ensemble Selection

**Algorithm**:
```python
def select_ensemble_output(dllm_output, qwen_output, prefix, suffix):
    # Calculate perplexity for both
    ppl_dllm = calculate_perplexity(prefix, dllm_output, suffix)
    ppl_qwen = calculate_perplexity(prefix, qwen_output, suffix)
    
    # Select output with lower perplexity
    if ppl_dllm < ppl_qwen:
        return dllm_output, "dllm"
    else:
        return qwen_output, "qwen"
```

**Selection Criteria**:
- **Deterministic**: Always selects lower perplexity (no randomness)
- **Per-Example**: Independent decision for each task
- **Model-Agnostic**: Works with any perplexity calculator

**Alternative Strategies** (future work):
1. **Weighted Combination**: Blend outputs proportional to inverse perplexity
2. **Threshold-Based**: Only use ensemble if perplexity difference > threshold
3. **Multi-Model PPL**: Average perplexity from both models
4. **Learned Selection**: Train a classifier on features (perplexity, length, etc.)

## 📊 Data Flow

### Batch Processing

```python
# Input batch
batch_size = 8
prefixes = ["def foo():\n    ", ...]  # 8 prefixes
suffixes = ["\n    return x", ...]     # 8 suffixes

# Step 1: Generate from DLLM
dllm_outputs = dllm_generator.generate_batch(
    prefixes, suffixes, middle_lens, steps=64
)  # ~2-3 seconds per batch on L4

# Step 2: Generate from Qwen
qwen_outputs = qwen_generator.generate_batch(
    prefixes, suffixes, max_new_tokens=512
)  # ~1-2 seconds per batch on L4

# Step 3: Calculate perplexities
for i in range(batch_size):
    ppl_dllm = calculator.calculate_perplexity(
        prefixes[i], dllm_outputs[i], suffixes[i]
    )
    ppl_qwen = calculator.calculate_perplexity(
        prefixes[i], qwen_outputs[i], suffixes[i]
    )
    
    # Step 4: Select
    if ppl_dllm < ppl_qwen:
        ensemble_outputs[i] = dllm_outputs[i]
        winners[i] = "dllm"
    else:
        ensemble_outputs[i] = qwen_outputs[i]
        winners[i] = "qwen"
```

### Memory Efficiency

**GPU Memory Usage** (approximate for batch_size=8):

| Component | Memory |
|-----------|--------|
| Open-dLLM Model | ~2 GB |
| Qwen Model | ~2 GB |
| Activations (DLLM) | ~1-2 GB |
| Activations (Qwen) | ~0.5-1 GB |
| **Total** | **~6-8 GB** |

**Optimization Strategies**:
1. **Sequential Loading**: Load models one at a time (saves 2 GB)
2. **Gradient Checkpointing**: Not needed (inference only)
3. **Mixed Precision**: Use `bfloat16` throughout
4. **Batch Size Tuning**: Reduce if OOM errors occur

## 🧪 Evaluation Pipeline

### Task: HumanEval-Infill

```python
# Load dataset (164 examples)
dataset = load_humaneval_infill()

# For each example:
for example in dataset:
    prefix = example["prompt"]
    suffix = example["suffix"]
    ground_truth = example["canonical_solution"]
    
    # Generate ensemble completion
    completion = ensemble_generate(prefix, suffix)
    
    # Construct full function
    full_code = prefix + completion + suffix + test_code
    
    # Execute tests
    result = execute_tests(full_code)
    
    # Record: passed (bool)
    results.append({
        "task_id": example["task_id"],
        "completion": completion,
        "passed": result
    })

# Calculate Pass@1
pass_at_1 = sum(r["passed"] for r in results) / len(results)
```

### Task: SantaCoder-FIM

```python
# Load dataset (Python subset, ~1000 examples)
dataset = load_santacoder_fim()

# For each example:
for example in dataset:
    prefix = example["prompt"]
    suffix = example["suffix"]
    canonical = example["canonical_solution"]
    
    # Generate ensemble completion
    completion = ensemble_generate(prefix, suffix)
    
    # Check exact match
    is_match = (completion.strip() == canonical.strip())
    
    results.append({
        "task_id": example["id"],
        "completion": completion,
        "exact_match": is_match
    })

# Calculate Exact Match rate
em_rate = sum(r["exact_match"] for r in results) / len(results)
```

## 🎯 Design Decisions

### Why Perplexity for Selection?

**Alternatives Considered**:

1. **Confidence Scores**: Less interpretable, varies by model
2. **Ensemble Voting**: Requires multiple samples (expensive)
3. **Learned Selector**: Requires training data and labels
4. **Rule-Based**: Hard to generalize across tasks

**Why Perplexity Won**:
- ✅ **No Training**: Works out-of-the-box
- ✅ **Interpretable**: Clear meaning (likelihood)
- ✅ **Calibrated**: Uses model's native confidence
- ✅ **Context-Aware**: Considers full surrounding code
- ✅ **Model-Agnostic**: Works with any LM

### Why These Two Models?

1. **Open-dLLM**: State-of-the-art diffusion LM for code
2. **Qwen 2.5 Coder**: Strong baseline, similar size (0.5B)
3. **Complementary**: Different architectures, different strengths
4. **Fair Comparison**: Similar parameter counts

### Oracle Length Setting

We use "oracle" length (ground truth token count) for Open-dLLM because:
- Fair comparison with autoregressive models (they stop at EOS)
- Represents upper bound on diffusion model performance
- Standard practice in infilling benchmarks

## 📈 Performance Analysis

### Expected Behavior

**When DLLM is Selected**:
- More structured, template-like code
- Deterministic algorithms
- Clear patterns from context

**When Qwen is Selected**:
- Complex, context-dependent logic
- Natural language-heavy comments
- Creative solutions

**Ensemble Advantage**:
- Combines strengths of both approaches
- Reduces model-specific biases
- Improves on edge cases

### Metrics to Track

1. **Model Selection Rate**: How often each model wins
2. **Average Perplexity**: Mean PPL for each model
3. **Pass@1 / Exact Match**: Final correctness metrics
4. **Perplexity Correlation**: PPL vs. correctness analysis

## 🔬 Future Research Directions

1. **Multi-Model Ensemble**: Add more models (CodeLlama, StarCoder2, etc.)
2. **Learned Weighting**: Train a meta-learner for selection
3. **Dynamic Routing**: Route examples to best model based on features
4. **Calibration**: Normalize perplexity scores across models
5. **Hybrid Generation**: Combine outputs token-by-token
6. **Confidence Intervals**: Use multiple samples for uncertainty estimation

## 📚 References

- **Diffusion Models**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **Open-dLLM**: [GitHub](https://github.com/pengzhangzhi/Open-dLLM)
- **Qwen 2.5**: [Technical Report](https://arxiv.org/abs/2309.16609)
- **Code Infilling**: [InCoder Paper](https://arxiv.org/abs/2204.05999)
- **Perplexity**: [Language Model Evaluation](https://aclanthology.org/P16-1021/)

---

**Implementation Status**: ✅ Complete and Ready for Experimentation

