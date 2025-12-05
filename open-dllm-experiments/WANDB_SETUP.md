# Wandb Setup and Integration Guide

## ✅ Wandb is Already Integrated!

Wandb logging is **enabled by default** in the evaluation scripts. Here's what gets logged:

### What Gets Logged to Wandb

#### 1. **Configuration Parameters**
- Model path: `fredzzp/open-dcoder-0.5B`
- Task: `humaneval_infill` or `santacoder-fim`
- Temperature, steps, algorithm
- Batch size, GPU settings
- All hyperparameters

#### 2. **Evaluation Metrics**
- **Pass@1** scores (HumanEval-Infill)
- **Exact Match** scores (SantaCoder-FIM)
- Sample counts

#### 3. **Artifacts**
- All prediction files (`.jsonl`)
- Evaluation results (`.json`)
- Complete timestamped results

### Wandb Project Name
```python
# In utils.py line 33
project="eval-infill-dllm-step64-latest"
```

All your experiments will appear under this project in Wandb.

---

## 🔧 Setup Wandb (One-Time)

### On Your GPU VM:

**Before running evaluations**, set up wandb authentication:

```bash
# Install wandb (already included in gpu_setup.sh)
pip install wandb

# Login to wandb
wandb login
```

You'll be prompted for your API key. Get it from: https://wandb.ai/authorize

**Or** set it as environment variable:
```bash
export WANDB_API_KEY="your-api-key-here"
```

---

## 📊 How It Works

### During Evaluation:

1. **Wandb initializes** with project name and run name:
   ```
   Run name: humaneval_infill_open-dcoder-0.5B_20251204_214523
   ```

2. **Logs metrics** as they're computed:
   ```python
   wandb.log({
       'pass@1': 0.325,  # For HumanEval-Infill
       'exact_match': 0.296  # For SantaCoder-FIM
   })
   ```

3. **Creates artifacts** with all results:
   - Prediction files
   - Evaluation JSON files
   - Timestamped for versioning

4. **Public sharing**: Your wandb runs can be made public for everyone to see!

---

## 🚀 Quick Start with Wandb

### Option 1: Login Before Running
```bash
cd open-dllm-experiments
wandb login  # Enter your API key
bash gpu_setup.sh
bash run_evaluations.sh
```

### Option 2: Set API Key as Environment Variable
```bash
export WANDB_API_KEY="your-key-from-https://wandb.ai/authorize"
cd open-dllm-experiments
bash gpu_setup.sh
bash run_evaluations.sh
```

### Option 3: Disable Wandb (Optional)
If you don't want wandb logging:
```bash
cd Open-dLLM/eval/eval_infill
python eval_infill.py --model_path fredzzp/open-dcoder-0.5B \
  --task humaneval_infill --no_wandb
```

---

## 📈 Viewing Results

### On Wandb Dashboard:

1. Go to https://wandb.ai
2. Navigate to project: `eval-infill-dllm-step64-latest`
3. See all runs with:
   - Interactive charts of Pass@1 and Exact Match
   - Configuration comparison
   - Artifact downloads
   - Run history

### Making Results Public:

In Wandb dashboard:
1. Go to your project settings
2. Change visibility to "Public"
3. Share the project link with anyone!

Example public project URL:
```
https://wandb.ai/your-username/eval-infill-dllm-step64-latest
```

---

## 🎯 What You'll See in Wandb

### Runs Tab:
- List of all evaluation runs
- Pass@1 and Exact Match metrics
- Timestamp and configuration

### Charts Tab:
- Pass@1 trends across runs
- Exact Match comparison
- Temperature vs. performance
- Steps vs. accuracy

### Artifacts Tab:
- All prediction files
- Evaluation results
- Downloadable for further analysis

### System Tab:
- GPU utilization
- Memory usage
- Runtime statistics

---

## 📋 Wandb Integration Code Reference

### Initialization (utils.py:32-36)
```python
wandb.init(
    project="eval-infill-dllm-step64-latest",
    name=run_name,
    config=config_params or {}
)
```

### Logging Metrics (utils.py:39-41)
```python
if eval_results:
    wandb.log(eval_results)
    print(f"Logged metrics to wandb: {eval_results}")
```

### Creating Artifacts (utils.py:44-63)
```python
artifact = wandb.Artifact(
    name=f"{task}_{model_name}_results",
    type="predictions",
    description=f"Prediction results for {task}"
)
artifact.add_file(prediction_path)
wandb.log_artifact(artifact)
```

---

## ✨ Summary

✅ **Wandb is fully integrated and enabled by default**  
✅ **Logs all metrics (Pass@1, Exact Match) automatically**  
✅ **Saves all predictions and results as artifacts**  
✅ **One-time setup with `wandb login`**  
✅ **Results viewable in beautiful dashboard**  
✅ **Can be made public for everyone to see**

### Next Step:
Just run `wandb login` on your GPU VM before running evaluations!

```bash
wandb login  # One-time setup
cd /path/to/Diffusion-Language-Models/open-dllm-experiments
bash run_evaluations.sh  # Everything logged automatically!
```
