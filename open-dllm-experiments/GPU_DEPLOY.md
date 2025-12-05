# GPU VM Quick Deploy - Complete Instructions

## 🚀 One-Command Setup and Run

On your GPU VM, run this single command to set up everything and start evaluations:

```bash
git clone https://github.com/msritian/Diffusion-Language-Models.git && \
cd Diffusion-Language-Models && \
git checkout feature/open-dllm-experiments && \
cd open-dllm-experiments && \
bash gpu_setup.sh && \
bash run_evaluations.sh
```

**Total time:** ~45-75 minutes (15 min setup + 30-60 min evaluation)

## 📋 What You Need

### On Your GPU VM:
- Ubuntu 20.04+ or similar Linux
- NVIDIA GPU (T4, V100, A100, or better)
- CUDA 12.1 or higher installed
- 16GB+ GPU VRAM (recommended)
- 50GB+ disk space
- Python 3.8+
- Git

### On Your Local Machine (Already Done):
✅ Committed to branch: `feature/open-dllm-experiments`  
✅ Ready to push to: `https://github.com/msritian/Diffusion-Language-Models`

## 📤 Push Your Branch to GitHub

Before running on GPU, push your local branch:

```bash
cd /Users/Shivam/Documents/Diffusion-Language-Models
git push -u origin feature/open-dllm-experiments
```

If this is the first push, you may need to authenticate with GitHub.

## 🖥️ On Your GPU VM

### Step 1: Clone Repository
```bash
git clone https://github.com/msritian/Diffusion-Language-Models.git
cd Diffusion-Language-Models
git checkout feature/open-dllm-experiments
```

### Step 2: Run Setup
```bash
cd open-dllm-experiments
bash gpu_setup.sh
```

This installs:
- PyTorch 2.5.0 with CUDA 12.1
- Flash Attention 2.7.4
- All evaluation packages
- Open-dLLM and dependencies

### Step 3: Run Evaluations
```bash
bash run_evaluations.sh
```

This runs:
1. HumanEval-Infill (Pass@1 metric)
2. SantaCoder-FIM (Exact Match metric)

### Step 4: View Results
```bash
cd ../Open-dLLM/eval/eval_infill/infill_results
ls -lh
```

Look for files:
- `humaneval_infill_*.jsonl` - All predictions
- `humaneval_infill_*_eval_results.json` - Metrics
- `santacoder-fim_*.jsonl` - All predictions  
- `santacoder-fim_*_eval_results.json` - Metrics

## 📊 Expected Results

From Open-dLLM paper (0.5B model):

| Benchmark | Metric | Expected Score |
|-----------|--------|----------------|
| HumanEval-Infill | Pass@1 | ~32.5% (fixed) / ~77.4% (oracle) |
| SantaCoder-FIM | Exact Match | ~29.6% (fixed) / ~56.4% (oracle) |

## ⚙️ Customization

### Use Different Parameters
```bash
export TEMPERATURE=0.8
export STEPS=128
export BATCH_SIZE=16
bash run_evaluations.sh
```

### Use Multiple GPUs
The script auto-detects available GPUs and uses all of them with DDP.

### Run Only One Benchmark
```bash
cd ../Open-dLLM/eval/eval_infill

# Just HumanEval-Infill
python eval_infill.py --model_path fredzzp/open-dcoder-0.5B \
  --task humaneval_infill --temperature 0.6 --steps 64

# Just SantaCoder-FIM
python eval_infill.py --model_path fredzzp/open-dcoder-0.5B \
  --task santacoder-fim --temperature 0.6 --steps 64
```

## 🔧 Troubleshooting

### Out of Memory
```bash
export BATCH_SIZE=8
bash run_evaluations.sh
```

### CUDA Not Found
Verify CUDA installation:
```bash
nvidia-smi
nvcc --version
```

### Model Download Slow
Use Hugging Face cache:
```bash
export HF_HOME=/path/to/large/disk
bash run_evaluations.sh
```

## 📦 Transferring Results Back

After evaluation:

```bash
# Copy results to experiment directory
cd /path/to/Diffusion-Language-Models
cp -r Open-dLLM/eval/eval_infill/infill_results open-dllm-experiments/results/

# Commit and push
git add open-dllm-experiments/results/
git commit -m "Add GPU evaluation results"
git push origin feature/open-dllm-experiments
```

## 🎯 Files Created for GPU

All in `open-dllm-experiments/`:

| File | Purpose |
|------|---------|
| `gpu_setup.sh` | Complete environment setup |
| `run_evaluations.sh` | Run both benchmarks |
| `QUICKSTART.md` | Detailed instructions |
| `README.md` | Full documentation |
| `sample_test.py` | CPU test (optional) |

## ✅ Verification Checklist

Before running evaluations:
- [ ] GPU VM has CUDA installed (`nvidia-smi` works)
- [ ] Pushed branch to GitHub
- [ ] Cloned repository on GPU VM
- [ ] Checked out `feature/open-dllm-experiments` branch
- [ ] Ran `gpu_setup.sh` successfully
- [ ] Verified PyTorch sees GPU: `python -c "import torch; print(torch.cuda.is_available())"`

After running evaluations:
- [ ] Both benchmarks completed without errors
- [ ] Results files created in `infill_results/`
- [ ] Metrics look reasonable (Pass@1 > 0%)
- [ ] Copied results back to repo
- [ ] Committed and pushed results

## 💡 Quick Tips

1. **Use tmux/screen** for long-running evaluations:
   ```bash
   tmux new -s eval
   bash run_evaluations.sh
   # Ctrl+B, D to detach
   ```

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Save logs**:
   ```bash
   bash run_evaluations.sh 2>&1 | tee eval_log.txt
   ```

## 🤝 Need Help?

See detailed documentation in:
- `QUICKSTART.md` - Quick start guide
- `README.md` - Complete documentation
- Open-dLLM repo: https://github.com/pengzhangzhi/Open-dLLM

---

**Ready to deploy!** Just push your branch and run on GPU VM. 🚀
