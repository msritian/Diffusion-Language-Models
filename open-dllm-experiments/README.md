# Open-dLLM Experimentation Setup

**Status:** ✅ GPU-Ready | Full evaluation environment configured

This directory contains all setup scripts, documentation, and results for experimenting with the Open-dLLM diffusion language model on code infilling benchmarks.

## 🚀 Quick Deploy on GPU

```bash
# From repository root
cd open-dllm-experiments
bash gpu_setup.sh      # ~15 minutes
bash run_evaluations.sh  # ~30-60 minutes
```

### Limitations Encountered
1. **No CUDA Support**: macOS doesn't support CUDA, limiting us to CPU-only inference
2. **Flash Attention Unavailable**: Flash attention (required for efficient inference) is not available on macOS
3. **Slow CPU Inference**: Diffusion models require many sampling steps (50-512), making CPU inference very slow
4. **PyTorch Version Conflicts**: Some dependencies (bytecheckpoint) require PyTorch <=2.5.0, but macOS ARM builds only start from 2.6.0+

### Successfully Installed Components
- ✅ Transformers 4.54.1
- ✅ Accelerate, Datasets, PEFT
- ✅ Evaluation packages: lm-evaluation-harness, human-eval-infilling
- ✅ Veomni package (Open-dLLM core)
- ✅ Model loading: fredzzp/open-dcoder-0.5B can be loaded successfully

### Partial/Skipped Components
- ⚠️ flash-attn (not compatible with macOS)
- ⚠️ liger-kernel (requires triton, which needs CUDA)
- ⚠️ bytecheckpoint (PyTorch version conflict)

## Repository Structure

```
Open-dLLM/
├── eval/
│   ├── eval_completion/     # HumanEval, MBPP benchmarks
│   └── eval_infill/         # HumanEval-Infill, SantaCoder-FIM
├── experiments/             # Our experiment results (created)
│   ├── visualizations/      # Graphs and plots
│   └── results_summary.md   # Experiment documentation
├── sample.py               # Basic inference script
├── sample_test.py          # Our test script (CPU-compatible)
└── veomni/                 # Core diffusion LM implementation
```

## Usage

### Quick Test (CPU-only)
```bash
python sample_test.py
```

This will:
1. Load the fredzzp/open-dcoder-0.5B model
2. Run a simple code generation task
3. Save results to `test_output.txt`

**Note**: This is slow on CPU (~few minutes per generation with reduced steps)

### Full Evaluation (Requires GPU)

To run the complete evaluation benchmarks, you'll need a Linux machine with CUDA:

#### HumanEval-Infill
```bash
cd eval/eval_infill
bash run_eval.sh
```

#### SantaCoder-FIM
The script `run_eval.sh` runs both benchmarks:
- HumanEval-Infill: Tests Fill-in-the-Middle (FIM) on 164 problems
- SantaCoder-FIM: Tests code infilling with exact match metrics

**Configuration** (in `run_eval.sh`):
```bash
MODEL_PATH="fredzzp/open-dcoder-0.5B"
TEMPERATURE=0.6
STEPS=64              # Diffusion sampling steps
ALG="p2"              # Sampling algorithm
BATCH_SIZE=32
```

### Running on GPU (Recommended Workflow)

1. **Transfer this repository to a GPU instance**:
   ```bash
   # On your local machine
   cd /Users/Shivam/Documents/Diffusion-Language-Models
   git add Open-dLLM/
   git commit -m "Add Open-dLLM setup for experimentation"
   git push origin feature/open-dllm-experiments
   
   # On GPU machine (e.g., Google Colab, AWS, etc.)
   git clone <your-repo>
   git checkout feature/open-dllm-experiments
   cd Open-dLLM
   ```

2. **Install GPU-specific dependencies**:
   ```bash
   # Install CUDA-enabled PyTorch
   pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121
   
   # Install flash attention
   pip install "flash-attn==2.7.4.post1" \\
     --extra-index-url https://github.com/Dao-AILab/flash-attention/releases/download
   
   # Install all remaining dependencies
   pip install -e .
   pip install -e lm-evaluation-harness human-eval-infilling
   ```

3. **Run evaluations**:
   ```bash
   cd eval/eval_infill
   bash run_eval.sh
   ```

## Expected Results

Based on the Open-dLLM paper, the expected performance metrics are:

| Benchmark | Metric | Expected Score (0.5B model) |
|-----------|--------|----------------------------|
| HumanEval-Infill | Pass@1 | ~32.5% (fixed length), ~77.4% (oracle length) |
| SantaCoder-FIM | Exact Match | ~29.6% (fixed length), ~56.4% (oracle length) |

Results will be saved in:
- `eval/eval_infill/infill_results/<task>/<model>/<temperature>/`
- JSON format with completions and evaluation metrics

## Next Steps

To complete the experimentation:

1. **Option A: Use GPU**
   - Transfer this repository to a machine with CUDA
   - Run the full evaluation suite
   - Results will be automatically saved and can be synced back

2. **Option B: Use Cloud Services**
   - Google Colab (free GPU access)
   - AWS/GCP with GPU instances
   - Paperspace, Lambda Labs, etc.

3. **Collect and Analyze Results**
   - Evaluation scripts generate JSONL files with all predictions
   - Automatic metrics calculation (Pass@1, Exact Match)
   - Optional: wandb logging for experiment tracking

## Resources

- [Open-dLLM GitHub](https://github.com/pengzhangzhi/Open-dLLM)
- [Open-dLLM Blog Post](https://oval-shell-31c.notion.site/Open-Diffusion-Large-Language-Model-25e03bf6136480b7a4ebe3d53be9f68a)
- [Model on Hugging Face](https://huggingface.co/fredzzp/open-dcoder-0.5B)
- [HumanEval-Infill Benchmark](https://github.com/openai/human-eval)
- [SantaCoder-FIM Dataset](https://huggingface.co/datasets/bigcode/santacoder-fim-task)
