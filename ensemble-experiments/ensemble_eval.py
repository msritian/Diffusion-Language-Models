"""
Ensemble Model Evaluation Script
Combines Open-dLLM (diffusion) and Qwen 2.5 Coder 0.5B (autoregressive)
Selects the output with lower perplexity as the ensemble prediction.
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import gzip
import shutil
import wandb
from perplexity_calculator import PerplexityCalculator
from typing import List, Dict, Tuple
import datetime

# Import Open-dLLM components
import sys
sys.path.append('../Open-dLLM')
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM as DiffusionQwen2
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate ensemble model on code infilling tasks")
    parser.add_argument("--dllm_model_path", type=str, default="fredzzp/open-dcoder-0.5B",
                        help="Path to Open-dLLM diffusion model")
    parser.add_argument("--qwen_model_path", type=str, default="Qwen/Qwen2.5-Coder-0.5B",
                        help="Path to Qwen 2.5 Coder model")
    parser.add_argument("--task", type=str, required=True, 
                        choices=["santacoder-fim", "humaneval_infill"],
                        help="Evaluation task")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature for both models")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum new tokens for Qwen model")
    parser.add_argument("--dllm_steps", type=int, default=64,
                        help="Diffusion steps for Open-dLLM")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--wandb_project", type=str, default="ensemble-code-infilling",
                        help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Wandb run name")
    parser.add_argument("--perplexity_model", type=str, default="qwen",
                        choices=["qwen", "dllm", "both"],
                        help="Which model to use for perplexity calculation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples for testing")
    return parser.parse_args()


class OpenDLLMGenerator:
    """Generator for Open-dLLM diffusion model"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading Open-dLLM model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = DiffusionQwen2.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(device).eval()
        self.device = device
        
    def generate_batch(
        self, 
        prefixes: List[str], 
        suffixes: List[str],
        middle_lens: List[int],
        steps: int = 64,
        temperature: float = 0.6
    ) -> List[str]:
        """Generate infill completions using diffusion"""
        # Tokenize prompts
        tokenized_prompts = [
            self.tokenizer.encode(p, add_special_tokens=True) for p in prefixes
        ]
        prefix_lens = [len(p) for p in tokenized_prompts]
        
        # Construct sequences: prefix + masks + suffix
        sequences = [
            p + [self.tokenizer.mask_token_id] * m +
            self.tokenizer.encode(s, add_special_tokens=False)
            for p, m, s in zip(tokenized_prompts, middle_lens, suffixes)
        ]
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in sequences)
        padded_seqs = torch.LongTensor([
            seq + [self.tokenizer.pad_token_id] * (max_len - len(seq))
            for seq in sequences
        ]).to(self.device)
        
        # Generate using diffusion
        generation_config = MDMGenerationConfig(
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=10000000,  # No new tokens, just infilling
            steps=steps,
            temperature=temperature,
            top_p=0.95,
            alg="p2",
            alg_temp=0.5,
            return_dict_in_generate=True
        )
        
        with torch.no_grad():
            outputs = self.model._mdm_sample(
                x=padded_seqs,
                attention_mask=None,
                generation_config=generation_config
            )
        
        # Extract middle parts and decode
        batch_results = outputs.sequences
        generations = []
        for result, pl, ml in zip(batch_results, prefix_lens, middle_lens):
            middle_part = result[pl:pl + ml]
            decoded = self.tokenizer.decode(middle_part.tolist(), skip_special_tokens=True)
            generations.append(decoded)
            
        return generations


class QwenFIMGenerator:
    """Generator for Qwen 2.5 Coder with FIM"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading Qwen model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,  # Use explicit device instead of auto
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="eager", # sdpa might cause issues on CPU sometimes
            use_safetensors=True
        )
        self.device = device
        
    def format_fim(self, prefix: str, suffix: str) -> str:
        """Format prefix and suffix for Qwen FIM"""
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    
    def generate_batch(
        self,
        prefixes: List[str],
        suffixes: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.6
    ) -> List[str]:
        """Generate infill completions using FIM"""
        prompts = [self.format_fim(p, s) for p, s in zip(prefixes, suffixes)]
        
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        generations = []
        for output in outputs:
            generated_ids = output[inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            generations.append(generated_text)
            
        return generations


def load_humaneval_data():
    """Load HumanEval-Infill dataset"""
    data_path = "HumanEval-SingleLineInfilling.jsonl"
    if not os.path.exists(data_path):
        print("Downloading HumanEval-Infill dataset...")
        os.system("wget -q https://github.com/openai/human-eval-infilling/raw/master/data/HumanEval-SingleLineInfilling.jsonl.gz")
        with gzip.open("HumanEval-SingleLineInfilling.jsonl.gz", 'rb') as f_in:
            with open(data_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    with open(data_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    return dataset


def load_santacoder_data():
    """Load SantaCoder-FIM dataset (Python only)"""
    dataset = load_dataset("bigcode/santacoder-fim-task", split="train")
    
    # Filter for Python only
    if "lang" in dataset.features:
        dataset = dataset.filter(lambda x: x["lang"] == "py")
    elif "language" in dataset.features:
        dataset = dataset.filter(lambda x: x["language"] == "py")
    
    return list(dataset)


def evaluate_ensemble(args):
    """Main evaluation function"""
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"ensemble-{args.task}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=vars(args)
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    dllm_generator = OpenDLLMGenerator(args.dllm_model_path, device)
    qwen_generator = QwenFIMGenerator(args.qwen_model_path, device)
    
    # Initialize perplexity calculator
    ppl_calculator = PerplexityCalculator(
        qwen_generator if args.perplexity_model == "qwen" else dllm_generator
    )
    
    # Load dataset
    print(f"Loading {args.task} dataset...")
    if args.task == "humaneval_infill":
        dataset = load_humaneval_data()
        prefixes = [ex["prompt"] for ex in dataset]
        suffixes = [ex["suffix"] for ex in dataset]
        canonical_solutions = [ex["canonical_solution"] for ex in dataset]
        task_ids = [ex["task_id"] for ex in dataset]
    else:  # santacoder-fim
        dataset = load_santacoder_data()
        prefixes = [ex["prompt"] for ex in dataset]
        suffixes = [ex["suffix"] for ex in dataset]
        canonical_solutions = [ex["canonical_solution"] for ex in dataset]
        task_ids = list(range(len(dataset)))
    
    print(f"Loaded {len(dataset)} examples")
    
    if args.limit:
        print(f"Limiting to {args.limit} examples")
        dataset = dataset[:args.limit]
        prefixes = prefixes[:args.limit]
        suffixes = suffixes[:args.limit]
        canonical_solutions = canonical_solutions[:args.limit]
        task_ids = task_ids[:args.limit]
    
    # Calculate middle lengths for Open-dLLM (oracle setting)
    middle_lens = [
        len(dllm_generator.tokenizer.encode(sol, add_special_tokens=False))
        for sol in canonical_solutions
    ]
    
    # Results storage
    results = []
    stats = {
        "dllm_wins": 0,
        "qwen_wins": 0,
        "total": 0
    }
    
    # Create wandb table
    table = wandb.Table(columns=[
        "Task ID", "Prefix", "Suffix", "Canonical", 
        "DLLM Output", "DLLM PPL", 
        "Qwen Output", "Qwen PPL",
        "Ensemble Output", "Winner"
    ])
    
    # Batch generation and evaluation
    print(f"\nGenerating and evaluating {len(dataset)} examples...")
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch_prefixes = prefixes[i:i + args.batch_size]
        batch_suffixes = suffixes[i:i + args.batch_size]
        batch_middle_lens = middle_lens[i:i + args.batch_size]
        batch_canonical = canonical_solutions[i:i + args.batch_size]
        batch_task_ids = task_ids[i:i + args.batch_size]
        
        # Generate from both models
        dllm_outputs = dllm_generator.generate_batch(
            batch_prefixes,
            batch_suffixes,
            batch_middle_lens,
            steps=args.dllm_steps,
            temperature=args.temperature
        )
        
        qwen_outputs = qwen_generator.generate_batch(
            batch_prefixes,
            batch_suffixes,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        # Calculate perplexity and select ensemble output
        for j, (prefix, suffix, dllm_out, qwen_out, task_id, canonical) in enumerate(
            zip(batch_prefixes, batch_suffixes, dllm_outputs, qwen_outputs, 
                batch_task_ids, batch_canonical)
        ):
            # Calculate perplexity for both outputs
            dllm_ppl = ppl_calculator.calculate_perplexity(prefix, dllm_out, suffix)
            qwen_ppl = ppl_calculator.calculate_perplexity(prefix, qwen_out, suffix)
            
            # Select output with lower perplexity
            if dllm_ppl < qwen_ppl:
                ensemble_output = dllm_out
                winner = "dllm"
                stats["dllm_wins"] += 1
            else:
                ensemble_output = qwen_out
                winner = "qwen"
                stats["qwen_wins"] += 1
            
            stats["total"] += 1
            
            # Store result
            result = {
                "task_id": task_id,
                "completion": ensemble_output,
                "dllm_output": dllm_out,
                "qwen_output": qwen_out,
                "dllm_perplexity": float(dllm_ppl),
                "qwen_perplexity": float(qwen_ppl),
                "winner": winner,
                "prefix": prefix,
                "suffix": suffix,
                "canonical_solution": canonical
            }
            results.append(result)
            
            # Add to wandb table (limit prefix/suffix length for readability)
            table.add_data(
                task_id,
                prefix[:200] + "..." if len(prefix) > 200 else prefix,
                suffix[:200] + "..." if len(suffix) > 200 else suffix,
                canonical,
                dllm_out,
                f"{dllm_ppl:.2f}",
                qwen_out,
                f"{qwen_ppl:.2f}",
                ensemble_output,
                winner
            )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_dir, f"ensemble_{args.task}_results_{timestamp}.jsonl")
    
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\nResults saved to {output_file}")
    
    # Calculate and log statistics
    dllm_win_rate = stats["dllm_wins"] / stats["total"] * 100
    qwen_win_rate = stats["qwen_wins"] / stats["total"] * 100
    
    print(f"\nEnsemble Statistics:")
    print(f"  DLLM wins: {stats['dllm_wins']} ({dllm_win_rate:.2f}%)")
    print(f"  Qwen wins: {stats['qwen_wins']} ({qwen_win_rate:.2f}%)")
    print(f"  Total: {stats['total']}")
    
    wandb.log({
        "dllm_win_rate": dllm_win_rate,
        "qwen_win_rate": qwen_win_rate,
        "samples": table
    })
    
    # Save stats
    stats_file = os.path.join(args.output_dir, f"ensemble_{args.task}_stats_{timestamp}.json")
    with open(stats_file, 'w') as f:
        json.dump({
            "config": vars(args),
            "statistics": stats
        }, f, indent=2)
    
    wandb.finish()
    
    return output_file


def main():
    args = get_args()
    output_file = evaluate_ensemble(args)
    print(f"\nEvaluation complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()

