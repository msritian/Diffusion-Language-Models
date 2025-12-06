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
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--task", type=str, required=True, choices=["santacoder-fim", "humaneval_infill"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--wandb_project", type=str, default="code-infilling-benchmarks")
    parser.add_argument("--wandb_name", type=str, default=None)
    return parser.parse_args()

class FIMFormatter:
    def __init__(self, model_path, tokenizer):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.fim_type = self._detect_fim_type()
        
    def _detect_fim_type(self):
        name = self.model_path.lower()
        if "qwen" in name:
            return "qwen"
        elif "deepseek" in name:
            return "deepseek"
        elif "starcoder" in name or "santa" in name:
            return "starcoder"
        else:
            print(f"Warning: Unknown model type for {self.model_path}. Defaulting to StarCoder format.")
            return "starcoder"

    def format(self, prefix, suffix):
        if self.fim_type == "qwen":
            # Qwen: <|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>
            return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
            
        elif self.fim_type == "deepseek":
            # DeepSeek: <｜fim hole｜>{suffix}<｜fim begin｜>{prefix}<｜fim end｜>
            # Based on debug results, SPM format (Format 2) seems to work best.
            return f"<｜fim hole｜>{suffix}<｜fim begin｜>{prefix}<｜fim end｜>"
            
        elif self.fim_type == "starcoder":
            # StarCoder: <fim_prefix>...<fim_suffix>...<fim_middle>
            return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
            
        return f"{prefix}{suffix}"

def evaluate_santacoder(model, tokenizer, args, formatter):
    dataset = load_dataset("bigcode/santacoder-fim-task", split="train")
    
    # Filter for Python only
    if "lang" in dataset.features:
        dataset = dataset.filter(lambda x: x["lang"] == "py")
    elif "language" in dataset.features:
        dataset = dataset.filter(lambda x: x["language"] == "py")
        
    print(f"Evaluating on {len(dataset)} examples (filtered for Python)...")
    
    results = []
    table = wandb.Table(columns=["Task ID", "Prompt", "Suffix", "Canonical Solution", "Generated Code", "Exact Match"])
    
    prompts = [formatter.format(ex["prompt"], ex["suffix"]) for ex in dataset]
    
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]
        batch_indices = range(i, min(i + args.batch_size, len(dataset)))
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True if args.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<file_sep>")] if "starcoder" in args.model_path.lower() else tokenizer.eos_token_id
            )
        
        for j, output in enumerate(outputs):
            # Decode only the new tokens
            generated_ids = output[inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            idx = batch_indices[j]
            example = dataset[idx]
            
            is_exact_match = generated_text.strip() == example["canonical_solution"].strip()
            
            results.append({
                "task_id": idx,
                "prompt": example["prompt"],
                "suffix": example["suffix"],
                "canonical_solution": example["canonical_solution"],
                "generated_code": generated_text,
                "exact_match": is_exact_match
            })
            
            table.add_data(
                idx, 
                example["prompt"][:500],
                example["suffix"][:500],
                example["canonical_solution"],
                generated_text,
                is_exact_match
            )
        
    exact_matches = sum(r["exact_match"] for r in results)
    accuracy = exact_matches / len(results)
    
    print(f"SantaCoder-FIM Accuracy: {accuracy:.2%}")
    
    wandb.log({
        "santacoder_accuracy": accuracy,
        "santacoder_exact_matches": exact_matches,
        "santacoder_total": len(results),
        "santacoder_samples": table
    })
    
    return results, {"accuracy": accuracy}

def evaluate_humaneval(model, tokenizer, args, formatter):
    data_path = "HumanEval-SingleLineInfilling.jsonl"
    if not os.path.exists(data_path):
        print("Downloading HumanEval-Infill dataset...")
        os.system("wget -q https://github.com/openai/human-eval-infilling/raw/master/data/HumanEval-SingleLineInfilling.jsonl.gz")
        with gzip.open("HumanEval-SingleLineInfilling.jsonl.gz", 'rb') as f_in:
            with open(data_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
    with open(data_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
        
    results = []
    table = wandb.Table(columns=["Task ID", "Prompt", "Suffix", "Canonical Solution", "Generated Code"])

    print(f"Evaluating on {len(dataset)} examples...")
    
    prompts = [formatter.format(ex["prompt"], ex["suffix"]) for ex in dataset]
    
    for i in tqdm(range(0, len(dataset), args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]
        batch_indices = range(i, min(i + args.batch_size, len(dataset)))
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True if args.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<file_sep>")] if "starcoder" in args.model_path.lower() else tokenizer.eos_token_id
            )
            
        for j, output in enumerate(outputs):
            generated_ids = output[inputs.input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            idx = batch_indices[j]
            example = dataset[idx]
            
            results.append({
                "task_id": example["task_id"],
                "completion": generated_text
            })
            
            table.add_data(
                example["task_id"],
                example["prompt"][:500],
                example["suffix"][:500],
                example["canonical_solution"],
                generated_text
            )
    
    wandb.log({
        "humaneval_generated_count": len(results),
        "humaneval_samples": table
    })
        
    return results, {}

def main():
    args = get_args()
    
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{args.task}-{args.model_path.split('/')[-1]}",
        config=vars(args)
    )
    
    print(f"Loading model {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Fix for DeepSeek FIM tokens being split
    if "deepseek" in args.model_path.lower():
        print("Adding DeepSeek FIM tokens to tokenizer...")
        fim_tokens = ["<｜fim begin｜>", "<｜fim hole｜>", "<｜fim end｜>"]
        tokenizer.add_special_tokens({"additional_special_tokens": fim_tokens})
        # No need to resize embeddings as these tokens are already in the vocab (just not marked special)

        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        use_safetensors=True
    )
    
    formatter = FIMFormatter(args.model_path, tokenizer)
    print(f"Detected FIM type: {formatter.fim_type}")
    
    if args.task == "santacoder-fim":
        results, metrics = evaluate_santacoder(model, tokenizer, args, formatter)
    elif args.task == "humaneval_infill":
        results, metrics = evaluate_humaneval(model, tokenizer, args, formatter)
        
    os.makedirs(args.output_dir, exist_ok=True)
    # Create model-specific output directory
    model_name = args.model_path.split('/')[-1]
    model_output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    output_file = os.path.join(model_output_dir, f"{args.task}_results.jsonl")
    
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    if metrics:
        with open(os.path.join(model_output_dir, f"{args.task}_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
            
    print(f"Results saved to {output_file}")
    wandb.finish()

if __name__ == "__main__":
    main()
