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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task", type=str, required=True, choices=["santacoder-fim", "humaneval_infill"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--wandb_project", type=str, default="qwen-code-infilling")
    parser.add_argument("--wandb_name", type=str, default=None)
    return parser.parse_args()

def format_fim(prefix, suffix, tokenizer):
    # Standard FIM tokens
    FIM_PREFIX = "<|fim_prefix|>"
    FIM_SUFFIX = "<|fim_suffix|>"
    FIM_MIDDLE = "<|fim_middle|>"
    
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

def evaluate_santacoder(model, tokenizer, args):
    dataset = load_dataset("bigcode/santacoder-fim-task", split="test")
    results = []
    
    # Create a WandB Table
    table = wandb.Table(columns=["Task ID", "Prompt", "Suffix", "Canonical Solution", "Generated Code", "Exact Match"])
    
    print(f"Evaluating on {len(dataset)} examples...")
    
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        prompt = format_fim(example["prompt"], example["suffix"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True if args.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        is_exact_match = generated_text.strip() == example["canonical_solution"].strip()
        
        results.append({
            "task_id": i,
            "prompt": example["prompt"],
            "suffix": example["suffix"],
            "canonical_solution": example["canonical_solution"],
            "generated_code": generated_text,
            "exact_match": is_exact_match
        })
        
        # Add to table (limit to first 100 to avoid huge tables if dataset is large, or log all)
        # Logging all might be heavy but useful. Let's log all.
        table.add_data(
            i, 
            example["prompt"][:500], # Truncate for display if needed
            example["suffix"][:500],
            example["canonical_solution"],
            generated_text,
            is_exact_match
        )
        
    # Calculate metrics
    exact_matches = sum(r["exact_match"] for r in results)
    accuracy = exact_matches / len(results)
    
    print(f"SantaCoder-FIM Accuracy: {accuracy:.2%}")
    
    # Log to wandb
    wandb.log({
        "santacoder_accuracy": accuracy,
        "santacoder_exact_matches": exact_matches,
        "santacoder_total": len(results),
        "santacoder_samples": table
    })
    
    return results, {"accuracy": accuracy}

def evaluate_humaneval(model, tokenizer, args):
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
    
    # Create WandB Table
    table = wandb.Table(columns=["Task ID", "Prompt", "Suffix", "Canonical Solution", "Generated Code"])

    print(f"Evaluating on {len(dataset)} examples...")
    
    for example in tqdm(dataset):
        prompt = format_fim(example["prompt"], example["suffix"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True if args.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
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
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{args.task}-{args.model_path.split('/')[-1]}",
        config=vars(args)
    )
    
    print(f"Loading model {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    if args.task == "santacoder-fim":
        results, metrics = evaluate_santacoder(model, tokenizer, args)
    elif args.task == "humaneval_infill":
        results, metrics = evaluate_humaneval(model, tokenizer, args)
        
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.task}_results.jsonl")
    
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    if metrics:
        with open(os.path.join(args.output_dir, f"{args.task}_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
            
    print(f"Results saved to {output_file}")
    wandb.finish()

if __name__ == "__main__":
    main()
