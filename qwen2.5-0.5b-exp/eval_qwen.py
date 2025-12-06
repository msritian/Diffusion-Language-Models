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
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task", type=str, required=True, choices=["santacoder-fim", "humaneval_infill"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--wandb_project", type=str, default="qwen-code-infilling")
    parser.add_argument("--wandb_name", type=str, default=None)
    return parser.parse_args()

def format_fim(prefix, suffix, tokenizer):
    FIM_PREFIX = "<|fim_prefix|>"
    FIM_SUFFIX = "<|fim_suffix|>"
    FIM_MIDDLE = "<|fim_middle|>"
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

def evaluate_santacoder(model, tokenizer, args):
    dataset = load_dataset("bigcode/santacoder-fim-task", split="train")
    
    # Filter for Python only (to match the ~1000 examples in Open-dLLM experiments)
    # The dataset uses 'py' for Python
    if "lang" in dataset.features:
        dataset = dataset.filter(lambda x: x["lang"] == "py")
    elif "language" in dataset.features:
        dataset = dataset.filter(lambda x: x["language"] == "py")
        
    print(f"Evaluating on {len(dataset)} examples (filtered for Python)...")
    
    results = []
    table = wandb.Table(columns=["Task ID", "Prompt", "Suffix", "Canonical Solution", "Generated Code", "Exact Match"])
    
    # Prepare data for batching
    prompts = [format_fim(ex["prompt"], ex["suffix"], tokenizer) for ex in dataset]
    
    # Batch processing
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
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode batch
        for j, output in enumerate(outputs):
            # The input length might vary per sample due to padding, but here we just skip the prompt length
            # A safer way is to decode the new tokens only
            input_len = inputs.input_ids[j].shape[0]
            # Actually, inputs.input_ids is padded, so we need to be careful.
            # But model.generate returns the full sequence.
            # We can just decode the part after the input.
            # However, with left padding (common for generation), it's tricky.
            # Qwen tokenizer might use right padding by default?
            # Let's decode everything and strip the prompt.
            
            full_text = tokenizer.decode(output, skip_special_tokens=False)
            # Remove the FIM special tokens and prompt manually or just use the skip_special_tokens=True on the new part
            # Let's try to just decode the new tokens.
            # We know the input length.
            # But wait, if we padded, the input_ids have padding.
            
            # Better approach:
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
        
    # Calculate metrics
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
    table = wandb.Table(columns=["Task ID", "Prompt", "Suffix", "Canonical Solution", "Generated Code"])

    print(f"Evaluating on {len(dataset)} examples...")
    
    # Prepare prompts
    prompts = [format_fim(ex["prompt"], ex["suffix"], tokenizer) for ex in dataset]
    
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
                pad_token_id=tokenizer.eos_token_id
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
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{args.task}-{args.model_path.split('/')[-1]}",
        config=vars(args)
    )
    
    print(f"Loading model {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left" # Important for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa"
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
