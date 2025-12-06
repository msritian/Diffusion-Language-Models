import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import gzip
import shutil

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--task", type=str, required=True, choices=["santacoder-fim", "humaneval_infill"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results")
    return parser.parse_args()

def format_fim(prefix, suffix, tokenizer):
    # Standard FIM tokens
    FIM_PREFIX = "<|fim_prefix|>"
    FIM_SUFFIX = "<|fim_suffix|>"
    FIM_MIDDLE = "<|fim_middle|>"
    
    # Check if tokenizer has these tokens, if not add them (though model might not understand them if not trained)
    # Qwen 2.5 Base usually supports these. Instruct might treat them as text.
    # For this experiment, we assume the model supports them or we use a prompt.
    
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

def evaluate_santacoder(model, tokenizer, args):
    dataset = load_dataset("bigcode/santacoder-fim-task", split="test")
    results = []
    
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
        
        # Extract generated text (remove prompt)
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Check for exact match
        # SantaCoder benchmark usually compares exact match after stripping
        is_exact_match = generated_text.strip() == example["canonical_solution"].strip()
        
        results.append({
            "task_id": i,
            "prompt": example["prompt"],
            "suffix": example["suffix"],
            "canonical_solution": example["canonical_solution"],
            "generated_code": generated_text,
            "exact_match": is_exact_match
        })
        
    # Calculate metrics
    exact_matches = sum(r["exact_match"] for r in results)
    accuracy = exact_matches / len(results)
    
    print(f"SantaCoder-FIM Accuracy: {accuracy:.2%}")
    
    return results, {"accuracy": accuracy}

def evaluate_humaneval(model, tokenizer, args):
    # Ensure dataset exists
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
        
        # For HumanEval, we just save the completion. 
        # The evaluation script will handle the rest.
        # Note: generated_text might contain the suffix if the model doesn't stop.
        # Ideally we should stop at the suffix or EOS.
        
        results.append({
            "task_id": example["task_id"],
            "completion": generated_text
        })
        
    return results, {}

def main():
    args = get_args()
    
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

if __name__ == "__main__":
    main()
