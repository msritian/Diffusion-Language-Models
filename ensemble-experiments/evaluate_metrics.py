"""
Evaluation Metrics Script for Ensemble Model

Runs functional correctness evaluation on ensemble results.
Supports both HumanEval-Infill and SantaCoder-FIM tasks.
"""

import argparse
import json
import os
import sys
import subprocess
from typing import Dict, Any

# Add paths for evaluation tools
sys.path.append('../small-model-experiments')
sys.path.append('../Open-dLLM/eval/eval_infill')

from evaluate_humaneval_infill import evaluate_functional_correctness


def evaluate_santacoder_em(results_file: str) -> Dict[str, float]:
    """
    Evaluate exact match for SantaCoder-FIM task.
    
    Args:
        results_file: Path to ensemble results JSONL file
        
    Returns:
        Dictionary with exact match metrics
    """
    print(f"\nEvaluating SantaCoder-FIM Exact Match...")
    
    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f]
    
    exact_matches = 0
    total = 0
    
    for result in results:
        completion = result["completion"].strip()
        canonical = result["canonical_solution"].strip()
        
        if completion == canonical:
            exact_matches += 1
        total += 1
    
    em_rate = exact_matches / total if total > 0 else 0
    
    metrics = {
        "exact_match": em_rate,
        "exact_matches": exact_matches,
        "total": total
    }
    
    print(f"Exact Match Rate: {em_rate:.2%} ({exact_matches}/{total})")
    
    return metrics


def evaluate_humaneval_pass_at_k(results_file: str) -> Dict[str, float]:
    """
    Evaluate pass@k for HumanEval-Infill task.
    
    Args:
        results_file: Path to ensemble results JSONL file
        
    Returns:
        Dictionary with pass@k metrics
    """
    print(f"\nEvaluating HumanEval-Infill Pass@k...")
    
    # Convert ensemble results to format expected by evaluation tool
    temp_file = results_file.replace('.jsonl', '_for_eval.jsonl')
    
    with open(results_file, 'r') as fin, open(temp_file, 'w') as fout:
        for line in fin:
            result = json.loads(line)
            eval_format = {
                "task_id": result["task_id"],
                "completion": result["completion"]
            }
            fout.write(json.dumps(eval_format) + '\n')
    
    try:
        # Try to import and use the evaluation function directly
        try:
            from evaluate_humaneval_infill import evaluate_functional_correctness as eval_func
            metrics = eval_func(temp_file, k=[1])
        except ImportError:
            # Fallback: use the imported function from the module
            metrics = evaluate_functional_correctness(temp_file, k=[1])
        
        print(f"Pass@1: {metrics.get('pass@1', 0):.2%}")
        if 'pass@10' in metrics:
            print(f"Pass@10: {metrics['pass@10']:.2%}")
        if 'pass@100' in metrics:
            print(f"Pass@100: {metrics['pass@100']:.2%}")
        
        return metrics
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print(f"Note: You can manually evaluate using:")
        print(f"  python ../small-model-experiments/evaluate_humaneval_infill.py {temp_file}")
        return {}
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)


def compare_with_baselines(results_file: str, task: str):
    """
    Compare ensemble results with individual model results.
    
    Args:
        results_file: Path to ensemble results
        task: Task name (humaneval_infill or santacoder-fim)
    """
    print(f"\n{'='*60}")
    print("Ensemble vs. Individual Model Comparison")
    print(f"{'='*60}")
    
    with open(results_file, 'r') as f:
        results = [json.loads(line) for line in f]
    
    # Analyze which model was selected more often
    dllm_selected = sum(1 for r in results if r["winner"] == "dllm")
    qwen_selected = sum(1 for r in results if r["winner"] == "qwen")
    total = len(results)
    
    print(f"\nModel Selection Statistics:")
    print(f"  Open-dLLM selected: {dllm_selected}/{total} ({dllm_selected/total*100:.2f}%)")
    print(f"  Qwen selected: {qwen_selected}/{total} ({qwen_selected/total*100:.2f}%)")
    
    # Average perplexities
    avg_dllm_ppl = sum(r["dllm_perplexity"] for r in results) / total
    avg_qwen_ppl = sum(r["qwen_perplexity"] for r in results) / total
    
    print(f"\nAverage Perplexities:")
    print(f"  Open-dLLM: {avg_dllm_ppl:.2f}")
    print(f"  Qwen: {avg_qwen_ppl:.2f}")
    
    # Perplexity distribution
    dllm_better = sum(1 for r in results if r["dllm_perplexity"] < r["qwen_perplexity"])
    qwen_better = sum(1 for r in results if r["qwen_perplexity"] < r["dllm_perplexity"])
    
    print(f"\nPerplexity Comparison:")
    print(f"  Open-dLLM had lower PPL: {dllm_better}/{total} ({dllm_better/total*100:.2f}%)")
    print(f"  Qwen had lower PPL: {qwen_better}/{total} ({qwen_better/total*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble model results")
    parser.add_argument("results_file", type=str, help="Path to ensemble results JSONL file")
    parser.add_argument("--task", type=str, required=True, 
                        choices=["santacoder-fim", "humaneval_infill"],
                        help="Evaluation task")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for metrics (default: results_file + _metrics.json)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return 1
    
    # Compare with baselines
    compare_with_baselines(args.results_file, args.task)
    
    # Run task-specific evaluation
    if args.task == "santacoder-fim":
        metrics = evaluate_santacoder_em(args.results_file)
    else:  # humaneval_infill
        metrics = evaluate_humaneval_pass_at_k(args.results_file)
    
    # Save metrics
    output_file = args.output_file or args.results_file.replace('.jsonl', '_metrics.json')
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

