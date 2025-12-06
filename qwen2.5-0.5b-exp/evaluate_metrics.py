import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_metrics.py <sample_file>")
        sys.exit(1)
        
    sample_file = sys.argv[1]
    
    try:
        from human_eval_infilling.evaluation import evaluate_functional_correctness
    except ImportError:
        try:
            from human_eval.evaluation import evaluate_functional_correctness
        except ImportError:
            print("Error: Could not import evaluate_functional_correctness. Please run setup_env.sh")
            sys.exit(1)

    print(f"Evaluating {sample_file}...")
    # Signature is: evaluate_functional_correctness(benchmark_name, sample_file, ...)
    # We use k=[1] because we only generate 1 sample per problem
    results = evaluate_functional_correctness("single-line", sample_file, k=[1])
    print(results)

if __name__ == "__main__":
    main()
