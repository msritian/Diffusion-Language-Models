import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_metrics.py <sample_file>")
        sys.exit(1)
        
    sample_file = sys.argv[1]
    
    try:
        from human_eval_infilling.evaluate_functional_correctness import evaluate_functional_correctness
    except ImportError:
        try:
            from human_eval.evaluate_functional_correctness import evaluate_functional_correctness
        except ImportError:
            print("Error: Could not import evaluate_functional_correctness. Please run setup_env.sh")
            sys.exit(1)

    print(f"Evaluating {sample_file}...")
    evaluate_functional_correctness(sample_file, benchmark_name="single-line")

if __name__ == "__main__":
    main()
