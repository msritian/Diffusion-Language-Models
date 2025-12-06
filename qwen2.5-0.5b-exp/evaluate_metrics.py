import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_metrics.py <sample_file>")
        sys.exit(1)
        
    sample_file = sys.argv[1]
    
    try:
        import human_eval_infilling
        print(f"Found human_eval_infilling at: {human_eval_infilling.__file__}")
        from human_eval_infilling.evaluate_functional_correctness import evaluate_functional_correctness
        print("Successfully imported evaluate_functional_correctness from human_eval_infilling")
    except ImportError as e:
        print(f"ImportError: {e}")
        try:
            import human_eval
            print(f"Found human_eval at: {human_eval.__file__}")
            from human_eval.evaluate_functional_correctness import evaluate_functional_correctness
            print("Successfully imported evaluate_functional_correctness from human_eval")
        except ImportError as e2:
            print(f"ImportError: {e2}")
            print("CRITICAL ERROR: Could not import evaluate_functional_correctness from anywhere.")
            print("Please check if 'human-eval-infilling' is installed in your environment.")
            sys.exit(1)

    print(f"Evaluating {sample_file}...")
    evaluate_functional_correctness(sample_file, benchmark_name="single-line")

if __name__ == "__main__":
    main()
