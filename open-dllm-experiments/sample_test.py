"""
Simple test script to verify Open-dCoder model can be loaded and used for inference.

NOTE: This is adapted for CPU-only inference since we're running on macOS without CUDA.
Flash attention and GPU-specific optimizations are disabled.
"""

import torch
from transformers import AutoTokenizer
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("=" * 60)
    print("Testing Open-dCoder Model Loading")
    print("=" * 60)
    
    model_id = "fredzzp/open-dcoder-0.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nEnvironment:")
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - Device: {device}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    
    print(f"\nLoading model: {model_id}")
    print("  - Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    print("  - Loading model... (this may take a while)")
    # Use float32 instead of bfloat16 for CPU compatibility
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Qwen2ForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(device).eval()
    
    print("  ✓ Model loaded successfully!")
    
    return tokenizer, model, device


def test_inference(tokenizer, model, device):
    """Test basic inference with the model"""
    print("\n" + "=" * 60)
    print("Testing Diffusion Generation")
    print("=" * 60)
    
    # Simple prompt
    prompt = "def quick_sort(arr):"
    print(f"\nPrompt: {prompt}")
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Use fewer steps for faster CPU inference
    print("\nGeneration config:")
    print("  - max_new_tokens: 64")
    print("  - steps: 50 (reduced for CPU)")
    print("  - temperature: 0.7")
    
    gen_cfg = MDMGenerationConfig(
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=64,
        steps=50,  # Reduced from 200 for faster CPU inference
        temperature=0.7
    )
    
    print("\nGenerating... (this will take a while on CPU)")
    with torch.no_grad():
        outputs = model.diffusion_generate(inputs=input_ids, generation_config=gen_cfg)
    
    result = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    print("\n" + "-" * 60)
    print("Generated code:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    
    return result


def main():
    """Main test function"""
    try:
        # Test 1: Model loading
        tokenizer, model, device = test_model_loading()
        
        # Test 2: Inference
        result = test_inference(tokenizer, model, device)
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        
        # Save result to file
        output_file = "test_output.txt"
        with open(output_file, "w") as f:
            f.write(f"Model: fredzzp/open-dcoder-0.5B\n")
            f.write(f"Device: {device}\n")
            f.write(f"PyTorch: {torch.__version__}\n\n")
            f.write(f"Generated output:\n{result}\n")
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
