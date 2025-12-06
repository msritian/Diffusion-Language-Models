import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model(model_path, fim_type):
    print(f"\n{'='*50}")
    print(f"Testing Model: {model_path}")
    print(f"{'='*50}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Apply DeepSeek Fix
    if "deepseek" in model_path.lower():
        print("Applying DeepSeek Fix: Adding special tokens...")
        fim_tokens = ["<｜fim begin｜>", "<｜fim hole｜>", "<｜fim end｜>"]
        tokenizer.add_special_tokens({"additional_special_tokens": fim_tokens})
        
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True,
        use_safetensors=True
    )
    
    prefix = "def add(a, b):\n    "
    suffix = "\n    return result"
    
    # Apply FIM Formatting
    if fim_type == "qwen":
        prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    elif fim_type == "deepseek":
        prompt = f"<｜fim begin｜>{prefix}<｜fim hole｜>{suffix}<｜fim end｜>"
    elif fim_type == "starcoder":
        prompt = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
        
    print(f"Prompt: {prompt!r}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Apply StarCoder Fix
    eos_token_id = tokenizer.eos_token_id
    if "starcoder" in model_path.lower():
        print("Applying StarCoder Fix: Adding <file_sep> as stop token...")
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<file_sep>")]
        
    print("Generating...")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=50, 
        do_sample=False, 
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=eos_token_id
    )
    
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    
    print(f"\nOutput: {generated_text!r}")
    
    # Simple verification
    if "return a + b" in generated_text or "result = a + b" in generated_text:
        print("\n✅ SUCCESS: Generated expected code.")
    else:
        print("\n❌ FAILURE: Did not generate expected code.")

def main():
    models = [
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "qwen"),
        ("deepseek-ai/deepseek-coder-1.3b-base", "deepseek"),
        ("bigcode/starcoder2-3b", "starcoder")
    ]
    
    for path, type in models:
        try:
            test_model(path, type)
        except Exception as e:
            print(f"\n❌ ERROR testing {path}: {e}")

if __name__ == "__main__":
    main()
