from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "AntonV/mamba2-130m-hf"

def test_model_load():
    try:
        print(f"Loading {model_id}...")
        # trust_remote_code=True is often needed for newer/custom architectures like Mamba2 in transformers
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True) 
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Model loaded successfully.")
        
        input_text = "The goal of Augmented Latent State Injection is"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        print("Running inference...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        
        print("Output:")
        print(tokenizer.decode(outputs[0]))
        return True
        
    except Exception as e:
        print(f"Error loading or running model: {e}")
        return False

if __name__ == "__main__":
    test_model_load()
