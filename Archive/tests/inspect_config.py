from transformers import AutoConfig
import torch

model_id = "AntonV/mamba2-130m-av"

def inspect_config():
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print("Config:")
        print(config)
        
        # Try to see what the model thinks its architecture is
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        print("\nModel Structure (simplified):")
        for name, param in model.named_parameters():
            if "conv1d.bias" in name:
                print(f"{name}: {param.shape}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_config()
