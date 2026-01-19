from safetensors.torch import load_file
import os

model_id = "AntonV/mamba2-130m-av"
from huggingface_hub import hf_hub_download

def inspect_safetensors():
    try:
        path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        state_dict = load_file(path)
        print("Shapes in safetensors:")
        for name, tensor in state_dict.items():
            if "conv1d.bias" in name:
                print(f"{name}: {tensor.shape}")
                break # Just need one to verify
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_safetensors()
