from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "AntonV/mamba2-130m-hf"

def inspect_cache():
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    input_text = "Hello"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    # We need to use use_cache=True to get the states
    outputs = model(**inputs, use_cache=True)
    print(f"Output attributes: {dir(outputs)}")
    
    # Try to find the cache
    cache = None
    if hasattr(outputs, "past_key_values"):
        cache = outputs.past_key_values
    elif hasattr(outputs, "cache_params"):
        cache = outputs.cache_params
    elif hasattr(outputs, "mamba_cache"): # Some versions might use this
        cache = outputs.mamba_cache
    
    if cache is None:
        print("Could not find cache in outputs.")
        return

    if hasattr(cache, "ssm_states"):
        print(f"SSM States type: {type(cache.ssm_states)}")
        if isinstance(cache.ssm_states, torch.Tensor):
            print(f"SSM States shape: {cache.ssm_states.shape}")
        elif isinstance(cache.ssm_states, list):
            print(f"SSM States is list of length {len(cache.ssm_states)}")
            print(f"First element shape: {cache.ssm_states[0].shape}")
            
    if hasattr(cache, "conv_states"):
        print(f"Conv States type: {type(cache.conv_states)}")
        if isinstance(cache.conv_states, torch.Tensor):
            print(f"Conv States shape: {cache.conv_states.shape}")
        elif isinstance(cache.conv_states, list):
            print(f"Conv States is list of length {len(cache.conv_states)}")
            print(f"First element shape: {cache.conv_states[0].shape}")

if __name__ == "__main__":
    inspect_cache()
