from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "AntonV/mamba2-130m-hf"

def verify():
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    text = "The quick brown fox"
    inputs = tokenizer(text, return_tensors="pt")
    
    input_ids = inputs.input_ids
    cache_position = torch.arange(input_ids.shape[1])
    
    with torch.no_grad():
        outputs = model(input_ids, cache_position=cache_position, use_cache=True)
    
    cache = outputs.cache_params
    print(f"SSM State shape: {cache.ssm_states.shape}")
    
    # Run one more token
    next_token = tokenizer(" jumps", return_tensors="pt")
    next_token_ids = next_token.input_ids
    next_cache_position = torch.tensor([input_ids.shape[1]])
    
    with torch.no_grad():
        outputs2 = model(next_token_ids, cache_params=cache, cache_position=next_cache_position, use_cache=True)
    
    print("Succesfully used cache for next token.")
    print(f"New SSM State shape: {outputs2.cache_params.ssm_states.shape}")

if __name__ == "__main__":
    verify()
