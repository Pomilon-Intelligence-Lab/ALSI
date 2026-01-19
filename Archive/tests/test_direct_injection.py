import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

model_id = "AntonV/mamba2-130m-hf"

def copy_cache(cache, config):
    new_cache = Mamba2Cache(config, cache.ssm_states.shape[1], device=cache.ssm_states.device, dtype=cache.ssm_states.dtype)
    new_cache.ssm_states = cache.ssm_states.clone()
    new_cache.conv_states = cache.conv_states.clone()
    return new_cache

def test_direct_injection():
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 1. Get state of JUST the secret
    secret_text = "BLUE-SKY-99"
    secret_ids = tokenizer(secret_text, return_tensors="pt").input_ids
    with torch.no_grad():
        secret_out = model(secret_ids, use_cache=True)
    secret_cache = secret_out.cache_params
    
    # 2. Probe
    probe_text = "The password is '"
    probe_ids = tokenizer(probe_text, return_tensors="pt").input_ids
    with torch.no_grad():
        probe_out = model(probe_ids, use_cache=True)
    probe_cache = probe_out.cache_params
    
    # 3. Target
    target_text = "BLUE-SKY-99'."
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids
    
    # Baseline
    cache_pos = torch.arange(probe_ids.shape[1], probe_ids.shape[1] + target_ids.shape[1], device=model.device)
    with torch.no_grad():
        base_out = model(target_ids, cache_params=copy_cache(probe_cache, model.config), cache_position=cache_pos)
        base_loss = torch.nn.functional.cross_entropy(base_out.logits.view(-1, base_out.logits.size(-1)), target_ids.view(-1)).item()
    print(f"Baseline Loss: {base_loss:.4f}")
    
    # Injection Loop
    g = 0.8 # High graft strength
    print(f"Injecting raw concept state with g={g}...")
    
    for L in range(model.config.num_hidden_layers):
        grafted_cache = copy_cache(probe_cache, model.config)
        
        # Inject secret state
        # h_new = (1-g)h_probe + g*h_secret
        h_probe = probe_cache.ssm_states[L]
        h_secret = secret_cache.ssm_states[L]
        
        grafted_cache.ssm_states[L] = (1-g)*h_probe + g*h_secret
        
        with torch.no_grad():
            out = model(target_ids, cache_params=grafted_cache, cache_position=cache_pos)
            loss = torch.nn.functional.cross_entropy(out.logits.view(-1, out.logits.size(-1)), target_ids.view(-1)).item()
        
        diff = loss - base_loss
        if diff < -0.1:
            print(f"Layer {L:02d}: {loss:.4f} (IMPROVED by {diff:.4f})")
        elif diff > 0.1:
             print(f"Layer {L:02d}: {loss:.4f} (Worsened by {diff:.4f})")
        else:
             print(f"Layer {L:02d}: {loss:.4f} (Neutral)")

if __name__ == "__main__":
    test_direct_injection()
