import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import numpy as np
import matplotlib.pyplot as plt
import copy

model_id = "AntonV/mamba2-130m-hf"

def get_state_at_end(model, input_ids):
    """Runs the model and returns the cache after processing input_ids."""
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    return outputs.cache_params

def copy_cache(cache, config):
    """Deep copies a Mamba2Cache."""
    # We need to create a new cache with the same config and shapes
    # cache.ssm_states shape: (n_layers, batch, n_heads, head_dim, d_state)
    new_cache = Mamba2Cache(config, cache.ssm_states.shape[1], device=cache.ssm_states.device, dtype=cache.ssm_states.dtype)
    new_cache.ssm_states = cache.ssm_states.clone()
    new_cache.conv_states = cache.conv_states.clone()
    return new_cache

def run_sensitivity_scan():
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Define stimuli
    # Fact: Context defining the secret
    fact_text = "Important security update: The password to open the main vault is 'BLUE-SKY-99'. Do not share this."
    # Probe: A query asking for the secret
    probe_text = "The guard asked for the vault password. I replied that the password is '"
    # Target: The answer
    target_text = "BLUE-SKY-99'."
    
    fact_ids = tokenizer(fact_text, return_tensors="pt").input_ids
    probe_ids = tokenizer(probe_text, return_tensors="pt").input_ids
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids
    
    print(f"Fact: {fact_text}")
    print(f"Probe: {probe_text}")
    print(f"Target: {target_text}")

    # 1. Get States
    print("Capturing states...")
    # State A: Natural state after seeing the probe (The recipient of the graft)
    probe_cache = get_state_at_end(model, probe_ids)
    
    # State B: State after seeing the FULL fact (The donor)
    # Ideally, we want the state *right before* the target tokens if we were cheating,
    # but here we are simulating "recalling" the fact. 
    # Let's take the state at the END of the fact.
    fact_cache = get_state_at_end(model, fact_ids)
    
    # 2. Baseline Performance
    # Run target_ids starting from probe_cache
    # We need to set cache_position correctly for the CONTINUATION
    # The next token is at index len(probe_ids)
    start_pos = probe_ids.shape[1]
    cache_pos = torch.arange(start_pos, start_pos + target_ids.shape[1], device=model.device)
    
    with torch.no_grad():
        baseline_outputs = model(target_ids, cache_params=copy_cache(probe_cache, model.config), cache_position=cache_pos)
        baseline_loss = torch.nn.functional.cross_entropy(
            baseline_outputs.logits.view(-1, baseline_outputs.logits.size(-1)), 
            target_ids.view(-1)
        ).item()
    
    print(f"Baseline Loss (Natural Continuation): {baseline_loss:.4f}")

    # 3. Layer-wise Grafting Scan
    num_layers = model.config.num_hidden_layers
    fact_losses = []
    random_losses = []
    
    g = 0.5 # Grafting strength (Mixing coefficient)
    
    print(f"Starting scan with g={g}...")
    
    for L in range(num_layers):
        # --- Condition A: Factual Graft ---
        # h_new = (1-g)*h_probe + g*h_fact
        # Note: We must be careful about shapes. 
        # ssm_states: [layers, batch, heads, head_dim, state_size]
        
        grafted_cache = copy_cache(probe_cache, model.config)
        
        # We need the donor state. 
        # CAUTION: fact_cache might have different temporal length implications if we aren't careful,
        # but Mamba states are fixed size per layer/head. They represent the "summary so far".
        # So we can directly mix them.
        
        h_probe = probe_cache.ssm_states[L]
        h_fact = fact_cache.ssm_states[L]
        
        # Normalize h_fact to match scale of h_probe roughly? 
        # Or just mix. Let's trust the mixing for now as they are from same model.
        
        grafted_state = (1 - g) * h_probe + g * h_fact
        grafted_cache.ssm_states[L] = grafted_state
        
        # We also need to decide what to do with conv_states. 
        # Usually SSM state is the long-term memory. Conv is local.
        # Let's graft ONLY SSM state to test long-term recall injection.
        
        with torch.no_grad():
            outputs = model(target_ids, cache_params=grafted_cache, cache_position=cache_pos)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)), 
                target_ids.view(-1)
            ).item()
        fact_losses.append(loss)
        
        # --- Condition B: Random Control ---
        # Inject random noise instead of fact
        # h_new = (1-g)*h_probe + g*Noise
        
        rand_cache = copy_cache(probe_cache, model.config)
        
        # Generate noise with same mean/std as h_fact to be fair
        noise = torch.randn_like(h_fact) * h_fact.std() + h_fact.mean()
        
        rand_state = (1 - g) * h_probe + g * noise
        rand_cache.ssm_states[L] = rand_state
        
        with torch.no_grad():
            outputs_rand = model(target_ids, cache_params=rand_cache, cache_position=cache_pos)
            loss_rand = torch.nn.functional.cross_entropy(
                outputs_rand.logits.view(-1, outputs_rand.logits.size(-1)), 
                target_ids.view(-1)
            ).item()
        random_losses.append(loss_rand)
        
        print(f"Layer {L:02d}: Fact={loss:.4f} | Rand={loss_rand:.4f} | Base={baseline_loss:.4f}")

    # 4. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_layers), fact_losses, 'b-o', label='Factual Graft')
    plt.plot(range(num_layers), random_losses, 'k--x', label='Random Graft (Control)')
    plt.axhline(y=baseline_loss, color='r', linestyle=':', label='Baseline (Natural)')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Target Cross-Entropy Loss')
    plt.title(f'ALSI Sensitivity Scan (g={g})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ALSI/sensitivity_curve.png')
    print("Saved plot to ALSI/sensitivity_curve.png")

if __name__ == "__main__":
    run_sensitivity_scan()
