import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import os
import pickle

model_id = "AntonV/mamba2-130m-hf"

# Configuration
LAYER_IDX = 7
PROBE_TEXT = "The password is '"
TARGETS = ["BLUE", "RED", "GREEN", "ORANGE", "YELLOW", "BLACK", "WHITE", "PURPLE", "GOLD", "SILVER"]
OUTPUT_FILE = "ALSI/data/phi_dataset.pkl"

def optimize_control_delta(model, tokenizer, h_prev_cache, target_str, steps=200, lr=1.0):
    target_id = tokenizer.encode(target_str, add_special_tokens=False)[0]
    print(f"Optimizing for target: {target_str} (ID: {target_id})")
    
    # Setup
    base_state_shape = h_prev_cache.ssm_states[LAYER_IDX].shape
    delta = torch.zeros(base_state_shape, device=model.device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    # Input for forward pass: The last token of probe ("'")
    # We inject into h_prev, then process "'", which yields logits for NEXT token (target)
    last_token_id = tokenizer(PROBE_TEXT, return_tensors="pt").input_ids[:, -1:]
    cache_pos = torch.tensor([tokenizer(PROBE_TEXT, return_tensors="pt").input_ids.shape[1]-1], device=model.device)
    
    best_rank = 99999
    final_delta = None
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Inject Delta
        base_states_tensor = h_prev_cache.ssm_states.detach()
        layers_list = [base_states_tensor[i] for i in range(model.config.num_hidden_layers)]
        layers_list[LAYER_IDX] = layers_list[LAYER_IDX] + delta
        injected_ssm_states = torch.stack(layers_list)
        
        # Mock Cache
        class MockCache:
            def __init__(self, ssm, conv):
                self.ssm_states = ssm
                self.conv_states = conv
                self.config = model.config
                self.conv_kernel_size = model.config.conv_kernel
            def update_ssm_state(self, layer_idx, new_ssm_state, cache_init=False): return
            def update_conv_state(self, layer_idx, new_conv_state, cache_init=False): return
            
        diff_cache = MockCache(injected_ssm_states, h_prev_cache.conv_states)
        
        # Forward
        outputs = model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
        logits = outputs.logits[0, -1]
        
        target_logit = logits[target_id]
        
        # Loss: Cross Entropy (forces relative probability, preventing global lift)
        loss = torch.nn.functional.cross_entropy(logits.view(1, -1), torch.tensor([target_id], device=model.device)) + 1e-5 * delta.norm()
        
        loss.backward()
        optimizer.step()
        
        # Check Rank
        with torch.no_grad():
            rank = (logits > target_logit).sum().item() + 1
            if rank < best_rank:
                best_rank = rank
            
            if rank <= 5:
                # Early success
                final_delta = delta.detach().cpu()
                print(f"  Success at step {step}: Rank {rank}")
                return final_delta, target_id, True
                
    print(f"  Finished {steps} steps. Best Rank: {best_rank}. Final Loss: {loss.item():.4f}")
    # Return even if not perfect, but mark success flag
    final_delta = delta.detach().cpu()
    success = best_rank <= 10 # Relaxed threshold for dataset inclusion
    return final_delta, target_id, success

def generate_dataset():
    os.makedirs("ALSI/data", exist_ok=True)
    
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    for param in model.parameters():
        param.requires_grad = False
        
    # 1. Get h_prev (Context State)
    probe_ids = tokenizer(PROBE_TEXT, return_tensors="pt").input_ids
    context_ids = probe_ids[:, :-1]
    
    with torch.no_grad():
        out_prev = model(context_ids, use_cache=True)
    h_prev_cache = out_prev.cache_params
    
    # Store h_prev (target layer only)
    h_prev_tensor = h_prev_cache.ssm_states[LAYER_IDX].detach().cpu()
    
    dataset = []
    
    # 2. Optimize for each target
    for target in TARGETS:
        delta, t_id, success = optimize_control_delta(model, tokenizer, h_prev_cache, target)
        if success:
            dataset.append({
                "h_prev": h_prev_tensor,
                "target_id": t_id,
                "delta": delta,
                "target_str": target
            })
            print(f"  Added {target} to dataset.")
        else:
            print(f"  Discarded {target} (Rank too low).")
            
    # Save
    print(f"Saving {len(dataset)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    generate_dataset()
