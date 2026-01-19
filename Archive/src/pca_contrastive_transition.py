import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import numpy as np

model_id = "AntonV/mamba2-130m-hf"

# Contrastive Targets
TARGET_TOKENS = ["BLUE", "RED", "GREEN", "ORANGE", "YELLOW", "BLACK", "WHITE", "PURPLE"]

def collect_contrastive_deltas(model, tokenizer, layer_idx=7):
    print(f"Collecting contrastive deltas for targets: {TARGET_TOKENS}")
    
    probe_text = "The password is '"
    probe_ids = tokenizer(probe_text, return_tensors="pt").input_ids
    
    # Context: "The password is " (without the quote)
    context_ids = probe_ids[:, :-1]
    
    # 1. Get h_prev (State before last token)
    with torch.no_grad():
        out_prev = model(context_ids, use_cache=True)
    h_prev_cache = out_prev.cache_params
    # Clone the specific layer state to avoid modification
    h_prev = h_prev_cache.ssm_states[layer_idx].clone()
    
    # 2. Collect delta for each target
    raw_deltas = {}
    
    print("Computing individual token updates...")
    for t_str in TARGET_TOKENS:
        t_id = tokenizer.encode(t_str, add_special_tokens=False)[0]
        t_tensor = torch.tensor([[t_id]], device=model.device)
        
        # We need to process 't' starting from h_prev
        # Note: We must be careful not to mutate h_prev_cache in place for other iterations
        # But Mamba2Cache updates in place.
        # So we need to deep copy cache for each run.
        
        # Helper to copy cache
        # We only need the conv state to be consistent? 
        # Actually, to get strictly correct delta, we need identical starting conditions.
        
        # Simpler: Just re-run context? No, that's slow.
        # Clone the cache tensors.
        curr_ssm = h_prev_cache.ssm_states.clone()
        curr_conv = h_prev_cache.conv_states.clone()
        
        # Mock cache object
        class MockCache:
            def __init__(self, ssm, conv):
                self.ssm_states = ssm
                self.conv_states = conv
                self.config = model.config
                self.conv_kernel_size = model.config.conv_kernel
                self.seq_len_offset = 0 
                self.dtype = ssm.dtype
            def update_ssm_state(self, layer_idx, new_ssm_state, cache_init=False): return
            def update_conv_state(self, layer_idx, new_conv_state, cache_init=False): return
        
        temp_cache = MockCache(curr_ssm, curr_conv)
        
        # Run forward for token 't'
        # cache_position: index is len(context_ids)
        cache_pos = torch.tensor([context_ids.shape[1]], device=model.device)
        
        with torch.no_grad():
            _ = model(t_tensor, cache_params=temp_cache, cache_position=cache_pos)
            
        # h_after is now in temp_cache.ssm_states[layer_idx]
        h_after = temp_cache.ssm_states[layer_idx]
        
        # Delta = h_after - h_prev
        delta = h_after - h_prev
        raw_deltas[t_str] = delta
        
    # 3. Build Contrastive Pairs
    print("Building contrastive pairs...")
    contrastive_vectors = []
    
    token_list = list(raw_deltas.keys())
    for i in range(len(token_list)):
        for j in range(len(token_list)):
            if i == j: continue
            
            t1 = token_list[i]
            t2 = token_list[j]
            
            # Contrast: Δh_i - Δh_j
            diff = raw_deltas[t1] - raw_deltas[t2]
            contrastive_vectors.append(diff.view(1, -1))
            
    X = torch.cat(contrastive_vectors, dim=0)
    print(f"Collected contrastive matrix shape: {X.shape}")
    return X, h_prev_cache

def compute_pca(X, k=16):
    print(f"Computing PCA on Contrastive Deltas (k={k})...")
    # rank might be less than k
    k_actual = min(k, X.shape[0], X.shape[1])
    
    mean = X.mean(dim=0)
    X_centered = X - mean
    
    U, S, V = torch.pca_lowrank(X_centered, q=k_actual, center=False, niter=10)
    return V.to(X.device), mean.to(X.device)

def run_contrastive_search():
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    for param in model.parameters():
        param.requires_grad = False
        
    # 1. Build Contrastive Basis
    # Note: collect_contrastive_deltas returns X and the cache state used as baseline
    target_layer = 7
    X, h_prev_cache = collect_contrastive_deltas(model, tokenizer, layer_idx=target_layer)
    
    k = min(16, X.shape[0])
    pca_basis, pca_mean = compute_pca(X, k=k)
    
    # 2. Setup Optimization Target
    # We want to steer towards "BLUE"
    target_str = "BLUE"
    target_id = tokenizer.encode(target_str, add_special_tokens=False)[0]
    print(f"Target Token: {target_str} (ID: {target_id})")
    
    probe_text = "The password is '"
    probe_ids = tokenizer(probe_text, return_tensors="pt").input_ids
    last_token_id = probe_ids[:, -1:] # "'"
    
    # Baseline Metrics
    # Run forward with NO injection
    cache_pos = torch.tensor([probe_ids.shape[1]-1], device=model.device) # Index of last token
    
    # We need a fresh cache copy because the previous function might have mutated the original object references
    # But collect_contrastive_deltas returned h_prev_cache which was the baseline
    
    with torch.no_grad():
        # Check baseline logits
        # We need to make sure we use the same cache state
        # The cache returned is clean? No, collect_contrastive_deltas cloned it.
        # h_prev_cache is the state after "The password is "
        
        # We construct a wrapper
        class InitialCache:
            def __init__(self, ssm, conv):
                self.ssm_states = ssm
                self.conv_states = conv
                self.config = model.config
                self.conv_kernel_size = model.config.conv_kernel
            def update_ssm_state(self, layer_idx, new_ssm_state, cache_init=False): return
            def update_conv_state(self, layer_idx, new_conv_state, cache_init=False): return
            
        # Clone tensors for safety
        base_ssm = h_prev_cache.ssm_states.clone()
        base_conv = h_prev_cache.conv_states.clone()
        
        temp_cache = InitialCache(base_ssm, base_conv)
        
        outputs = model(last_token_id, cache_params=temp_cache, cache_position=cache_pos)
        logits = outputs.logits[0, -1]
        
        base_logit = logits[target_id].item()
        base_prob = torch.softmax(logits, dim=-1)[target_id].item()
        base_rank = (logits > base_logit).sum().item() + 1
        
    print(f"Baseline: Logit={base_logit:.4f} | Prob={base_prob:.6f} | Rank={base_rank}")
    
    base_state_shape = h_prev_cache.ssm_states[target_layer].shape
    
    # 3. Optimize z
    print(f"Optimizing z in {k}-dim contrastive subspace...")
    
    z = torch.zeros((k,), device=model.device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=1.0)
    
    for step in range(300):
        optimizer.zero_grad()
        
        # Construct Delta
        delta_flat = torch.matmul(pca_basis, z)
        delta = delta_flat.view(base_state_shape)
        
        # Inject into h_prev
        base_states_tensor = h_prev_cache.ssm_states.detach() # (L, B, H, D, S)
        
        # We have to reconstruct the stack manually to be differentiable
        layers_list = [base_states_tensor[i] for i in range(model.config.num_hidden_layers)]
        layers_list[target_layer] = layers_list[target_layer] + delta
        injected_ssm_states = torch.stack(layers_list)
        
        diff_cache = InitialCache(injected_ssm_states, h_prev_cache.conv_states)
        
        # Forward Last Token
        outputs = model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
        logits = outputs.logits[0, -1]
        
        target_val = logits[target_id]
        
        # Loss: -Logit + Regularization
        # User requested lambda * ||z||. 
        # Previous experiment used 0.001 * z.norm()
        loss = -target_val + 0.001 * z.norm()
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                prob = probs[target_id].item()
                rank = (logits > target_val).sum().item() + 1
                d_norm = delta.norm().item()
                z_norm = z.norm().item()
                print(f"Step {step}: Logit={target_val.item():.4f} | Prob={prob:.6f} | Rank={rank} | zNorm={z_norm:.4f} | dNorm={d_norm:.4f}")

    # 4. Final Result
    print("\n--- Final Results ---")
    with torch.no_grad():
        delta_flat = torch.matmul(pca_basis, z)
        delta = delta_flat.view(base_state_shape)
        
        layers_list = [h_prev_cache.ssm_states[i] for i in range(model.config.num_hidden_layers)]
        layers_list[target_layer] = layers_list[target_layer] + delta
        final_ssm = torch.stack(layers_list)
        
        final_cache = InitialCache(final_ssm, h_prev_cache.conv_states)
        
        outputs = model(last_token_id, cache_params=final_cache, cache_position=cache_pos)
        logits = outputs.logits[0, -1]
        
        final_logit = logits[target_id].item()
        final_prob = torch.softmax(logits, dim=-1)[target_id].item()
        final_rank = (logits > final_logit).sum().item() + 1
        
        # Top 5
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5)
        top_tokens = [tokenizer.decode(idx) for idx in top_indices]
        
        print(f"Rank(BLUE)={final_rank} (Baseline: {base_rank})")
        print(f"Prob(BLUE)={final_prob:.6f} (Baseline: {base_prob:.6f})")
        print(f"Logit(BLUE)={final_logit:.4f}")
        print(f"Top-5: {', '.join(top_tokens)}")
        print(f"Final Delta Norm: {delta.norm().item():.4f}")

if __name__ == "__main__":
    run_contrastive_search()
