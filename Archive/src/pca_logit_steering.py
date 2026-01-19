import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import matplotlib.pyplot as plt
import numpy as np

model_id = "AntonV/mamba2-130m-hf"

NATURAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "I love programming in Python.",
    "Artificial intelligence is the future.",
    "The weather is nice today.",
    "Paris is the capital of France.",
    "The cat sat on the mat.",
    "Hello world, this is a test.",
    "Learning to fly is my dream.",
    "Music is the food of love.",
    "Deep learning requires large amounts of data.",
    "The recipe calls for three large eggs.",
    "Unexpected error occurred in the system.",
    "Once upon a time in a galaxy far away.",
    "The stock market saw significant gains today.",
    "Please enter your password to continue.",
    "The universe is expanding at an accelerating rate.",
    "Quantum entanglement is a spooky phenomenon.",
    "Javascript is the language of the web.",
    "Rust provides memory safety guarantees.",
    "Climate change is a pressing global issue.",
    "Photosynthesis converts light into chemical energy.",
    "The mitochondria is the powerhouse of the cell.",
    "Effective communication is key to leadership.",
    "Machine learning models can be biased.",
    "Data structures and algorithms are fundamental.",
    "The history of Rome spans over a thousand years.",
    "Modern art challenges traditional aesthetics.",
    "Renewable energy sources are becoming cheaper.",
    "Space exploration has yielded many technologies."
]

def collect_transition_deltas(model, tokenizer, layer_idx=7):
    print(f"Collecting transition deltas from {len(NATURAL_TEXTS)} samples...")
    deltas = []
    
    for i, text in enumerate(NATURAL_TEXTS):
        if i % 10 == 0: print(f"Processing {i+1}/{len(NATURAL_TEXTS)}...")
        
        tokens = tokenizer(text, return_tensors="pt").input_ids
        if tokens.shape[1] < 2: continue
        
        # h_before: State after processing all but last token
        input_before = tokens[:, :-1]
        with torch.no_grad():
            out_before = model(input_before, use_cache=True)
        h_before = out_before.cache_params.ssm_states[layer_idx].clone()
        
        # h_after: State after processing full sequence
        last_token = tokens[:, -1:]
        cache_pos = torch.tensor([input_before.shape[1]], device=model.device)
        
        with torch.no_grad():
            out_after = model(last_token, cache_params=out_before.cache_params, cache_position=cache_pos)
            
        h_after = out_before.cache_params.ssm_states[layer_idx]
        
        delta = h_after - h_before
        deltas.append(delta.view(1, -1))
        
    X = torch.cat(deltas, dim=0)
    print(f"Collected delta shape: {X.shape}")
    return X

def compute_pca(X, k=30):
    print(f"Computing PCA on Deltas (k={k})...")
    mean = X.mean(dim=0)
    X_centered = X - mean
    k_actual = min(k, X.shape[0])
    U, S, V = torch.pca_lowrank(X_centered, q=k_actual, center=False, niter=10)
    return V.to(X.device), mean.to(X.device)

def run_logit_steering():
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    for param in model.parameters():
        param.requires_grad = False
        
    # 1. Build Delta PCA Basis
    X = collect_transition_deltas(model, tokenizer, layer_idx=7)
    k = min(30, X.shape[0])
    pca_basis, pca_mean = compute_pca(X, k=k)
    
    # 2. Setup Task
    probe_text = "The password is '"
    # We want to boost "BLUE"
    target_token_str = "BLUE"
    target_token_id = tokenizer.encode(target_token_str, add_special_tokens=False)[0]
    
    print(f"Target Token: '{target_token_str}' (ID: {target_token_id})")
    
    probe_ids = tokenizer(probe_text, return_tensors="pt").input_ids
    
    # 3. Get Initial State (h_prev) - State BEFORE last token
    context_ids_prev = probe_ids[:, :-1] # "The password is"
    last_token_id = probe_ids[:, -1:]    # "'"
    
    with torch.no_grad():
        out_prev = model(context_ids_prev, use_cache=True)
    h_prev_cache = out_prev.cache_params
    
    # Baseline Check
    with torch.no_grad():
        base_out = model(last_token_id, cache_params=h_prev_cache, cache_position=torch.tensor([context_ids_prev.shape[1]]))
        base_logits = base_out.logits[0, -1]
        base_target_logit = base_logits[target_token_id].item()
        base_probs = torch.softmax(base_logits, dim=-1)
        base_target_prob = base_probs[target_token_id].item()
        base_rank = (base_logits > base_target_logit).sum().item() + 1
        
    print(f"Baseline: Logit={base_target_logit:.4f}, Prob={base_target_prob:.6f}, Rank={base_rank}")
    
    target_layer = 7
    base_state_shape = h_prev_cache.ssm_states[target_layer].shape
    
    # 4. Optimize z
    print(f"Optimizing z in {k}-dim transition subspace...")
    
    z_init = torch.zeros((k,), device=model.device)
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=1.0)
    
    cache_pos = torch.tensor([context_ids_prev.shape[1]], device=model.device)
    
    for step in range(300):
        optimizer.zero_grad()
        
        # Construct Delta
        delta_flat = torch.matmul(pca_basis, z)
        delta = delta_flat.view(base_state_shape)
        
        # Inject into h_prev
        base_states = h_prev_cache.ssm_states.detach()
        layers_list = [base_states[i] for i in range(model.config.num_hidden_layers)]
        layers_list[target_layer] = layers_list[target_layer] + delta
        injected_ssm_states = torch.stack(layers_list)
        
        class DifferentiableCache:
            def __init__(self, ssm, conv):
                self.ssm_states = ssm
                self.conv_states = conv
                self.config = model.config
                self.conv_kernel_size = model.config.conv_kernel
            def update_ssm_state(self, layer_idx, new_ssm_state, cache_init=False): return
            def update_conv_state(self, layer_idx, new_conv_state, cache_init=False): return
            
        diff_cache = DifferentiableCache(injected_ssm_states, h_prev_cache.conv_states)
        
        # Forward ONLY last token
        outputs = model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
        
        logits = outputs.logits[0, -1]
        target_logit = logits[target_token_id]
        
        # Loss: Maximize target logit
        # Add L2 on z
        loss = -target_logit + 0.001 * z.norm()
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            probs = torch.softmax(logits, dim=-1)
            target_prob = probs[target_token_id].item()
            rank = (logits > target_logit).sum().item() + 1
            print(f"Step {step}: Logit={target_logit.item():.4f} | Prob={target_prob:.6f} | Rank={rank} | z-Norm={z.norm().item():.4f}")

    # 5. Final Report
    print("\n--- Final Results ---")
    with torch.no_grad():
        # Re-run with final z
        delta_flat = torch.matmul(pca_basis, z)
        delta = delta_flat.view(base_state_shape)
        
        final_layers = [base_states[i] for i in range(model.config.num_hidden_layers)]
        final_layers[target_layer] = final_layers[target_layer] + delta
        final_ssm = torch.stack(final_layers)
        final_cache = DifferentiableCache(final_ssm, h_prev_cache.conv_states)
        
        outputs = model(last_token_id, cache_params=final_cache, cache_position=cache_pos)
        logits = outputs.logits[0, -1]
        
        target_logit = logits[target_token_id].item()
        probs = torch.softmax(logits, dim=-1)
        target_prob = probs[target_token_id].item()
        rank = (logits > target_logit).sum().item() + 1
        
        # Top 5
        top_probs, top_indices = torch.topk(probs, 5)
        top_tokens = [tokenizer.decode(idx) for idx in top_indices]
        
        print(f"Rank(BLUE)={rank} | P={target_prob:.6f} | Logit={target_logit:.4f}")
        print(f"Top-5: {', '.join(top_tokens)}")
        print(f"Delta stats: Min={delta.min():.4f}, Max={delta.max():.4f}, Norm={delta.norm().item():.4f}")

if __name__ == "__main__":
    run_logit_steering()
