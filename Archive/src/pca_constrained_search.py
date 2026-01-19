import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import matplotlib.pyplot as plt
import numpy as np

model_id = "AntonV/mamba2-130m-hf"

def get_more_texts(n=50):
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "I love programming in Python.",
        "Artificial intelligence is the future.",
        "The weather is nice today.",
        "Paris is the capital of France.",
        "The cat sat on the mat.",
        "Hello world, this is a test.",
        "Learning to fly is my dream.",
        "Music is the food of love."
    ]
    # Simple augmentation
    texts = []
    for i in range(n):
        t = base_texts[i % len(base_texts)]
        texts.append(f"{t} {i}") # Make unique
    return texts

def collect_natural_states(model, tokenizer, layer_idx=7, n_samples=50):
    texts = get_more_texts(n_samples)
    print(f"Collecting natural states from {len(texts)} samples...")
    states = []
    
    for i, text in enumerate(texts):
        if i % 10 == 0: print(f"Processing {i}/{len(texts)}...")
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs.input_ids, use_cache=True)
        state = outputs.cache_params.ssm_states[layer_idx]
        flat_state = state.view(1, -1)
        states.append(flat_state)
        
    X = torch.cat(states, dim=0)
    print(f"Collected data shape: {X.shape}")
    return X

def compute_pca(X, k=32):
    print(f"Computing PCA (k={k})...")
    mean = X.mean(dim=0)
    X_centered = X - mean
    U, S, V = torch.pca_lowrank(X_centered, q=k, center=False, niter=10)
    return V.to(X.device), mean.to(X.device)

def run_pca_search():
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    for param in model.parameters():
        param.requires_grad = False
        
    # 1. Build PCA Basis
    X = collect_natural_states(model, tokenizer, layer_idx=7, n_samples=40)
    k = 32
    pca_basis, pca_mean = compute_pca(X, k=k)
    
    # 2. Setup
    probe_text = "The password is '"
    target_text = "BLUE-SKY" 
    
    probe_ids = tokenizer(probe_text, return_tensors="pt").input_ids
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids
    
    context_ids = probe_ids[:, :-1]
    last_probe_token = probe_ids[:, -1:]
    with torch.no_grad():
        context_out = model(context_ids, use_cache=True)
    probe_cache = context_out.cache_params
    
    target_layer = 7
    base_state_shape = probe_cache.ssm_states[target_layer].shape 
    base_state_flat = probe_cache.ssm_states[target_layer].flatten()
    print(f"Base State Norm: {base_state_flat.norm().item():.4f}")
    
    full_sequence = torch.cat([last_probe_token, target_ids], dim=1)
    input_seq = full_sequence[:, :-1]
    label_seq = full_sequence[:, 1:]
    
    # 3. Optimize z
    print(f"Optimizing z in {k}-dim subspace...")
    
    # Initialize z. 
    # Try initializing with larger variance?
    # Must be leaf tensor
    z_init = torch.randn((k,), device=model.device) * 0.1
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=1.0)
    
    cache_pos = torch.arange(context_ids.shape[1], context_ids.shape[1] + input_seq.shape[1], device=model.device)
    
    for step in range(500):
        optimizer.zero_grad()
        
        delta_flat = torch.matmul(pca_basis, z)
        delta = delta_flat.view(base_state_shape)
        
        # Inject
        base_states = probe_cache.ssm_states.detach()
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
            
        diff_cache = DifferentiableCache(injected_ssm_states, probe_cache.conv_states)
        
        outputs = model(input_seq, cache_params=diff_cache, cache_position=cache_pos)
        loss = torch.nn.functional.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), label_seq.view(-1))
        
        # No reg for now, just see if we can move the loss
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}: CE={loss.item():.4f} | z-Norm={z.norm().item():.4f} | Delta-Norm={delta.norm().item():.4f}")

    # 4. Verify
    print("Verifying generation...")
    with torch.no_grad():
        delta_flat = torch.matmul(pca_basis, z)
        delta = delta_flat.view(base_state_shape)
        final_layers = [base_states[i] for i in range(model.config.num_hidden_layers)]
        final_layers[target_layer] = final_layers[target_layer] + delta
        final_ssm = torch.stack(final_layers)
        final_cache = DifferentiableCache(final_ssm, probe_cache.conv_states)
        
        out = model.generate(input_ids=last_probe_token, cache_params=final_cache, max_new_tokens=20)
        gen_text = tokenizer.decode(out[0])
        print(f"Generated: {gen_text}")

if __name__ == "__main__":
    run_pca_search()