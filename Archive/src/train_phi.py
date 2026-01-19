import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import pickle
import numpy as np
import os
from phi_model import PhiProjector

model_id = "AntonV/mamba2-130m-hf"
DATASET_PATH = "ALSI/data/phi_dataset.pkl"
LAYER_IDX = 7

def train_phi():
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Freeze Base Model
    for param in model.parameters():
        param.requires_grad = False
        
    embedding_layer = model.get_input_embeddings()
    embed_dim = embedding_layer.weight.shape[1]
    
    # Load Dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)
        
    # Prepare Data
    # h_prev is the same for all samples in this specific dataset (single context)
    # But generally could vary.
    # dataset item: {'h_prev': tensor, 'target_id': int, 'delta': tensor, ...}
    
    # Determine State Dim
    sample_h = dataset[0]['h_prev']
    state_shape = sample_h.shape # (1, 24, 64, 128)
    state_dim = sample_h.numel()
    
    print(f"State Dim: {state_dim}, Embed Dim: {embed_dim}")
    
    # Initialize Phi
    phi = PhiProjector(state_dim, embed_dim, hidden_dim=1024).to(model.device)
    optimizer = optim.Adam(phi.parameters(), lr=1e-4)
    
    # Training Loop
    epochs = 100
    alpha = 1.0 # Match weight
    beta = 1e-5 # Weight decay (handled by optimizer usually, but explicit reg on output norm?)
    
    # Cache Context State (h_prev is static for this experiment)
    # But we need the FULL cache structure for forward pass, not just layer 7
    # We need to recreate the cache object.
    # We can run the probe once to get the baseline cache structure
    probe_text = "The password is '"
    probe_ids = tokenizer(probe_text, return_tensors="pt").input_ids
    context_ids = probe_ids[:, :-1]
    last_token_id = probe_ids[:, -1:]
    
    with torch.no_grad():
        out_prev = model(context_ids, use_cache=True)
    base_cache_struct = out_prev.cache_params
    
    print("Starting Training...")
    
    for epoch in range(epochs):
        total_loss = 0
        total_control = 0
        total_match = 0
        
        # Shuffle? Small dataset, doesn't matter much.
        
        for sample in dataset:
            optimizer.zero_grad()
            
            target_id = sample['target_id']
            gt_delta = sample['delta'].to(model.device)
            h_prev = sample['h_prev'].to(model.device) # (1, 24, 64, 128)
            
            # Get Target Embedding
            target_embed = embedding_layer(torch.tensor([[target_id]], device=model.device)) # (1, 1, 768)
            target_embed_flat = target_embed.view(1, -1)
            
            h_prev_flat = h_prev.view(1, -1)
            
            # Phi Forward
            pred_delta_flat = phi(h_prev_flat, target_embed_flat)
            pred_delta = pred_delta_flat.view(state_shape)
            
            # L_match (MSE)
            l_match = nn.functional.mse_loss(pred_delta, gt_delta)
            
            # L_control (Cross Entropy)
            # Inject pred_delta
            # We need to reconstruct the full layer stack
            
            # NOTE: We can't modify base_cache_struct in place safely if we loop?
            # We should detach/clone base states
            base_states = base_cache_struct.ssm_states.detach().clone()
            layers_list = [base_states[i] for i in range(model.config.num_hidden_layers)]
            layers_list[LAYER_IDX] = layers_list[LAYER_IDX] + pred_delta
            injected_ssm_states = torch.stack(layers_list)
            
            class MockCache:
                def __init__(self, ssm, conv):
                    self.ssm_states = ssm
                    self.conv_states = conv
                    self.config = model.config
                    self.conv_kernel_size = model.config.conv_kernel
                def update_ssm_state(self, layer_idx, new_ssm_state, cache_init=False): return
                def update_conv_state(self, layer_idx, new_conv_state, cache_init=False): return
            
            diff_cache = MockCache(injected_ssm_states, base_cache_struct.conv_states)
            
            cache_pos = torch.tensor([context_ids.shape[1]], device=model.device)
            
            # Forward
            outputs = model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
            logits = outputs.logits[0, -1]
            
            # Cross Entropy on Target
            l_control = nn.functional.cross_entropy(logits.view(1, -1), torch.tensor([target_id], device=model.device))
            
            loss = l_control + alpha * l_match
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_control += l_control.item()
            total_match += l_match.item()
            
        if epoch % 10 == 0:
            avg_loss = total_loss / len(dataset)
            avg_ctl = total_control / len(dataset)
            avg_mat = total_match / len(dataset)
            print(f"Epoch {epoch}: Loss={avg_loss:.4f} (Ctl={avg_ctl:.4f}, Match={avg_mat:.4f})")

    # Evaluation
    print("\n--- Evaluation ---")
    phi.eval()
    with torch.no_grad():
        for sample in dataset:
            target_str = sample['target_str']
            target_id = sample['target_id']
            h_prev = sample['h_prev'].to(model.device)
            target_embed = embedding_layer(torch.tensor([[target_id]], device=model.device)).view(1, -1)
            
            pred_delta = phi(h_prev.view(1, -1), target_embed).view(state_shape)
            
            # Inject
            base_states = base_cache_struct.ssm_states.detach().clone()
            layers_list = [base_states[i] for i in range(model.config.num_hidden_layers)]
            layers_list[LAYER_IDX] = layers_list[LAYER_IDX] + pred_delta
            injected_ssm_states = torch.stack(layers_list)
            diff_cache = MockCache(injected_ssm_states, base_cache_struct.conv_states)
            cache_pos = torch.tensor([context_ids.shape[1]], device=model.device)
            
            outputs = model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
            logits = outputs.logits[0, -1]
            target_val = logits[target_id].item()
            rank = (logits > target_val).sum().item() + 1
            prob = torch.softmax(logits, dim=-1)[target_id].item()
            
            print(f"Target: {target_str} | Rank: {rank} | Prob: {prob:.4f}")
            
    # Save Model
    torch.save(phi.state_dict(), "ALSI/data/phi_model.pt")
    print("Saved Phi model.")

if __name__ == "__main__":
    train_phi()
