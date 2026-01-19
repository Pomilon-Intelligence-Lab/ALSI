import json

# Define the cells separately to avoid escaping nightmare
cell1_md = "# ALSI: Phi Projector Training (Phase 2)\n\nThis notebook trains a non-linear projector (Phi) to perform Augmented Latent State Injection (ALSI) on Mamba-2.\n\n## 1. Setup Environment"

cell2_code = "!pip install -q transformers accelerate einops"

cell3_md = "## 2. Imports and Configuration"

cell4_code = """import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pickle

model_id = \"AntonV/mamba2-130m-hf\"
LAYER_IDX = 7
PROBE_TEXT = \"The password is '\""
TARGETS = [\"BLUE\", \"RED\", \"GREEN\", \"ORANGE\", \"YELLOW\", \"BLACK\", \"WHITE\", \"PURPLE\", \"GOLD\", \"SILVER\"]
DATA_DIR = \"./ALSI_data\"
os.makedirs(DATA_DIR, exist_ok=True)

device = \"cuda\" if torch.cuda.is_available() else \"cpu\"
print(f\"Using device: {device}\")"""

cell5_md = "## 3. Dataset Generation (Ground Truth Optimization)\n\nWe first find the 'Golden Deltas' for each target via direct optimization."

cell6_code = """def optimize_control_delta(model, tokenizer, h_prev_cache, target_str, steps=200, lr=1.0):
    target_id = tokenizer.encode(target_str, add_special_tokens=False)[0]
    print(f\"Optimizing for target: {target_str} (ID: {target_id})")
    
    base_state_shape = h_prev_cache.ssm_states[LAYER_IDX].shape
    delta = torch.zeros(base_state_shape, device=model.device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)
    
    last_token_id = tokenizer(PROBE_TEXT, return_tensors=\"pt\").input_ids[:, -1:].to(device)
    cache_pos = torch.tensor([tokenizer(PROBE_TEXT, return_tensors=\"pt\").input_ids.shape[1]-1], device=model.device)
    
    best_rank = 99999
    
    for step in range(steps):
        optimizer.zero_grad()
        
        base_states_tensor = h_prev_cache.ssm_states.detach()
        layers_list = [base_states_tensor[i] for i in range(model.config.num_hidden_layers)]
        layers_list[LAYER_IDX] = layers_list[LAYER_IDX] + delta
        injected_ssm_states = torch.stack(layers_list)
        
        class MockCache:
            def __init__(self, ssm, conv):
                self.ssm_states = ssm
                self.conv_states = conv
                self.config = model.config
                self.conv_kernel_size = model.config.conv_kernel
            def update_ssm_state(self, layer_idx, new_ssm_state, cache_init=False): return
            def update_conv_state(self, layer_idx, new_conv_state, cache_init=False): return
            
        diff_cache = MockCache(injected_ssm_states, h_prev_cache.conv_states)
        outputs = model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
        logits = outputs.logits[0, -1]
        
        # Cross Entropy Loss
        loss = torch.nn.functional.cross_entropy(logits.view(1, -1), torch.tensor([target_id], device=model.device)) + 1e-5 * delta.norm()
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            rank = (logits > logits[target_id]).sum().item() + 1
            if rank < best_rank: best_rank = rank
            if rank <= 5: return delta.detach().cpu(), target_id, True
                
    return delta.detach().cpu(), target_id, best_rank <= 10

def generate_dataset(model, tokenizer):
    probe_ids = tokenizer(PROBE_TEXT, return_tensors=\"pt\").input_ids.to(device)
    context_ids = probe_ids[:, :-1]
    with torch.no_grad():
        out_prev = model(context_ids, use_cache=True)
    h_prev_cache = out_prev.cache_params
    
    h_prev_tensor = h_prev_cache.ssm_states[LAYER_IDX].detach().cpu()
    dataset = []
    for target in TARGETS:
        delta, t_id, success = optimize_control_delta(model, tokenizer, h_prev_cache, target)
        if success:
            dataset.append({\"h_prev\": h_prev_tensor, \"target_id\": t_id, \"delta\": delta, \"target_str\": target})
    return dataset"""

cell7_md = "## 4. Phi Projector Architecture"

cell8_code = """class PhiProjector(nn.Module):
    def __init__(self, state_dim, embed_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, h_prev, target_embed):
        x = torch.cat([h_prev, target_embed], dim=-1)
        return self.net(x)"""

cell9_md = "## 5. Main Execution Loop"

cell10_code = """print(f\"Loading model {model_id}...\")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 1. Generate Dataset
dataset = generate_dataset(model, tokenizer)
print(f\"Generated {len(dataset)} samples.\")

# 2. Initialize Phi
embedding_layer = model.get_input_embeddings()
sample_h = dataset[0]['h_prev']
state_dim = sample_h.numel()
embed_dim = embedding_layer.weight.shape[1]
phi = PhiProjector(state_dim, embed_dim).to(device)
optimizer = optim.Adam(phi.parameters(), lr=1e-4)

# 3. Training Loop
probe_ids = tokenizer(PROBE_TEXT, return_tensors=\"pt\").input_ids.to(device)
context_ids = probe_ids[:, :-1]
last_token_id = probe_ids[:, -1:]
with torch.no_grad():
    out_prev = model(context_ids, use_cache=True)
base_cache_struct = out_prev.cache_params

for epoch in range(101):
    total_l = 0
    for sample in dataset:
        optimizer.zero_grad()
        t_id = sample['target_id']
        gt_delta = sample['delta'].to(device)
        h_prev = sample['h_prev'].to(device).view(1, -1)
        t_embed = embedding_layer(torch.tensor([[t_id]], device=device)).view(1, -1)
        
        pred_delta_flat = phi(h_prev, t_embed)
        pred_delta = pred_delta_flat.view(sample_h.shape)
        
        # Match Loss
        l_match = nn.functional.mse_loss(pred_delta, gt_delta)
        
        # Control Loss
        base_states = base_cache_struct.ssm_states.detach().clone()
        layers_list = [base_states[i] for i in range(model.config.num_hidden_layers)]
        layers_list[LAYER_IDX] = layers_list[LAYER_IDX] + pred_delta
        
        class MockCache: # Re-defined for training
            def __init__(self, ssm, conv): self.ssm_states, self.conv_states, self.config, self.conv_kernel_size = ssm, conv, model.config, model.config.conv_kernel
            def update_ssm_state(self, *args, **kwargs): pass
            def update_conv_state(self, *args, **kwargs): pass
            
        diff_cache = MockCache(torch.stack(layers_list), base_cache_struct.conv_states)
        outputs = model(last_token_id, cache_params=diff_cache, cache_position=torch.tensor([context_ids.shape[1]], device=device))
        l_control = nn.functional.cross_entropy(outputs.logits[0, -1].view(1, -1), torch.tensor([t_id], device=device))
        
        loss = l_control + 1.0 * l_match
        loss.backward()
        optimizer.step()
        total_l += loss.item()
        
    if epoch % 20 == 0: print(f"Epoch {epoch}: Loss {total_l/len(dataset):.4f}")

# 4. Final Evaluation
phi.eval()
with torch.no_grad():
    for sample in dataset:
        t_embed = embedding_layer(torch.tensor([[sample['target_id']]], device=device)).view(1, -1)
        pred_delta = phi(sample['h_prev'].to(device).view(1, -1), t_embed).view(sample_h.shape)
        base_states = base_cache_struct.ssm_states.detach().clone()
        layers_list = [base_states[i] for i in range(model.config.num_hidden_layers)]
        layers_list[LAYER_IDX] = layers_list[LAYER_IDX] + pred_delta
        diff_cache = MockCache(torch.stack(layers_list), base_cache_struct.conv_states)
        logits = model(last_token_id, cache_params=diff_cache, cache_position=torch.tensor([context_ids.shape[1]], device=device)).logits[0, -1]
        rank = (logits > logits[sample['target_id']]).sum().item() + 1
        print(f"Target: {sample['target_str']} | Final Rank: {rank}")"""

def make_cell(type, source):
    return {"cell_type": type, "metadata": {}, "outputs": [], "source": [s + "\n" for s in source.split("\n")]}

nb = {
    "cells": [
        make_cell("markdown", cell1_md),
        make_cell("code", cell2_code),
        make_cell("markdown", cell3_md),
        make_cell("code", cell4_code),
        make_cell("markdown", cell5_md),
        make_cell("code", cell6_code),
        make_cell("markdown", cell7_md),
        make_cell("code", cell8_code),
        make_cell("markdown", cell9_md),
        make_cell("code", cell10_code)
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.12.11"}
    },
    "nbformat": 4, "nbformat_minor": 4
}

with open("ALSI_Phi_Training.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Notebook generated successfully.")