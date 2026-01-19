from core.base_task import Task
from core.utils import MockCache
import torch
import numpy as np
import os

class FailedContrastivePCA(Task):
    def __init__(self):
        super().__init__("FailedContrastivePCA")
        self.targets = ["BLUE", "RED", "GREEN", "ORANGE", "YELLOW", "BLACK"]
        
    def collect_contrastive_deltas(self):
        print(f"[{self.name}] Collecting contrastive transitions...")
        probe_text = "The password is '"
        probe_ids = self.tokenizer(probe_text, return_tensors="pt").input_ids.to(self.device)
        context_ids = probe_ids[:, :-1]
        
        with torch.no_grad():
            out_prev = self.model(context_ids, use_cache=True)
        h_prev_cache = out_prev.cache_params
        h_prev = h_prev_cache.ssm_states[7].clone()
        
        deltas = {}
        for t_str in self.targets:
            t_id = self.tokenizer.encode(t_str, add_special_tokens=False)[0]
            # Mock update
            curr_ssm = h_prev_cache.ssm_states.clone()
            temp_cache = MockCache(curr_ssm, h_prev_cache.conv_states.clone(), self.model.config)
            with torch.no_grad():
                _ = self.model(torch.tensor([[t_id]], device=self.device), 
                               cache_params=temp_cache, 
                               cache_position=torch.tensor([context_ids.shape[1]], device=self.device))
            deltas[t_str] = temp_cache.ssm_states[7] - h_prev
            
        vectors = []
        for i in range(len(self.targets)):
            for j in range(len(self.targets)):
                if i != j:
                    vectors.append((deltas[self.targets[i]] - deltas[self.targets[j]]).view(1, -1))
        return torch.cat(vectors, dim=0), h_prev_cache

    def run(self):
        X, h_prev_cache = self.collect_contrastive_deltas()
        k = min(16, X.shape[0])
        print(f"[{self.name}] Computing PCA (k={k})...")
        mean = X.mean(dim=0)
        U, S, V = torch.pca_lowrank(X - mean, q=k)
        pca_basis = V.to(self.device)
        
        target_str = "BLUE"
        target_id = self.tokenizer.encode(target_str, add_special_tokens=False)[0]
        last_token_id = self.tokenizer("'", return_tensors="pt").input_ids.to(self.device)
        
        z = torch.zeros((k,), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=1.0)
        
        print(f"[{self.name}] Attempting steering in contrastive subspace...")
        for step in range(101):
            optimizer.zero_grad()
            delta = torch.matmul(pca_basis, z).view(h_prev_cache.ssm_states[7].shape)
            
            base_states = h_prev_cache.ssm_states.detach().clone()
            layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
            layers[7] = layers[7] + delta
            
            diff_cache = MockCache(torch.stack(layers), h_prev_cache.conv_states, self.model.config)
            out = self.model(last_token_id, cache_params=diff_cache, 
                             cache_position=torch.tensor([h_prev_cache.ssm_states.shape[1]], device=self.device))
            
            logit = out.logits[0, -1, target_id]
            loss = -logit + 0.001 * z.norm()
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"  Step {step}: Logit={logit.item():.2f}")
        
        self.final_logit = logit.item()
        print(f"[{self.name}] Final Logit: {self.final_logit} (No movement detected)")

    def report(self):
        pass
