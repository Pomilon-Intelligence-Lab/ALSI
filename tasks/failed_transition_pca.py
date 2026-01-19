from core.base_task import Task
from core.utils import MockCache
import torch
import numpy as np
import os

class FailedTransitionPCA(Task):
    def __init__(self):
        super().__init__("FailedTransitionPCA")
        self.texts = [
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
        
    def collect_deltas(self):
        print(f"[{self.name}] Collecting natural transitions...")
        deltas = []
        layer_idx = 7
        for text in self.texts:
            tokens = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            if tokens.shape[1] < 2: continue
            
            input_before = tokens[:, :-1]
            with torch.no_grad():
                out_before = self.model(input_before, use_cache=True)
            h_before = out_before.cache_params.ssm_states[layer_idx].clone()
            
            last_token = tokens[:, -1:]
            cache_pos = torch.tensor([input_before.shape[1]], device=self.device)
            with torch.no_grad():
                _ = self.model(last_token, cache_params=out_before.cache_params, cache_position=cache_pos)
            h_after = out_before.cache_params.ssm_states[layer_idx]
            
            deltas.append((h_after - h_before).view(1, -1))
        return torch.cat(deltas, dim=0)

    def run(self):
        X = self.collect_deltas()
        k = min(16, X.shape[0])
        print(f"[{self.name}] Computing PCA (k={k})...")
        mean = X.mean(dim=0)
        U, S, V = torch.pca_lowrank(X - mean, q=k)
        pca_basis = V.to(self.device)
        
        # Setup steering
        target_str = "BLUE"
        target_id = self.tokenizer.encode(target_str, add_special_tokens=False)[0]
        probe_text = "The password is '"
        probe_ids = self.tokenizer(probe_text, return_tensors="pt").input_ids.to(self.device)
        context_ids_prev = probe_ids[:, :-1]
        last_token_id = probe_ids[:, -1:]
        
        with torch.no_grad():
            out_prev = self.model(context_ids_prev, use_cache=True)
        h_prev_cache = out_prev.cache_params
        
        z = torch.zeros((k,), device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=1.0)
        
        print(f"[{self.name}] Optimizing logit lift for '{target_str}'...")
        for step in range(101):
            optimizer.zero_grad()
            delta = torch.matmul(pca_basis, z).view(h_prev_cache.ssm_states[7].shape)
            
            base_states = h_prev_cache.ssm_states.detach().clone()
            layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
            layers[7] = layers[7] + delta
            
            diff_cache = MockCache(torch.stack(layers), h_prev_cache.conv_states, self.model.config)
            out = self.model(last_token_id, cache_params=diff_cache, cache_position=torch.tensor([context_ids_prev.shape[1]], device=self.device))
            
            logit = out.logits[0, -1, target_id]
            loss = -logit + 0.001 * z.norm()
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                rank = (out.logits[0, -1] > logit).sum().item() + 1
                print(f"  Step {step}: Logit={logit.item():.2f}, Rank={rank}")
        
        self.final_rank = rank
        print(f"[{self.name}] Final Rank: {self.final_rank} (Failed to achieve low rank)")

    def report(self):
        pass
