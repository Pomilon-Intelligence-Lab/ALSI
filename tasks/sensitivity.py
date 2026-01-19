from core.base_task import Task
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

class SensitivityScan(Task):
    def __init__(self):
        super().__init__("SensitivityScan")
    
    def run(self):
        print(f"[{self.name}] Running sensitivity scan...")
        
        # Stimuli
        fact_text = "Important security update: The password to open the main vault is 'BLUE-SKY-99'. Do not share this."
        probe_text = "The guard asked for the vault password. I replied that the password is '"
        target_text = "BLUE-SKY-99'."
        
        fact_ids = self.tokenizer(fact_text, return_tensors="pt").input_ids.to(self.device)
        probe_ids = self.tokenizer(probe_text, return_tensors="pt").input_ids.to(self.device)
        target_ids = self.tokenizer(target_text, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Capture States
        # Fact State (Donor)
        with torch.no_grad():
            out_fact = self.model(fact_ids, use_cache=True)
        fact_cache = out_fact.cache_params
        
        # Probe State (Recipient)
        with torch.no_grad():
            out_probe = self.model(probe_ids, use_cache=True)
        probe_cache = out_probe.cache_params
        
        # 2. Baseline
        # Run target prediction starting from clean probe
        cache_pos = torch.arange(probe_ids.shape[1], probe_ids.shape[1] + target_ids.shape[1], device=self.device)
        # We need to copy probe cache
        base_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
        base_cache.ssm_states = probe_cache.ssm_states.clone()
        base_cache.conv_states = probe_cache.conv_states.clone()
        
        with torch.no_grad():
            base_out = self.model(target_ids, cache_params=base_cache, cache_position=cache_pos)
            base_loss = torch.nn.functional.cross_entropy(
                base_out.logits.view(-1, base_out.logits.size(-1)), 
                target_ids.view(-1)
            ).item()
            
        print(f"[{self.name}] Baseline Loss: {base_loss:.4f}")
        
        # 3. Scan
        num_layers = self.model.config.num_hidden_layers
        self.losses = []
        g = 0.5
        
        for L in range(num_layers):
            # Graft
            grafted_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
            grafted_cache.ssm_states = probe_cache.ssm_states.clone()
            grafted_cache.conv_states = probe_cache.conv_states.clone()
            
            h_probe = probe_cache.ssm_states[L]
            h_fact = fact_cache.ssm_states[L]
            
            # Simple linear mix
            grafted_cache.ssm_states[L] = (1 - g) * h_probe + g * h_fact
            
            with torch.no_grad():
                out = self.model(target_ids, cache_params=grafted_cache, cache_position=cache_pos)
                loss = torch.nn.functional.cross_entropy(
                    out.logits.view(-1, out.logits.size(-1)), 
                    target_ids.view(-1)
                ).item()
            
            self.losses.append(loss)
            print(f"[{self.name}] Layer {L:02d}: {loss:.4f}")
            
    def report(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.losses)), self.losses, marker='o')
        plt.title("Layer-wise Sensitivity to Naive State Grafting")
        plt.xlabel("Layer Index")
        plt.ylabel("Target Cross-Entropy Loss")
        plt.grid(True)
        path = os.path.join(self.output_dir, "sensitivity_curve.png")
        plt.savefig(path)
        print(f"[{self.name}] Plot saved to {path}")
