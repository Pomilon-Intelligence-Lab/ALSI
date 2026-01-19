from core.base_task import Task
from core.phi import PhiProjector
from core.psi import PsiMonitor
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import os
import pickle
import json
import matplotlib.pyplot as plt

class RefusalABTestMultiTarget(Task):
    def __init__(self):
        super().__init__("RefusalABTestMultiTarget")
        self.phi_path = "ALSI/results/PhiTraining/phi_model.pt"
        self.dataset_path = "ALSI/results/PhiTraining/dataset.pkl"
        self.layer_idx = 7
        self.prompt = "The password is '"
        self.targets = ["BLUE", "RED", "GREEN", "ORANGE", "BLACK"] 
        self.results = []

    def run_condition(self, target, condition_name, use_stored_state, use_sampling):
        psi = PsiMonitor(self.tokenizer)
        
        # Load Resources (Optimization: Load once in run(), but keeping here for isolation)
        with open(self.dataset_path, "rb") as f:
            dataset = pickle.load(f)
        h_stored = dataset[0]['h_prev'] 
        
        embed_layer = self.model.get_input_embeddings()
        state_dim = h_stored.numel()
        embed_dim = embed_layer.weight.shape[1]
        
        phi = PhiProjector(state_dim, embed_dim).to(self.device)
        phi.load_state_dict(torch.load(self.phi_path))
        phi.eval()
        
        # Prepare Inputs
        probe_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        context_ids = probe_ids[:, :-1]
        last_token_id = probe_ids[:, -1:]
        
        with torch.no_grad():
            out = self.model(context_ids, use_cache=True)
        base_cache = out.cache_params
        h_live = base_cache.ssm_states[self.layer_idx].detach().clone().to(self.device).view(1, -1)
        
        # Select State
        if use_stored_state:
            h_input = h_stored.to(self.device).view(1, -1)
        else:
            h_input = h_live
            
        # Calculate Delta
        t_id = self.tokenizer.encode(target, add_special_tokens=False)[0]
        t_embed = embed_layer(torch.tensor([[t_id]], device=self.device)).view(1, -1)
        pred_delta = phi(h_input, t_embed).view(base_cache.ssm_states[self.layer_idx].shape)
        
        # Apply Delta to LIVE State
        base_states = base_cache.ssm_states.detach().clone()
        layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
        layers[self.layer_idx] = layers[self.layer_idx] + pred_delta
        
        # Generate
        gen_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
        gen_cache.ssm_states = torch.stack(layers)
        gen_cache.conv_states = base_cache.conv_states.detach().clone()
        
        out_gen = self.model.generate(
            input_ids=last_token_id, 
            cache_params=gen_cache, 
            max_new_tokens=20,
            do_sample=use_sampling,
            temperature=0.7 if use_sampling else 1.0
        )
        generated_text = self.tokenizer.decode(out_gen[0])
        
        is_refusal, phrase = psi.check_refusal(generated_text)
        
        print(f"[{target} | {condition_name}] -> Refusal: {is_refusal}")
        if is_refusal:
             print(f"   Generated: {generated_text.replace(chr(10), ' ')}")

        self.results.append({
            "target": target,
            "condition": condition_name,
            "refusal": is_refusal,
            "generated": generated_text
        })

    def run(self):
        print(f"[{self.name}] Running Refusal Factor A/B Test (Multi-Target)...")
        
        for target in self.targets:
            # We only test the 'Clean' condition (Live State) as we proved State Source didn't matter
            # We focus on Decoding Strategy: Greedy vs Sampling
            self.run_condition(target, "Greedy", use_stored_state=False, use_sampling=False)
            self.run_condition(target, "Sampling", use_stored_state=False, use_sampling=True)
        
        with open(os.path.join(self.output_dir, "ab_test_multi_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

    def report(self):
        pass
