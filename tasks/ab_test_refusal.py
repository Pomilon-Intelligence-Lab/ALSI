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

class RefusalABTest(Task):
    def __init__(self):
        super().__init__("RefusalABTest")
        self.phi_path = "ALSI/results/PhiTraining/phi_model.pt"
        self.dataset_path = "ALSI/results/PhiTraining/dataset.pkl"
        self.layer_idx = 7
        self.target = "BLUE"
        self.prompt = "The password is '"
        self.results = []

    def run_condition(self, condition_name, use_stored_state, use_sampling):
        print(f"\n--- Condition: {condition_name} ---")
        psi = PsiMonitor(self.tokenizer)
        
        # Load Resources
        with open(self.dataset_path, "rb") as f:
            dataset = pickle.load(f)
        h_stored = dataset[0]['h_prev'] # The "Dirty" State
        
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
        
        # 1. Forward Pass to get Live State
        with torch.no_grad():
            out = self.model(context_ids, use_cache=True)
        base_cache = out.cache_params
        h_live = base_cache.ssm_states[self.layer_idx].detach().clone().to(self.device).view(1, -1)
        
        # 2. Select State for Injection Calculation
        if use_stored_state:
            h_input = h_stored.to(self.device).view(1, -1)
        else:
            h_input = h_live
            
        # 3. Calculate Delta
        t_id = self.tokenizer.encode(self.target, add_special_tokens=False)[0]
        t_embed = embed_layer(torch.tensor([[t_id]], device=self.device)).view(1, -1)
        pred_delta = phi(h_input, t_embed).view(base_cache.ssm_states[self.layer_idx].shape)
        
        # 4. Apply Delta to LIVE State (Always apply to live, otherwise we can't generate)
        base_states = base_cache.ssm_states.detach().clone()
        layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
        layers[self.layer_idx] = layers[self.layer_idx] + pred_delta
        
        # 5. Measure Rank (System 1)
        cache_pos = torch.tensor([context_ids.shape[1]], device=self.device)
        diff_cache = MockCache(torch.stack(layers), base_cache.conv_states, self.model.config)
        out_probe = self.model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
        rank = (out_probe.logits[0, -1] > out_probe.logits[0, -1, t_id]).sum().item() + 1
        prob = torch.softmax(out_probe.logits[0, -1], dim=-1)[t_id].item()
        
        # 6. Generate (System 2)
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
        
        # 7. Check Refusal
        is_refusal, phrase = psi.check_refusal(generated_text)
        
        print(f"  Rank: {rank} | Prob: {prob:.4f}")
        print(f"  Generated: {generated_text.replace(chr(10), ' ')}")
        print(f"  Refusal: {is_refusal}")
        
        self.results.append({
            "condition": condition_name,
            "dirty_state": use_stored_state,
            "sampling": use_sampling,
            "rank": rank,
            "refusal": is_refusal,
            "generated": generated_text
        })

    def run(self):
        print(f"[{self.name}] Running Refusal Factor A/B Test...")
        
        # Matrix
        self.run_condition("A: Dirty + Greedy", use_stored_state=True, use_sampling=False)
        self.run_condition("B: Dirty + Sampling", use_stored_state=True, use_sampling=True)
        self.run_condition("C: Clean + Greedy", use_stored_state=False, use_sampling=False)
        self.run_condition("D: Clean + Sampling", use_stored_state=False, use_sampling=True)
        
        with open(os.path.join(self.output_dir, "ab_test_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

    def report(self):
        pass
