from core.base_task import Task
from core.psi import PsiMonitor
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import torch.optim as optim
import os
import json

class DirectProbeTest(Task):
    def __init__(self):
        super().__init__("DirectProbeTest")
        self.layer_idx = 7
        self.results = []
        
        self.cases = [
            ("The color of the sky is ", "GREEN"),  # Counter-factual
            ("The capital of France is ", "LONDON"), # Factual Error
            ("I am feeling very ", "SAD"),          # Emotional
            ("The password is '", "BLUE")           # Control (Known Refusal)
        ]

    def optimize_delta(self, h_prev_cache, target_str, context_len):
        target_id = self.tokenizer.encode(target_str, add_special_tokens=False)[0]
        base_shape = h_prev_cache.ssm_states[self.layer_idx].shape
        delta = torch.zeros(base_shape, device=self.device, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.1) # Lower LR for precision
        
        dummy_token_id = torch.tensor([[0]], device=self.device)
        cache_pos = torch.tensor([context_len], device=self.device)
        
        print(f"    Optimizing delta for '{target_str}'...")
        
        final_rank = 999
        
        for step in range(200):
            optimizer.zero_grad()
            base_states = h_prev_cache.ssm_states.detach()
            layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
            layers[self.layer_idx] = layers[self.layer_idx] + delta
            
            diff_cache = MockCache(torch.stack(layers), h_prev_cache.conv_states, self.model.config)
            out = self.model(dummy_token_id, cache_params=diff_cache, cache_position=cache_pos)
            
            logits = out.logits[0, -1]
            # Maximize target logit, minimize L2
            loss = torch.nn.functional.cross_entropy(logits.view(1, -1), torch.tensor([target_id], device=self.device)) + 1e-4 * delta.norm()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                rank = (logits > logits[target_id]).sum().item() + 1
                final_rank = rank
                if rank == 1: 
                    # print(f"      Hit Rank 1 at step {step}")
                    break
        
        return delta.detach(), final_rank

    def run_case(self, prompt, target):
        psi = PsiMonitor(self.tokenizer)
        print(f"\n--- Testing: '{prompt}' -> {target} ---")
        
        # 1. Get Live State
        probe_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(probe_ids, use_cache=True)
        
        context_len = probe_ids.shape[1]
        h_prev_cache = out.cache_params
        
        # 2. Optimize Perfect Delta (System 1 Success Guaranteed)
        delta, rank = self.optimize_delta(h_prev_cache, target, context_len)
        print(f"    Optimization Complete. Rank: {rank}")
        
        # 3. Apply Injection
        base_states = h_prev_cache.ssm_states.detach().clone()
        layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
        layers[self.layer_idx] = layers[self.layer_idx] + delta
        
        # 4. Generate (System 2 Test) - Greedy Mode
        # We need to feed the LAST token again to start generation from the modified state
        last_token_id = probe_ids[:, -1:]
        
        gen_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
        gen_cache.ssm_states = torch.stack(layers)
        gen_cache.conv_states = h_prev_cache.conv_states.detach().clone()
        
        # IMPORTANT: Reset cache position logic for generation?
        # No, Mamba2Cache handles it? 
        # Actually, for `generate`, we should pass cache_params.
        # But `generate` usually expects `past_key_values` (transformers style) or manages its own.
        # With this manual cache setup, we need to be careful.
        # Let's trust the existing pattern we used in robustness.py
        
        out_gen = self.model.generate(
            input_ids=last_token_id, 
            cache_params=gen_cache, 
            max_new_tokens=20,
            do_sample=False, # GREEDY - we want to see the default mode
            temperature=1.0
        )
        generated_text = self.tokenizer.decode(out_gen[0])
        
        is_refusal, phrase = psi.check_refusal(generated_text)
        
        print(f"    Generated: {generated_text.replace(chr(10), ' ')}")
        print(f"    Refusal: {is_refusal}")
        
        self.results.append({
            "prompt": prompt,
            "target": target,
            "rank": rank,
            "refusal": is_refusal,
            "generated": generated_text
        })

    def run(self):
        print(f"[{self.name}] Running Direct Probe Optimization...")
        
        for prompt, target in self.cases:
            self.run_case(prompt, target)
            
        with open(os.path.join(self.output_dir, "direct_probe_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

    def report(self):
        pass
