from core.base_task import Task
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import torch.nn.functional as F
import os

class CacheAlignmentTest(Task):
    def __init__(self):
        super().__init__("CacheAlignmentTest")
        self.prompt = "The password is '"

    def run(self):
        print(f"[{self.name}] Diagnosing Cache Alignment...")
        
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Native Generation (The Baseline)
        # We let the model handle everything internally
        print("\n--- 1. Native Generation (Reference) ---")
        torch.manual_seed(42)
        out_native = self.model.generate(
            input_ids, 
            max_new_tokens=10, 
            do_sample=False
        )
        text_native = self.tokenizer.decode(out_native[0])
        print(f"Native: {text_native.replace(chr(10), ' ')}")
        
        # 2. Manual Cache Initialization (The "Broken" Method)
        # This replicates how we've been doing injection (even with Zero Delta)
        print("\n--- 2. Manual Cache (Replicating Bug) ---")
        
        # Step 2a: Run forward pass to fill cache
        with torch.no_grad():
            out_fwd = self.model(input_ids[:, :-1], use_cache=True)
        
        base_cache = out_fwd.cache_params
        
        # Step 2b: Setup Generator Cache
        # We manually construct Mamba2Cache from the forward pass results
        gen_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
        
        # KEY: Are we copying correctly?
        # ssms are usually (B, N, D, N) or similar.
        # We need to ensure deep copy and correct assignment
        gen_cache.ssm_states = base_cache.ssm_states.detach().clone()
        gen_cache.conv_states = base_cache.conv_states.detach().clone()
        
        # Step 2c: Generate
        last_token_id = input_ids[:, -1:]
        
        torch.manual_seed(42)
        # IMPORTANT: 'cache_params' is argument for 'generate' IF it supports it. 
        # But Mamba2 'generate' might handle cache differently or re-init if not passed correctly.
        # In `transformers`, `past_key_values` is usually the arg.
        # For Mamba, it is `cache_params`.
        
        try:
            out_manual = self.model.generate(
                input_ids=last_token_id, 
                cache_params=gen_cache, # We pass our manual cache
                max_new_tokens=10,
                do_sample=False
            )
            text_manual = self.tokenizer.decode(out_manual[0])
            print(f"Manual: {text_manual.replace(chr(10), ' ')}")
        except Exception as e:
            print(f"Manual Generation Failed: {e}")

        # 3. Cache State Inspection
        print("\n--- 3. State Inspection ---")
        # Let's compare the last state of Native vs Manual
        # Actually, Native doesn't expose the cache unless we return it.
        # Let's run a single forward step on both and compare logits.
        
        # Native Step
        with torch.no_grad():
            out_nat_step = self.model(input_ids, use_cache=True)
        logit_nat = out_nat_step.logits[0, -1]
        
        # Manual Step (using our cache from 2a)
        # We feed the LAST token using the cache populated by the PREFIX
        with torch.no_grad():
            out_man_step = self.model(last_token_id, cache_params=gen_cache, cache_position=torch.tensor([input_ids.shape[1]-1], device=self.device))
        logit_man = out_man_step.logits[0, -1]
        
        diff = (logit_nat - logit_man).abs().max().item()
        print(f"Logit Difference (Max Abs): {diff:.6f}")
        
        if diff > 1e-4:
            print("FAILURE: Manual cache state does not match Native state.")
        else:
            print("SUCCESS: Cache states match. The bug is in 'generate' loop handling.")

    def report(self):
        pass
