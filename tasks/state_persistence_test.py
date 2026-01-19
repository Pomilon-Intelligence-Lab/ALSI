from core.base_task import Task
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch

class StatePersistenceTest(Task):
    def __init__(self):
        super().__init__("StatePersistenceTest")
        self.prompt = "The password is '"

    def run(self):
        print(f"[{self.name}] Verifying Cache Write Access...")
        
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Pre-fill
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        cache = out.cache_params
        
        last_token_id = input_ids[:, -1:]
        cache_pos = torch.tensor([input_ids.shape[1] - 1], device=self.device)
        
        # 2. Control Run (Normal)
        print("\n--- Control Run (No Modification) ---")
        with torch.no_grad():
            # Clone cache for control run to not affect the next step
            # Mamba2Cache structure is complex, manual deep copy of tensors needed
            control_ssm = cache.ssm_states.clone()
            control_conv = cache.conv_states.clone()
            
            # We can't easily deepcopy the whole object without potentially breaking internal refs
            # So we will modify it, run, then restore it.
            
            out_control = self.model(last_token_id, cache_params=cache, cache_position=cache_pos)
            top_token = torch.argmax(out_control.logits[0, -1]).item()
            print(f"Next Token: '{self.tokenizer.decode(top_token)}' (ID: {top_token})")
            
        # 3. Nuke Run (Zero out Cache)
        print("\n--- Nuke Run (Zeroed Cache) ---")
        cache.ssm_states.zero_()
        cache.conv_states.zero_()
        
        with torch.no_grad():
            out_nuke = self.model(last_token_id, cache_params=cache, cache_position=cache_pos)
            top_token_nuke = torch.argmax(out_nuke.logits[0, -1]).item()
            print(f"Next Token: '{self.tokenizer.decode(top_token_nuke)}' (ID: {top_token_nuke})")
            
        if top_token != top_token_nuke:
            print("SUCCESS: Cache modification impacts output. We have the steering wheel.")
        else:
            print("FAILURE: Cache modification ignored. We are turning a disconnected wheel.")

    def report(self):
        pass
