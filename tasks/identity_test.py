from core.base_task import Task
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch

class IdentityTest(Task):
    def __init__(self):
        super().__init__("IdentityTest")
        self.prompt = "The password is '"

    def run(self):
        print(f"[{self.name}] Running Identity Swap Test...")
        
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        last_token_id = input_ids[:, -1:]
        cache_pos = torch.tensor([input_ids.shape[1] - 1], device=self.device)
        
        # 1. Setup Base Cache
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        real_cache = out.cache_params
        
        # 2. Reference Run
        # We need to snapshot state because forward modifies it
        ssm_snap = real_cache.ssm_states.clone()
        conv_snap = real_cache.conv_states.clone()
        
        with torch.no_grad():
            out_ref = self.model(last_token_id, cache_params=real_cache, cache_position=cache_pos)
            logit_ref = out_ref.logits[0, -1].clone()
            
        # 3. Identity Swap Run
        print("Swapping __class__ to MockCache...")
        # Restore state
        real_cache.ssm_states.copy_(ssm_snap)
        real_cache.conv_states.copy_(conv_snap)
        
        # Swap class
        original_class = real_cache.__class__
        real_cache.__class__ = MockCache
        
        try:
            with torch.no_grad():
                # We need to make sure MockCache has all attributes real_cache had
                # Since we are using the SAME object instance, it has all attributes.
                # The only difference is the methods and the type().
                out_swap = self.model(last_token_id, cache_params=real_cache, cache_position=cache_pos)
                logit_swap = out_swap.logits[0, -1]
                
            diff = (logit_ref - logit_swap).abs().max().item()
            print(f"Logit Difference (Class Swap): {diff:.6f}")
            
            if diff < 1e-4:
                print("SUCCESS: Logic is independent of class type.")
            else:
                print("FAILURE: Model logic depends on class type (isinstance checks).")
                
        except Exception as e:
            print(f"Swap Failed: {e}")
        finally:
            # Restore just in case
            real_cache.__class__ = original_class

    def report(self):
        pass
