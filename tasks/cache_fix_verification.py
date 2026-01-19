from core.base_task import Task
import torch
import torch.nn.functional as F

class CacheFixVerification(Task):
    def __init__(self):
        super().__init__("CacheFixVerification")
        self.prompt = "The password is '"

    def run(self):
        print(f"[{self.name}] Verifying Cache Object Reuse...")
        
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Forward Pass to get Cache
        # We process the prompt MINUS the last token
        with torch.no_grad():
            outputs = self.model(input_ids[:, :-1], use_cache=True)
        
        # This object now contains the state history
        live_cache = outputs.cache_params
        
        # 2. Native Next Step (The Reference)
        # We run the LAST token using the cache from the prefix
        # We clone the cache to avoid modifying the one we want to use later
        # Wait, Mamba2Cache doesn't have a .clone() method usually. 
        # Let's just run it and see.
        
        last_token_id = input_ids[:, -1:]
        
        # We need a reference logit
        with torch.no_grad():
            # We must use a copy of the tensors if we want to reuse the object
            ref_ssm = live_cache.ssm_states.clone()
            ref_conv = live_cache.conv_states.clone()
            
            out_ref = self.model(
                last_token_id, 
                cache_params=live_cache, 
                cache_position=torch.tensor([input_ids.shape[1]-1], device=self.device)
            )
            logit_ref = out_ref.logits[0, -1].clone()
            
            # Reset live_cache to pre-last-token state for the "Manual" test
            live_cache.ssm_states.copy_(ref_ssm)
            live_cache.conv_states.copy_(ref_conv)

        # 3. Manual Generation using the REUSED object
        print("\n--- Testing Reused Cache Object ---")
        torch.manual_seed(42)
        out_gen = self.model.generate(
            input_ids=last_token_id, 
            cache_params=live_cache, 
            max_new_tokens=10,
            do_sample=False
        )
        text_gen = self.tokenizer.decode(out_gen[0])
        print(f"Generated: {text_gen.replace(chr(10), ' ')}")
        
        if "I'm not sure" in text_gen:
            print("FAILURE: Model still refuses even with reused cache.")
        else:
            print("SUCCESS: Model produced natural text!")

    def report(self):
        pass
