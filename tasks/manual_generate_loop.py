from core.base_task import Task
from core.psi import PsiMonitor
import torch
import torch.nn.functional as F

class ManualGenerateLoop(Task):
    def __init__(self):
        super().__init__("ManualGenerateLoop")
        self.prompt = "The password is '"

    def run(self):
        print(f"[{self.name}] Testing Custom Generation Loop...")
        
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Native Generate (Reference)
        torch.manual_seed(42)
        out_native = self.model.generate(input_ids, max_new_tokens=10, do_sample=False)
        print(f"Native: {self.tokenizer.decode(out_native[0]).replace(chr(10), ' ')}")
        
        # 2. Custom Loop
        print("\n--- Custom Loop ---")
        
        # A. Pre-fill
        # Run everything EXCEPT the last token
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        cache = out.cache_params
        
        # B. First Step (Last Token of Prompt)
        # We need to feed the last token to get the first NEW token
        curr_input_id = input_ids[:, -1:]
        cache_pos = torch.tensor([input_ids.shape[1] - 1], device=self.device)
        
        generated_tokens = []
        
        torch.manual_seed(42) # Reset seed for comparison
        
        for _ in range(10):
            with torch.no_grad():
                out = self.model(curr_input_id, cache_params=cache, cache_position=cache_pos)
            
            # Greedy Decode
            next_token = torch.argmax(out.logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())
            
            # Update for next step
            curr_input_id = next_token
            cache_pos += 1
            
        custom_text = self.tokenizer.decode(generated_tokens)
        print(f"Custom: {custom_text.replace(chr(10), ' ')}")
        
        if "I'm not sure" in custom_text:
            print("FAILURE: Refusal persists even in custom loop.")
        else:
            print("SUCCESS: Refusal bypassed! The issue was in model.generate()")

    def report(self):
        pass
