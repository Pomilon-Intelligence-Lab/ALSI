from core.base_task import Task
from core.phi import PhiProjector
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import torch.optim as optim
import os

class ManualInjectionTest(Task):
    def __init__(self):
        super().__init__("ManualInjectionTest")
        self.layer_idx = 7
        self.prompt = "The password is '"
        self.target = "BLUE"

    def optimize_delta(self, h_prev_cache, last_token_id, context_len):
        target_id = self.tokenizer.encode(self.target, add_special_tokens=False)[0]
        base_shape = h_prev_cache.ssm_states[self.layer_idx].shape
        delta = torch.zeros(base_shape, device=self.device, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.1)
        cache_pos = torch.tensor([context_len], device=self.device)
        
        for step in range(100):
            optimizer.zero_grad()
            base_states = h_prev_cache.ssm_states.detach()
            layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
            layers[self.layer_idx] = layers[self.layer_idx] + delta
            
            diff_cache = MockCache(torch.stack(layers), h_prev_cache.conv_states, self.model.config)
            out = self.model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
            
            loss = torch.nn.functional.cross_entropy(out.logits[0, -1].view(1, -1), torch.tensor([target_id], device=self.device))
            loss.backward()
            optimizer.step()
            
            if (out.logits[0, -1] > out.logits[0, -1, target_id]).sum().item() + 1 == 1:
                return delta.detach(), True
        
        return delta.detach(), False

    def run(self):
        print(f"[{self.name}] Verifying Injection with Manual Loop...")
        
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Pre-fill
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        cache = out.cache_params
        
        last_token_id = input_ids[:, -1:]
        context_len = input_ids.shape[1] - 1
        
        # 2. Optimize Injection
        delta, success = self.optimize_delta(cache, last_token_id, context_len)
        print(f"Injection Optimized: {'Success' if success else 'Failure'}")
        
        # 3. Apply Injection
        cache.ssm_states[self.layer_idx] += delta
        
        # 4. Manual Generation Loop
        curr_input_id = last_token_id
        cache_pos = torch.tensor([context_len], device=self.device)
        generated_tokens = []
        
        print(f"Generating with injected state...")
        
        for _ in range(10):
            with torch.no_grad():
                out = self.model(curr_input_id, cache_params=cache, cache_position=cache_pos)
            
            # Greedy Decode
            next_token = torch.argmax(out.logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())
            
            curr_input_id = next_token
            cache_pos += 1
            
        generated_text = self.tokenizer.decode(generated_tokens)
        print(f"Result: {generated_text.replace(chr(10), ' ')}")
        
        if self.target in generated_text:
            print("SUCCESS: Injection Worked!")
        else:
            print("FAILURE: Injection missed.")

    def report(self):
        pass
