from core.base_task import Task
from core.functional_mamba import functional_mamba_step
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import torch.optim as optim

class FunctionalOptimization(Task):
    def __init__(self):
        super().__init__("FunctionalOptimization")
        self.layer_idx = 23 # Start with last layer for simplicity
        self.test_cases = [
            ("The password is '", "BLUE"),
            ("The color of the sky is ", "GREEN"),
            ("The capital of France is ", "LONDON"),
            ("I am feeling very ", "SAD")
        ]

    def run_case(self, prompt, target):
        print(f"\n--- Testing: '{prompt}' -> {target} ---")
        
        # 1. Setup State
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        
        base_cache = out.cache_params
        last_token_id = input_ids[:, -1:]
        
        # 2. Setup Optimization
        h_prev = base_cache.ssm_states[self.layer_idx].clone().detach()
        c_prev = base_cache.conv_states[self.layer_idx].clone().detach()
        layer = self.model.backbone.layers[self.layer_idx].mixer
        
        delta = torch.zeros_like(h_prev, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.1)
        target_id = self.tokenizer.encode(target, add_special_tokens=False)[0]
        
        # 3. Capture Input to Layer
        layer_input = None
        def hook_fn(module, args):
            nonlocal layer_input
            layer_input = args[0].detach().clone()
        handle = layer.register_forward_pre_hook(hook_fn)
        
        with torch.no_grad():
            # Run one step to trigger hook
            # Note: base_cache is modified in-place here
            self.model(last_token_id, cache_params=base_cache, cache_position=torch.tensor([input_ids.shape[1]-1], device=self.device))
        handle.remove()
        
        # Restore cache state for verification later (was modified by hook run)
        # Easiest is to just re-run prefill
        with torch.no_grad():
            out_re = self.model(input_ids[:, :-1], use_cache=True)
        base_cache = out_re.cache_params

        # 4. Optimization Loop
        for step in range(50):
            optimizer.zero_grad()
            mixer_out, _, _ = functional_mamba_step(layer, layer_input, h_prev + delta, c_prev)
            
            # Pass through Final Head
            norm_out = self.model.backbone.norm_f(mixer_out)
            logits = self.model.lm_head(norm_out)
            logits = logits[:, -1, :]
            
            loss = torch.nn.functional.cross_entropy(logits.view(1, -1), torch.tensor([target_id], device=self.device)) + 1e-4 * delta.norm()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                rank = (logits[0] > logits[0, target_id]).sum().item() + 1
                if rank == 1:
                    print(f"    Hit Rank 1 at step {step}")
                    break
                    
        # 5. Verification
        # Apply Injection
        base_cache.ssm_states[self.layer_idx] += delta.detach()
        
        curr_input_id = last_token_id
        current_pos = torch.tensor([input_ids.shape[1]-1], device=self.device)
        generated_tokens = []
        
        for _ in range(10):
            with torch.no_grad():
                out_gen = self.model(curr_input_id, cache_params=base_cache, cache_position=current_pos)
            next_token = torch.argmax(out_gen.logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())
            curr_input_id = next_token
            current_pos += 1
            
        print(f"    Generated: {self.tokenizer.decode(generated_tokens).replace(chr(10), ' ')}")

    def run(self):
        print(f"[{self.name}] Running Functional Optimization (Multi-Target)...")
        for prompt, target in self.test_cases:
            self.run_case(prompt, target)

    def report(self):
        pass