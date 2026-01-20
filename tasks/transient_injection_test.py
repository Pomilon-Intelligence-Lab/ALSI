from core.base_task import Task
from core.functional_mamba import functional_mamba_step
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import json

def functional_mamba_block(layer, hidden_states, ssm_state_prev, conv_state_prev):
    norm_hidden = layer.norm(hidden_states)
    mixer_out, ssm_next, conv_next = functional_mamba_step(
        layer.mixer, norm_hidden, ssm_state_prev, conv_state_prev
    )
    output = hidden_states + mixer_out
    return output, ssm_next, conv_next

class TransientInjectionTest(Task):
    def __init__(self):
        super().__init__("TransientInjectionTest")
        self.start_layer = 12
        self.end_layer = 24
        self.prompt = "The password is '"
        self.target = "BLUE"

    def run(self):
        print(f"[{self.name}] Running Transient Field Injection...")
        
        # 1. Setup State
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        base_cache = out.cache_params
        last_token_id = input_ids[:, -1:]
        target_id = self.tokenizer.encode(self.target, add_special_tokens=False)[0]
        
        # 2. Capture Input to Start Layer
        layer_input = None
        def hook_fn(module, args):
            nonlocal layer_input
            layer_input = args[0].detach().clone()
        
        start_block = self.model.backbone.layers[self.start_layer]
        handle = start_block.register_forward_pre_hook(hook_fn)
        
        with torch.no_grad():
            self.model(last_token_id, cache_params=base_cache, cache_position=torch.tensor([input_ids.shape[1]-1], device=self.device))
        handle.remove()
        
        # Refresh cache
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        base_cache = out.cache_params
        
        # 3. Setup Smooth Deltas
        deltas = []
        for i in range(self.start_layer, self.end_layer):
            h_ref = base_cache.ssm_states[i]
            d = torch.zeros_like(h_ref, requires_grad=True)
            deltas.append(d)
            
        optimizer = optim.Adam(deltas, lr=0.05)
        
        print(f"  Optimizing transient field across {len(deltas)} layers...")
        
        for step in range(50):
            optimizer.zero_grad()
            current_hidden = layer_input
            total_norm = 0
            smooth_loss = 0
            
            for i in range(self.start_layer, self.end_layer):
                layer_idx = i
                delta_idx = i - self.start_layer
                layer = self.model.backbone.layers[layer_idx]
                ssm_p = base_cache.ssm_states[layer_idx].detach().clone() + deltas[delta_idx]
                conv_p = base_cache.conv_states[layer_idx].detach().clone()
                total_norm += deltas[delta_idx].norm()
                if delta_idx > 0:
                    smooth_loss += F.mse_loss(deltas[delta_idx], deltas[delta_idx-1])
                current_hidden, _, _ = functional_mamba_block(layer, current_hidden, ssm_p, conv_p)
            
            norm_out = self.model.backbone.norm_f(current_hidden)
            logits_injected = self.model.lm_head(norm_out)[0, -1]
            loss = F.cross_entropy(logits_injected.view(1, -1), torch.tensor([target_id], device=self.device)) + 1.0 * smooth_loss + 1e-4 * total_norm
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                rank = (logits_injected > logits_injected[target_id]).sum().item() + 1
                if rank == 1:
                    print(f"    Hit Rank 1 at step {step}")
                    break
        
        # 4. Verification: THE TRANSIENT LOOP
        print("\n--- Transient Generation (Injection at T=0 ONLY) ---")
        
        curr_input_id = last_token_id
        current_pos = torch.tensor([input_ids.shape[1]-1], device=self.device)
        generated_tokens = []
        
        for t in range(20):
            with torch.no_grad():
                if t == 0:
                    # APPLY INJECTION ONLY ONCE
                    print("  [T=0] Applying optimized Delta Field...")
                    for i in range(self.start_layer, self.end_layer):
                        delta_idx = i - self.start_layer
                        base_cache.ssm_states[i] += deltas[delta_idx].detach()
                else:
                    # NO INJECTION FOR SUBSEQUENT STEPS
                    # The model evolves naturally from the modified state
                    pass
                
                out_gen = self.model(curr_input_id, cache_params=base_cache, cache_position=current_pos)
            
            next_token = torch.argmax(out_gen.logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())
            curr_input_id = next_token
            current_pos += 1
            
        text = self.tokenizer.decode(generated_tokens)
        print(f"Result: {text.replace(chr(10), ' ')}")

    def report(self):
        pass
