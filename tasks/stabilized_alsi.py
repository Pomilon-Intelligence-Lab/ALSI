from core.base_task import Task
from core.functional_mamba import functional_mamba_step
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import json

def functional_mamba_block(layer, hidden_states, ssm_state_prev, conv_state_prev):
    """
    Functional implementation of a full Mamba2Block.
    x = x + Mixer(Norm(x))
    """
    # 1. Norm
    norm_hidden = layer.norm(hidden_states)
    
    # 2. Mixer (Functional Step)
    mixer_out, ssm_next, conv_next = functional_mamba_step(
        layer.mixer, 
        norm_hidden, 
        ssm_state_prev, 
        conv_state_prev
    )
    
    # 3. Residual Connection
    # mixer_out is already the output of the mixer branch
    output = hidden_states + mixer_out
    
    return output, ssm_next, conv_next

class StabilizedALSI(Task):
    def __init__(self):
        super().__init__("StabilizedALSI")
        self.start_layer = 12
        self.end_layer = 24 # Process up to the end
        self.prompt = "The password is '"
        self.target = "BLUE"

    def run(self):
        print(f"[{self.name}] Running Stabilized ALSI (Layer {self.start_layer} -> 23)...")
        
        # 1. Setup State
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        
        base_cache = out.cache_params
        last_token_id = input_ids[:, -1:]
        target_id = self.tokenizer.encode(self.target, add_special_tokens=False)[0]
        
        # 2. Capture Input to the Start Layer
        # We need the hidden_states entering Layer 12 for the last token
        layer_input = None
        def hook_fn(module, args):
            nonlocal layer_input
            layer_input = args[0].detach().clone()
        
        start_block = self.model.backbone.layers[self.start_layer]
        handle = start_block.register_forward_pre_hook(hook_fn)
        
        with torch.no_grad():
            self.model(last_token_id, cache_params=base_cache, cache_position=torch.tensor([input_ids.shape[1]-1], device=self.device))
        handle.remove()
        
        # Re-run prefill to get clean base cache
        with torch.no_grad():
            out_clean = self.model(input_ids[:, :-1], use_cache=True)
        base_cache = out_clean.cache_params
        
        # 3. Setup Optimization
        # Optimized variable: Delta at start layer
        h_start = base_cache.ssm_states[self.start_layer].detach().clone()
        delta = torch.zeros_like(h_start, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.05)
        
        print(f"  Optimizing for '{self.target}' with Stability Constraint...")
        
        # Capture Natural Logits for KL Divergence
        with torch.no_grad():
            out_nat = self.model(last_token_id, cache_params=base_cache, cache_position=torch.tensor([input_ids.shape[1]-1], device=self.device))
            logits_nat = out_nat.logits[0, -1]
            
        for step in range(50):
            optimizer.zero_grad()
            
            # Chain layers functionally
            current_hidden = layer_input
            
            # States for layer i
            # We only optimize ssm_state of the start layer
            # Conv states and other layer SSM states are constants for this step
            
            for i in range(self.start_layer, self.end_layer):
                layer = self.model.backbone.layers[i]
                ssm_p = base_cache.ssm_states[i].detach().clone()
                if i == self.start_layer:
                    ssm_p = ssm_p + delta
                
                conv_p = base_cache.conv_states[i].detach().clone()
                
                current_hidden, _, _ = functional_mamba_block(layer, current_hidden, ssm_p, conv_p)
            
            # Final Head
            norm_out = self.model.backbone.norm_f(current_hidden)
            logits_injected = self.model.lm_head(norm_out)[0, -1]
            
            # Losses
            l_target = F.cross_entropy(logits_injected.view(1, -1), torch.tensor([target_id], device=self.device))
            
            # KL Divergence (Stability)
            # Minimize change to the rest of the distribution
            l_kl = F.kl_div(
                F.log_softmax(logits_injected, dim=-1),
                F.softmax(logits_nat, dim=-1),
                reduction='batchmean'
            )
            
            loss = l_target + 0.1 * l_kl + 1e-4 * delta.norm()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                rank = (logits_injected > logits_injected[target_id]).sum().item() + 1
                if rank == 1:
                    print(f"    Hit Rank 1 at step {step} (KL: {l_kl.item():.4f})")
                    break
                    
        # 4. Final Verification
        print("\n--- Verification (Layer 12 Injection) ---")
        base_cache.ssm_states[self.start_layer] += delta.detach()
        
        curr_input_id = last_token_id
        current_pos = torch.tensor([input_ids.shape[1]-1], device=self.device)
        generated_tokens = []
        
        for _ in range(20):
            with torch.no_grad():
                out_gen = self.model(curr_input_id, cache_params=base_cache, cache_position=current_pos)
            next_token = torch.argmax(out_gen.logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())
            curr_input_id = next_token
            current_pos += 1
            
        result_text = self.tokenizer.decode(generated_tokens)
        print(f"Result: {result_text.replace(chr(10), ' ')}")

    def report(self):
        pass
