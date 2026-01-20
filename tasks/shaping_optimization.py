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

class ShapingOptimization(Task):
    def __init__(self):
        super().__init__("ShapingOptimization")
        self.start_layer = 12
        self.end_layer = 24
        self.prompt = "The password is '"
        self.target = "BLUE"
        self.window = 3 # Optimize across 3 steps

    def run(self):
        print(f"[{self.name}] Running Multi-Step Trajectory Shaping (Window={self.window})...")
        
        # 1. Setup State
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        base_cache = out.cache_params
        last_token_id = input_ids[:, -1:]
        target_id = self.tokenizer.encode(self.target, add_special_tokens=False)[0]
        
        # 2. Capture Natural Rollout (The "Heal" Target)
        print("  Capturing natural rollout baseline...")
        natural_rollout = []
        temp_cache = out.cache_params
        curr_id = last_token_id
        curr_pos = torch.tensor([input_ids.shape[1]-1], device=self.device)
        
        with torch.no_grad():
            for _ in range(self.window):
                out_step = self.model(curr_id, cache_params=temp_cache, cache_position=curr_pos)
                nat_next_id = torch.argmax(out_step.logits[0, -1], dim=-1)
                natural_rollout.append(nat_next_id.item())
                curr_id = nat_next_id.unsqueeze(0).unsqueeze(0)
                curr_pos += 1
        
        # Refresh cache for optimization
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        base_cache = out.cache_params
        
        # 3. Setup Distributed Deltas
        deltas = []
        for i in range(self.start_layer, self.end_layer):
            h_ref = base_cache.ssm_states[i]
            d = torch.zeros_like(h_ref, requires_grad=True)
            deltas.append(d)
            
        optimizer = optim.Adam(deltas, lr=0.05)
        
        print(f"  Shaping trajectory to force '{self.target}' then recover...")
        
        for step in range(50):
            optimizer.zero_grad()
            
            # Step 1: Force Token
            # We need to run the model up to start_layer for step 1
            # For simplicity, we'll re-capture layer_input inside the loop or assume it's constant
            # (In a real BPTT we'd unroll everything, but here we only optimize the t=0 injection)
            
            # Capture layer_input for t=0
            layer_input = None
            def hook_fn(module, args):
                nonlocal layer_input
                layer_input = args[0]
            handle = self.model.backbone.layers[self.start_layer].register_forward_pre_hook(hook_fn)
            with torch.no_grad():
                # This is a bit slow but ensures correct gradients for step 1
                self.model(last_token_id, cache_params=base_cache, cache_position=torch.tensor([input_ids.shape[1]-1], device=self.device))
            handle.remove()
            
            # CURRENT STATES (t=0)
            current_ssms = [base_cache.ssm_states[i].detach().clone() for i in range(24)]
            current_convs = [base_cache.conv_states[i].detach().clone() for i in range(24)]
            
            # UNROLL 1: Force Token
            h = layer_input
            next_ssms = []
            next_convs = []
            for i in range(self.start_layer, self.end_layer):
                layer_idx = i
                delta_idx = i - self.start_layer
                ssm_p = current_ssms[layer_idx] + deltas[delta_idx]
                h, s_n, c_n = functional_mamba_block(self.model.backbone.layers[i], h, ssm_p, current_convs[layer_idx])
                next_ssms.append(s_n)
                next_convs.append(c_n)
            
            logits_t1 = self.model.lm_head(self.model.backbone.norm_f(h))[0, -1]
            l_force = F.cross_entropy(logits_t1.view(1, -1), torch.tensor([target_id], device=self.device))
            
            # UNROLL 2: Natural Recovery
            # Feed the FORCED token (Target) to see where the trajectory goes
            # and penalize it if it continues to loop Target.
            curr_id_t2 = torch.tensor([[target_id]], device=self.device)
            # We'd need to run the whole model functionally for t=2... 
            # This is getting very deep. 
            # Let's use a simpler heuristic for now: 
            # Penalize the Target token probability at t=2 using the updated state from t=1.
            
            # Note: We'd need the hidden_states input to start_layer at t=2.
            # This depends on the LM Head -> Embeddings -> Layers 0-11.
            # We approximate: input_t2 = Embed(Target) -> Layers 0-11.
            
            embeds_t2 = self.model.backbone.embeddings(curr_id_t2)
            h_t2 = embeds_t2
            for i in range(self.start_layer):
                h_t2, _, _ = functional_mamba_block(self.model.backbone.layers[i], h_t2, base_cache.ssm_states[i], base_cache.conv_states[i])
            
            # Now run layers 12-23 with the states produced at t=1
            for i in range(self.start_layer, self.end_layer):
                idx = i - self.start_layer
                h_t2, _, _ = functional_mamba_block(self.model.backbone.layers[i], h_t2, next_ssms[idx], next_convs[idx])
            
            logits_t2 = self.model.lm_head(self.model.backbone.norm_f(h_t2))[0, -1]
            # Anti-Loop: Minimize probability of target at t=2
            l_anti_loop = -torch.log(1.0 - torch.softmax(logits_t2, dim=-1)[target_id] + 1e-7)
            
            loss = l_force + 0.5 * l_anti_loop + 1e-4 * sum(d.norm() for d in deltas)
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"    Step {step}: Force {l_force.item():.4f}, Anti-Loop {l_anti_loop.item():.4f}")
                
        # 4. Verification
        print("\n--- Shaped Injection Verification ---")
        for i in range(self.start_layer, self.end_layer):
            delta_idx = i - self.start_layer
            base_cache.ssm_states[i] += deltas[delta_idx].detach()
            
        curr_input_id = last_token_id
        current_pos = torch.tensor([input_ids.shape[1]-1], device=self.device)
        generated_tokens = []
        
        for t in range(20):
            with torch.no_grad():
                out_gen = self.model(curr_input_id, cache_params=base_cache, cache_position=current_pos)
            next_token = torch.argmax(out_gen.logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())
            curr_input_id = next_token
            current_pos += 1
            
        print(f"Result: {self.tokenizer.decode(generated_tokens).replace(chr(10), ' ')}")

    def report(self):
        pass
