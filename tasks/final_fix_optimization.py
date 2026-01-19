from core.base_task import Task
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import torch.optim as optim

class DiffCache(Mamba2Cache):
    """
    A subclass of Mamba2Cache to pass isinstance checks
    but allow for custom behavior if needed.
    """
    pass

class FinalFixOptimization(Task):
    def __init__(self):
        super().__init__("FinalFixOptimization")
        self.layer_idx = 7
        self.prompt = "The password is '"
        self.target = "BLUE"

    def run(self):
        print(f"[{self.name}] Running Final Optimization Fix...")
        
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        
        base_cache = out.cache_params
        last_token_id = input_ids[:, -1:]
        context_len = input_ids.shape[1] - 1
        cache_pos = torch.tensor([context_len], device=self.device)
        
        # Base SSM State
        base_ssm = base_cache.ssm_states.detach().clone()
        base_conv = base_cache.conv_states.detach().clone()
        
        delta = torch.zeros(base_ssm[self.layer_idx].shape, device=self.device, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.1)
        target_id = self.tokenizer.encode(self.target, add_special_tokens=False)[0]
        
        print(f"Optimizing for '{self.target}'...")
        
        for step in range(50):
            optimizer.zero_grad()
            
            # 1. Functional Construction
            # Create a list of tensors. Each is a CLONE or NEW tensor.
            # NO VIEWS of base_ssm allowed.
            ssm_layers = []
            for i in range(base_ssm.shape[0]):
                if i == self.layer_idx:
                    # New tensor from addition
                    layer = base_ssm[i] + delta 
                else:
                    # Detached Clone (Constant)
                    layer = base_ssm[i].detach().clone()
                ssm_layers.append(layer)
            
            # Stack creates a new tensor from the list
            full_ssm = torch.stack(ssm_layers)
            
            # 2. Cache Setup (Using DiffCache to pass isinstance)
            temp_cache = DiffCache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
            
            # 3. CRITICAL: Copy tensors into cache
            # Mamba2Cache initializes empty tensors. We overwrite them.
            # But we must ensure 'full_ssm' stays in the graph.
            # Assigning to the attribute works.
            temp_cache.ssm_states = full_ssm
            temp_cache.conv_states = base_conv.clone() # Clone conv to be safe
            
            # 4. Forward
            # Mamba2Model will modify temp_cache.ssm_states in-place.
            # This means 'full_ssm' gets modified in-place.
            # But 'full_ssm' was created by stack.
            # Does modifying output of stack break gradients of inputs?
            # Yes, usually.
            
            # WORKAROUND: Pass a CLONE to the model, but keep 'full_ssm' for graph?
            # No, we need the graph to flow THROUGH the model.
            
            # Wait. The model updates the state for the NEXT token.
            # It computes logits using the CURRENT state.
            # If the update operation happens *after* logit computation, we are fine.
            # If it happens *during* (fused kernel), we are screwed.
            
            # Let's hope DiffCache works.
            out = self.model(last_token_id, cache_params=temp_cache, cache_position=cache_pos)
            
            loss = torch.nn.functional.cross_entropy(out.logits[0, -1].view(1, -1), torch.tensor([target_id], device=self.device))
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                rank = (out.logits[0, -1] > out.logits[0, -1, target_id]).sum().item() + 1
                if rank == 1:
                    print(f"  Hit Rank 1 at step {step}")
                    break
        
        print("Optimization finished.")
        
        # Verify
        print("\n--- Verification ---")
        final_ssm = base_ssm.clone()
        final_ssm[self.layer_idx] += delta.detach()
        base_cache.ssm_states = final_ssm
        
        with torch.no_grad():
            out_final = self.model(last_token_id, cache_params=base_cache, cache_position=cache_pos)
            rank = (out_final.logits[0, -1] > out_final.logits[0, -1, target_id]).sum().item() + 1
            print(f"Final Rank (Real Cache): {rank}")

    def report(self):
        pass
