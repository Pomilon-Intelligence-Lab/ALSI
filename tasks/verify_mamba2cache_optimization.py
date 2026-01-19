from core.base_task import Task
from core.psi import PsiMonitor
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import torch.optim as optim
import os

class VerifyMamba2CacheOptimization(Task):
    def __init__(self):
        super().__init__("VerifyMamba2CacheOptimization")
        self.layer_idx = 7
        self.prompt = "The password is '"
        self.target = "BLUE"

    def run(self):
        print(f"[{self.name}] Verifying Real-Cache Optimization...")
        torch.autograd.set_detect_anomaly(True)
        
        # 1. Setup Live State
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        
        # The REAL cache object
        cache = out.cache_params
        last_token_id = input_ids[:, -1:]
        context_len = input_ids.shape[1] - 1
        cache_pos = torch.tensor([context_len], device=self.device)
        
        # 2. Optimization Loop (Directly on Mamba2Cache)
        print(f"  Optimizing delta for '{self.target}'...")
        
        # Snapshot base state
        base_ssm = cache.ssm_states.detach().clone()
        
        # Delta shape: [1, 24, 64, 128] (Batch, Heads, HeadDim, StateSize)
        delta_shape = base_ssm[self.layer_idx].shape
        delta = torch.zeros(delta_shape, device=self.device, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.1)
        target_id = self.tokenizer.encode(self.target, add_special_tokens=False)[0]
        
        final_rank = 999
        
        for step in range(100):
            optimizer.zero_grad()
            
            # Purely functional construction of the SSM state tensor
            # base_ssm: [24, 1, 24, 64, 128]
            # CLONE here to ensure this is a new allocation, not a view dependent on base_ssm
            injected_layer = (base_ssm[self.layer_idx] + delta).clone() 
            
            # Construct full SSM tensor
            ssm_list = [base_ssm[i].detach().clone() for i in range(base_ssm.shape[0])]
            ssm_list[self.layer_idx] = injected_layer
            full_ssm = torch.stack(ssm_list)
            
            # Fresh cache for this step
            temp_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
            temp_cache.ssm_states = full_ssm
            temp_cache.conv_states = cache.conv_states.detach().clone()
            
            # Forward pass
            out_step = self.model(last_token_id, cache_params=temp_cache, cache_position=cache_pos)
            logits = out_step.logits[0, -1]
            
            loss = torch.nn.functional.cross_entropy(logits.view(1, -1), torch.tensor([target_id], device=self.device)) + 1e-4 * delta.norm()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                rank = (logits > logits[target_id]).sum().item() + 1
                final_rank = rank
                if rank == 1:
                    print(f"    Hit Rank 1 at step {step}")
                    break
        
        # 3. Verification: Manual Generation Loop
        print("\n--- Generation Verification (Manual Loop) ---")
        
        # Apply final delta to cache
        cache.ssm_states = base_ssm.clone()
        cache.ssm_states[self.layer_idx] += delta.detach()
        
        curr_input_id = last_token_id
        current_pos = cache_pos.clone()
        generated_tokens = []
        
        for _ in range(10):
            with torch.no_grad():
                # Note: model.forward will update 'cache' in-place for the next step
                out_gen = self.model(curr_input_id, cache_params=cache, cache_position=current_pos)
            
            next_token = torch.argmax(out_gen.logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())
            
            curr_input_id = next_token
            current_pos += 1
            
        result_text = self.tokenizer.decode(generated_tokens)
        print(f"Generated: {result_text.replace(chr(10), ' ')}")
        
        if self.target in result_text:
            print("SUCCESS: Injection Worked! Real-Cache Optimization is Valid.")
        else:
            print("FAILURE: Injection still missed. There is another hidden variable.")

    def report(self):
        pass
