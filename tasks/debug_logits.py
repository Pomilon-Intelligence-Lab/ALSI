from core.base_task import Task
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import torch.optim as optim
import os

class DebugLogits(Task):
    def __init__(self):
        super().__init__("DebugLogits")
        self.layer_idx = 7
        self.prompt = "The password is '"
        self.target = "BLUE"

    def run(self):
        print(f"[{self.name}] Debugging Logit Mismatch...")
        
        # 1. Setup State
        probe_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(probe_ids, use_cache=True)
        h_prev_cache = out.cache_params
        context_len = probe_ids.shape[1]
        
        # 2. Optimize Delta using DUMMY token (0) - reproducing original flaw
        print("--- Optimization (with Dummy Token 0) ---")
        base_shape = h_prev_cache.ssm_states[self.layer_idx].shape
        delta = torch.zeros(base_shape, device=self.device, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.1)
        target_id = self.tokenizer.encode(self.target, add_special_tokens=False)[0]
        dummy_token_id = torch.tensor([[0]], device=self.device)
        cache_pos = torch.tensor([context_len], device=self.device)
        
        for step in range(50):
            optimizer.zero_grad()
            base_states = h_prev_cache.ssm_states.detach()
            layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
            layers[self.layer_idx] = layers[self.layer_idx] + delta
            diff_cache = MockCache(torch.stack(layers), h_prev_cache.conv_states, self.model.config)
            
            # Forward with DUMMY
            out = self.model(dummy_token_id, cache_params=diff_cache, cache_position=cache_pos)
            loss = torch.nn.functional.cross_entropy(out.logits[0, -1].view(1, -1), torch.tensor([target_id], device=self.device))
            loss.backward()
            optimizer.step()
            
            if (out.logits[0, -1] > out.logits[0, -1, target_id]).sum().item() + 1 == 1:
                print(f"  Hit Rank 1 at step {step}")
                break
                
        # 3. Verify Logits with DUMMY vs REAL token
        print("\n--- Verification ---")
        
        # Apply Delta
        final_layers = [h_prev_cache.ssm_states[i].detach().clone() for i in range(self.model.config.num_hidden_layers)]
        final_layers[self.layer_idx] = final_layers[self.layer_idx] + delta.detach()
        verify_cache = MockCache(torch.stack(final_layers), h_prev_cache.conv_states, self.model.config)
        
        # Case A: Dummy Token (What we optimized for)
        out_dummy = self.model(dummy_token_id, cache_params=verify_cache, cache_position=cache_pos)
        rank_dummy = (out_dummy.logits[0, -1] > out_dummy.logits[0, -1, target_id]).sum().item() + 1
        top_dummy = torch.topk(out_dummy.logits[0, -1], 5)
        print(f"Input [0] (Dummy) -> Rank: {rank_dummy}")
        print(f"  Top 5: {[self.tokenizer.decode(t) for t in top_dummy.indices]}")
        
        # Case B: Real Token (What generate uses)
        # The prompt ends with "'", so the input to generate is "'".
        last_token_id = probe_ids[:, -1:]
        out_real = self.model(last_token_id, cache_params=verify_cache, cache_position=cache_pos)
        rank_real = (out_real.logits[0, -1] > out_real.logits[0, -1, target_id]).sum().item() + 1
        top_real = torch.topk(out_real.logits[0, -1], 5)
        print(f"Input ['] (Real)  -> Rank: {rank_real}")
        print(f"  Top 5: {[self.tokenizer.decode(t) for t in top_real.indices]}")

    def report(self):
        pass
