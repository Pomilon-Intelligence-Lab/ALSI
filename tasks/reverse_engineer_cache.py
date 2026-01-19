from core.base_task import Task
import torch

class ReverseEngineerCache(Task):
    def __init__(self):
        super().__init__("ReverseEngineerCache")
        self.layer_idx = 7
        self.prompt = "The password is '"

    def run(self):
        print(f"[{self.name}] Reverse Engineering Mamba2Cache Flow...")
        
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Baseline State (Pre-fill)
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        cache = out.cache_params
        
        last_token_id = input_ids[:, -1:]
        cache_pos = torch.tensor([input_ids.shape[1] - 1], device=self.device)
        
        # Capture Baseline Logits
        # We need to CLONE the cache because forward() modifies it
        # But we can't easily clone Mamba2Cache deep structures without manual copy
        # So we will "snapshot" the tensors we care about
        
        ssm_snap = cache.ssm_states[self.layer_idx].detach().clone()
        conv_snap = cache.conv_states[self.layer_idx].detach().clone()
        
        with torch.no_grad():
            # Run baseline step
            # We need to restore cache after this because it will update
            out_base = self.model(last_token_id, cache_params=cache, cache_position=cache_pos)
            base_logit = out_base.logits[0, -1].clone()
            
            # Check if cache updated
            ssm_post = cache.ssm_states[self.layer_idx]
            update_diff = (ssm_post - ssm_snap).abs().max().item()
            print(f"\n[Timing Probe] Cache Update Magnitude during Forward: {update_diff:.6f}")
            if update_diff > 0:
                print("  -> Cache is UPDATED during forward pass (Read-Modify-Write).")
            else:
                print("  -> Cache is STATIC (Read-Only? Unlikely for SSM).")

        # 2. SSM Sensitivity Probe
        print("\n[Sensitivity Probe] Modifying SSM State...")
        # Restore cache
        cache.ssm_states[self.layer_idx].copy_(ssm_snap)
        cache.conv_states[self.layer_idx].copy_(conv_snap)
        
        # Inject large noise into SSM
        noise_ssm = torch.randn_like(ssm_snap) * 100.0
        cache.ssm_states[self.layer_idx] += noise_ssm
        
        with torch.no_grad():
            out_ssm = self.model(last_token_id, cache_params=cache, cache_position=cache_pos)
            diff_ssm = (out_ssm.logits[0, -1] - base_logit).abs().max().item()
            print(f"  Logit Delta (SSM Injection): {diff_ssm:.6f}")
            
        # 3. Conv Sensitivity Probe
        print("\n[Sensitivity Probe] Modifying Conv State...")
        # Restore cache
        cache.ssm_states[self.layer_idx].copy_(ssm_snap)
        cache.conv_states[self.layer_idx].copy_(conv_snap)
        
        # Inject large noise into Conv
        noise_conv = torch.randn_like(conv_snap) * 100.0
        cache.conv_states[self.layer_idx] += noise_conv
        
        with torch.no_grad():
            out_conv = self.model(last_token_id, cache_params=cache, cache_position=cache_pos)
            diff_conv = (out_conv.logits[0, -1] - base_logit).abs().max().item()
            print(f"  Logit Delta (Conv Injection): {diff_conv:.6f}")

    def report(self):
        pass
