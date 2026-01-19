from core.base_task import Task
from core.utils import MockCache
import torch

class VerifyMockEquivalence(Task):
    def __init__(self):
        super().__init__("VerifyMockEquivalence")
        self.prompt = "The password is '"

    def run(self):
        print(f"[{self.name}] Verifying MockCache vs Mamba2Cache Equivalence...")
        
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Setup Real Cache
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        real_cache = out.cache_params
        
        last_token_id = input_ids[:, -1:]
        cache_pos = torch.tensor([input_ids.shape[1] - 1], device=self.device)
        
        # 2. Run Real Forward
        with torch.no_grad():
            # Clone to preserve state for Mock
            # (We need to manually reconstruct inputs for Mock because it wraps tensors)
            # Actually, MockCache TAKES the tensors.
            
            # We need independent copies for the test
            real_ssm = real_cache.ssm_states.clone()
            real_conv = real_cache.conv_states.clone()
            
            out_real = self.model(last_token_id, cache_params=real_cache, cache_position=cache_pos)
            logit_real = out_real.logits[0, -1]
            
        # 3. Setup Mock Cache
        # MockCache expects:
        # ssm_states: [24, 1, 24, 64, 128] (Monolithic) -> Split into list?
        # Let's see core/utils.py again.
        # "self.ssm_states = ssm_states"
        # It just stores whatever we give it.
        
        # The critical part is how `modeling_mamba2.py` USES the cache.
        # It accesses `cache_params.ssm_states[layer_idx]`.
        # If passed a Tensor, `tensor[i]` gives the i-th slice.
        # If passed a List, `list[i]` gives the i-th element.
        
        # So we construct MockCache with the CLONED tensors.
        mock_cache = MockCache(real_ssm, real_conv, self.model.config)
        # MockCache needs seq_len_offset
        # mock_cache.seq_len_offset = real_cache.seq_len_offset # Mamba2Cache doesn't have this
        
        # 4. Run Mock Forward
        with torch.no_grad():
            out_mock = self.model(last_token_id, cache_params=mock_cache, cache_position=cache_pos)
            logit_mock = out_mock.logits[0, -1]
            
        # 5. Compare
        diff = (logit_real - logit_mock).abs().max().item()
        print(f"Logit Difference (Real vs Mock): {diff:.6f}")
        
        if diff < 1e-4:
            print("SUCCESS: MockCache is equivalent.")
        else:
            print("FAILURE: MockCache behavior diverges from RealCache.")

    def report(self):
        pass
