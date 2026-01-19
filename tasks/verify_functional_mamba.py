from core.base_task import Task
from core.functional_mamba import functional_mamba_step
import torch

class VerifyFunctionalMamba(Task):
    def __init__(self):
        super().__init__("VerifyFunctionalMamba")
        self.layer_idx = 7
        self.prompt = "The password is '"

    def run(self):
        print(f"[{self.name}] Verifying Functional Mamba Step...")
        
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Pre-fill to get state
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        cache = out.cache_params
        
        last_token_id = input_ids[:, -1:]
        cache_pos = torch.tensor([input_ids.shape[1] - 1], device=self.device)
        
        # 2. Native Step
        print("Running Native Step...")
        # Clone state for comparison
        ssm_prev = cache.ssm_states[self.layer_idx].clone()
        conv_prev = cache.conv_states[self.layer_idx].clone()
        
        with torch.no_grad():
            # We need to access the MIXER output, not the full model output
            # But the full model output depends on the mixer.
            # Let's run the full layer manually using the module.
            
            layer = self.model.backbone.layers[self.layer_idx]
            
            # We need hidden_states input to the layer.
            # We can get this by running the model up to layer_idx-1.
            # Or we can just mock the input for this test?
            # Mocking input is safer for unit testing the function itself.
            
            hidden_dim = self.model.config.hidden_size
            hidden_states = torch.randn(1, 1, hidden_dim, device=self.device, dtype=self.model.dtype)
            
            # Native Forward on Mixer
            # Note: layer.mixer modifies cache in-place
            native_out = layer.mixer(hidden_states, cache_params=cache, cache_position=cache_pos)
            
            ssm_next_native = cache.ssm_states[self.layer_idx]
            conv_next_native = cache.conv_states[self.layer_idx]
            
        # 3. Functional Step
        print("Running Functional Step...")
        # Restore prev state inputs
        # functional_step takes: layer, hidden_states, ssm_prev, conv_prev
        
        func_out, ssm_next_func, conv_next_func = functional_mamba_step(
            layer.mixer,
            hidden_states,
            ssm_prev,
            conv_prev
        )
        
        # 4. Compare
        diff_out = (native_out - func_out).abs().max().item()
        diff_ssm = (ssm_next_native - ssm_next_func).abs().max().item()
        diff_conv = (conv_next_native - conv_next_func).abs().max().item()
        
        print(f"\n--- Results ---")
        print(f"Output Diff: {diff_out:.6f}")
        print(f"SSM State Diff: {diff_ssm:.6f}")
        print(f"Conv State Diff: {diff_conv:.6f}")
        
        if diff_out < 1e-4 and diff_ssm < 1e-4:
            print("SUCCESS: Functional implementation matches Native!")
        else:
            print("FAILURE: Implementation divergence.")

    def report(self):
        pass
