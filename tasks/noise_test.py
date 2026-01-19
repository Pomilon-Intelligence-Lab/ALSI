from core.base_task import Task
from core.psi import PsiMonitor
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import torch.optim as optim
import os
import json

class NoiseTest(Task):
    def __init__(self):
        super().__init__("NoiseTest")
        self.layer_idx = 7
        self.prompt = "The password is '"
        self.target = "BLUE"
        self.results = []

    def get_control_delta(self, h_prev_cache, last_token_id, context_len):
        """Optimizes a delta for 'BLUE' to get a reference norm."""
        target_id = self.tokenizer.encode(self.target, add_special_tokens=False)[0]
        base_shape = h_prev_cache.ssm_states[self.layer_idx].shape
        delta = torch.zeros(base_shape, device=self.device, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.1)
        cache_pos = torch.tensor([context_len], device=self.device)
        
        for _ in range(100):
            optimizer.zero_grad()
            base_states = h_prev_cache.ssm_states.detach()
            layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
            layers[self.layer_idx] = layers[self.layer_idx] + delta
            diff_cache = MockCache(torch.stack(layers), h_prev_cache.conv_states, self.model.config)
            out = self.model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
            loss = torch.nn.functional.cross_entropy(out.logits[0, -1].view(1, -1), torch.tensor([target_id], device=self.device))
            loss.backward()
            optimizer.step()
            if (out.logits[0, -1] > out.logits[0, -1, target_id]).sum().item() + 1 == 1:
                break
        return delta.detach()

    def run(self):
        print(f"[{self.name}] Comparing Semantic Forcing vs Random Noise...")
        psi = PsiMonitor(self.tokenizer)
        
        # 1. Setup State
        probe_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(probe_ids, use_cache=True)
        h_prev_cache = out.cache_params
        last_token_id = probe_ids[:, -1:]
        context_len = probe_ids.shape[1]
        
        # 2. Get Control (Semantic) Delta
        control_delta = self.get_control_delta(h_prev_cache, last_token_id, context_len)
        control_norm = control_delta.norm().item()
        print(f"  Control (BLUE) Norm: {control_norm:.2f}")
        
        # 3. Test Conditions
        conditions = [
            ("Semantic (BLUE)", control_delta),
            ("Random Noise 1", torch.randn_like(control_delta) * (control_norm / torch.randn_like(control_delta).norm())),
            ("Random Noise 2", torch.randn_like(control_delta) * (control_norm / torch.randn_like(control_delta).norm())),
            ("Zero Delta (Natural)", torch.zeros_like(control_delta))
        ]
        
        for name, delta in conditions:
            print(f"\n--- Condition: {name} ---")
            
            base_states = h_prev_cache.ssm_states.detach().clone()
            layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
            layers[self.layer_idx] = layers[self.layer_idx] + delta
            
            gen_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
            gen_cache.ssm_states = torch.stack(layers)
            gen_cache.conv_states = h_prev_cache.conv_states.detach().clone()
            
            # Greedy Generation
            out_gen = self.model.generate(
                input_ids=last_token_id, 
                cache_params=gen_cache, 
                max_new_tokens=15,
                do_sample=False
            )
            text = self.tokenizer.decode(out_gen[0])
            is_refusal, _ = psi.check_refusal(text)
            
            print(f"  Generated: {text.replace(chr(10), ' ')}")
            print(f"  Refusal: {is_refusal}")
            
            self.results.append({"condition": name, "text": text, "refusal": is_refusal})

        with open(os.path.join(self.output_dir, "noise_test_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

    def report(self):
        pass
