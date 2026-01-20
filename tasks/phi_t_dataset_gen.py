from core.base_task import Task
from core.functional_mamba import functional_mamba_step
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle

def functional_mamba_block(layer, hidden_states, ssm_state_prev, conv_state_prev):
    norm_hidden = layer.norm(hidden_states)
    mixer_out, ssm_next, conv_next = functional_mamba_step(
        layer.mixer, norm_hidden, ssm_state_prev, conv_state_prev
    )
    output = hidden_states + mixer_out
    return output, ssm_next, conv_next

class PhiTDatasetGen(Task):
    def __init__(self):
        super().__init__("PhiTDatasetGen")
        self.start_layer = 12
        self.end_layer = 24
        self.dataset_path = os.path.join(self.output_dir, "phi_t_dataset.pkl")
        
        self.prompts = ["The password is '", "The color of the sky is ", "The capital of France is "]
        self.targets = ["BLUE", "RED", "GREEN", "PARIS", "LONDON"]

    def optimize_field(self, h_prev_cache, last_token_id, context_len, target_str):
        target_id = self.tokenizer.encode(target_str, add_special_tokens=False)[0]
        
        # Capture layer_input
        layer_input = None
        def hook_fn(module, args):
            nonlocal layer_input
            layer_input = args[0].detach().clone()
        handle = self.model.backbone.layers[self.start_layer].register_forward_pre_hook(hook_fn)
        with torch.no_grad():
            self.model(last_token_id, cache_params=h_prev_cache, cache_position=torch.tensor([context_len], device=self.device))
        handle.remove()
        
        deltas = [torch.zeros_like(h_prev_cache.ssm_states[i], requires_grad=True) for i in range(self.start_layer, self.end_layer)]
        optimizer = optim.Adam(deltas, lr=0.1)
        
        for step in range(30):
            optimizer.zero_grad()
            h = layer_input
            for i in range(self.start_layer, self.end_layer):
                idx = i - self.start_layer
                ssm_p = h_prev_cache.ssm_states[i].detach().clone() + deltas[idx]
                h, _, _ = functional_mamba_block(self.model.backbone.layers[i], h, ssm_p, h_prev_cache.conv_states[i].detach())
            
            logits = self.model.lm_head(self.model.backbone.norm_f(h))[0, -1]
            loss = F.cross_entropy(logits.view(1, -1), torch.tensor([target_id], device=self.device)) + 1e-4 * sum(d.norm() for d in deltas)
            loss.backward()
            optimizer.step()
            if (logits > logits[target_id]).sum().item() == 0: break
                
        return torch.stack([d.detach().cpu() for d in deltas])

    def run(self):
        print(f"[{self.name}] Generating Phi-T Dataset...")
        dataset = []
        for prompt in self.prompts:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad(): out = self.model(ids[:, :-1], use_cache=True)
            cache = out.cache_params
            last_token_id = ids[:, -1:]
            ctx_len = ids.shape[1] - 1
            for target in self.targets:
                field = self.optimize_field(cache, last_token_id, ctx_len, target)
                dataset.append({"h": cache.ssm_states[self.start_layer].detach().cpu(), "t": self.tokenizer.encode(target, add_special_tokens=False)[0], "field": field})
        
        with open(self.dataset_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Generated {len(dataset)} samples.")

    def report(self):
        pass