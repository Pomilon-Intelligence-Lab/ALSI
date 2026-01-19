from core.base_task import Task
from core.functional_mamba import functional_mamba_step
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import json
import matplotlib.pyplot as plt

def functional_mamba_block(layer, hidden_states, ssm_state_prev, conv_state_prev):
    norm_hidden = layer.norm(hidden_states)
    mixer_out, ssm_next, conv_next = functional_mamba_step(
        layer.mixer, norm_hidden, ssm_state_prev, conv_state_prev
    )
    output = hidden_states + mixer_out
    return output, ssm_next, conv_next

class FunctionalSensitivityScan(Task):
    def __init__(self):
        super().__init__("FunctionalSensitivityScan")
        self.layers_to_scan = [15, 17] # Refining the Sweet Spot
        self.prompt = "The password is '"
        self.target = "BLUE"
        self.results = []

    def run_layer(self, layer_idx):
        print(f"\n--- Scanning Layer {layer_idx} ---")
        
        # 1. Setup State
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        base_cache = out.cache_params
        last_token_id = input_ids[:, -1:]
        target_id = self.tokenizer.encode(self.target, add_special_tokens=False)[0]
        
        # 2. Capture Input to Start Layer
        layer_input = None
        def hook_fn(module, args):
            nonlocal layer_input
            layer_input = args[0].detach().clone()
        
        block = self.model.backbone.layers[layer_idx]
        handle = block.register_forward_pre_hook(hook_fn)
        
        with torch.no_grad():
            self.model(last_token_id, cache_params=base_cache, cache_position=torch.tensor([input_ids.shape[1]-1], device=self.device))
        handle.remove()
        
        # Refresh cache
        with torch.no_grad():
            out = self.model(input_ids[:, :-1], use_cache=True)
        base_cache = out.cache_params
        
        # 3. Optimize
        h_start = base_cache.ssm_states[layer_idx].detach().clone()
        delta = torch.zeros_like(h_start, requires_grad=True)
        optimizer = optim.Adam([delta], lr=0.1)
        
        final_rank = 9999
        steps_to_rank1 = 9999
        
        for step in range(50):
            optimizer.zero_grad()
            current_hidden = layer_input
            
            # Functional Chain
            for i in range(layer_idx, 24):
                layer = self.model.backbone.layers[i]
                ssm_p = base_cache.ssm_states[i].detach().clone()
                if i == layer_idx:
                    ssm_p = ssm_p + delta
                conv_p = base_cache.conv_states[i].detach().clone()
                current_hidden, _, _ = functional_mamba_block(layer, current_hidden, ssm_p, conv_p)
            
            norm_out = self.model.backbone.norm_f(current_hidden)
            logits = self.model.lm_head(norm_out)[0, -1]
            
            loss = F.cross_entropy(logits.view(1, -1), torch.tensor([target_id], device=self.device)) + 1e-4 * delta.norm()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                rank = (logits > logits[target_id]).sum().item() + 1
                final_rank = rank
                if rank == 1:
                    steps_to_rank1 = step
                    break
        
        print(f"  Rank: {final_rank} (Steps: {steps_to_rank1})")
        
        # 4. Verification (Coherence)
        base_cache.ssm_states[layer_idx] += delta.detach()
        curr_input_id = last_token_id
        current_pos = torch.tensor([input_ids.shape[1]-1], device=self.device)
        generated_tokens = []
        
        for _ in range(10):
            with torch.no_grad():
                out_gen = self.model(curr_input_id, cache_params=base_cache, cache_position=current_pos)
            next_token = torch.argmax(out_gen.logits[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())
            curr_input_id = next_token
            current_pos += 1
            
        gen_text = self.tokenizer.decode(generated_tokens)
        print(f"  Gen: {gen_text.replace(chr(10), ' ')}")
        
        # Repetition Metric (Simple unique token ratio)
        unique_ratio = len(set(generated_tokens)) / len(generated_tokens)
        
        self.results.append({
            "layer": layer_idx,
            "rank": final_rank,
            "steps": steps_to_rank1 if steps_to_rank1 < 9999 else 50,
            "coherence": unique_ratio,
            "text": gen_text
        })

    def run(self):
        print(f"[{self.name}] Running Functional Sensitivity Scan...")
        for layer in self.layers_to_scan:
            self.run_layer(layer)
            
        with open(os.path.join(self.output_dir, "functional_sensitivity.json"), "w") as f:
            json.dump(self.results, f, indent=2)

    def report(self):
        layers = [r['layer'] for r in self.results]
        ranks = [r['rank'] for r in self.results]
        coherence = [r['coherence'] for r in self.results]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Layer Depth')
        ax1.set_ylabel('Rank (Lower is Better)', color=color)
        ax1.plot(layers, ranks, color=color, marker='o', label='Control Rank')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log')
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Coherence (Unique Token Ratio)', color=color)
        ax2.plot(layers, coherence, color=color, marker='x', linestyle='--', label='Coherence')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1.1)
        
        plt.title("The ALSI Sweet Spot: Control vs Coherence")
        fig.tight_layout()
        path = os.path.join(self.output_dir, "sweet_spot.png")
        plt.savefig(path)
        print(f"[{self.name}] Plot saved to {path}")
