from core.base_task import Task
from core.phi import PhiProjector
from core.psi import PsiMonitor
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import os
import pickle
import json
import matplotlib.pyplot as plt

class RobustnessPsiTest(Task):
    def __init__(self):
        super().__init__("RobustnessPsiTest")
        self.phi_path = "ALSI/results/PhiTraining/phi_model.pt"
        self.dataset_path = "ALSI/results/PhiTraining/dataset.pkl"
        self.layer_idx = 7
        self.probe_text = "The password is '"
        self.unseen_targets = ["PINK", "CYAN", "BROWN", "NAVY"]
        self.generation_targets = [
            "BLUE", "RED", "GREEN", "ORANGE", "YELLOW", "BLACK", 
            "WHITE", "PURPLE", "GOLD", "SILVER", 
            "APPLE", "SKY", "DOG", "CAT"
        ]
        self.ranks = {}
        self.failures = []
        
    def run(self):
        print(f"[{self.name}] Running robustness tests with Psi Monitor...")
        psi = PsiMonitor(self.tokenizer)
        
        with open(self.dataset_path, "rb") as f:
            dataset = pickle.load(f)
        h_prev_tensor = dataset[0]['h_prev']
        
        embed_layer = self.model.get_input_embeddings()
        state_dim = h_prev_tensor.numel()
        embed_dim = embed_layer.weight.shape[1]
        
        phi = PhiProjector(state_dim, embed_dim).to(self.device)
        phi.load_state_dict(torch.load(self.phi_path))
        phi.eval()
        
        probe_ids = self.tokenizer(self.probe_text, return_tensors="pt").input_ids.to(self.device)
        context_ids = probe_ids[:, :-1]
        last_token_id = probe_ids[:, -1:]
        
        with torch.no_grad():
            out = self.model(context_ids, use_cache=True)
        base_cache = out.cache_params
        cache_pos = torch.tensor([context_ids.shape[1]], device=self.device)
        
        print("--- Zero Shot Generalization ---")
        for t in self.unseen_targets:
            t_id = self.tokenizer.encode(t, add_special_tokens=False)[0]
            t_embed = embed_layer(torch.tensor([[t_id]], device=self.device)).view(1, -1)
            h_prev = h_prev_tensor.to(self.device).view(1, -1)
            
            pred_delta = phi(h_prev, t_embed).view(h_prev_tensor.shape)
            
            base_states = base_cache.ssm_states.detach().clone()
            layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
            layers[self.layer_idx] = layers[self.layer_idx] + pred_delta
            
            diff_cache = MockCache(torch.stack(layers), base_cache.conv_states, self.model.config)
            out = self.model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
            logits = out.logits[0, -1]
            
            rank = (logits > logits[t_id]).sum().item() + 1
            prob = torch.softmax(logits, dim=-1)[t_id].item()
            print(f"Target: {t} | Rank: {rank} | Prob: {prob:.4f}")
            self.ranks[t] = rank
            
        print("--- Generation Stability & Psi Check ---")
        for gen_target in self.generation_targets:
            t_id = self.tokenizer.encode(gen_target, add_special_tokens=False)[0]
            t_embed = embed_layer(torch.tensor([[t_id]], device=self.device)).view(1, -1)
            
            # Injection
            pred_delta = phi(h_prev_tensor.to(self.device).view(1, -1), t_embed).view(h_prev_tensor.shape)
            
            base_states = base_cache.ssm_states.detach().clone()
            layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
            layers[self.layer_idx] = layers[self.layer_idx] + pred_delta
            
            # Measure Rank
            diff_cache = MockCache(torch.stack(layers), base_cache.conv_states, self.model.config)
            out = self.model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
            rank = (out.logits[0, -1] > out.logits[0, -1, t_id]).sum().item() + 1
            
            # Generate
            gen_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
            gen_cache.ssm_states = torch.stack(layers)
            gen_cache.conv_states = base_cache.conv_states.detach().clone()
            
            out_gen = self.model.generate(input_ids=last_token_id, cache_params=gen_cache, max_new_tokens=20)
            generated_text = self.tokenizer.decode(out_gen[0])
            
            # Psi Check
            is_refusal, phrase = psi.check_refusal(generated_text)
            
            print(f"[{gen_target}] Rank: {rank} | Refusal: {is_refusal}")
            if is_refusal:
                print(f"  -> Generated: {generated_text.replace(chr(10), ' ')}")
                
            if rank <= 5 and is_refusal:
                self.failures.append({
                    "target": gen_target,
                    "rank": rank,
                    "generated": generated_text,
                    "trigger_phrase": phrase
                })
        
        with open(os.path.join(self.output_dir, "psi_failures.json"), "w") as f:
            json.dump(self.failures, f, indent=2)
        print(f"[{self.name}] Saved {len(self.failures)} collision cases to psi_failures.json")

    def report(self):
        plt.figure(figsize=(8, 5))
        names = list(self.ranks.keys())
        values = list(self.ranks.values())
        
        plt.bar(names, values, color='skyblue')
        plt.yscale('log')
        plt.title("Zero-Shot Injection Ranks (Log Scale)")
        plt.ylabel("Rank (Lower is Better)")
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(values):
            plt.text(i, v, str(v), ha='center', va='bottom')
            
        path = os.path.join(self.output_dir, "robustness_psi_ranks.png")
        plt.savefig(path)
        print(f"[{self.name}] Robustness plot saved to {path}")
