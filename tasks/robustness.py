from core.base_task import Task
from core.phi import PhiProjector
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import os
import pickle

class RobustnessTest(Task):
    def __init__(self):
        super().__init__("RobustnessTest")
        self.phi_path = "ALSI/results/PhiTraining/phi_model.pt"
        self.dataset_path = "ALSI/results/PhiTraining/dataset.pkl"
        self.layer_idx = 7
        self.probe_text = "The password is '"
        self.unseen_targets = ["PINK", "CYAN", "BROWN", "NAVY"]
        
    def run(self):
        print(f"[{self.name}] Running robustness tests...")
        
        # Load Data context
        with open(self.dataset_path, "rb") as f:
            dataset = pickle.load(f)
        h_prev_tensor = dataset[0]['h_prev']
        
        # Load Phi
        embed_layer = self.model.get_input_embeddings()
        state_dim = h_prev_tensor.numel()
        embed_dim = embed_layer.weight.shape[1]
        
        phi = PhiProjector(state_dim, embed_dim).to(self.device)
        phi.load_state_dict(torch.load(self.phi_path))
        phi.eval()
        
        # Context Setup
        probe_ids = self.tokenizer(self.probe_text, return_tensors="pt").input_ids.to(self.device)
        context_ids = probe_ids[:, :-1]
        last_token_id = probe_ids[:, -1:]
        
        with torch.no_grad():
            out = self.model(context_ids, use_cache=True)
        base_cache = out.cache_params
        cache_pos = torch.tensor([context_ids.shape[1]], device=self.device)
        
        # 1. Zero Shot
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
            
        # 2. Generation Stability
        print("--- Generation Stability ---")
        gen_target = "BLUE"
        t_id = self.tokenizer.encode(gen_target, add_special_tokens=False)[0]
        t_embed = embed_layer(torch.tensor([[t_id]], device=self.device)).view(1, -1)
        pred_delta = phi(h_prev, t_embed).view(h_prev_tensor.shape)
        
        base_states = base_cache.ssm_states.detach().clone()
        layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
        layers[self.layer_idx] = layers[self.layer_idx] + pred_delta
        
        # Use real cache for generation
        gen_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
        gen_cache.ssm_states = torch.stack(layers)
        gen_cache.conv_states = base_cache.conv_states.detach().clone()
        
        out_gen = self.model.generate(input_ids=last_token_id, cache_params=gen_cache, max_new_tokens=20)
        print(f"Injected: {gen_target}")
        print(f"Generated: {self.tokenizer.decode(out_gen[0])}")

    def report(self):
        pass
