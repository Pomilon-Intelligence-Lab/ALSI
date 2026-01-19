from core.base_task import Task
from core.phi import PhiProjector
from core.utils import MockCache
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import matplotlib.pyplot as plt

class PhiTraining(Task):
    def __init__(self, targets=None):
        super().__init__("PhiTraining")
        self.targets = targets or ["BLUE", "RED", "GREEN", "ORANGE", "YELLOW", "BLACK", "WHITE", "PURPLE", "GOLD", "SILVER"]
        self.dataset_path = os.path.join(self.output_dir, "dataset.pkl")
        self.phi_path = os.path.join(self.output_dir, "phi_model.pt")
        self.layer_idx = 7
        self.probe_text = "The password is '"
        self.epoch_losses = []
        
    def optimize_delta(self, h_prev_cache, target_str):
        target_id = self.tokenizer.encode(target_str, add_special_tokens=False)[0]
        base_shape = h_prev_cache.ssm_states[self.layer_idx].shape
        delta = torch.zeros(base_shape, device=self.device, requires_grad=True)
        optimizer = optim.Adam([delta], lr=1.0)
        
        last_token_id = self.tokenizer(self.probe_text, return_tensors="pt").input_ids[:, -1:].to(self.device)
        ctx_len = self.tokenizer(self.probe_text, return_tensors="pt").input_ids.shape[1] - 1
        cache_pos = torch.tensor([ctx_len], device=self.device)
        
        for step in range(200):
            optimizer.zero_grad()
            base_states = h_prev_cache.ssm_states.detach()
            layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
            layers[self.layer_idx] = layers[self.layer_idx] + delta
            diff_cache = MockCache(torch.stack(layers), h_prev_cache.conv_states, self.model.config)
            out = self.model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
            logits = out.logits[0, -1]
            loss = torch.nn.functional.cross_entropy(logits.view(1, -1), torch.tensor([target_id], device=self.device)) + 1e-5 * delta.norm()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                rank = (logits > logits[target_id]).sum().item() + 1
                if rank <= 5: return delta.detach().cpu(), target_id, True
        return delta.detach().cpu(), target_id, rank <= 10

    def generate_dataset(self):
        print(f"[{self.name}] Generating ground truth dataset...")
        probe_ids = self.tokenizer(self.probe_text, return_tensors="pt").input_ids.to(self.device)
        context_ids = probe_ids[:, :-1]
        with torch.no_grad():
            out = self.model(context_ids, use_cache=True)
        h_prev_cache = out.cache_params
        self.h_prev_tensor = h_prev_cache.ssm_states[self.layer_idx].detach().cpu()
        dataset = []
        for t in self.targets:
            delta, t_id, success = self.optimize_delta(h_prev_cache, t)
            if success:
                dataset.append({"h_prev": self.h_prev_tensor, "target_id": t_id, "delta": delta, "target_str": t})
                print(f"  + {t}")
            else:
                print(f"  - {t} (Failed)")
        with open(self.dataset_path, "wb") as f:
            pickle.dump(dataset, f)
        return dataset

    def run(self):
        if os.path.exists(self.dataset_path):
            print(f"[{self.name}] Loading existing dataset...")
            with open(self.dataset_path, "rb") as f:
                self.dataset = pickle.load(f)
            self.h_prev_tensor = self.dataset[0]['h_prev']
        else:
            self.dataset = self.generate_dataset()
            
        embed_layer = self.model.get_input_embeddings()
        state_dim = self.h_prev_tensor.numel()
        embed_dim = embed_layer.weight.shape[1]
        self.phi = PhiProjector(state_dim, embed_dim).to(self.device)
        optimizer = optim.Adam(self.phi.parameters(), lr=1e-4)
        
        print(f"[{self.name}] Training Phi...")
        probe_ids = self.tokenizer(self.probe_text, return_tensors="pt").input_ids.to(self.device)
        context_ids = probe_ids[:, :-1]
        last_token_id = probe_ids[:, -1:]
        with torch.no_grad():
            out = self.model(context_ids, use_cache=True)
        base_cache = out.cache_params
        cache_pos = torch.tensor([context_ids.shape[1]], device=self.device)
        assert cache_pos.item() == context_ids.shape[1], "Cache position mismatch"
        
        self.epoch_losses = []
        
        for epoch in range(101):
            total_loss = 0
            for sample in self.dataset:
                optimizer.zero_grad()
                t_id = sample['target_id']
                gt_delta = sample['delta'].to(self.device)
                h_prev = sample['h_prev'].to(self.device).view(1, -1)
                t_embed = embed_layer(torch.tensor([[t_id]], device=self.device)).view(1, -1)
                pred_delta = self.phi(h_prev, t_embed).view(self.h_prev_tensor.shape)
                l_match = nn.functional.mse_loss(pred_delta, gt_delta)
                
                base_states = base_cache.ssm_states.detach().clone()
                layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
                layers[self.layer_idx] = layers[self.layer_idx] + pred_delta
                diff_cache = MockCache(torch.stack(layers), base_cache.conv_states, self.model.config)
                out = self.model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
                l_ctl = nn.functional.cross_entropy(out.logits[0, -1].view(1, -1), torch.tensor([t_id], device=self.device))
                loss = l_ctl + l_match
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss/len(self.dataset)
            self.epoch_losses.append(avg_loss)
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Loss {avg_loss:.4f}")
                
        torch.save(self.phi.state_dict(), self.phi_path)
        print(f"[{self.name}] Phi model saved.")

    def report(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_losses, label='Training Loss')
        plt.title("Phi Projector Training Convergence")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Control + Match)")
        plt.legend()
        plt.grid(True)
        path = os.path.join(self.output_dir, "phi_training_curve.png")
        plt.savefig(path)
        
        # Copy to docs/images
        docs_path = "docs/images/phi_training_curve.png"
        plt.savefig(docs_path)
        print(f"[{self.name}] Training plot saved to {path} and {docs_path}")