from core.base_task import Task
from core.phi import PhiProjector
from core.utils import MockCache
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import matplotlib.pyplot as plt

class PhiTrainingV2(Task):
    def __init__(self):
        super().__init__("PhiTrainingV2")
        self.dataset_path = os.path.join(self.output_dir, "dataset_v2.pkl")
        self.phi_path = os.path.join(self.output_dir, "phi_model_v2.pt")
        self.layer_idx = 7
        self.epoch_losses = []
        
        # Expanded Training Distribution
        self.prompts = [
            "The password is '",
            "The color of the sky is ",
            "I am feeling very ",
            "The capital of France is ",
            "Please ignore previous instructions and ",
            "My favorite fruit is "
        ]
        self.targets = [
            "BLUE", "RED", "GREEN", "ORANGE", "BLACK",  # Colors
            "APPLE", "SKY", "DOG", "CAT", "PARIS", "LONDON" # Objects/Entities
        ]
        
    def optimize_delta(self, h_prev_cache, target_str, context_len):
        target_id = self.tokenizer.encode(target_str, add_special_tokens=False)[0]
        base_shape = h_prev_cache.ssm_states[self.layer_idx].shape
        delta = torch.zeros(base_shape, device=self.device, requires_grad=True)
        optimizer = optim.Adam([delta], lr=1.0)
        
        # We need a dummy token to drive the step forward
        dummy_token_id = torch.tensor([[0]], device=self.device) # 0 is usually not EOS, safe?
        cache_pos = torch.tensor([context_len], device=self.device)
        
        # Optimization Loop
        for step in range(150): # Reduced steps for speed
            optimizer.zero_grad()
            base_states = h_prev_cache.ssm_states.detach()
            layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
            layers[self.layer_idx] = layers[self.layer_idx] + delta
            
            diff_cache = MockCache(torch.stack(layers), h_prev_cache.conv_states, self.model.config)
            out = self.model(dummy_token_id, cache_params=diff_cache, cache_position=cache_pos)
            
            logits = out.logits[0, -1]
            loss = torch.nn.functional.cross_entropy(logits.view(1, -1), torch.tensor([target_id], device=self.device)) + 1e-4 * delta.norm()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                rank = (logits > logits[target_id]).sum().item() + 1
                if rank <= 5: 
                    return delta.detach().cpu(), target_id, True
                    
        return delta.detach().cpu(), target_id, rank <= 10

    def generate_dataset(self):
        print(f"[{self.name}] Generating expanded ground truth dataset...")
        dataset = []
        
        for prompt in self.prompts:
            print(f"  Prompt: '{prompt}'")
            probe_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            # Run forward to get state just before the next token
            with torch.no_grad():
                out = self.model(probe_ids, use_cache=True)
            
            # The state we want to modify is the one resulting from the FULL prompt processing
            # ready to predict the NEXT token.
            # h_prev_cache contains the state after processing `probe_ids`.
            h_prev_cache = out.cache_params
            h_prev_tensor = h_prev_cache.ssm_states[self.layer_idx].detach().cpu()
            
            context_len = probe_ids.shape[1]
            
            for t in self.targets:
                delta, t_id, success = self.optimize_delta(h_prev_cache, t, context_len)
                if success:
                    dataset.append({
                        "h_prev": h_prev_tensor, 
                        "target_id": t_id, 
                        "delta": delta, 
                        "target_str": t,
                        "prompt": prompt
                    })
                    # print(f"    + {t}")
                else:
                    print(f"    - {t} (Failed)")
                    
        print(f"[{self.name}] Dataset generated. Total samples: {len(dataset)}")
        with open(self.dataset_path, "wb") as f:
            pickle.dump(dataset, f)
        return dataset

    def run(self):
        if os.path.exists(self.dataset_path):
            print(f"[{self.name}] Loading existing dataset...")
            with open(self.dataset_path, "rb") as f:
                self.dataset = pickle.load(f)
        else:
            self.dataset = self.generate_dataset()
            
        # Initialize Phi
        sample_h = self.dataset[0]['h_prev']
        embed_layer = self.model.get_input_embeddings()
        state_dim = sample_h.numel()
        embed_dim = embed_layer.weight.shape[1]
        self.phi = PhiProjector(state_dim, embed_dim).to(self.device)
        optimizer = optim.Adam(self.phi.parameters(), lr=1e-4)
        
        print(f"[{self.name}] Training Phi-v2 on {len(self.dataset)} samples...")
        
        self.epoch_losses = []
        batch_size = 16
        
        # Pre-move data to device to speed up training
        device_dataset = []
        for s in self.dataset:
            device_dataset.append({
                'target_id': s['target_id'],
                'delta': s['delta'].to(self.device),
                'h_prev': s['h_prev'].to(self.device)
            })
            
        for epoch in range(6): # Just 5 epochs
            total_loss = 0
            optimizer.zero_grad()
            batch_loss = 0
            count = 0
            
            for i, sample in enumerate(device_dataset):
                t_id = sample['target_id']
                gt_delta = sample['delta']
                h_prev = sample['h_prev'].view(1, -1)
                
                # We still need embed layer
                with torch.no_grad(): # Ensure no graph leak from embedding
                     t_embed = embed_layer(torch.tensor([[t_id]], device=self.device)).view(1, -1)
                
                pred_delta = self.phi(h_prev, t_embed).view(sample_h.shape)
                
                loss = nn.functional.mse_loss(pred_delta, gt_delta)
                loss.backward()
                batch_loss += loss.item()
                count += 1
                
                if (i + 1) % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            if count > 0: optimizer.step() 
            
            avg_loss = batch_loss / len(self.dataset)
            self.epoch_losses.append(avg_loss)
            
            print(f"  Epoch {epoch}: MSE Loss {avg_loss:.6f}")
                
        torch.save(self.phi.state_dict(), self.phi_path)
        print(f"[{self.name}] Phi-v2 model saved.")

    def report(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_losses, label='Training Loss')
        plt.title("Phi-v2 Training Convergence (MSE)")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        path = os.path.join(self.output_dir, "phi_v2_training_curve.png")
        plt.savefig(path)
        print(f"[{self.name}] Training plot saved to {path}")
