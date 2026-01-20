from core.base_task import Task
from core.phi_t import PhiFieldProjector
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import matplotlib.pyplot as plt

class PhiTTrainer(Task):
    def __init__(self):
        super().__init__("PhiTTrainer")
        self.dataset_path = "ALSI/results/PhiTDatasetGen/phi_t_dataset.pkl"
        self.model_path = os.path.join(self.output_dir, "phi_t_model.pt")
        self.epoch_losses = []

    def run(self):
        print(f"[{self.name}] Training Recursive Phi-T Projector...")
        if not os.path.exists(self.dataset_path): return
        with open(self.dataset_path, "rb") as f: dataset = pickle.load(f)
            
        h_dim = dataset[0]['h'].numel()
        e_dim = self.model.get_input_embeddings().weight.shape[1]
        self.phi_t = PhiFieldProjector(h_dim, e_dim).to(self.device)
        optimizer = optim.Adam(self.phi_t.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        for epoch in range(101):
            total_loss = 0
            for s in dataset:
                h = s['h'].to(self.device).view(1, -1)
                t_id = torch.tensor([[s['t']]], device=self.device)
                t_e = self.model.get_input_embeddings()(t_id).squeeze(1).detach()
                
                optimizer.zero_grad()
                sample_loss = 0
                for l in range(12):
                    pred = self.phi_t(h, t_e, l)
                    target = s['field'][l].to(self.device).view(1, -1)
                    loss = criterion(pred, target)
                    loss.backward()
                    sample_loss += loss.item()
                optimizer.step()
                total_loss += sample_loss
            
            avg_loss = total_loss / (len(dataset) * 12)
            self.epoch_losses.append(avg_loss)
            if epoch % 20 == 0: print(f"Epoch {epoch} | Loss: {avg_loss:.6f}")
                
        torch.save(self.phi_t.state_dict(), self.model_path)

    def report(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_losses)
        plt.title("Phi-T Recursive Training Loss")
        plt.show()