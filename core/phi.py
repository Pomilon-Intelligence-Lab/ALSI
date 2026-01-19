import torch.nn as nn
import torch

class PhiProjector(nn.Module):
    def __init__(self, state_dim, embed_dim, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, h_prev, target_embed):
        # h_prev: [Batch, State_Dim]
        # target_embed: [Batch, Embed_Dim]
        x = torch.cat([h_prev, target_embed], dim=-1)
        return self.net(x)
