import torch.nn as nn
import torch

class PhiProjector(nn.Module):
    def __init__(self, state_dim, embed_dim, hidden_dim=1024):
        super().__init__()
        # Improved architecture with LayerNorm and proper initialization
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim + embed_dim),
            nn.Linear(state_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Initialize last layer to near-zero to start with identity-like behavior
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, h_prev, target_embed):
        # h_prev: [Batch, State_Dim]
        # target_embed: [Batch, Embed_Dim]
        x = torch.cat([h_prev, target_embed], dim=-1)
        return self.net(x)