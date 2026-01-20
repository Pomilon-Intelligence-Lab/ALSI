import torch
import torch.nn as nn

class PhiFieldProjector(nn.Module):
    """
    Phi-T: Trajectory-Aware Field Projector (Memory-Efficient Recursive Version).
    """
    def __init__(self, state_dim, embed_dim, num_layers=12, hidden_dim=1024):
        super().__init__()
        self.state_dim = state_dim
        self.num_layers = num_layers
        
        self.layer_embed = nn.Embedding(num_layers, 64)
        
        # 196k -> 1024 (The Bottleneck)
        self.input_proj = nn.Linear(state_dim + embed_dim + 64, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.mid_net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # 1024 -> 196k (The Expansion)
        self.output_proj = nn.Linear(hidden_dim, state_dim)
        
        # Initialize output to zero to start with 'Identity' behavior
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, h, t, l): 
        # h: [B, State_Dim], t: [B, Embed_Dim], l: int or LongTensor
        if isinstance(l, int):
            l = torch.full((h.size(0),), l, device=h.device, dtype=torch.long)
        
        l_e = self.layer_embed(l)
        x = torch.cat([h, t, l_e], dim=-1)
        
        x = self.norm1(self.input_proj(x))
        x = self.mid_net(x)
        return self.output_proj(x)