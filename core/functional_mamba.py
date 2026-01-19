import torch
import torch.nn.functional as F

def functional_mamba_step(layer, hidden_states, ssm_state_prev, conv_state_prev):
    """
    Functional implementation of a single Mamba-2 step.
    Does NOT modify states in-place.
    
    Args:
        layer: The Mamba2Mixer layer module (contains weights).
        hidden_states: [batch, 1, hidden_size] (Input x_t)
        ssm_state_prev: [batch, num_heads, head_dim, state_size]
        conv_state_prev: [batch, conv_dim, kernel_size]
        
    Returns:
        output: [batch, 1, hidden_size]
        ssm_state_next: [batch, num_heads, head_dim, state_size]
        conv_state_next: [batch, conv_dim, kernel_size]
    """
    batch_size, seq_len, _ = hidden_states.shape
    dtype = hidden_states.dtype
    device = hidden_states.device
    
    # 1. Gated MLP's linear projection
    # hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask) # Assume no padding/mask for single step
    projected_states = layer.in_proj(hidden_states)
    
    # Calculate splits
    # d_mlp = (projected_states.shape[-1] - 2 * layer.intermediate_size - 2 * layer.n_groups * layer.ssm_state_size - layer.num_heads) // 2
    # The split logic in source:
    # _, _, gate, hidden_states_B_C, dt = projected_states.split(...)
    
    # We need to replicate the split logic exactly.
    # From source:
    # conv_dim = intermediate_size + 2 * n_groups * ssm_state_size
    # projection_size = intermediate_size + conv_dim + num_heads
    # So split is: [d_mlp, d_mlp, intermediate, conv_dim, num_heads]?
    # Wait, the source code had:
    # d_mlp = (projected_states.shape[-1] - 2 * intermediate - 2 * n_groups * ssm - num_heads) // 2
    # And split: [d_mlp, d_mlp, intermediate, conv_dim, num_heads]
    
    # Let's re-calculate to be safe.
    d_mlp = (projected_states.shape[-1] - 2 * layer.intermediate_size - 2 * layer.n_groups * layer.ssm_state_size - layer.num_heads) // 2
    
    _, _, gate, hidden_states_B_C, dt = projected_states.split(
        [d_mlp, d_mlp, layer.intermediate_size, layer.conv_dim, layer.num_heads], dim=-1
    )
    
    # 2. Convolution (Functional Update)
    # hidden_states_B_C shape: [batch, 1, conv_dim]
    # conv_state_prev shape: [batch, conv_dim, kernel_size]
    
    # Shift: Drop first column, append new input
    # Input to conv needs to be transposed?
    # Source: cache_params.update_conv_state(..., new_conv_state=hidden_states_B_C)
    # Source Conv Update: conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    # conv_state[:, :, -1] = new_conv_state
    
    # Functional equivalent:
    # hidden_states_B_C is [B, 1, D]. We need [B, D, 1] for concat?
    new_conv_input = hidden_states_B_C.transpose(1, 2) # [B, D, 1]
    conv_state_next = torch.cat([conv_state_prev[:, :, 1:], new_conv_input], dim=-1)
    
    # Apply Conv1d
    # Source: torch.sum(conv_states * self.conv1d.weight.squeeze(1), dim=-1)
    # conv1d.weight shape: [out_C, in_C/groups, K] -> [D, 1, K] since groups=D
    # squeeze(1) -> [D, K]
    
    # conv_state_next: [B, D, K]
    # weight: [D, K]
    # Element-wise mul then sum over K
    
    conv_out = torch.sum(conv_state_next * layer.conv1d.weight.squeeze(1), dim=-1) # [B, D]
    
    if layer.use_conv_bias:
        conv_out = conv_out + layer.conv1d.bias
        
    # Activation
    hidden_states_B_C_act = layer.act(conv_out) # [B, D]
    
    # Split B, C, hidden
    # D = intermediate + 2 * groups * ssm_state
    groups_time_state_size = layer.n_groups * layer.ssm_state_size
    
    hidden_states_inner, B, C = torch.split(
        hidden_states_B_C_act,
        [layer.intermediate_size, groups_time_state_size, groups_time_state_size],
        dim=-1
    )
    
    # 3. SSM Transformation
    A = -torch.exp(layer.A_log.float()) # [num_heads]
    
    # Dimensions
    # dt: [B, 1, num_heads]
    # A: [num_heads]
    
    # Prepare dt
    # dt = dt[:, 0, :][:, None, ...] # [B, 1, num_heads]
    # dt = dt.transpose(1, 2) # [B, num_heads, 1]
    # dt = dt.expand(batch_size, num_heads, head_dim) # [B, H, P] ?
    
    # Source logic:
    # dt = dt[:, 0, :][:, None, ...] # [B, 1, H]
    # dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], layer.head_dim) # [B, H, P]
    dt_in = dt[:, 0, :].unsqueeze(1).transpose(1, 2).expand(batch_size, layer.num_heads, layer.head_dim)
    
    dt_bias = layer.dt_bias[..., None].expand(layer.num_heads, layer.head_dim)
    
    dt_soft = torch.nn.functional.softplus(dt_in + dt_bias.to(dtype))
    # Time step limits? Not critical for now, usually default.
    
    # Prepare A
    # A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size)
    A_exp = A[..., None, None].expand(layer.num_heads, layer.head_dim, layer.ssm_state_size).to(dtype=torch.float32)
    
    # Calculate dA
    # dA = torch.exp(dt * A)
    # dt_soft: [B, H, P]
    # A_exp: [H, P, N]
    # dt[..., None]: [B, H, P, 1]
    dA = torch.exp(dt_soft[..., None] * A_exp) # [B, H, P, N]
    
    # Prepare B
    # B: [B, n_groups * state_size]
    # Reshape logic from source:
    # B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
    # B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
    # B = B.reshape(batch_size, -1, B.shape[-1])
    
    B_reshaped = B.reshape(batch_size, layer.n_groups, -1).unsqueeze(-2)
    B_expanded = B_reshaped.expand(batch_size, layer.n_groups, layer.num_heads // layer.n_groups, -1).contiguous()
    B_final = B_expanded.reshape(batch_size, -1, layer.ssm_state_size) # [B, H, N]
    
    # Calculate dB
    # dB = dt[..., None] * B[..., None, :]
    # dt_soft: [B, H, P] -> [B, H, P, 1]
    # B_final: [B, H, N] -> [B, H, 1, N]
    dB = dt_soft[..., None] * B_final.unsqueeze(-2) # [B, H, P, N]
    
    # Discretize x (hidden_states_inner) into dBx
    # hidden: [B, intermediate] -> [B, H, P]
    x_reshaped = hidden_states_inner.reshape(batch_size, -1, layer.head_dim)
    
    # dBx = dB * x[..., None]
    # x: [B, H, P] -> [B, H, P, 1]
    dBx = dB * x_reshaped.unsqueeze(-1)
    
    # STATE UPDATE
    # ssm_state_prev: [B, H, P, N]
    ssm_state_next = ssm_state_prev * dA + dBx
    
    # Compute Output y
    # Prepare C
    C_reshaped = C.reshape(batch_size, layer.n_groups, -1).unsqueeze(-2)
    C_expanded = C_reshaped.expand(batch_size, layer.n_groups, layer.num_heads // layer.n_groups, -1).contiguous()
    C_final = C_expanded.reshape(batch_size, -1, layer.ssm_state_size) # [B, H, N]
    
    # y = ssm_state @ C
    # ssm: [B, H, P, N]
    # C: [B, H, N] -> [B, H, N, 1]
    y = torch.matmul(ssm_state_next, C_final.unsqueeze(-1)).squeeze(-1) # [B, H, P]
    
    # Skip connection (D)
    # D: [H] -> [H, P]
    D = layer.D[..., None].expand(layer.num_heads, layer.head_dim)
    y = y + x_reshaped * D
    
    # Reshape to [B, 1, intermediate]
    y = y.reshape(batch_size, -1).unsqueeze(1)
    
    # Final Norm and Gate
    # gate comes from original split: [B, 1, intermediate]
    # norm(y, gate)
    scan_output = layer.norm(y, gate)
    
    # Final Projection
    out = layer.out_proj(scan_output)
    
    return out, ssm_state_next, conv_state_next
