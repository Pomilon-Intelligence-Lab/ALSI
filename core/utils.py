import torch

class MockCache:
    """
    A helper class to wrap SSM states for Mamba2 forward passes.
    Compatible with transformers.models.mamba2.modeling_mamba2
    """
    def __init__(self, ssm_states, conv_states, config):
        self.ssm_states = ssm_states
        self.conv_states = conv_states
        self.config = config
        self.conv_kernel_size = config.conv_kernel
        
        # Mamba2Cache attributes for kernel dispatch/equivalence
        self.n_groups = config.n_groups if hasattr(config, 'n_groups') else 1
        self.state_size = config.state_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_heads
        # Mamba2 uses hidden_size * expand for intermediate_size
        self.intermediate_size = config.hidden_size * config.expand
        self.dtype = ssm_states.dtype
        self.seq_len_offset = 0 # Dummy for compatibility

    def reset(self):
        pass

    def update_ssm_state(self, layer_idx, new_ssm_state, cache_init=False):
        pass

    def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
        pass
