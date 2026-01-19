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
        # Mamba2Cache attributes
        self.seq_len_offset = 0
        self.dtype = ssm_states.dtype

    def update_ssm_state(self, layer_idx, new_ssm_state, cache_init=False):
        # We generally don't want side effects during optimization/probing
        pass

    def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
        pass
