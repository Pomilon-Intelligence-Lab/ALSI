from core.base_task import Task
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch

class DebugCacheInternals(Task):
    def __init__(self):
        super().__init__("DebugCacheInternals")

    def run(self):
        print(f"[{self.name}] Comparing Real vs Mock Internals...")
        
        # 1. Create Real Cache
        real_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
        
        # 2. Create Mock Cache with same tensors
        mock_cache = MockCache(real_cache.ssm_states.clone(), real_cache.conv_states.clone(), self.model.config)
        
        print("\n--- Attribute Comparison ---")
        real_attrs = set(dir(real_cache))
        mock_attrs = set(dir(mock_cache))
        
        missing = real_attrs - mock_attrs
        print(f"Attributes in Real missing from Mock: {missing}")
        
        print("\n--- Tensor Stride/Layout Comparison ---")
        print(f"Real SSM Stride: {real_cache.ssm_states.stride()}")
        print(f"Mock SSM Stride: {mock_cache.ssm_states.stride()}")
        
        print(f"Real Conv Stride: {real_cache.conv_states.stride()}")
        print(f"Mock Conv Stride: {mock_cache.conv_states.stride()}")

    def report(self):
        pass
