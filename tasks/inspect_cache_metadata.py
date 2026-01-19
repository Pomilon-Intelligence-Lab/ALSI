from core.base_task import Task
import torch

class InspectCacheMetadata(Task):
    def __init__(self):
        super().__init__("InspectCacheMetadata")
        self.prompt = "The password is '"

    def run(self):
        print(f"[{self.name}] Inspecting Mamba2Cache Metadata...")
        input_ids = self.tokenizer(self.prompt, return_tensors="pt").input_ids.to(self.device)
        
        # 1. Native forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        
        cache = outputs.cache_params
        
        print("\n--- Cache Object Properties ---")
        print(f"Type: {type(cache)}")
        
        # Check all attributes
        for attr in dir(cache):
            if not attr.startswith("__") and not callable(getattr(cache, attr)):
                val = getattr(cache, attr)
                if isinstance(val, torch.Tensor):
                    print(f"{attr}: Tensor shape {val.shape}")
                else:
                    print(f"{attr}: {val}")

    def report(self):
        pass
