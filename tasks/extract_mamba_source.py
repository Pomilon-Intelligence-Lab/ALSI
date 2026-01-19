from core.base_task import Task
import inspect
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

class ExtractMambaSource(Task):
    def __init__(self):
        super().__init__("ExtractMambaSource")

    def run(self):
        print(f"[{self.name}] Extracting Mamba2Mixer source...")
        source = inspect.getsource(Mamba2Mixer)
        print(source)

    def report(self):
        pass
