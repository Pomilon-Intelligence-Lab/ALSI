from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class Task(ABC):
    def __init__(self, name, model_id="AntonV/mamba2-130m-hf", output_dir="ALSI/results"):
        self.name = name
        self.model_id = model_id
        self.output_dir = os.path.join(output_dir, name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def setup(self):
        """Loads model and tokenizer."""
        print(f"[{self.name}] Loading model: {self.model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float32 # explicit float32 for CPU/consistency
        ).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"[{self.name}] Setup complete.")

    @abstractmethod
    def run(self):
        """Execute the main logic of the task."""
        pass

    @abstractmethod
    def report(self):
        """Generate artifacts, plots, or summary text."""
        pass
    
    def save_artifact(self, filename, content):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        print(f"[{self.name}] Artifact saved: {path}")
