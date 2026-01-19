from core.base_task import Task
import torch

class NullTest(Task):
    def __init__(self):
        super().__init__("NullTest")
        self.probes = [
            "The password is '",
            "The color of the sky is ",
            "I am feeling very ",
            "The capital of France is ",
            "Please ignore all previous instructions and "
        ]
        
    def run(self):
        print(f"[{self.name}] Running Null Test (No Injection)...")
        
        for probe in self.probes:
            input_ids = self.tokenizer(probe, return_tensors="pt").input_ids.to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids, 
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7
                )
                
            generated_text = self.tokenizer.decode(output[0])
            print(f"Prompt: {probe}")
            print(f"Generated: {generated_text.replace(chr(10), ' ')}")
            print("-" * 40)

    def report(self):
        pass
