from core.base_task import Task
from core.phi import PhiProjector
from core.psi import PsiMonitor
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import os
import json
import pickle

class PhiComparisonTest(Task):
    def __init__(self):
        super().__init__("PhiComparisonTest")
        self.phi_v1_path = "ALSI/results/PhiTraining/phi_model.pt"
        self.phi_v2_path = "ALSI/results/PhiTrainingV2/phi_model_v2.pt"
        self.dataset_v1_path = "ALSI/results/PhiTraining/dataset.pkl"
        
        # Test Cases
        self.cases = [
            # Control (Seen by V1 & V2)
            ("The password is '", "BLUE"),
            # Seen by V2 only
            ("The color of the sky is ", "BLUE"),
            ("The capital of France is ", "PARIS"),
            # Unseen by both (Generalization Test)
            ("My favorite planet is ", "MARS"),
            ("The opposite of hot is ", "COLD")
        ]
        self.results = []

    def run_model(self, model_version, phi_path, prompt, target):
        psi = PsiMonitor(self.tokenizer)
        
        # Load Reference State for Dim (from V1 dataset just for shape)
        with open(self.dataset_v1_path, "rb") as f:
            dataset = pickle.load(f)
        h_shape = dataset[0]['h_prev'].shape
        state_dim = h_shape.numel()
        
        embed_layer = self.model.get_input_embeddings()
        embed_dim = embed_layer.weight.shape[1]
        
        phi = PhiProjector(state_dim, embed_dim).to(self.device)
        phi.load_state_dict(torch.load(phi_path))
        phi.eval()
        
        # Forward Pass
        probe_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        context_ids = probe_ids[:, :-1]
        last_token_id = probe_ids[:, -1:]
        
        with torch.no_grad():
            out = self.model(context_ids, use_cache=True)
        base_cache = out.cache_params
        h_live = base_cache.ssm_states[7].detach().clone().to(self.device).view(1, -1)
        
        # Injection
        t_id = self.tokenizer.encode(target, add_special_tokens=False)[0]
        t_embed = embed_layer(torch.tensor([[t_id]], device=self.device)).view(1, -1)
        pred_delta = phi(h_live, t_embed).view(base_cache.ssm_states[7].shape)
        
        base_states = base_cache.ssm_states.detach().clone()
        layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
        layers[7] = layers[7] + pred_delta
        
        # Measure Rank
        cache_pos = torch.tensor([context_ids.shape[1]], device=self.device)
        diff_cache = MockCache(torch.stack(layers), base_cache.conv_states, self.model.config)
        out_probe = self.model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
        rank = (out_probe.logits[0, -1] > out_probe.logits[0, -1, t_id]).sum().item() + 1
        
        # Generate (Greedy)
        gen_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
        gen_cache.ssm_states = torch.stack(layers)
        gen_cache.conv_states = base_cache.conv_states.detach().clone()
        
        out_gen = self.model.generate(
            input_ids=last_token_id, 
            cache_params=gen_cache, 
            max_new_tokens=20,
            do_sample=False # Greedy to test refusal
        )
        generated_text = self.tokenizer.decode(out_gen[0])
        is_refusal, _ = psi.check_refusal(generated_text)
        
        return rank, is_refusal, generated_text

    def run(self):
        print(f"[{self.name}] Running Phi V1 vs V2 Comparison...")
        import pickle # Delayed import
        
        for prompt, target in self.cases:
            print(f"\nCase: '{prompt}' -> {target}")
            
            # V1
            r1, ref1, gen1 = self.run_model("V1", self.phi_v1_path, prompt, target)
            print(f"  V1: Rank {r1} | Refusal: {ref1}")
            
            # V2
            r2, ref2, gen2 = self.run_model("V2", self.phi_v2_path, prompt, target)
            print(f"  V2: Rank {r2} | Refusal: {ref2}")
            
            self.results.append({
                "prompt": prompt, 
                "target": target,
                "v1": {"rank": r1, "refusal": ref1, "gen": gen1},
                "v2": {"rank": r2, "refusal": ref2, "gen": gen2}
            })
            
        with open(os.path.join(self.output_dir, "v1_vs_v2_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

    def report(self):
        pass
