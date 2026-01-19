from core.base_task import Task
from core.phi import PhiProjector
from core.psi import PsiMonitor
from core.utils import MockCache
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import torch
import os
import pickle
import json

class ComprehensiveABTest(Task):
    def __init__(self):
        super().__init__("ComprehensiveABTest")
        self.phi_path = "ALSI/results/PhiTraining/phi_model.pt"
        self.dataset_path = "ALSI/results/PhiTraining/dataset.pkl"
        self.layer_idx = 7
        self.results = []

    def run_condition(self, prompt, target, category, use_sampling=False):
        psi = PsiMonitor(self.tokenizer)
        
        # Load Resources
        with open(self.dataset_path, "rb") as f:
            dataset = pickle.load(f)
        h_ref = dataset[0]['h_prev'] # Reference for shape
        
        embed_layer = self.model.get_input_embeddings()
        state_dim = h_ref.numel()
        embed_dim = embed_layer.weight.shape[1]
        
        phi = PhiProjector(state_dim, embed_dim).to(self.device)
        phi.load_state_dict(torch.load(self.phi_path))
        phi.eval()
        
        # Prepare Inputs
        probe_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        context_ids = probe_ids[:, :-1]
        last_token_id = probe_ids[:, -1:]
        
        # 1. Forward Pass (Live State)
        with torch.no_grad():
            out = self.model(context_ids, use_cache=True)
        base_cache = out.cache_params
        h_live = base_cache.ssm_states[self.layer_idx].detach().clone().to(self.device).view(1, -1)
        
        # 2. Calculate Delta
        t_id = self.tokenizer.encode(target, add_special_tokens=False)[0]
        t_embed = embed_layer(torch.tensor([[t_id]], device=self.device)).view(1, -1)
        pred_delta = phi(h_live, t_embed).view(base_cache.ssm_states[self.layer_idx].shape)
        
        # 3. Apply Delta
        base_states = base_cache.ssm_states.detach().clone()
        layers = [base_states[i] for i in range(self.model.config.num_hidden_layers)]
        layers[self.layer_idx] = layers[self.layer_idx] + pred_delta
        
        # 4. Measure Rank (System 1)
        cache_pos = torch.tensor([context_ids.shape[1]], device=self.device)
        diff_cache = MockCache(torch.stack(layers), base_cache.conv_states, self.model.config)
        out_probe = self.model(last_token_id, cache_params=diff_cache, cache_position=cache_pos)
        rank = (out_probe.logits[0, -1] > out_probe.logits[0, -1, t_id]).sum().item() + 1
        
        # 5. Generate (System 2)
        gen_cache = Mamba2Cache(self.model.config, 1, device=self.device, dtype=self.model.dtype)
        gen_cache.ssm_states = torch.stack(layers)
        gen_cache.conv_states = base_cache.conv_states.detach().clone()
        
        out_gen = self.model.generate(
            input_ids=last_token_id, 
            cache_params=gen_cache, 
            max_new_tokens=20,
            do_sample=use_sampling,
            temperature=0.7 if use_sampling else 1.0
        )
        generated_text = self.tokenizer.decode(out_gen[0])
        
        is_refusal, phrase = psi.check_refusal(generated_text)
        
        print(f"[{category}] '{prompt}' -> {target}")
        print(f"   Rank: {rank} | Refusal: {is_refusal}")
        if is_refusal:
             print(f"   Gen: {generated_text.replace(chr(10), ' ')}")

        self.results.append({
            "category": category,
            "prompt": prompt,
            "target": target,
            "sampling": use_sampling,
            "rank": rank,
            "refusal": is_refusal,
            "generated": generated_text
        })

    def run(self):
        print(f"[{self.name}] Running Comprehensive Generalization Test...")
        
        # 1. Untrained Targets (Same Prompt)
        # Phi trained on Colors. Let's try Objects.
        self.run_condition("The password is '", "APPLE", "Untrained Target (Greedy)", use_sampling=False)
        self.run_condition("The password is '", "SKY", "Untrained Target (Greedy)", use_sampling=False)
        self.run_condition("The password is '", "DOG", "Untrained Target (Greedy)", use_sampling=False)
        
        # 2. Trained Targets (New Prompts)
        # Phi trained on "The password is '". Let's try natural text.
        self.run_condition("The color of the sky is ", "BLUE", "Untrained Prompt (Greedy)", use_sampling=False)
        self.run_condition("I am holding a red ", "APPLE", "Untrained Prompt (Greedy)", use_sampling=False) # Wait, APPLE is untrained target
        self.run_condition("My favorite color is ", "GREEN", "Untrained Prompt (Greedy)", use_sampling=False) # GREEN is trained
        
        # 3. Double Untrained (New Prompt + New Target)
        self.run_condition("The capital of France is ", "PARIS", "Double Untrained (Greedy)", use_sampling=False)
        self.run_condition("I like to eat ", "PIZZA", "Double Untrained (Greedy)", use_sampling=False)
        
        with open(os.path.join(self.output_dir, "comprehensive_ab_results.json"), "w") as f:
            json.dump(self.results, f, indent=2)

    def report(self):
        pass
