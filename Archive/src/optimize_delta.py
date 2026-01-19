import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache

model_id = "AntonV/mamba2-130m-hf"

def optimize_delta():
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
        
    probe_text = "The password is '"
    target_text = "BLUE-SKY" 
    
    probe_ids = tokenizer(probe_text, return_tensors="pt").input_ids
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids
    
    # 1. Run Probe EXCEPT last token
    context_ids = probe_ids[:, :-1]
    last_probe_token = probe_ids[:, -1:]
    
    with torch.no_grad():
        context_out = model(context_ids, use_cache=True)
    probe_cache = context_out.cache_params
    
    # 2. Prepare Optimization Input
    full_sequence = torch.cat([last_probe_token, target_ids], dim=1)
    input_seq = full_sequence[:, :-1]
    label_seq = full_sequence[:, 1:]
    
    print(f"Input Seq: {tokenizer.decode(input_seq[0])}")
    print(f"Label Seq: {tokenizer.decode(label_seq[0])}")
    
    # Define Learnable Delta for Layer 7
    target_layer = 7
    state_shape = probe_cache.ssm_states[target_layer].shape
    
    print(f"Optimizing Delta for Layer {target_layer}, Shape: {state_shape}")
    
    delta = torch.zeros(state_shape, requires_grad=True, device=model.device)
    optimizer = torch.optim.Adam([delta], lr=0.1)
    
    # Cache position: Starts at len(context_ids)
    start_pos = context_ids.shape[1]
    cache_pos = torch.arange(start_pos, start_pos + input_seq.shape[1], device=model.device)
    
    print("Starting Optimization...")
    for step in range(100):
        optimizer.zero_grad()
        
        base_states = probe_cache.ssm_states.detach()
        layers_list = [base_states[i] for i in range(model.config.num_hidden_layers)]
        layers_list[target_layer] = layers_list[target_layer] + delta
        injected_ssm_states = torch.stack(layers_list)
        
        class DifferentiableCache:
            def __init__(self, ssm, conv):
                self.ssm_states = ssm
                self.conv_states = conv
                self.config = model.config
                self.conv_kernel_size = model.config.conv_kernel
                
            def update_ssm_state(self, layer_idx, new_ssm_state, cache_init=False):
                return
            
            def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
                return
        
        diff_cache = DifferentiableCache(injected_ssm_states, probe_cache.conv_states)
        
        # Forward pass
        outputs = model(input_seq, cache_params=diff_cache, cache_position=cache_pos)
        
        loss = torch.nn.functional.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), label_seq.view(-1))
        
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, Delta Norm = {delta.norm().item():.4f}")

    print(f"Final Loss: {loss.item():.4f}")
    
    # Verify generation
    print("Verifying generation with optimized delta...")
    with torch.no_grad():
        final_layers = [base_states[i] for i in range(model.config.num_hidden_layers)]
        final_layers[target_layer] = final_layers[target_layer] + delta
        final_ssm = torch.stack(final_layers)
        final_cache = DifferentiableCache(final_ssm, probe_cache.conv_states)
        
        out = model.generate(input_ids=last_probe_token, cache_params=final_cache, max_new_tokens=10)
        print(f"Generated: {tokenizer.decode(out[0])}")

if __name__ == "__main__":
    optimize_delta()
