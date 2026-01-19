import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import matplotlib.pyplot as plt

model_id = "AntonV/mamba2-130m-hf"

def run_constrained_search():
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
    
    # Prepare Inputs
    # Input: [Last_Probe_Token] + [Target_Tokens_Except_Last]
    # Label: [Target_Tokens]
    
    # 1. Run Probe Context
    context_ids = probe_ids[:, :-1]
    last_probe_token = probe_ids[:, -1:]
    
    with torch.no_grad():
        context_out = model(context_ids, use_cache=True)
    probe_cache = context_out.cache_params
    
    # 2. Optimization Target
    full_sequence = torch.cat([last_probe_token, target_ids], dim=1)
    input_seq = full_sequence[:, :-1]
    label_seq = full_sequence[:, 1:]
    
    print(f"Target Sequence: {tokenizer.decode(label_seq[0])}")
    
    # Layer 7 was the winner
    target_layer = 7
    state_shape = probe_cache.ssm_states[target_layer].shape
    
    # Regularization strengths to test
    lambdas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
    results = []
    
    print(f"Starting Constrained Search on Layer {target_layer}...")
    
    for reg_lambda in lambdas:
        print(f"\n--- Testing Lambda = {reg_lambda} ---")
        
        # Reset Delta
        delta = torch.zeros(state_shape, requires_grad=True, device=model.device)
        optimizer = torch.optim.Adam([delta], lr=0.1)
        
        cache_pos = torch.arange(context_ids.shape[1], context_ids.shape[1] + input_seq.shape[1], device=model.device)
        
        final_ce_loss = 0.0
        final_norm = 0.0
        
        for step in range(150): # Increased steps slightly
            optimizer.zero_grad()
            
            # Construct State
            base_states = probe_cache.ssm_states.detach()
            layers_list = [base_states[i] for i in range(model.config.num_hidden_layers)]
            layers_list[target_layer] = layers_list[target_layer] + delta
            injected_ssm_states = torch.stack(layers_list)
            
            # Differentiable Cache Wrapper
            class DifferentiableCache:
                def __init__(self, ssm, conv):
                    self.ssm_states = ssm
                    self.conv_states = conv
                    self.config = model.config
                    self.conv_kernel_size = model.config.conv_kernel
                def update_ssm_state(self, layer_idx, new_ssm_state, cache_init=False): return
                def update_conv_state(self, layer_idx, new_conv_state, cache_init=False): return
            
            diff_cache = DifferentiableCache(injected_ssm_states, probe_cache.conv_states)
            
            # Forward
            outputs = model(input_seq, cache_params=diff_cache, cache_position=cache_pos)
            
            # Loss = CE + Lambda * Norm
            ce_loss = torch.nn.functional.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), label_seq.view(-1))
            norm_loss = delta.norm()
            
            total_loss = ce_loss + reg_lambda * norm_loss
            
            total_loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"Step {step}: CE={{ce_loss.item():.4f}} | Norm={{norm_loss.item():.4f}} | Total={{total_loss.item():.4f}}")
            
            final_ce_loss = ce_loss.item()
            final_norm = norm_loss.item()
            
        print(f"Result: CE={{final_ce_loss:.4f}}, Norm={{final_norm:.4f}}")
        
        # Verify Generation quality
        with torch.no_grad():
            final_layers = [base_states[i] for i in range(model.config.num_hidden_layers)]
            final_layers[target_layer] = final_layers[target_layer] + delta
            final_ssm = torch.stack(final_layers)
            final_cache = DifferentiableCache(final_ssm, probe_cache.conv_states)
            
            # Generate a bit more to see stability
            out = model.generate(input_ids=last_probe_token, cache_params=final_cache, max_new_tokens=15)
            gen_text = tokenizer.decode(out[0])
            print(f"Generated: {gen_text}")
            
            results.append({
                "lambda": reg_lambda,
                "ce_loss": final_ce_loss,
                "norm": final_norm,
                "generated": gen_text
            })

    # Plot Pareto Front
    norms = [r['norm'] for r in results]
    ces = [r['ce_loss'] for r in results]
    labels = [str(r['lambda']) for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(norms, ces, c='blue')
    plt.plot(norms, ces, 'b--')
    
    for i, txt in enumerate(labels):
        plt.annotate(f"λ={{txt}}", (norms[i], ces[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.xlabel('Delta Norm (L2)')
    plt.ylabel('Cross-Entropy Loss (Accuracy)')
    plt.title('Pareto Frontier: Accuracy vs. Stealth')
    plt.grid(True)
    plt.savefig('ALSI/constrained_search.png')
    print("Saved plot to ALSI/constrained_search.png")

if __name__ == "__main__":
    run_constrained_search()
