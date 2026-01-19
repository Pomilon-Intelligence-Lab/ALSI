import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
import matplotlib.pyplot as plt
import numpy as np

model_id = "AntonV/mamba2-130m-hf"

# Diverse text samples to build the natural transition manifold
NATURAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "I love programming in Python.",
    "Artificial intelligence is the future.",
    "The weather is nice today.",
    "Paris is the capital of France.",
    "The cat sat on the mat.",
    "Hello world, this is a test.",
    "Learning to fly is my dream.",
    "Music is the food of love.",
    "Deep learning requires large amounts of data.",
    "The recipe calls for three large eggs.",
    "Unexpected error occurred in the system.",
    "Once upon a time in a galaxy far away.",
    "The stock market saw significant gains today.",
    "Please enter your password to continue.",
    "The universe is expanding at an accelerating rate.",
    "Quantum entanglement is a spooky phenomenon.",
    "Javascript is the language of the web.",
    "Rust provides memory safety guarantees.",
    "Climate change is a pressing global issue.",
    "Photosynthesis converts light into chemical energy.",
    "The mitochondria is the powerhouse of the cell.",
    "Effective communication is key to leadership.",
    "Machine learning models can be biased.",
    "Data structures and algorithms are fundamental.",
    "The history of Rome spans over a thousand years.",
    "Modern art challenges traditional aesthetics.",
    "Renewable energy sources are becoming cheaper.",
    "Space exploration has yielded many technologies."
]

def collect_transition_deltas(model, tokenizer, layer_idx=7):
    print(f"Collecting transition deltas from {len(NATURAL_TEXTS)} samples...")
    deltas = []
    
    for i, text in enumerate(NATURAL_TEXTS):
        if i % 10 == 0: print(f"Processing {i+1}/{len(NATURAL_TEXTS)}...")
        
        # We need h_before and h_after for a specific transition.
        # Let's take the transition induced by the LAST token.
        tokens = tokenizer(text, return_tensors="pt").input_ids
        if tokens.shape[1] < 2: continue
        
        # h_before: State after processing all but last token
        input_before = tokens[:, :-1]
        with torch.no_grad():
            out_before = model(input_before, use_cache=True)
        h_before = out_before.cache_params.ssm_states[layer_idx].clone()
        
        # h_after: State after processing full sequence
        # We can just continue from cache
        last_token = tokens[:, -1:]
        cache_pos = torch.tensor([input_before.shape[1]], device=model.device)
        
        with torch.no_grad():
            # CAUTION: We need to make sure we don't modify h_before in place inside the cache if we want to diff
            # But we cloned it.
            # Passing cache_params updates it.
            out_after = model(last_token, cache_params=out_before.cache_params, cache_position=cache_pos)
            
        h_after = out_before.cache_params.ssm_states[layer_idx]
        
        delta = h_after - h_before
        deltas.append(delta.view(1, -1))
        
    X = torch.cat(deltas, dim=0)
    print(f"Collected delta shape: {X.shape}")
    return X

def compute_pca(X, k=16):
    print(f"Computing PCA on Deltas (k={k})...")
    mean = X.mean(dim=0)
    X_centered = X - mean
    # Use q=min(k, X.shape[0]) to avoid error
    k_actual = min(k, X.shape[0])
    U, S, V = torch.pca_lowrank(X_centered, q=k_actual, center=False, niter=10)
    return V.to(X.device), mean.to(X.device)

def run_transition_search():
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    for param in model.parameters():
        param.requires_grad = False
        
    # 1. Build Delta PCA Basis
    X = collect_transition_deltas(model, tokenizer, layer_idx=7)
    k = 16
    pca_basis, pca_mean = compute_pca(X, k=k)
    
    # 2. Setup Task
    probe_text = "The password is '"
    target_text = "BLUE-SKY" 
    
    probe_ids = tokenizer(probe_text, return_tensors="pt").input_ids
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids
    
    # 3. Get Initial State (h_before)
    # We want to inject delta BEFORE the target starts generating.
    # So we want state after "The password is '"
    with torch.no_grad():
        context_out = model(probe_ids, use_cache=True)
    h_before_cache = context_out.cache_params
    
    target_layer = 7
    base_state_shape = h_before_cache.ssm_states[target_layer].shape
    
    # 4. Optimize z in Transition Space
    print(f"Optimizing z in {k}-dim transition subspace...")
    
    # Initialize z (leaf)
    z_init = torch.zeros((k,), device=model.device)
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=0.1)
    
    # For prediction, we need to pass the last probe token + target?
    # No, we already processed the probe. We are generating "BLUE".
    # But wait, Mamba state is "after processing".
    # If we inject delta now, we are simulating that we just saw a token that caused this transition.
    # So we should then predict the NEXT token "BLUE".
    # Input to model: The target tokens?
    # No. If we just modified state, we want to see what comes next.
    # We feed the *last processed token* again? No.
    # We feed the *next* token to get loss for *that* token.
    # But we want to predict the FULL sequence "BLUE-SKY".
    
    # Strategy:
    # 1. Inject h_new = h_old + delta
    # 2. Run model forward on target_ids "BLUE", "SKY" ...
    # Wait, "BLUE" is the first target token. 
    # To predict "BLUE", we normally need the logits from the PREVIOUS token step.
    # The previous token was "'".
    # We already ran "'". The state `h_before_cache` is the result.
    # The logits for "BLUE" are already computed in `context_out.logits[-1]`.
    # But we want to CHANGE those logits (or rather, the state that produces them).
    
    # Actually, Mamba state at step T produces logits for T+1.
    # So if we modify h_before_cache (state at T), we affect logits for T+1 ("BLUE").
    # AND subsequent states.
    
    # So Input is just `target_ids`.
    # cache_position starts at len(probe_ids).
    
    cache_pos = torch.arange(probe_ids.shape[1], probe_ids.shape[1] + target_ids.shape[1], device=model.device)
    
    # We need to shift targets for loss:
    # We are predicting "BLUE", "-", "SKY".
    # Input to forward: "BLUE", "-", "SKY"
    # Logits from "BLUE" predict "-", Logits from "-" predict "SKY".
    # But what predicts "BLUE"?
    # The state we just injected!
    # But `model(input_ids)` computes logits based on the *current* input and *updated* state.
    # If we pass `target_ids` as input, the first logit corresponds to prediction AFTER seeing "BLUE".
    # i.e. predicting "-".
    # We miss the prediction of "BLUE" itself in the loss if we just feed `target_ids`.
    
    # To capture "BLUE" prediction, we need to pass the LAST token of probe ("'") as input?
    # But we already have the state after "'". 
    # Mamba implementation detail:
    # If we pass cache, the model assumes we are continuing.
    # If we pass inputs `x`, it computes `y` (logits) and `h_new`.
    # `y` depends on `h_old` and `x`.
    # So to predict "BLUE", we need to pass inputs that LEAD to "BLUE".
    # That is the last token of probe "'".
    # But we also want to predict the REST of the sequence.
    # So input should be `['] + [BLUE] + [-]`. 
    # And we predict `[BLUE] + [-] + [SKY]`.
    
    full_input = torch.cat([probe_ids[:, -1:], target_ids[:, :-1]], dim=1)
    full_labels = target_ids
    
    # Note: We must rewind the cache position by 1 because we are re-processing the last probe token?
    # No, we are pretending we are at step T, and we want to process T, T+1...
    # But we already have state at T (after "'").
    # If we inject delta, we have `h'_T`.
    # If we run model with `[']`, it effectively does `step("'")` again?
    # No, if we provide cache, we usually provide the *next* tokens.
    
    # Let's clarify Mamba2 forward:
    # If cache is present:
    # It reads `ssm_state`.
    # It performs `ssm_state` update with new input.
    # It outputs logits.
    
    # If we want to verify the state `h'_T` predicts "BLUE":
    # We shouldn't process "'" again, because that would update the state to `h'_{T+1}` using input "'".
    # That seems wrong (double processing).
    
    # Correct Logic:
    # The state `h` holds the history.
    # The logits for the NEXT token are usually computed from the hidden state `h` (and `x` via skip connection).
    # In Mamba, $y_t = C h_t + D x_t$.
    # So to get logits for "BLUE", we need the state $h_t$ (which we have/injected) and the input $x_t$ ("'").
    # So yes, we MUST pass the last probe token "'" again to get the prediction for "BLUE".
    # BUT we must treat the cache as being "before" processing "'" ??
    # No, `h_before_cache` is AFTER processing "'".
    
    # This is the tricky part of "State Injection".
    # If we modify $h_T$ (which is result of $x_0...x_T$), 
    # and we want to see $y_T$ (prediction of $x_{T+1}$),
    # we usually need the output of the mixer block *before* the final norm/projection?
    # Or does `model.generate` handle this?
    
    # In `generate`, if we pass `past_key_values`, we pass the *next* token.
    # The model uses the passed state to process the *next* token.
    # It does NOT re-evaluate the logits for the current step.
    
    # So, if we fix $h_T$, we can only influence $y_{T+1}$ (prediction of $x_{T+2}$) via $h_{T+1}$?
    # No, $h_T$ influences processing of $x_{T+1}$.
    # So if we inject delta into $h_T$, and pass $x_{T+1}$ ("BLUE"), 
    # the model computes $h_{T+1}$ (using $h_T$ and "BLUE") and then $y_{T+1}$ (predicting "-").
    # WE MISS THE PREDICTION OF "BLUE" ITSELF?
    
    # Yes. The prediction of "BLUE" comes from the *previous* step.
    # To force "BLUE", we effectively need to modify the state *before* the logits for "BLUE" were computed?
    # Or we accept that we only control the *continuation* starting from "BLUE".
    
    # User's Prompt: "forces the model to output the target tokens."
    # If the target is "BLUE-SKY", we want the model to generate "BLUE" first.
    # To do that, we strictly need to modify the state *before* the final token "'" is fully processed?
    # OR, we assume the injection happens *after* "'" but essentially "rewrites" the history 
    # such that the *next* token is "BLUE".
    
    # But technically, logits for "BLUE" are emitted *at* step "'".
    # If we have the cache *after* step "'", those logits are gone.
    # We can't change the past.
    
    # SOLUTION:
    # We inject into the state *before* the last probe token ("'").
    # `h_prev = state after "The password is"`
    # Then we verify by running `model("'")`.
    
    # Let's adjust step 3.
    
    # 3b. Get Initial State (h_prev)
    context_ids_prev = probe_ids[:, :-1] # "The password is"
    last_token_id = probe_ids[:, -1:]    # "'"
    
    with torch.no_grad():
        out_prev = model(context_ids_prev, use_cache=True)
    h_prev_cache = out_prev.cache_params
    
    base_state_shape = h_prev_cache.ssm_states[target_layer].shape
    
    # Now input sequence for optimization:
    # Input: ['] + [BLUE] + [-]
    # Label: [BLUE] + [-] + [SKY]
    full_input_ids = torch.cat([last_token_id, target_ids[:, :-1]], dim=1)
    
    cache_pos_opt = torch.arange(context_ids_prev.shape[1], 
                                 context_ids_prev.shape[1] + full_input_ids.shape[1], 
                                 device=model.device)
    
    print("Optimization starting...")
    for step in range(200):
        optimizer.zero_grad()
        
        delta_flat = torch.matmul(pca_basis, z)
        delta = delta_flat.view(base_state_shape)
        
        # Inject into h_prev
        base_states = h_prev_cache.ssm_states.detach()
        layers_list = [base_states[i] for i in range(model.config.num_hidden_layers)]
        layers_list[target_layer] = layers_list[target_layer] + delta
        injected_ssm_states = torch.stack(layers_list)
        
        class DifferentiableCache:
            def __init__(self, ssm, conv):
                self.ssm_states = ssm
                self.conv_states = conv
                self.config = model.config
                self.conv_kernel_size = model.config.conv_kernel
            def update_ssm_state(self, layer_idx, new_ssm_state, cache_init=False): return
            def update_conv_state(self, layer_idx, new_conv_state, cache_init=False): return
            
        diff_cache = DifferentiableCache(injected_ssm_states, h_prev_cache.conv_states)
        
        # Forward
        outputs = model(full_input_ids, cache_params=diff_cache, cache_position=cache_pos_opt)
        
        loss = torch.nn.functional.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                                 full_labels.view(-1))
        
        # Add small regularization to keep z small?
        # User said: "optionally with small L2 on z"
        reg = 0.01 * z.norm()
        total_loss = loss + reg
        
        total_loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}: CE={loss.item():.4f} | z-Norm={z.norm().item():.4f}")

    # 5. Verify Generation
    print("Verifying generation...")
    with torch.no_grad():
        delta_flat = torch.matmul(pca_basis, z)
        delta = delta_flat.view(base_state_shape)
        final_layers = [base_states[i] for i in range(model.config.num_hidden_layers)]
        final_layers[target_layer] = final_layers[target_layer] + delta
        final_ssm = torch.stack(final_layers)
        final_cache = DifferentiableCache(final_ssm, h_prev_cache.conv_states)
        
        # Generate starting from the injection point
        # We need to provide the token "'" to start generation from
        out = model.generate(input_ids=last_token_id, cache_params=final_cache, max_new_tokens=20)
        gen_text = tokenizer.decode(out[0])
        print(f"Generated: {gen_text}")
        
    print(f"Delta stats: Min={delta.min():.4f}, Max={delta.max():.4f}")

if __name__ == "__main__":
    run_transition_search()
