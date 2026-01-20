# The ALSI Vision: Implicit Latent Memory

## The Original Spark
The inspiration for ALSI came from the intersection of **CRSM** (decoupling reasoning from generation) and MIT's **RLM** (Recursive Language Models).

If a State Space Model's recurrent state $h_t$ is a compressed representation of the conversation context, then mathematically modifying that state should be equivalent to "reminding" the model of forgotten or external information.

## Core Hypothesis
If we can map external facts to latent state deltas ($\Delta$), we can create a model that:
1.  **Never Forgets:** Key information is periodically re-injected into the state.
2.  **Scales Infinitely:** Memory is handled externally but accessed internally.
3.  **Zero-Latency:** Information is available during the forward pass, not via slow tool-calls.

## Why This Might Work
Unlike Transformers, Mamba models have a fixed-size bottleneck. This makes them the perfect candidates for "state surgery"—there is a single, well-defined point of intervention.

## Why This Might Fail
1.  **Manifold Brittleness:** The state space might be too sensitive to perturbations (The Coherence Gap).
2.  **Encoding Complexity:** Mapping "John lives in Paris" to a vector that actually updates the model's "beliefs" may be non-linear beyond our current projection capacity.
