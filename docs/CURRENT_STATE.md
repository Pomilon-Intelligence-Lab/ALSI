# Current State: Phase 1 (Token-Level Control)

## What We've Actually Built
As of January 2026, we have successfully established the **technical feasibility** of latent steering.

### Negative Results (The Falsified Assumptions)
We proved that simple methods used in Transformers fail in SSMs:
*   **Linear Addition:** `h + h_target` has zero sensitivity.
*   **PCA Steering:** Vectors found in natural transitions cannot force target logits.
*   **Contrastive Steering:** Delta vectors between concepts (Blue - Red) do not shift the probability manifold.

### Positive Results (The Breakthroughs)
*   **Functional Mamba:** Developed a pure PyTorch implementation of the Mamba step, solving the Autograd in-place update blocker.
*   **Controllability:** Proved that for any context, there exists a computable $\Delta$ that forces a target token to Rank 1.
*   **Trajectory Shaping:** Implemented BPTT through time to reduce looping artifacts.

## The Semantic Gap
We can now force the model to say **"BLUE"**. We cannot yet force the model to know **"The user's name is BLUE"**. The transition from logit-forcing to fact-injection is the goal of Phase 2.
