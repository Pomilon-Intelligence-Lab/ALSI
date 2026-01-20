# ALSI Project: Executive Summary

**Augmented Latent State Injection (ALSI)**
*A technical feasibility study of non-linear state steering in Mamba-2.*

---

## 1. Core Findings: Mechanics of Control
We have successfully demonstrated that the recurrent state of a Mamba-2 SSM is a **differentiable control surface**.

*   **Learned Steering:** We can force target tokens with **Rank 1** success by training non-linear projectors to find "Golden Deltas" in the latent space.
*   **Linear Failure:** We proved that simple linear methods (PCA, vector addition) are insufficient for SSM control due to the complex, gated nature of the recurrent update.
*   **Functional Requirement:** True latent steering requires treating the Mamba step as a pure functional operation $(h_t, x_t) \rightarrow (y_t, h_{t+1})$ to enable gradient-based optimization.

## 2. Research Limitations: The Semantic Gap
While we have mastered the **mechanics** of token forcing, we have identified two critical hurdles:

1.  **The Coherence Gap:** Injected states frequently destabilize the model's trajectory, resulting in garbled or repetitive outputs after the forced token.
2.  **The Semantic Gap:** We have not yet proven that these latent perturbations can encode complex **facts** (e.g., memory recall) rather than just immediate logit biases.

---

## 3. The Future: Toward Latent Memory
The project's ultimate goal remains the internalization of MIT's **Recursive Language Models (RLM)** paradigm. 

*   **Current State:** Successful "Token Forcer."
*   **Next Milestone:** Proving "Fact Injection"—allowing the model to answer questions about forgotten context via latent grafting.

See [`docs/vision.md`](../vision.md) for the complete roadmap.
