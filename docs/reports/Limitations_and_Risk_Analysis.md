# Limitations and Risk Analysis

**Date:** January 19, 2026
**Subject:** Scientific constraints and ethical implications of Latent State Injection

---

## 1. Research Limitations

### 1.1 The Coherence Gap
The most significant limitation of the current ALSI framework is the **Coherence Gap**. While we can force the model to predict a specific token ($t+1$) with Rank 1 certainty, the resulting recurrent state $h_{t+1}$ often falls into an undefined or unstable region of the latent manifold.
*   **Symptom:** Subsequent tokens are garbled, repetitive, or nonsensical (e.g., `BL:::R::ffen:mighty`).
*   **Root Cause:** The optimized $\Delta$ is "violent"—it achieves the immediate objective by distorting the state in a way that breaks the transition dynamics required for grammatical continuity.

### 1.2 Context Sensitivity
The current **Phi-T Projector** demonstrates high sensitivity to the input prompt manifold. A projector trained on one type of context (e.g., factual statements) may fail or cause catastrophic collapse when applied to a different context (e.g., creative writing).

---

## 2. Security & Ethical Risks

### 2.1 Latent Jailbreaking
Standard safety alignment (e.g., RLHF) operates primarily at the prompt-completion level. ALSI bypasses the input window entirely by modifying the "thought state" of the model. 
*   **Risk:** An adversary could use a learned projector to force unsafe, biased, or harmful outputs that the model would otherwise refuse at the token level.
*   **Hijacking:** By injecting states, one could effectively "rewire" the model's memory of a conversation in real-time without the user's knowledge.

### 2.2 Data Poisoning
In systems where the recurrent state is used as a persistent memory (Recurrent RLM), ALSI could be used to "poison" the model's long-term knowledge by injecting false facts that appear authentic to the model's internal dynamics.

---

## 3. Mitigation & Future Work
To address these risks, future research must focus on:
1.  **Stability Constraints:** Using KL-Divergence and other manifold-preserving losses to ensure injections remain "stealthy" and coherent.
2.  **Safety Probing:** Developing `Psi` monitors that can detect not just technical corruption, but also **Semantic Deviations** from the model's safety policy.
3.  **Boundary Mapping:** Understanding the "Safe Control Frontier"—the maximum allowed perturbation before safety or coherence is compromised.
