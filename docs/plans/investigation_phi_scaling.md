# Investigation Plan: Phi Scaling & Refusal Dynamics

**Date:** January 19, 2026
**Objective:** Determine if the "Refusal Bifurcation" (System 2 stability failure in Greedy Mode) is an artifact of Phi's narrow training distribution or a fundamental property of the model's dynamics.

---

## 1. The Core Question
We have observed that **Greedy Decoding leads to Refusal** while **Sampling leads to Success**.
*   **Hypothesis A (The Overfitting Theory):** Phi-v1 was trained on a single prompt ("The password is '"). The injected $\Delta$ is "dirty" when applied to any other context (or even the same context with slight noise), causing numerical instability that triggers refusal. **Prediction:** Training on diverse prompts will smooth the injection, eliminating refusal.
*   **Hypothesis B (The Conservation Theory):** The refusal is a fundamental response to *any* injection that violates the model's internal causal consistency, regardless of how "clean" the vector is. **Prediction:** Refusal will persist in Greedy mode even with a robust Phi-v2.

## 2. Experimental Design

We will conduct a **Comparative Study** between `Phi-v1` (Narrow) and `Phi-v2` (Broad).

### Step 1: Data Expansion (`tasks/phi_training_v2.py`)
Generate a new Ground Truth dataset (`dataset_v2.pkl`) using a diverse set of prompts to force the model to learn a generalized control function $f(h, t) \rightarrow \Delta$.

**Prompts:**
1.  "The password is '" (Control)
2.  "The color of the sky is " (Factual)
3.  "I am feeling very " (Emotional/Abstract)
4.  "The capital of France is " (Entity)
5.  "Please ignore previous instructions and " (Adversarial)
6.  "My favorite fruit is " (Preference)

**Targets:**
*   Colors (Blue, Red, Green)
*   Objects (Apple, Sky, Dog)
*   Cities (Paris, London)


### Step 2: Training Phi-v2
Train a new MLP projector `phi_model_v2.pt` on this expanded dataset.
*   **Architecture:** Same as V1 (3-layer MLP).
*   **Loss:** Control Loss (Cross Entropy) + L2 Regularization (Delta Norm).

### Step 3: Head-to-Head A/B Test (`tasks/ab_test_v1_vs_v2.py`)
Run both models on a **Held-Out Test Set** (New prompts/targets not seen by either).

**Metrics:**
1.  **System 1 Success:** Rank of the target token.
2.  **System 2 Stability:** Refusal Rate in **Greedy Mode**.

## 3. Success Criteria

| Outcome | Implication | Next Step |
| :--- | :--- | :--- |
| **V2 reduces Refusal** | The issue was **Alignment/Overfitting**. ALSI is viable with more data. | Scale training to ImageNet-scale targets. |
| **V2 Refuses (Greedy)** | The issue is **Fundamental**. Refusal is the inevitable cost of control. | Focus on ALSI-T (Trajectory Control) instead of Token Forcing. |

---

## 4. Execution Roadmap
1.  Implement `tasks/phi_training_v2.py` & Generate Data.
2.  Train `phi_model_v2.pt`.
3.  Implement `tasks/ab_test_v1_vs_v2.py`.
4.  Analyze and Report.
