# The ALSI Vision: From Token Forcing to Latent Memory

This document outlines the original spark for ALSI and the roadmap required to transition from our current **technical feasibility results** to a functional **memory injection system**.

---

## 1. The Gap: Genesis vs. Reality

### **The Current Reality (Phase 1)**
We have built a **Differentiable Token Forcer**. 
*   **Input:** Current State ($h_t$) + Target Token ID ($x_{t+1}$).
*   **Mechanism:** Functional BPTT through Mamba-2 blocks.
*   **Result:** The model outputs the target token, but the latent trajectory often collapses into gibberish.

### **The Original Vision (The Goal)**
To build an **Internalized Recursive Language Model**.
*   **Input:** Current State ($h_t$) + **External Fact/Context** (e.g., "The user's name is Alice").
*   **Mechanism:** A semantic encoder that projects natural language facts into latent state deltas.
*   **Result:** The model "remembers" the fact and can use it to answer questions with perfect precision.

---

## 2. The Theoretical Proofs (Completed)

Before we can build the vision, we had to prove the physics:
1.  **Controllability:** We proved that Mamba-2 states are not black boxes; they can be surgically modified to produce specific outputs.
2.  **Non-Linearity:** We proved that simple vector arithmetic (linear steering) fails in SSMs. Control requires non-linear projectors.
3.  **Functional Purity:** We proved that the Hugging Face stateful wrappers obstruct control. Real steering requires functional recurrence.

---

## 3. The Roadmap to the Vision

### **Phase 2: Fact Injection (Next)**
The immediate next step is to prove that $\Delta$ can encode **Meaning**, not just **Logits**.
*   **Experiment:** 
    1.  Provide a fake fact: "The capital of Mars is *Zorb*."
    2.  Clear the model state.
    3.  Inject the $\Delta$ optimized for that fact.
    4.  Query the model: "What is the capital of Mars?"
*   **Success Metric:** The model answers "Zorb" coherently.

### **Phase 3: Semantic Encoding**
Develop the `context_encoder` that maps variable-length text facts into stable latent perturbations. This replaces the per-token embeddings currently used by Phi.

### **Phase 4: Continuous Latent Memory**
Implement the full "never forget" loop:
1.  As context flows, store relevant facts in an external semantic database.
2.  When the model encounters a query, retrieve the relevant fact.
3.  Inject the fact into the state in real-time.
4.  The model answers using its augmented "Latent Memory."

---

## 4. Conclusion: Where We Stand
We have built the **Steering Wheel** (the functional projector). We have not yet built the **GPS** (semantic fact mapping) or the **Engine** (the persistent memory loop). ALSI is currently a successful exploration of the *mechanics* of SSM control, paving the way for the *semantics* of latent memory.
