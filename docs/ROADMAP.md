# Roadmap: Baby Steps to the Vision

## Phase 2: Fact Injection 🔄 (Next)
**Goal:** Prove that $\Delta$ can encode meaning, not just logits.
*   **Experiment 1: Fake Fact Recall.**
    1.  Inject fact: "The password is ZORB."
    2.  Clear state.
    3.  Query: "What is the password?"
*   **Experiment 2: Embedding Alignment.**
    *   Can we predict the optimal $\Delta$ field directly from a sentence embedding (e.g., SBERT)?

## Phase 3: Multi-Hop Reasoning 🔮
**Goal:** Inject multiple facts simultaneously.
*   **Experiment:** Inject "Alice is Bob's sister" and "Alice is a doctor." Verify if model concludes "Bob's sister is a doctor."

## Phase 4: Continuous Latent Memory 🔮
**Goal:** Rolling context injection.
*   **Implementation:** Integration with a vector database (FAISS) to trigger latent grafts based on conversation drift.
