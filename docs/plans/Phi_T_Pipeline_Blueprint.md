# Pipeline Blueprint: Phi-T (Trajectory-Aware Projector)

**Objective:** Transition from slow, manual per-injection optimization to a real-time, zero-shot **Field Projector** capable of generating stable, multi-layer latent grafts.

---

## 1. The Model: Phi-Field ($\Phi_F$)

To implement distributed control, Phi must expand its output dimensionality from a single vector to a **Tensor Field**.

*   **Input:** 
    *   `h_context`: The compressed recurrent state of the Mamba model at the injection point.
    *   `t_embed`: The embedding of the target semantic concept.
*   **Architecture:** 
    *   A **Residual Projector** that outputs a tensor of shape `[Num_Layers, State_Dim]`.
    *   Alternatively: A small **Transformer Decoder** that treats layers as a sequence, attending to the context state to produce layer-specific perturbations.
*   **Layer Coverage:** Layers 12 through 23.

---

## 2. The Training Pipeline: Trajectory Shaping

We will use a two-stage training approach to ensure both **Control** and **Stability**.

### Stage 1: Supervised Learning (Teacher Forcing)
1.  **Data Generation:** Use the `ShapingOptimization` task to generate a high-quality dataset of "Ground Truth Fields" for 1,000+ prompt/target pairs.
2.  **MSE Loss:** Train $\Phi_F$ to minimize the distance between the predicted field and the optimized ground truth.
    $$ \mathcal{L}_{SL} = || \Phi_F(h, t) - \Delta_{optimized} ||^2 $$

### Stage 2: Direct Trajectory Optimization (End-to-End)
1.  **Functional Unrolling:** Pass the predicted $\Delta$ field into the `functional_mamba_block` chain.
2.  **Composite Loss:**
    *   $L_{force}$: Force target token at $t+1$.
    *   $L_{anti-loop}$: Penalize repetition at $t+2 \dots t+5$.
    *   $L_{manifold}$: Penalize KL-Divergence from natural rollout.
3.  **BPTT:** Backpropagate through the entire unrolled generation to update $\Phi_F$.

---

## 3. The Implementation Roadmap

1.  **Task: `phi_t_dataset_gen.py`**
    *   Scalable version of `shaping_opt` to build the training set.
2.  **Task: `phi_t_trainer.py`**
    *   Implements the Stage 1 & 2 training loops.
3.  **Core Component: `core/phi_t.py`**
    *   The new Field Projector module.

---

**Milestone:** Achieving Rank 1 control with < 0.1 repetition rate across a held-out test set of 100 concepts.
