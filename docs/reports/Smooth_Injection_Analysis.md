# Smooth Injection Analysis: The Sticky Attractor Problem

**Date:** January 19, 2026
**Subject:** Evaluation of Depth-Wise Smooth Regularization for Manifold Stability

---

## 1. The Experiment
Based on the hypothesis that "Tuned/Smooth" multi-layer injection could prevent dynamical collapse, we implemented **Smooth Distributed Injection**.

**Methodology:**
1.  **Field Optimization:** Optimized $\Delta$ across 12 layers (12-23).
2.  **Smoothness Penalty:** Added $L_{smooth} = \sum ||\delta_i - \delta_{i+1}||^2$ to the objective.
3.  **Stability Constraint:** Maintained KL-Divergence penalty to preserve the natural logit distribution.

## 2. Results

| Experiment | Injection Type | Avg Norm | Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Impulse (L23)** | Point | ~180.0 | `BL \ [BLBL...` | **Shattered** |
| **Distributed (L16-23)** | Random Field | ~20.88 | `BLBLBLBL...` | **Sticky** |
| **Smooth (L12-23)** | Continuous Field | **~15.4** | `BLBLLYBLometimes...` | **Sticky but Recovering** |

### Key Findings
1.  **Energy Reduction:** Smoothness further reduced the energy required per layer.
2.  **Partial Recovery:** The model started to generate real words ("sometimes") between loops, proving that the underlying context is more preserved than in the Impulse case.
3.  **The Sticky Attractor:** Despite being smooth and distributed, the injection creates a "Magnet" in the state space. Once the model outputs the target sub-token, the recurrent state remains biased toward that sub-token, creating a limit cycle (loop).

## 3. Conclusion: Tuning through Time
The "Tuned Injection" hypothesis is partially validated: distributing and smoothing the delta across **depth** makes the model more resilient (it tries to speak). However, to prevent looping, the tuning must happen across **time**.

**The Final Frontier (ALSI-T):**
We must optimize for a **Transient Injection**—a pulse that is strong at $t$ to force the token, but dissipates or explicitly penalizes its own repetition at $t+1$ and $t+2$. 

This moves ALSI from a **Static Graft** to a **Dynamic Trajectory Control** system.
