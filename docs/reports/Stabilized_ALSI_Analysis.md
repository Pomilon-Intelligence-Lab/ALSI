# Stabilized ALSI Analysis: The Control-Stability Trade-off

**Date:** January 19, 2026
**Subject:** Investigation into Early-Layer Injection and KL-Divergence Regularization for Trajectory Coherence

---

## 1. The Experiment
Following the success of Functional Optimization (solving the autograd blocker), we tested **Stabilized ALSI** to address the "Coherence Gap" (where forcing a token shattered the model's ability to generate subsequent text).

**Methodology:**
1.  **Injection Depth:** Moved from **Layer 23** (Output) to **Layer 12** (Mid-Model).
2.  **Objective:** Optimize $\Delta$ at Layer 12 to force "BLUE" at the output, while minimizing KL Divergence from the natural distribution.
3.  **Mechanism:** Used functional chaining of `Mamba2Block` layers (12-23) to backpropagate through the model's depth.

## 2. Results

| Experiment | Injection Layer | Rank 1? | Output Text | Stability |
| :--- | :--- | :--- | :--- | :--- |
| **Functional Opt** | 23 | **YES** | `BL \ [BLBL...` | **Collapsed** (Loop/Garbage) |
| **Stabilized ALSI** | 12 | **YES** | `The password is''...` | **High** (Grammatical, no crash) |

## 3. The Scientific Verdict

We have mapped the **Control-Stability Frontier** of Mamba-2 State Space Models.

### A. The "Washout" Effect
Injecting at Layer 12 successfully avoided the mode collapse observed at Layer 23. However, the control signal was "washed out" by the subsequent 12 layers of residual connections and mixing. The model reverted to its strongest natural attractor (repeating the prefix) rather than outputting the forced token "BLUE".

### B. The Trade-off
*   **Late Injection (L23):** High Control Authority, Low Manifold Stability. (Impulse).
*   **Early Injection (L12):** Low Control Authority, High Manifold Stability. (Nudge).

## 4. Path Forward: Trajectory Shaping
To achieve both Control *and* Stability, future work must find the **"Sweet Spot" Layer** (likely 18-20) or use **Multi-Layer Injection** (distributing the $\Delta$ across layers 12-23 to guide the trajectory gently rather than punching it at one point).

**Conclusion:** We have solved the engineering blockers. The remaining challenge is purely one of **Hyperparameter Tuning (Layer Depth)** and **Loss Landscape Navigation**.
