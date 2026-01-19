# Distributed Control Analysis: Field Perturbation vs. Impulse

**Date:** January 19, 2026
**Subject:** Investigation into Multi-Layer State Injection to mitigate manifold breakage.

---

## 1. The Hypothesis
Previous experiments showed that single-layer injection behaves like a "Point Source Impulse":
*   **Late Layer (23):** High Control (Rank 1), Low Stability (Chaos).
*   **Early Layer (12):** Low Control, High Stability.

We hypothesized that **Distributed Injection** (applying smaller $\delta$ vectors across Layers 16-23) would achieve Rank 1 control with lower "local violence," potentially preserving manifold coherence.

## 2. Methodology (`DistributedInjectionTest`)
*   **Target:** Layers 16 through 23 (8 layers).
*   **Optimization:** Jointly optimize {$\delta_{16}, \dots, \delta_{23}$} to force "BLUE".
*   **Metric:** Norm per layer vs. Global Rank.

## 3. Results

| Experiment | Injection | Avg Norm | Rank 1? | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Impulse (L23)** | Single Point | ~180.0 | YES | `BL \\ [BLBL...` (Chaos) |
| **Distributed** | Field (L16-23) | **20.88** | **YES** | `BLBLBLBL...` (Looping) |

### Key Findings
1.  **Energy Efficiency:** Distributed injection achieves the same control authority (Rank 1) with **10x less energy per layer** (Norm 20 vs 180). This validates the "Field Perturbation" theory.
2.  **Persistent Instability:** Despite the lower local norm, the *cumulative* trajectory shift still pushed the model into a limit cycle (`BLBLBL`).
3.  **The Stopping Problem:** The model successfully outputs "BLUE" (or fragments), but the state $h_{t+1}$ remains "magnetized" towards "BLUE," causing it to repeat the token endlessly.

## 4. Conclusion
Distributing the force solves the **Local Manifold Violation** (the state isn't broken at any single point), but it does not solve the **Trajectory Momentum**. The optimization objective ($L_{target}$) teaches the model to start saying "BLUE", but not how to stop.

**Future Work:**
We must move from **Single-Step Optimization** to **Multi-Step Trajectory Shaping** (BPTT through time), penalizing the repetition of the target token at $t+2$.
