# The Sweet Spot Analysis [PHASE 1]

**Date:** January 19, 2026
**Subject:** High-Resolution Functional Sensitivity Scan of Mamba-2 (130m)

---

## 1. Objective
To find the specific layer depth where state injection ($\Delta$) achieves the maximum balance between **Control Authority** (forcing the token) and **Trajectory Stability** (maintaining coherence).

## 2. Experimental Data
We scanned layers 14 through 23 using functional optimization.

| Layer | Control Rank | Result | Stability Verdict |
| :--- | :--- | :--- | :--- |
| **14-15** | Med (Rank 1 in 3 steps) | `BLACK'. The password...` | **Resilient but Heavy.** Model fights back. |
| **16** | **High (Rank 1 in 1 step)** | **`BLOOM' and the password is...`** | **Optimal.** Fast control + Partial recovery. |
| **17-22** | High (Rank 1 in 1 step) | `BLACK' and the password is 'BLACK...` | **Over-Controlled.** Immediate looping collapse. |
| **23** | High (Rank 1 in 1 step) | `BLobsBLobs...` | **Shattered.** Token fragmentation. |

## 3. The Discovery: Layer 16
Layer 16 represents the **Semantic-Lexical Boundary** in Mamba-2 (130m).
*   Injecting **deeper** (L14) results in the residual stream washing out the signal.
*   Injecting **shallower** (L18+) results in the model losing the ability to "reason" beyond the forced token, leading to loops.

## 4. Conclusion
For zero-shot context injection in Mamba-2, **Layer 16 is the "Steering Wheel."** It allows us to insert a fact with enough force to change the immediate future without destroying the model's memory of the past.
