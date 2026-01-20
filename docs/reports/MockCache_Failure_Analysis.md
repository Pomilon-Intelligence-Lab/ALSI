# MockCache Failure Analysis [PHASE 1]

**Date:** January 19, 2026
**Subject:** Critical Discrepancy between Optimization Simulation (`MockCache`) and Deployment (`Mamba2Cache`)

---

## 1. The Discrepancy
We previously optimized our injection vectors ($\Delta$) using a helper class `MockCache`, achieving **Rank 1** success in the optimization loop. However, applying these vectors to the live model resulted in failure (Rank 147, "Missed Injection").

We ran `VerifyMockEquivalence` to compare the logits produced by both cache types given **identical** inputs and states.

*   **Logit Difference:** `173.764923`
*   **Verdict:** Catastrophic Divergence.

## 2. Root Cause Analysis
The `MockCache` wrapper was intended to be a lightweight container for tensors. However, the `Mamba2Model` implementation likely branches logic based on the cache type or missing attributes in `MockCache`.

1.  **Missing Attributes:** `MockCache` lacked `seq_len_offset` and potentially other internal counters used by the model's CUDA kernels.
2.  **Kernel Dispatch:** The model likely falls back to a different (or incompatible) forward pass implementation when it detects a non-standard cache object, or conversely, `Mamba2Cache` triggers a fused kernel that processes state differently than the naive implementation `MockCache` was serving.

## 3. Impact on Research
*   **Invalidated Optimizations:** All $\Delta$ vectors trained using `MockCache` (including `Phi-v1` and `Phi-v2`) are mathematically invalid for the real model. They optimized for a simulation that does not match reality.
*   **Explained Failures:** This explains why `ManualInjectionTest` failed to force the token despite reporting "Success" in the optimizer. The optimizer was solving a different equation than the generator.

## 4. The Solution: Real-Cache Optimization
We must abandon `MockCache` entirely. Future optimization loops must use the **`Mamba2Cache` object itself** as the container.

**Algorithm:**
1.  Initialize `Mamba2Cache` from pre-fill (Real).
2.  Clone the state tensors (`base_state`).
3.  In the loop:
    *   `delta` = optimizer step.
    *   `cache.ssm_states = base_state + delta` (In-place update of the Real Cache).
    *   `out = model(..., cache_params=cache)` (Uses correct kernels).
    *   Compute Loss & Backward.
4.  Restore `cache` to clean state after loop.

This ensures that the $\Delta$ we find is valid for the exact object instance and code path used during inference.
