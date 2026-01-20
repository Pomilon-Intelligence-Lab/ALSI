# Optimization Autograd Blocker [PHASE 1]

**Date:** January 19, 2026
**Subject:** Persistent Autograd Failures in Real-Cache Optimization

---

## 1. The Problem
To enable ALSI-T (Trajectory Control), we must optimize the injection vector $\Delta$ using the **Real Model** (not a Mock). This requires backpropagating through the `Mamba2Model.forward()` pass while modifying the hidden state.

**Constraint:** The Mamba-2 implementation (via `transformers`) performs **In-Place Updates** on the `cache_params.ssm_states` tensor during the forward pass.

**Result:** `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation.`

## 2. Failed Mitigation Strategies

We attempted four distinct strategies to bypass this error, all failing:

1.  **Fresh Cache Per Step:** Re-initializing `Mamba2Cache` at every optimization step.
    *   *Result:* Failed. The model modifies the tensor assigned to the cache, which is linked to `delta`.
2.  **Cloning State Tensors:** Passing `base_ssm.clone()` to the cache.
    *   *Result:* Failed. The gradient graph tracks the clone, and the in-place update on the clone invalidates the graph for the backward pass.
3.  **Functional Construction:** Building the `ssm_states` tensor from a list of clones (`torch.stack`).
    *   *Result:* Failed. The in-place update on the stacked tensor propagates back to the inputs of the stack operation (specifically `AsStridedBackward`).
4.  **Subclassing (`DiffCache`):** Inheriting from `Mamba2Cache` to pass `isinstance` checks while controlling attributes.
    *   *Result:* Failed. The issue is in the CUDA kernel execution path, not the Python class logic.

## 3. The Diagnosis
The `mamba2` kernel (likely fused) is designed for inference efficiency, not state optimization. It assumes the state $h_t$ is a mutable buffer. When we try to make $h_t$ a function of $\Delta$ ($h_t = h_{base} + \Delta$), the in-place update $h_{t+1} \leftarrow f(h_t, x)$ overwrites the memory of $h_t$ that Autograd needs to compute $\partial h_{t+1} / \partial \Delta$.

## 4. The Recommended Solution: Functional Mamba Step

The only definitive way to enable real-cache optimization is to bypass the stateful `Mamba2Model` wrapper and implement a **Functional Mamba Step**. 

Instead of passing a mutable cache object, we must define the transition as a pure function:
$$(h_t, conv_t, x_t) \rightarrow (y_t, h_{t+1}, conv_{t+1})$$

### Implementation Strategy (Option A):
Use the low-level `mamba_ssm` kernels directly or re-implement the Mamba-2 recurrent step in pure PyTorch.

1.  **Extract the Core Mixer:** Isolate the SSM, Convolution, and Gating logic from the transformer layer.
2.  **Pass State as Input:** Treat $h_t$ as a standard tensor input in the computation graph.
3.  **Calculate Injection:**
    *   `h_injected = h_base + delta`
    *   `logits, h_next = functional_mamba_step(h_injected, x)`
4.  **Differentiate:** Since $h_{injected}$ is a pure input and no in-place mutation occurs on tensors linked to `delta`, `loss.backward()` will flow perfectly.

**Status:** This is the architecturally correct path for ALSI Phase 2. It moves the project from "state grafting" to "latent trajectory steering" by treating the recurrent state as a first-class differentiable variable.
