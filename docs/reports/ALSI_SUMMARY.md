# ALSI Project: Executive Summary

**Augmented Latent State Injection (ALSI)**
*A framework for deterministic, non-linear control of Mamba-2 State Space Models.*

---

## 1. The Core Discovery
**Logit Control $\neq$ Trajectory Control.**

We successfully proved that while Mamba-2's recurrent state is resistant to linear steering (PCA/Vector Arithmetic), it contains a **learnable, non-linear control surface**.

*   **Linear Steering Fails:** Simple addition or subspace optimization cannot steer the model. (See [`Why_Linear_Steering_Fails_in_SSMs.md`](../Why_Linear_Steering_Fails_in_SSMs.md))
*   **Non-Linear Projection Works:** A small MLP ($\Phi$) can learn to map `(State, Target)` $\rightarrow$ `Delta`, forcing specific tokens with **Rank 1** success.

## 2. The "Two-System" Dynamics
Our experiments revealed a fundamental split in the model's operation:

1.  **System 1 (Prediction):** We can hijack this. Phi successfully forces the next token (e.g., "BLUE").
2.  **System 2 (Stability):** This fights back. When an injection is semantically unsupported by history, the model's dynamics often collapse into a refusal attractor ("I'm not sure...").

## 3. The Refusal Bifurcation
We rigorously investigated why the model refuses injections. (See [`The_Refusal_Bifurcation.md`](reports/The_Refusal_Bifurcation.md))

**Key Finding:** Refusal is a **dynamical signature of successful injection**.
*   **Weak Injection:** Model ignores it (No Refusal, Normal Generation).
*   **Strong Injection:** Model feels the violation and Refuses (in Greedy Mode).
*   **Solution:** Sampling allows the model to "escape" the refusal attractor and find the injected reality.

## 4. The Artifacts

### Codebase
*   `core/phi.py`: The trained non-linear projector.
*   `core/psi.py`: The behavioral stability monitor.
*   `tasks/`: Reproducible scripts for every claim (Sensitivity, Robustness, A/B Tests).

### Key Reports
1.  **[Blueprint](../blueprint.md):** The original theoretical architecture.
2.  **[Final Report](FINAL_REPORT.md):** The detailed technical breakdown of Phase 1-3.
3.  **[The Refusal Bifurcation](The_Refusal_Bifurcation.md):** The investigation into stability dynamics and the "Safety Reflex" retraction.
4.  **[Scaling Attempt Analysis](Scaling_Attempt_Analysis.md):** The investigation into generalization and compute costs.
5.  **[MockCache Failure Analysis](MockCache_Failure_Analysis.md):** The critical diagnosis of why optimization failed to transfer to inference.
6.  **[Optimization Autograd Blocker](Optimization_Autograd_Blocker.md):** The final engineering hurdle preventing Real-Cache optimization.
7.  **[Functional Control Breakthrough](Functional_Control_Breakthrough.md):** The solution to the Autograd Blocker, proving consistent, differentiable steering.
8.  **[Stabilized ALSI Analysis](Stabilized_ALSI_Analysis.md):** Mapping the trade-off between control strength (Layer 23) and manifold stability (Layer 12).
9.  **[Why Linear Steering Fails](../Why_Linear_Steering_Fails_in_SSMs.md):** The negative results that motivated ALSI.

---

**Final Status:**
The project has successfully defined the **Phase Diagram of Latent Control** for SSMs and solved the engineering challenges of differentiable state injection. We have moved from "Does this work?" to "Here is the physics of how it fails and succeeds." The Autograd Blocker is resolved; ALSI is now technically viable. The final frontier is tuning injection depth to balance forcing strength with trajectory coherence.
