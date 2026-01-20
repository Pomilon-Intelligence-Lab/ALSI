# ALSI Project: Executive Summary

**Augmented Latent State Injection (ALSI)**
*A framework for deterministic, non-linear control of Mamba-2 State Space Models.*

---

## 1. The Core Discovery
**Logit Control $\neq$ Trajectory Control.**

We successfully proved that while Mamba-2's recurrent state is resistant to linear steering (PCA/Vector Arithmetic), it contains a **learnable, non-linear control surface**.

*   **Linear Steering Fails:** Simple addition or subspace optimization cannot steer the model. (See [`Why_Linear_Steering_Fails_in_SSMs.md`](../Why_Linear_Steering_Fails_in_SSMs.md))
*   **Non-Linear Projection Works:** A small MLP ($\Phi$) can learn to map `(State, Target)` $\rightarrow$ `Delta`, forcing specific tokens with **Rank 1** success.

## 2. The Journey of Falsification: From "Two Systems" to Cache Truth

Early in the project, we observed that forcing a target token often resulted in the model immediately outputting "I'm not sure" (a refusal). This led to the **Two-System Hypothesis** (documented in [`The_Refusal_Bifurcation.md`](reports/The_Refusal_Bifurcation.md)):
1.  **System 1 (Logits):** We hijacked the immediate prediction.
2.  **System 2 (Stability):** The model "rejected" the inconsistent state via a safety reflex.

**The Correction:** Rigorous A/B testing and cache inspection proved this was a **technical artifact**. The "refusal" was a symptom of **Cache Misalignment** (Logit Diff ~69.5) caused by improper manual state handling. Once we moved to a **Functional Mamba Step**, the model accepted the injections without refusal, proving that Mamba-2 does not inherently "fight" state grafting if the manifold integrity is maintained.

## 3. Technical Breakthroughs

1.  **Functional Recurrence:** We bypassed the stateful limitations of standard wrappers by re-implementing Mamba-2 as a pure function, enabling true end-to-end gradients.
2.  **The Sweet Spot:** Identified **Layer 16** as the optimal depth for injection, where control authority and manifold stability are in balance.
3.  **Trajectory Shaping:** Solved the "Sticky Attractor" problem (looping). By optimizing across multiple unrolled steps (BPTT), we achieved **Transient Injection**—forcing a token and then allowing the model to recover naturally.

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
9.  **[Distributed Control Analysis](Distributed_Control_Analysis.md):** Proving that multi-layer injection reduces local state stress but requires multi-step objectives to prevent looping.
10. **[Smooth Injection Analysis](Smooth_Injection_Analysis.md):** Demonstrating that depth-wise smoothing improves manifold resilience but confirms the "Sticky Attractor" problem.
11. **[Trajectory Shaping Success](Trajectory_Shaping_Success.md):** The final breakthrough: achieving non-looping, coherent steering via multi-step functional optimization.
12. **[The Sweet Spot Analysis](The_Sweet_Spot_Analysis.md):** High-resolution layer-wise sensitivity mapping identifying Layer 16 as the optimal injection depth.
13. **[Why Linear Steering Fails](../Why_Linear_Steering_Fails_in_SSMs.md):** The negative results that motivated ALSI.

---

**Final Status:**
The project has successfully achieved **Coherent Latent Steering**. By implementing a functional Mamba recurrence and optimizing across spatio-temporal fields, we have solved the Engineering Blocker (Autograd), the Semantic Blocker (Refusal), and the Dynamical Blocker (Looping). ALSI is now a validated primitive for "Internalized RLM." The next phase is the construction of the **Phi-T Training Pipeline** to scale this control to arbitrary contexts.
