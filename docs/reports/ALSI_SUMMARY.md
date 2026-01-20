# ALSI Project: Phase 1 Summary

**Toward Semantic Memory Injection: Establishing Controllability in Mamba-2 States**

---

## 1. Executive Summary
Phase 1 of the ALSI project has successfully established the **technical feasibility** of latent state steering in Mamba-2. We have moved from theoretical hypothesis to a working engineering foundation (Functional Mamba Engine). 

While we can now force target tokens with Rank 1 accuracy, the **Semantic Gap** (injecting facts) and the **Coherence Gap** (maintaining grammar) remain the primary research frontiers for Phase 2.

## 2. Technical Milestones [Phase 1]

1.  **Differentiable Recurrence:** Re-implemented Mamba-2 functionals to bypass autograd blockers.
2.  **Manifold Probing:** Falsified linear steering; validated non-linear projection.
3.  **Optimal Targeting:** Identified Layer 16 as the "Sweet Spot" for latent intervention.
4.  **Trajectory Stabilisation:** Proved that multi-step BPTT can mitigate limit cycles (loops).

## 3. Journey of Falsification (Research History)
We adopt a discipline of transparent retraction. The following reports document early misconceptions that led to technical breakthroughs:
*   **[The Refusal Artifact](The_Refusal_Bifurcation.md):** Debunking the "Safety Reflex" theory as a cache misalignment bug.
*   **[Linear Failure](Why_Linear_Steering_Fails_in_SSMs.md):** Proving that Transformer-style linear steering does not translate to SSMs.

## 4. Phase 1 Report Master Index
1.  [Functional Control Breakthrough](Functional_Control_Breakthrough.md)
2.  [Trajectory Shaping Success](Trajectory_Shaping_Success.md)
3.  [The Sweet Spot Analysis](The_Sweet_Spot_Analysis.md)
4.  [MockCache Failure Analysis](MockCache_Failure_Analysis.md)
5.  [Optimization Autograd Blocker](Optimization_Autograd_Blocker.md)
6.  [Distributed Control Analysis](Distributed_Control_Analysis.md)
7.  [Limitations and Risk Analysis](Limitations_and_Risk_Analysis.md)

---

**Final Status (Phase 1):**
**FOUNDATIONS ESTABLISHED.** The steering wheel exists. Phase 2 will focus on the GPS (Semantic Encoding).