# Augmented Latent State Injection (ALSI)

> [!CAUTION]
> **Research Artifact Disclaimer**: This repository documents exploratory research into Mamba-2 state dynamics. It is a **technical feasibility study** demonstrating single-token latent steering. It is **not** a production-ready framework, a memory system, or a finished product.

> **"Logit control ≠ trajectory control"**

ALSI investigates whether Mamba-2 State Space Models can be steered via learned non-linear latent perturbations. We prove that while the recurrent state is a differentiable control surface, achieving coherent, multi-step steering remains a significant research challenge.

---

## Current Status: Phase 1 (Technical Feasibility)

We have successfully built the **engineering foundation** for latent steering in SSMs, but the system is currently limited to **single-token forcing**. 

### What has been proven:
*   ✅ **SSM states are controllable** via off-manifold, non-linear perturbations (where linear methods fail).
*   ✅ **Functional differentiation is required** to bypass stateful cache limitations in standard model wrappers.
*   ✅ **Target tokens can be forced** with Rank 1 accuracy across various contexts.

### What is NOT yet proven:
*   ❌ **Trajectory Coherence:** Steering a token currently results in garbled/hallucinatory continuations.
*   ❌ **Semantic Fact Injection:** We have not yet demonstrated that injections can encode meaningful facts (e.g., memory recall) beyond statistical logit bias.

---

## The Long-Term Vision: Recurrent RLM

ALSI was inspired by the goal of internalizing the recursive context processing of MIT's **Recursive Language Models (RLM)**.

*   **The Idea:** To move from "Context as an external database" (RLM) to "Context as internal latent memory" (ALSI).
*   **The Goal:** A system where external facts are projected directly into the model's state, allowing it to "remember" forgotten context without re-prompting.

See `docs/vision.md` for the roadmap from our current "Token Forcer" to this ultimate goal.

## Key Breakthroughs

*   **Functional Recurrence:** A pure PyTorch implementation of the Mamba-2 step, enabling backpropagation through time (BPTT) for state optimization.
*   **Spatio-Temporal Mapping:** Discovery of the "Phase Diagram" of control stability and identifying Layer 16 as the optimal steering depth.
*   **Loop Mitigation:** Proving that multi-step objectives can break the "Sticky Attractor" limit cycles.

---

## Limitations & Risks

*   ❌ **No Coherent Generation:** Injected states currently result in garbled fallbacks or hallucinations after the forced token.
*   ❌ **Context Sensitivity:** The current Phi projector is highly sensitive to the initial prompt manifold.
*   ⚠️ **Security Risk:** This technique, if scaled, represents a potential vector for model hijacking or jailbreaking. See `docs/reports/Limitations_and_Risk_Analysis.md`.

## Key Visualizations

### The Failure of Linearity vs. The Success of Phi

![Sensitivity Curve](docs/images/sensitivity_curve.png)
*Naive linear addition (blue/flat) does nothing. The Phi Projector (learning curve, see full report) successfully finds the control surface.*

### The Cost of Control

![Pareto Frontier](docs/images/pareto_frontier.png)
*Control requires high-energy deltas (Y-axis) that fight the model's natural compression (X-axis).*

### Generalization & Refusal (Historical Artifact)

![Robustness](docs/images/robustness_ranks.png)
*Phi generalizes to semantic neighbors (PINK) but distant targets (CYAN) remain hard to steer. Early results showed the model "rejecting" the graft; this was later proven to be a **cache misalignment bug** rather than a semantic mechanism. Functional steering eliminates this refusal.*

---

## Repository Structure

* `core/`: Shared infrastructure.
  * `functional_mamba.py`: Pure PyTorch differentiable Mamba-2 step.
  * `phi_t.py`: The Trajectory-Aware Projector ($\Phi_T$).
* `tasks/`: Implementation of specific experiments.
  * `shaping_optimization.py`: **Final Breakthrough**: Multi-step BPTT for coherent control.
  * `functional_sensitivity_scan.py`: High-resolution "Sweet Spot" analysis.
  * `ab_test_refusal.py`: **Critical Diagnosis**: Proving refusal was a bug.
* `docs/`: Technical reports and blueprints.
  * **[EXECUTIVE SUMMARY](docs/reports/ALSI_SUMMARY.md):** Start here.
  * `reports/Trajectory_Shaping_Success.md`: Deep dive into non-looping control.
  * `reports/Functional_Control_Breakthrough.md`: Solving the Autograd blocker.
  * `reports/Phase1_Linear_Failure_Report.md`: Early research history.

## Quick Start

### Reproducing Results

1. Install dependencies: `pip install -r requirements.txt`
2. Run the full experimental pipeline: `python main.py --task all`
3. Train Phi: `python main.py --task train_phi`

To generate plots locally: `ALSI_Plots.ipynb` (in Archive).

### Hardware

* Validated on AMD Ryzen 5 PRO (CPU-only execution supported but slow).
* Recommended: 12GB+ VRAM GPU.