# Augmented Latent State Injection (ALSI)

> [!CAUTION]
> **Research Artifact Disclaimer**: This repository is a collection of research experiments and technical artifacts documenting a specific investigation into Mamba-2 state dynamics. It is **not** a production-ready framework, a stable product, or a library for general use. The methods, code, and conclusions contained herein are exploratory and provided as-is for scientific transparency.

> **"Logit control ≠ trajectory control"**

This repository contains the experimental history and findings regarding the discovery of a non-linear control surface within Mamba-2 State Space Models. It serves as a narrative artifact that falsifies the assumption of linear state steering and explores a new primitive for "grafting" semantic intentions directly into the model's recurrent dynamics.

---

## The Project Goal: Recurrent RLM

**Vision:** To internalize the recursive context processing of MIT's **Recursive Language Models (RLM)** directly into the latent dynamics of the model.

*   **RLM (MIT):** "Context as Database." The model uses an external Python REPL to query, split, and summarize massive contexts. It requires explicit reasoning and code generation to access memory.
*   **ALSI (Our Vision):** "Context as Memory." We use a trained projector ($\Phi$) to mathematically inject external information ($\Delta$) directly into the model's recurrent state ($h_t$).

**The Outcome:** A model that "never forgets" because the relevant context is not just available via search, but is **implanted into its immediate state of mind**, allowing for zero-latency, implicit access to infinite context.

---

---

## The Project Goal: Recurrent RLM

**Vision:** To internalize the recursive context processing of MIT's **Recursive Language Models (RLM)** directly into the latent dynamics of the model.

*   **RLM (MIT):** "Context as Database." The model uses an external Python REPL to query, split, and summarize massive contexts. It requires explicit reasoning and code generation to access memory.
*   **ALSI (Our Vision):** "Context as Memory." We use a trained projector ($\Phi$) to mathematically inject external information ($\Delta$) directly into the model's recurrent state ($h_t$).

**The Outcome:** A model that "never forgets" because the relevant context is not just available via search, but is **implanted into its immediate state of mind**, allowing for zero-latency, implicit access to infinite context.

---

## The Question

**Can we deterministically control the output of a State Space Model (Mamba-2) by surgically modifying its recurrent state?**

Unlike Transformers, which have a queryable "residual stream" at every token, Mamba compresses history into a fixed-size state. We asked if we could calculate a `Delta` such that:

$$ State_{new} = State_{old} + \Delta \rightarrow \text{Target Output} $$

## The Failed Assumptions

We initially hypothesized that semantic features (like "color") resided in linear subspaces, accessible via:
1.  **Linear Addition:** `h_new = h_old + h_target` (Result: **Failed**, no sensitivity)
2.  **Contrastive Steering:** `Delta = h_blue - h_red` (Result: **Failed**, zero logit movement)
3.  **Natural Transition PCA:** Steering within the manifold of natural state updates (Result: **Failed**, rank did not improve)

See `docs/Why_Linear_Steering_Fails_in_SSMs.md` for the technical breakdown.

## The Discovery

Semantic control in Mamba-2 is **non-linear**, **off-manifold**, and **functionally differentiable**.

*   **Golden Delta:** For any target token, there exists a specific, high-magnitude state perturbation that forces the model to output it (Loss < 0.02).
*   **The Phi Projector:** This non-linear mapping is learnable. We proved that a recursive MLP ($\Phi$) can predict these deltas across multiple layers.
*   **Functional Recurrence:** By reverse-engineering the Mamba-2 kernel into a pure functional step, we bypassed the stateful limitations of standard wrappers, enabling true end-to-end control.

## The New Primitive

ALSI is a framework for **State-Space Grafting**. It allows us to insert a semantic intention into the model's recurrent stream without seeing the full prompt context.

---

## Terminology

To avoid confusion with "jailbreaking" or activation patching:

*   **State Injection:** Operationally, we add a vector to the state. However, this is best understood as **Transient State Grafting**. We are not overwriting the model's memory; we are inserting a "phantom" transition that forces the dynamics to evolve towards a specific immediate future.
*   **Phi ($\Phi$) Projector:** The trained MLP that calculates the necessary graft vector.
*   **Psi ($\Psi$) Reader:** (Conceptual/Experimental) A stability monitor that detects when a graft forces the trajectory off-manifold, triggering refusal or collapse.

---

## The Psi Evolution Ladder

ALSI treats monitoring as an evolving instrument rather than a single feature. The project progresses through three stages of $\Psi$:

1.  **$\Psi$-B (Behavioral):** *[Current Implementation]* Detects trajectory rejection by monitoring output entropy and refusal patterns in generated text.
2.  **$\Psi$-L (Latent):** *[Phase 4 Goal]* Detects instability inside the hidden state manifold (e.g., Cosine Similarity, KL Divergence vs. natural transitions) *before* tokens are sampled.
3.  **$\Psi$-T (Trajectory):** *[Conceptual]* Closed-loop enforcement where $\Psi$ modulates the $\Phi$ injection strength in real-time to maintain state stability.

---

## What ALSI Is *Not*

*   ❌ **Not Full Semantic Overwrite:** It does not erase the model's previous history.
*   ❌ **Not Prompt Replacement:** It acts on the compressed state, not the input tokens.
*   ❌ **Not Linear Interpretability:** It proves that semantic control directions are non-linear w.r.t. the natural state manifold.

## What ALSI Demonstrates

*   **Local Controllability:** The transition operator $A$ and $B$ matrices in Mamba allow for arbitrary next-token forcing if the state perturbation is precise enough.
*   **The "Cache Alignment" Challenge:** The primary barrier to state injection is not semantic safety but **technical state management**. Manually modifying the Mamba-2 cache requires precise handling of convolutional and SSM states; even slight misalignments (Logit Diff > 60) cause the model to collapse into default fallback behaviors (refusal/hallucination).
*   **Retraction of "Safety Reflex":** Earlier claims that the model actively "refuses" injections were proven to be artifacts of cache corruption.
*   **Trajectory Shaping:** We solved the "looping problem" (`BLBLBL`) by moving from single-token forcing to **Multi-Step Optimization**. By explicitly penalizing token repetition during training, we achieved **Transient Latent Steering**—injections that turn on, flip the token, and then extinguish themselves to let the model recover naturally.

---

## Breakthrough: Functional ALSI (Jan 2026)

We have successfully implemented **Differentiable State Injection**. By reverse-engineering the Mamba-2 kernel into a pure functional step, we bypassed the autograd limitations of the stateful model.

*   **Result:** Consistent **Rank 1** forcing of target tokens across all prompts.
*   **Status:** "Steering Wheel" is active. Next challenge is maintaining manifold coherence (preventing loops/hallucinations after injection).

See `docs/reports/Functional_Control_Breakthrough.md` for details.

---

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