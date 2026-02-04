# ALSI: First Steps Toward Semantic Memory Injection

> **"Establishing Controllability in Mamba-2 Recurrent States"**

> [!CAUTION]
> **Research Artifact Disclaimer**: This repository documents exploratory research into Mamba-2 state dynamics. It is a **Phase 1: Technical Feasibility Study**. It is **not** a production-ready framework or a functional memory system.
> 
> [!WARNING]
> **SECURITY ALERT: SCAM REPOSITORIES**
> It has come to our attention that several malicious forks and clones (e.g., `AnagamiZz/ALSI`) are circulating. These repositories use SEO-bait tags (like "motor control" or "csharp") and fake READMEs to trick users into downloading "installers" or "executables."
> **ALSI is a Python research project.**
> * There is **NO** `.exe`, `.dmg`, or "Installer" for ALSI.
> * There is **NO** "User Interface" or "Application" version.
> * Official research code is only hosted here: [github.com/Pomilon-Intelligence-Lab/ALSI](https://github.com/Pomilon-Intelligence-Lab/ALSI).
> 
> 
> If you encounter a version of ALSI asking you to "Visit the Releases page to download," **do not run the file.** It is likely malware.

---

## 🎯 The Vision: Internalized RLM
The long-term goal of ALSI is to internalize the context processing of MIT's **Recursive Language Models (RLM)** directly into the latent dynamics of State Space Models.

*   **The Dream:** A model that mathematically "uploads" external facts into its recurrent state ($h_t$), allowing it to process unbounded context with zero-latency implicit recall.
*   **The Reality:** We are currently at the **infrastructure layer**, proving that such steering is physically and mathematically possible.

---

## 🔬 Current Reality: Phase 1 Complete
We have established the foundational capabilities required for latent steering. 

### ✅ What Works (Foundations)
*   **Controllability Proof:** SSM states are controllable via non-linear, off-manifold perturbations (linear methods like PCA fail).
*   **Functional Engine:** A custom differentiable Mamba-2 implementation that bypasses stateful cache limitations.
*   **Token Forcing:** A trained projector ($\Phi$) can force target tokens with Rank 1 accuracy.

### ❌ What Doesn't Work Yet (The Research Frontier)
*   **Coherence:** Model output often becomes garbled after the forced token (The Coherence Gap).
*   **Semantic Encoding:** We can force a specific token (e.g., "BLUE") but haven't yet proven we can inject a factual statement (e.g., "John lives in Paris").
*   **Memory Validation:** No experiments have been conducted on long-range recall or QA tasks.

---

## 🛤️ The Roadmap: Baby Steps to the Vision

| Phase | Milestone | Status |
| :--- | :--- | :--- |
| **Phase 1: Token Control** | Prove states are differentiable and controllable. | **COMPLETE** ✅ |
| **Phase 2: Fact Injection** | Inject semantic facts and verify recall in QA tasks. | *PLANNED* 🔄 |
| **Phase 3: Multi-Hop Reasoning** | Inject multiple compositional facts simultaneously. | *CONCEPTUAL* 🔮 |
| **Phase 4: Continuous Memory** | Rolling latent injection in long-range conversations. | *CONCEPTUAL* 🔮 |
| **Phase 5: RLM Parity** | Match/Exceed RLM performance on long-context benchmarks. | *CONCEPTUAL* 🔮 |

---

## Repository Navigation

### Core Foundations [Phase 1]
*   `core/functional_mamba.py`: Differentiable Mamba-2 recurrence.
*   `core/phi_t.py`: Recursive Trajectory Projector.

### Technical Reports
1.  **[EXECUTIVE SUMMARY](docs/reports/ALSI_SUMMARY.md):** Start here for the high-level research arc.
2.  **[The Vision](docs/VISION.md):** The original spark and core hypothesis.
3.  **[Current State](docs/CURRENT_STATE.md):** Detailed breakdown of Phase 1 results.
4.  **[Roadmap](docs/ROADMAP.md):** Immediate experiments to close the gap.

---

## Quick Start
1.  `pip install -r requirements.txt`
2.  Run Phase 1 validation: `python main.py --phase 1-token-control`
3.  View negative results: `docs/Why_Linear_Steering_Fails_in_SSMs.md`
