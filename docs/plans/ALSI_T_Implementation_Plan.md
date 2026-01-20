# Implementation Plan: ALSI-T (Trajectory Control)

**Date:** January 19, 2026
**Objective:** Evolve ALSI from a static "State Grafting" experiment into a robust "Latent Trajectory Control" system.

---

## 1. Architectural Rethink

To solve the **Sticky Attractor** (looping) and **Coherence Gap** problems, we must evolve our core components:

### **1.1 Phi ($\Phi$) - The Field Projector**
*   **Old Role:** Map `(Target) -> Single Delta`.
*   **New Role:** Map `(Context, Target, Layer_ID) -> Delta_Field`.
*   **Design:** A coordinated system that generates a continuous perturbation field across multiple layers. This allows the model to inject "Concepts" in early layers and "Tokens" in late layers simultaneously.

### **1.2 Psi ($\Psi$) - The Dynamic Governor**
*   **Old Role:** Post-generation failure reporting.
*   **New Role:** Real-time temporal dampening.
*   **Design:** A "Temporal Decay Gate" that monitors the accumulation of injection energy. Once the target token is generated, Psi aggressively decays the $\Delta$ field to zero to prevent the model from falling into a limit cycle (looping).

---

## 2. Implementation Roadmap

### **Step 1: Spatio-Temporal Optimization (Task: Transient Injection)**
Before scaling, we must prove that **Transience** solves the looping.
*   **Experiment:** Optimize a field at $t=0$, then generate at $t=1 \dots N$ with $\Delta=0$.
*   **Goal:** Force the first sub-token, then allow the model's natural dynamics to "heal" the trajectory.

### **Step 2: Multi-Step Functional Training**
Upgrade the training pipeline to "Trajectory Shaping."
*   **Loss Function:** 
    $$\mathcal{L} = \text{Force}(t+1) + \text{Natural\_Rollout}(t+2 \dots t+k)$$
*   **Mechanism:** Backpropagate through time (BPTT) using the `functional_mamba_block` to ensure the injection is stable across multiple steps.

### **Step 3: The ALSI-T Inference Engine**
Construct a standalone inference loop that replaces `model.generate()`.
1.  **Project:** Phi generates the $\Delta$ field for the target context.
2.  **Govern:** Psi modulates the field strength step-by-step.
3.  **Execute:** Functional Mamba blocks perform the state transitions.

---

## 3. The "Holy Grail" Target
**Implicit Context Injection:** The model should be able to answer "The password is BLUE" in greedy mode, with high coherence, without ever seeing the word "BLUE" in the input prompt.

**Next Milestone:** Validate the **Temporal Decay** hypothesis via `tasks/transient_injection_test.py`.
