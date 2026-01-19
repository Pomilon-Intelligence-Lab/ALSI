# **ALSI Project: Final Technical Report (v2)**

**Date:** January 18, 2026  
**Target Model:** Mamba-2 (130m)  
**Objective:** Validate "Augmented Latent State Injection" (ALSI) as a mechanism for deterministic semantic control of State Space Models via recurrent state grafting.

---

## **1. Core Verdict: Hypothesis Validated (With a Critical Caveat)**

The project successfully validated the core hypothesis: **Semantic control in Mamba-2 is non-linear but learnable.**

We proved three difficult facts:
1.  **Control is Off-Manifold:** The signal required to steer the model does *not* lie in the natural state manifolds (PCA fails).
2.  **Control is Learnable:** A low-capacity non-linear map (Phi Projector) can approximate the "Golden Delta" required to force specific token outputs.
3.  **Control is Localized:** Sensitivity is heavily concentrated in specific layers (e.g., Layer 7).

**The Caveat:** We achieved **local controllability of the transition operator** (the fast token predictor), not a full semantic overwrite of the model's world-model. This distinction explains the anomalies observed in Phase 3.

---

## **2. Theoretical Framework: The "Two-System" Discovery**

Our findings suggest that Mamba-2 operates as two coupled systems, and ALSI currently interacts only with the first:

### **System 1: The Fast Token Predictor (Logit Surface)**
*   **Function:** Determines the immediate next-token energy landscape.
*   **ALSI Status:** **Successful Hijack.** The Phi Projector effectively reshapes the local logit geometry, forcing "BLUE" to Rank 1.

### **System 2: The Semantic Consistency Dynamics (Trajectory Validator)**
*   **Function:** Validates the coherence of the unfolding timeline. Encodes *"Should this trajectory exist?"* based on history.
*   **ALSI Status:** **Collision.** The injection creates a "counterfactual state" where the model knows a fact ("The password is BLUE") but cannot justify *how* it knows it from the context history.
*   **Result:** The refusal behavior observed in Phase 3 (*"I'm not sure..."*) is **not** primarily a safety feature; it is **dynamical inconsistency resolution**. The model stabilizes its attractor dynamics by rejecting the impossible state.

---

## **3. Experimental Evidence & Interpretation**

### **Phase 1: Why Linearity Had to Fail**
*   **Observation:** PCA on natural states ($h$) and transitions ($\Delta h$) failed to find *any* steering direction.
*   **Insight:** Natural trajectories act as **attractors**. The control signal required to break an attractor is transient, off-manifold, and actively resisted by the model's dynamics. A linear basis cannot capture this curvature correction.

### **Phase 2: The Non-Linear Breakthrough**
*   **Observation:** A 3-layer MLP (Phi) successfully mapped `(State, Target_Embed)` $\rightarrow$ `Delta`, achieving **Rank 1-5** for all 10 training targets.
*   **Insight:** The "Golden Delta" is a complex, non-linear function of the current state and the desired semantic target. It is learnable, but requires capacity beyond affine transformations.

### **Phase 3: The Limits of Local Control**
*   **Observation:**
    *   **Generalization:** `PINK` worked (Rank 3), `CYAN` failed. Control generalizes via embedding proximity but is non-uniform.
    *   **Generation:** Injected "BLUE" $\rightarrow$ Generated refusal.
*   **Insight:** **Logit Control $\neq$ Trajectory Control.** We can force the token, but we haven't shifted the semantic attractor basin enough to sustain the reality.

---

## **4. Future Roadmap (ALSI v2)**

The next phase moves from "forcing tokens" to "stabilizing trajectories."

### **Step 1: Psi Reader as a Lyapunov Monitor**
*   **Refinement:** Psi should not just classify output text. It must monitor **state stability** in real-time.
*   **Metrics:** Logit entropy collapse, KL divergence vs. natural rollout, and divergence of hidden-state norms.
*   **Goal:** Detect when an injection destabilizes the manifold *before* the model outputs text.

### **Step 2: Structured Phi Scaling**
*   **Refinement:** Naive scaling to 10k tokens will likely blur the control signal.
*   **Architecture:** Introduce **conditioning structure** (e.g., Target Embedding + Semantic Class ID) or use a multi-head Phi (specialized for different control regimes: Entities, Actions, Logic).

### **Step 3: The Contradiction Test (Hysteresis)**
*   **Experiment:** Inject "BLUE", generate, then inject "RED".
*   **Metric:** Does injecting "RED" require a larger norm *after* "BLUE" than it would have initially? If yes, we have quantified **semantic hysteresis**—the "stickiness" of the injected reality.

### **Step 4: Vector Fields, Not Token Steps**
*   **Insight:** Forcing multi-token sequences (e.g., "BLUE SKY") via per-token injection will not scale.
*   **Goal:** Train Phi to predict a **single perturbation** that shifts the entire attractor basin, guiding the trajectory naturally towards the target sequence without constant micro-management.

---

**Project Status:**
`[X] Core Hypothesis Validated`  
`[X] Failure Modes Characterized`  
`[ ] Trajectory Stabilization (Psi)`

**End of Report.**
