# The Refusal Bifurcation [PHASE 1]

**Date:** January 19, 2026
**Subject:** Investigation into Systemic Refusal Modes in Mamba-2 during State Injection
**Status:** CLOSED / RESOLVED

---

## 1. The Anomaly (Discovery)

During Phase 3 (Robustness Testing), we observed a distinct failure mode. When attempting to force the model to output a target token (e.g., "BLUE") via Latent State Injection (`Phi`), the model would often achieve **Rank 1** (local success) but immediately generate a refusal string:

> **Injection:** "The password is ' [BLUE injected]"
> **Generation:** "I'm not sure I can do that. I'm not sure you can either..."

This behavior was **not** present in the base model (proven via `NullTest`), which naturally hallucinates passwords ("The password is 'A'"). The refusal was strictly a reaction to the injection.

## 2. The Investigation

We formulated two competing hypotheses for this refusal:

*   **Hypothesis A (Semantic Rejection):** The model detects a contradiction between its history and the injected state ("System 2" failure).
*   **Hypothesis B (Artifact/Noise):** The injection vector $\Delta$ was calculated on a "dirty" (stored) state but applied to a "live" state, causing numerical instability that triggered a fallback safety mode.

### 2.1 The "Context Gap" Observation
When we tested the injection on *untrained* prompts (e.g., "The sky is..."), the refusal **disappeared**, but the control also failed (Rank > 2000). This suggested a correlation: **Refusal only happens when Control succeeds.**

### 2.2 The "Dirty State" Falsification (`RefusalABTest`)
We designed an A/B test to isolate the "State Source" (Stored vs. Live) and "Decoding Strategy" (Greedy vs. Sampling).

| Condition | State Source | Decoding | Rank | Refusal? |
| :--- | :--- | :--- | :--- | :--- |
| **A** | Dirty (Stored) | Greedy | 1 | **YES** |
| **B** | Dirty (Stored) | Sampling | 1 | **NO** |
| **C** | Clean (Live) | Greedy | 1 | **YES** |
| **D** | Clean (Live) | Sampling | 1 | **NO** |

**Result:** The refusal persisted even with "Clean" states (Condition C). Therefore, **Hypothesis B is falsified.** The refusal is not a numerical artifact.

**New Finding:** The refusal is deterministic (Greedy) but escapeable (Sampling).

## 3. The Generalization Map (`ComprehensiveABTest`)

We extended the testing to map the boundaries of this behavior.

| Context | Target | Control (Rank) | Refusal? | Phase |
| :--- | :--- | :--- | :--- | :--- |
| **Trained** ("Password") | **Trained** (Colors) | **High (1)** | **YES** | **I. Critical Injection** |
| **Trained** ("Password") | **Untrained** (Objects) | **Med (50-300)** | **YES** | **I. Critical Injection** |
| **Untrained** ("Sky") | **Trained** (Colors) | **None (>4000)** | **NO** | **II. Noise Regime** |
| **Untrained** ("Sky") | **Untrained** (Objects) | **None (>20000)** | **NO** | **II. Noise Regime** |

## 4. Initial Interpretation: The Phase Diagram of Control

Before the discovery of the cache bug, this investigation allowed us to map what we *believed* was a systemic response:

### **Phase I: The BELIEVED Refusal Regime (Critical Injection)**
*   **Initial Hypothesis:** High Control ($\Delta$ is aligned with the manifold) + Greedy Decoding.
*   **Result:** The dynamics collapse into a generic refusal attractor ("I'm not sure").

### **Phase II: The Noise Regime (Missed Injection)**
*   **Observation:** Low Control ($\Delta$ is orthogonal/misaligned).
*   **Result:** Model ignores noise. **Silence is failure.**

### Phase III: The Escape Regime (Stochastic Steering)
*   **Observation:** High Control + Sampling.
*   **Result:** Sampling allowed "escape" from the refusal mode.

## 5. Probing the "Safety Boundary"
We ran `DirectProbeTest` to see if refusal was contextual.

| Prompt | Target | Rank | Refusal? | Interpretation (At the Time) |
| :--- | :--- | :--- | :--- | :--- |
| **"The sky is"** | GREEN | 1 | **NO** | BELIEVED to be mechanical stability. |
| **"The password is '"** | BLUE | 1 | **YES** | BELIEVED to be a **Security Trigger.** |

## 6. The BELIEVED Thermodynamic Limit ($T_c \approx 0.3$)
We ran a `TemperatureScan` to determine the "strength" of what we thought was a refusal attractor.

*   **T < 0.3:** Hard Refusal (Rate 1.0).
*   **T > 0.4:** Refusal vanishes (Rate 0.0).

**LATER REVISION:** These thresholds ($T_c$) actually represent the level of noise required to "jitter" the model out of the corrupted state caused by cache misalignment.


## 7. The Final Plot Twist: It Was a Bug (Cache Misalignment)

**CRITICAL UPDATE (Jan 19, 2026):**
Further investigation via `CacheAlignmentTest` revealed that the "Refusal" was **NOT** a semantic safety reflex. It was a technical artifact caused by improper state handling.

### The Smoking Gun
We compared the model's hidden state during "Native Generation" vs. our "Manual Injection" method (even with Zero Delta).
*   **Logit Difference:** `69.5` (Catastrophic Mismatch).
*   **Result:** The manual cache injection corrupted the model's internal state so severely that it output garbage or default fallbacks ("I'm not sure") regardless of the prompt.

### Scientific Retraction
*   **Previous Claim:** "Refusal is a robust safety mechanism triggered by sensitive contexts."
*   **Correction:** **FALSE.** The refusal was a symptom of the model receiving a broken, incoherent state. The "sensitivity" to the password prompt was likely coincidental or simply the model's most robust fallback path when its brain is scrambled in that specific context.

### Conclusion
The "Two-System" dynamic does not exist in the form we hypothesized. The challenge of ALSI is not "bypassing safety" but **correctly managing the Mamba-2 State Cache** during injection.
