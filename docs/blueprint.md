# **Project Blueprint: Augmented Latent State Injection (ALSI)**

Technical Alias: Recurrent Latent Caching (RLC)  
Target Model: Mamba-2 (AntonV/mamba2-130m-hf)

## **1\. Executive Summary**

**Augmented Latent State Injection (ALSI)** is a closed-loop memory architecture for State Space Models (SSMs). It treats the Mamba-2 recurrent state ($h\_t$) as a **Recurrent Latent Cache**. Unlike RAG, which expands the input window, ALSI performs **Inference-Time State Grafting** to manage context density and prevent "State Saturation."

The system employs a $\\Phi$ **(Phi) Writer** to project external information into the latent manifold and a $\\Psi$ **(Psi) Reader** to monitor state redundancy and entropy, utilizing the model's fixed-size state with maximum efficiency. ALSI adds $O(k \cdot d)$ overhead per injection, where $k$ is the number of targeted heads.

## **2\. Core Components: The Writer-Reader Symmetry**

### **2.1 The $\\Phi$ (Phi) Writer (The Injection Module)**

* **Role:** Transforms natural language into a state-delta vector ($\\Delta h$) for specific SSD (Structured State Duality) heads.  
* **Mechanism:** A **Cross-Attention Projector**. It attends to the current layer state to ensure the injection is semantically compatible with the existing manifold. $\Phi$ is trained using the base model’s own embedding layer to avoid representation mismatch.  
* **Gated Update:** $h\_{new} \= (1 \- g) \\cdot h\_{old} \+ g \\cdot \\Delta h$.

### **2.2 The $\\Psi$ (Psi) Reader (The Cache Manager)**

* **Role:** Measures state redundancy and **Latent Entropy**. *Latent entropy is operationalized as head-wise activation variance over a sliding window.*  
* **Operational Pivot:** Instead of "detecting truth," $\\Psi$ projects the incoming $\\Delta h$ onto the current state's subspace.  
* **Logic:** If **Cosine Similarity** $(\\Delta h, h\_{old}) \> \\tau$, the injection is suppressed as redundant. This prevents "State Stuffing."  
* **State Interrogation:** Uses relative metrics (sparsity, activation variance) to determine if a specific SSD head is "saturated" and requires an eviction (reset via the $1-g$ gate).

## **3\. Technical Refinements & Evaluation**

### **3.1 Hierarchical Sensitivity Mapping**

Instead of static layer targeting, ALSI utilizes an empirical **Sensitivity Curve**.

* **Objective:** Measure $\\Delta \\text{Loss}$ relative to injection at each layer index.  
* **Implementation:** During Phase 1, the system identifies the "Semantic Sweet Spot" (expected around middle layers) where factual grafts have the highest impact on next-token prediction without collapsing grammatical coherence.

### **3.2 The Blackout & Contradiction Tests**

* **Blackout Test:** Evaluate if the model can recall facts not present in its training weights via injection.  
* **Contradiction Test:** Observe the system's ability to "Overwrite" a prior graft. If ALSI can switch the model's answer from "Fact A" to "Fact B" cleanly using the Forget Gate ($g$), the architecture is validated.

### **3.3 Closed-Loop Homeostasis (The CRSM Extension)**

The Gating mechanism ($g$) is optionally conditioned on **loss sensitivity**. If a graft causes an immediate spike in next-token perplexity, the system triggers a **Rollback** or attenuates the injection strength, allowing the model to protect its own reasoning state.

## **4\. Execution Plan (Updated Phase 1\)**

### **Phase 1: Manifold Probing & Sensitivity (Weeks 1-2)**

1. **Baseline:** Load mamba2-130m-av.  
2. **Sensitivity Scan:** Systematically inject noise vs. factual vectors across all 24 layers. Plot the sensitivity curve.  
3. $\\Phi$ **Training:** Train the Cross-Attention Projector to align the base model's own text embeddings with the target SSD heads identified in the scan.

### **Phase 2: Gated Injection & Overwrite (Weeks 3-4)**

1. **Validation:** Perform the "Blackout Test."  
2. **Forget-Gate Tuning:** Optimize the $g$ parameter to find the balance between "Remembering the New" and "Preserving the Conversation."

## **5\. Citations & Theoretical Foundation**

* **\[1\] Hernandez, E., et al. (2023).** *REMEDI: Representation Editing for Multi-entity Knowledge.* \[Latent mapping concepts\].  
* **\[2\] Gu, A., & Dao, T. (2024).** *Mamba-2: Structured State Space Duality.* \[SSD architecture\].  
* **\[3\] Bricken, T., et al. (2023).** *Towards Monosemanticity.* \[Probing and sparse autoencoders\].  
* **\[4\] Anonymous. (2024).** *Stuffed Mamba: State Capacity and Information Interference in SSMs.* \[The "State Saturation" problem definition\].  
* **\[5\] Zhang, A., et al. (2024).** *Recursive Language Models.* \[Context-as-environment thesis\].

**Project ALSI** | *Recurrent Latent Caching for the Post-Attention Era.*