# Why Linear Steering Fails in State Space Models

**TL;DR:** Unlike Transformer residual streams, the recurrent state of Mamba-2 does not encode semantic features in globally linear subspaces. Attempts to steer the model using vector arithmetic (Linear Addition, PCA) failed completely.

## 1. The Sensitivity Vacuum (Naive Addition)

Our first hypothesis was that we could simply "add" a concept to the state, similar to word vector arithmetic (`King - Man + Woman = Queen`).

$$ h_{new} = h_{old} + \alpha \cdot h_{concept} $$

**Experiment:** We took the hidden state of "The sky is " and added the state vector for "BLUE".
**Result:**
*   **Sensitivity:** ~0.0 (Flat)
*   **Logit Movement:** None.
*   **Conclusion:** The state manifold is extremely high-dimensional and non-linear. A simple additive vector gets "swallowed" by the normalization and gating mechanisms of the SSM layer.

## 2. The Contrastive Failure (Delta PCA)

We attempted to find a "control direction" by analyzing the difference between states representing contrasting concepts.

$$ \vec{v}_{color} = \Delta h_{BLUE} - \Delta h_{RED} $$

**Hypothesis:** Moving along $\vec{v}_{color}$ should shift the probability from Red to Blue.
**Experiment:** We performed PCA on the difference vectors of 6 color tokens and optimized a steering vector within this subspace.
**Result:**
*   **Rank Improvement:** 7328 $\rightarrow$ 7328 (Zero change).
*   **Conclusion:** The direction that differentiates "Blue" from "Red" in the *output* space is not linearly accessible from the *transition* space. The mapping $f(h_t) \rightarrow h_{t+1}$ is too complex for linear contrastive vectors to hold valid semantics across different contexts.

## 3. The Natural Manifold Failure (Transition PCA)

We hypothesized that maybe we can't create *new* directions, but we can steer along *existing* valid transition directions.

**Hypothesis:** The "valid" state space is a lower-dimensional manifold defined by natural transitions.
**Experiment:** We collected 1000s of natural state updates ($h_{t+1} - h_t$) and computed their principal components. We then tried to optimize a delta within this "valid" subspace to force a specific token.
**Result:**
*   **Optimization:** The optimizer failed to reduce the loss significantly.
*   **Rank:** Remained high (>1000).
*   **Conclusion:** The "Golden Delta" (the specific perturbation needed to force a target) lies **off-manifold**. It is not a combination of typical natural updates. To control the model, you must inject a "hyper-natural" state that forces the dynamics into a specific attractor, often violating the statistical properties of natural text generation.

## Summary

Control in Mamba-2 requires **non-linear, off-manifold grafting**.
This motivated the creation of the **Phi Projector**, a neural network capable of finding these complex, non-linear control surfaces that linear algebra could not.
