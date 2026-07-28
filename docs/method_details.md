# Method Details: Evidential HDC & Online Prior Estimation for TTA

**Location:** `EvidentialHDC/docs/method_details.md`
**Status:** Core Documentation

## 1. The Problem Setting: Unsupervised Online Test-Time Adaptation under Spatiotemporal Corruption
In real-world autonomous navigation and spatial perception, deep learning models deployed in open-world environments inevitably encounter severe out-of-distribution (OOD) shifts and sensor degradations (e.g., LiDAR beam dropouts, heavy snow, wet ground reflections, and sensor misalignments as formalized in SemanticKITTI-C).

We address the problem of **Unsupervised Online Test-Time Adaptation (TTA)**, governed by three critical operational constraints:
1. **Absence of Ground Truth Supervision:** The model must adapt its internal decision boundaries and class prototypes on the fly using *only* streaming test point clouds and self-generated pseudo-labels.
2. **Streaming Batch Adaptation (Memory & Compute Bounds):** The model operates on sequential frames or disjoint temporal chunks without access to source training data or large replay buffers, precluding heavy offline optimization or multi-pass re-training.
3. **Severe Class Imbalance & Structural Degradation:** Standard perception datasets exhibit extreme Pareto class imbalance (e.g., *road* and *building* outnumber *motorcyclist* or *bicyclist* by orders of magnitude). Under sensor corruption, spatial boundaries blur and minority representations undergo differential collapse, leading to majority-class bleed.

---

## 2. The Core Mechanism: The Precision Paradigm & Prior Calibration
When applying standard cosine similarity in Hyperdimensional Computing (HDC) under extreme spatial corruption, feature distortion scatters points randomly across the hypersphere. Because majority classes dominate the scene geometry, an uncalibrated argmax assignment:
$$ \hat{y}_i = \arg\max_c \left( \kappa \cdot \mathbf{z}_i^\top \tilde{\mathbf{w}}_c \right) $$
drowns rare tail classes in hundreds of thousands of false-positive hallucinations (**The Precision Paradigm**). 

Instead of treating this as a representation failure, we formulate it as an exact **Bayesian Posterior Adjustment**. We shift the decision boundary using the source frequency prior $\pi$:
$$ \mathcal{L}_{i, c} = \kappa \cdot \mathbf{z}_i^\top \tilde{\mathbf{w}}_c - \tau \log \pi_c $$

At $\tau = -1.0$, this explicitly penalizes majority-class assignments and structurally protects tail classes. **This pure logit-adjustment yields a massive +6.8 mIoU improvement on a strictly frozen model with zero gradients.** The prior shift immediately and reliably corrects the precision collapse induced by corruption.

---

## 3. Geometric Adaptation vs. Headroom Exhaustion
Traditionally, TTA assumes the target representation should be updated via gradient-based geometric momentum (moving prototype centroids toward incoming pseudo-labels). However, rigorous evaluation across the SemanticKITTI-C panel revealed a fundamental limitation: **Headroom Exhaustion**.

The empirical gain from geometric adaptation follows a strict negative correlation against the model's initial frozen performance (`r = -0.503`):
$$ \text{gain} = +2.88 - 0.0972 \times \text{frozen\_mIoU} $$

For corruptions where the initial frozen mIoU is above ~30, there is virtually no headroom left that geometric pseudo-labels can fix. The errors in those regimes consist of fundamentally degraded structures and ambiguous boundaries where pseudo-label confidence is uninformative or actively harmful. The geometric adaptation mechanism is capped by the ceiling of what is correctable without ground truth.

*(For detailed ablations of the geometric mechanisms—Bayesian Momentum, IC4, and Asymmetric Dual Gating—see `docs/geometric_method/method_details.md`).*

---

## 4. The Path Forward: Online Prior Estimation (OPE)
Given that explicit inter-class calibration ($\tau = -1$ shift) is the only lever with substantial, robust magnitude (+6.8 mIoU), the most effective test-time adaptation strategy is to **adapt the prior rather than the prototype geometry**.

Domain shift in LiDAR corruption fundamentally manifests as a shift in the *effective class prior* of the visible scene (e.g., dense snow occludes distant background structures, inflating the relative frequency of foreground objects).

Our unified framework achieves TTA by estimating the **target prior** online:
1. **Static Baseline:** Shift logits using the pre-computed source prior.
2. **Expectation-Maximization (EM):** Iteratively re-estimate the target prior $\hat{\pi}^{(t)}$ over a sliding window of softmax predictions without requiring gradients.
3. **Black Box Shift Estimation (BBSE):** De-bias raw predicted label distributions using the source confusion matrix to infer true scene frequencies.

By shifting the locus of adaptation from high-dimensional prototype geometry to 1D prior estimation, the model rapidly aligns its Bayesian decision boundaries to the corrupted target distribution, avoiding confirmation bias entirely.

*(For full implementation plans regarding the EM and BBSE mechanisms, see `docs/pivot_details.md`).*
