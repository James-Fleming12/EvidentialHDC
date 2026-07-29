# Fourth Iteration: Falsification of the Geometric TTA Headroom Hypothesis

**Date:** July 2026
**Focus:** Evaluating the Headroom Exhaustion Hypothesis for Geometric Test-Time Adaptation (TTA).

## 1. Background Information

### 1.1 Overview
Throughout previous iterations, the unified geometric adaptation method (combining Bayesian Momentum, IC4, and Asymmetric Dual Gating) appeared to flatten out under heavy corruption (Severity 3). The prevailing hypothesis was **Headroom Exhaustion**: that on severely degraded point clouds, the residual errors are concentrated in fundamentally destroyed structures where no pseudo-labeling scheme could possibly succeed without ground truth. 

To formally test this, we executed a suite of "Tier Tests" to establish the absolute upper-bound ceilings (`oracle` gating) and to track adaptation performance across varying difficulty levels (Severities 1 through 3).

### 1.2 Finding 1: Correctable Headroom *Does* Exist
To determine if adaptation was theoretically possible, we ran an `oracle` gate test that admitted pseudo-labels if and only if they matched the ground truth, perfectly bounding the absolute ceiling of the gating mechanism.

**Severity 3 Results:**
* **Frozen Baseline:** 33.68 mIoU
* **Full Geometric Method:** 33.28 mIoU (Net **-0.40 mIoU**)
* **Oracle Gate:** 35.44 mIoU (Net **+1.76 mIoU**)

**Conclusion:** Headroom *does* exist. The model is capable of gaining nearly +2.0 mIoU through prototype updates alone. The stagnation of the `full_method` is not due to an intrinsic lack of correctable points, but rather the geometric gating mechanism failing to correctly identify and admit them without accumulating fatal false-positives.

### 1.3 Finding 2: Cross-Severity Collapse
If the headroom hypothesis were true, we would expect a *negative* correlation between frozen mIoU and adaptation gain (i.e., as the frozen model performs worse, adaptation has more room to help). To test this, we swept the method across Severities 1 (Light), 2 (Moderate), and 3 (Heavy).

**Cross-Severity Results:**
* **Severity 1:** Frozen 41.17 $\rightarrow$ Full 41.43 (Gain: **+0.27 mIoU**)
* **Severity 2:** Frozen 35.18 $\rightarrow$ Full 35.38 (Gain: **+0.20 mIoU**)
* **Severity 3:** Frozen 33.68 $\rightarrow$ Full 33.28 (Gain: **-0.40 mIoU**)

**Regression Fit:** 
$$ \text{gain} = -2.42 + 0.0667 \times \text{frozen\_mIoU} $$

**Conclusion:** The correlation is actually **positive**. As the data gets harder (lower frozen mIoU), the geometric adaptation performs strictly worse. The method collapses entirely once the frozen performance drops below the crossover boundary of **36.3 mIoU**. The negative correlation previously observed *within* Severity 3 was an artifact of specific corruptions.

### 1.4 Strategic Takeaway
The hypothesis that the geometric method failed due to a lack of headroom is decisively falsified. The mechanism doesn't flatten because it runs out of correctable errors; it flattens because **geometric prototype adaptation itself is fundamentally too fragile to withstand heavy (Severity 3+) corruption.** 

At high severities, feature distortion causes spatial gating (both epistemic and geometric) to break down, allowing confirmation bias to poison the prototypes. Future iterations must pivot away from geometry-based pseudo-labeling in favor of more robust alignment strategies.

---

## 2. Diagnostic Tests
*This section tracks active investigations and diagnostic tests to uncover the root causes of the geometric method's collapse and inform the next strategic pivot.*

### 2.1 Planned Diagnostics
- **Prototype Statistics Under Corruption:** Analyze the statistics (norms, intra-class variance, angular drift) of the learned HDC prototypes for each specific corruption condition to observe exactly how spatial distortion breaks the geometry.
- **Prior vs. TTA Mechanisms:** Compile stats to deeply analyze *why* the fixed prior ($\tau = -1$) works exponentially better than active Test-Time Adaptation, and evaluate if applying active TTA actually breaks or overrides the benefits of the prior.
- **Component Interference:** Check if each piece (Bayesian Momentum, IC4, Dual Gating, Temporal Consistency) works well in isolation, but when combined, they begin to impede each other and artificially constrain the model into a near-frozen state.
- **Problem Setting Validation:** Check for an incorrect problem setting—evaluate whether testing on Severity 3 exclusively masked structural issues, or if the entire dataset flow (such as SemanticKITTI-C applying corruptions to unchanged label distributions) is fundamentally flawed for the intended TTA task.
