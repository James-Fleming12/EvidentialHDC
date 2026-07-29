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
*This section tracks active investigations to uncover the root causes of the geometric method's collapse and separates fundamental method failures from issues with the problem setting.*

### D0. The Prime Suspect: Label/Weight Mismatch
The immediate lead is a suspected wiring fault in the adaptation loop. Currently, the pseudo-label uses the $\tau$-corrected logit, but the confidence weight gating the update uses the raw cosine similarity:
```python
pseudo_labels = argmax( cos_sims*kappa - tau*log(pi) )   # tau-CORRECTED label
base_weights  = softmax(cos_sims * 100).max()            # tau-FREE weight
```
This means a majority-class point (e.g., road) that $\tau$ flips to a rare class carries a **high** weight (due to its raw cosine to road) while being bound to the rare prototype. The update then writes a road-shaped hypervector into the rare prototype at high weight, corrupting exactly the classes $\tau$ was protecting.
* **D0a (Measure Disagreement):** Per frame, log the fraction of points where `argmax(cos_sims) != pseudo_labels`, and their mean `base_weight`.
* **D0b (The Fix-Test):** Make the gate weight $\tau$-consistent by setting `base_weights = softmax(pl_logits).max()`. If performance substantially recovers (e.g., on `wet_ground`), this was the root cause.
* **D0c (The Confirming Picture):** Log per-class angular drift. Prediction: under current code, tail prototypes rotate the most and lose IoU. Under the D0b fix, tail rotation drops.

### D1. Prototype Statistics Per Condition
Compile the following existing metrics into per-corruption tables:
* **Norms:** `‖w_c‖` initial vs. final to check for accumulation blowout.
* **Angular Drift:** `∠(w_t^c, w_0^c)` to see how far prototypes moved (tail should not move more than head).
* **Step Dilution:** Mean `_update_magnitude_log`. Expect ~5e-3; if ~1e-4, the veto isn't firing.
* **Correlation:** Map drift° vs. $\Delta$IoU per class. If classes that moved most lost the most IoU, adaptation is actively harming.

### D2. Why the Prior Beats TTA
Compare the quantity both mechanisms try to improve: pseudo-label accuracy.
* **D2a:** On a frozen model, measure accuracy of `argmax(cos_sims)` vs. $\tau$-corrected labels.
* **D2b:** Measure $\tau$-corrected accuracy *after* adaptation. If it falls, TTA is degrading the labels $\tau$ fixed.
* **D2c:** Report three numbers: `frozen` $\rightarrow$ `frozen+τ` $\rightarrow$ `frozen+τ+TTA`. If adaptation consistently degrades performance, the paper's honest claim becomes: *"The gain is an inference-time prior correction; gradient adaptation is not required."*

### D3. Component Interference
* **D3a (Pairwise Interaction):** For pieces like $\tau$, gating, BM, and IC4, run pairs to compute interaction: `(A+B - base) - (A - base) - (B - base)`. High negative values mean pieces are fighting.
* **D3b (Churn Test):** Compare net prototype displacement `‖Σ Δw_c‖` vs total displacement `Σ ‖Δw_c‖`. If net $\ll$ total, updates are cancelling out, mimicking a frozen model while actually churning.

### D4. Problem Setting Validation
* **Severity Guard:** Confirm on-disk directories actually exist (already patched via `run_tier_tests.sh` Stage 0) to avoid silent fallbacks to easier severities.
* **Prior Drift ($L_1$):** Measure $L_1(\pi_{chunk\_GT}, \pi_{source})$. If $\approx 0$, KITTI-C preserves the source distribution, meaning the prior pivot is fundamentally dead due to the dataset setting.
* **Data Flow Invariants:**
  1. Clean validation frozen mIoU must match pretraining val mIoU.
  2. The same corruption via a 3-item vs 8-item list must match (chunk determinism).
  3. Verify the published baseline frozen mIoU (e.g., fog at 6.04 is suspiciously low; could be a broken loader).

### D5. Is it *Completely* Broken? (The Ceilings)
Run the Gate Oracle (`gate_mode='oracle'`) and Prior Oracle at matched `fire_th` to bound the problem:
* **Gate $\approx 0$, Prior $\approx 0$:** No headroom. Saturated. Move to a harder setting.
* **Gate Large, Prior $\approx 0$:** Gating is the lever. Prior pivot is dead. Fix D0.
* **Gate $\approx 0$, Prior Large:** Prior is the lever. Pivot to online prior estimation.
* **Gate Large, Prior Large:** Both help, but the method is underperforming its ceiling.

### Execution Order
1. **D0b Fix-Test & Logging** (Most likely to explain everything).
2. **D5 Ceilings** (Decides "broken" vs. "no headroom").
3. **D4 Invariants** (Cheap pipeline confirmation).
4. **D2 Comparison** (Formalizes the performance drop).
5. **D1 / D3 Tables** (Deep dives only if D0/D5 don't close the case).
