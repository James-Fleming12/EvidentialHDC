# Method Pivot: Online Prior Estimation for TTA

**Location:** `EvidentialHDC/docs/pivot_details.md`
**Status:** In Development

## 1. The Headroom Hypothesis & Pivot Justification
Detailed analysis of Test-Time Adaptation (TTA) using the geometric prototype mechanism (Bayesian Momentum, IC4, Dual Gating) reveals a fundamental bottleneck: **headroom exhaustion**.

When evaluating the frozen (unadapted) model's performance against its potential TTA gain, a clear negative correlation emerges (`r = -0.503`):
* `gain = +2.88 - 0.0972 × frozen_mIoU`

The crossover point occurs at `29.6` mIoU. For corruptions where the initial frozen mIoU is above 30 (e.g., `wet_ground` at 42.02, `snow` at 50.22), there is virtually no headroom left that pseudo-labels can fix, because the residual errors are concentrated in hard boundaries or fundamentally degraded structures. The mean frozen mIoU across the SemanticKITTI-C panel at severity 3 is ~33.68, sitting squarely in the zero-gain regime.

Conversely, explicit inter-class calibration (the $\tau = -1$ prior shift) reliably contributes **+6.8 mIoU** on a strictly frozen model with zero gradients, operating entirely as a logit adjustment.

**Strategic Conclusion:** If the prior ($\pi$) is the only lever with substantial magnitude, then the TTA mechanism should adapt **the prior**, rather than the prototype geometry. Domain shift in LiDAR corruption is primarily a shift in the *effective class prior*. Estimating it online can recover the gap without gradients and without the risk of confirmation bias.

---

## 2. Testing the Ceiling (Tier 0 & Tier 1)
Before fully committing to the pivot, the headroom exhaustion hypothesis must be falsified or confirmed.

1. **Tier 0: Oracle Ceiling**
   * Execute `gate_mode='oracle'` to bound the absolute maximum capability of any gradient-based geometric gating. If the oracle mIoU is barely higher than the frozen mIoU (e.g., on `wet_ground`), it confirms that no adaptation scheme could have succeeded.
2. **Tier 1: Severity Sweep**
   * Execute the full method at severities 4 and 5. As the initial frozen mIoU collapses below the 29.6 crossover point (estimated 26.0 for sev 4 and 19.8 for sev 5), the geometric TTA method should flip from a net-loss to a net-gain. If the sign flips, headroom exhaustion is formally verified as the limiting factor.

---

## 3. Tier 5: Online Prior Estimation (The New Method)
Instead of relying on fixed source class frequencies, we replace `source_class_freq` with an online estimate of the target prior $\hat{\pi}^{(t)}$ updated incrementally.

### 3.1 T5a: Chunk-GT-Prior Oracle
First, establish the upper bound for online prior adaptation by passing the true ground-truth class frequencies for each specific temporal chunk to the model. If using the exact target distribution only improves performance marginally over the source prior, prior adaptation is not worth pursuing.

### 3.2 T5b: Expectation-Maximization (EM)
Implement the Saerens–Latinne–Decaestecker (2002) EM procedure to iteratively re-estimate the prior over a sliding window of softmax predictions.
* Initialize $\hat{\pi}_c^{(0)}$ using the source prior.
* E-step: Compute calibrated posteriors $\hat{P}(y=c|x, \hat{\pi}^{(k)})$ for all $x$ in the current window.
* M-step: Update the prior $\hat{\pi}_c^{(k+1)} = \frac{1}{N} \sum \hat{P}(y=c|x, \hat{\pi}^{(k)})$.
* *Advantage:* Highly stable, zero gradients, cheap.

### 3.3 T5c: Black Box Shift Estimation (BBSE)
De-bias the raw predicted label distribution using the source confusion matrix $C_{y,\hat{y}}$:
* $\hat{\pi}_{target} = C^{-1} \cdot \mu_{\hat{y}}$
* Where $\mu_{\hat{y}}$ is the empirical distribution of predicted labels in the target window.

### 3.4 T5d: Sliding Window Calibration
Sweep the temporal window length (e.g., $N=10, 50, 100, 500$ frames) to optimize the trade-off between rapid responsiveness to local distribution shifts and stability against transient anomalies.

---

## 4. Implementation Steps
1. **Tier 0/1 Execution:** Complete and log `run_tier_tests.sh`.
2. **Oracle Baseline (T5a):** Plumb the ground-truth chunk label distribution into `unsup_kitti-c.py` as an explicit override for `source_class_freq`.
3. **EM Integration (T5b):** Write an EM accumulator in `modules/HDC_utils.py` that intercepts the logit scaling pass, replacing the static $\tau \log \pi$ with a dynamically updated $\tau \log \hat{\pi}^{(t)}$.
