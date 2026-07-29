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

## 5. Post-Mortem: Tier Test Results (Hypotheses Falsified)
**Update (July 2026): Both the Headroom Hypothesis and the Prior Pivot have been decisively falsified by the Tier 0 and Tier 1 tests.**

### Falsification 1: The Prior Pivot is Dead
The `prior_oracle` test substituted the *true* ground-truth class prior of each temporal chunk in place of the static source prior. 
* **Result:** At severity 3, the baseline `frozen` model achieved **33.68 mIoU**. The `prior_oracle` model achieved exactly **33.68 mIoU** ($\Delta = +0.00$).
* **Conclusion:** SemanticKITTI-C applies synthetic corruptions to the exact same scenes without altering the underlying class distribution. Because $L_1(\pi_{chunk}, \pi_{source}) \approx 0$, there is no target prior drift to estimate. The massive +6.8 mIoU gain from $\tau = -1.0$ is a static calibration fix for the dataset's base imbalance, not an adaptation to a domain shift. Online prior estimation (EM/BBSE) cannot succeed because there is nothing to recover. 

### Falsification 2: The Headroom Hypothesis is Dead
We hypothesized that geometric adaptation flattens out due to a lack of correctable headroom on hard corruptions (expecting a negative correlation between frozen mIoU and adaptation gain). However, the GT-gated `oracle` test proved that headroom *does* exist, and the severity sweep proved the geometric mechanism simply fails to extract it.
* **Oracle Ceiling:** At Severity 3, the GT-gated `oracle` yielded a **+1.76 mIoU** gain (raising performance to 35.44 mIoU). However, the `full_method` suffered a **-0.40 mIoU** loss. Headroom exists, but the geometric gate fails to capture it.
* **Cross-Severity Fit:** Across severities 1, 2, and 3, the linear fit is actually **positive**: `gain = -2.42 + 0.0667 * frozen_mIoU`.
  * **Sev 1:** Frozen 41.17 $\rightarrow$ Full 41.43 (Gain: **+0.27**)
  * **Sev 2:** Frozen 35.18 $\rightarrow$ Full 35.38 (Gain: **+0.20**)
  * **Sev 3:** Frozen 33.68 $\rightarrow$ Full 33.28 (Gain: **-0.40**)
  * **Crossover Point:** The fit crosses zero at a frozen mIoU of **36.3**.
* **Conclusion:** Geometric adaptation actually performs *worse* as the data gets harder. The negative correlation previously observed within severity 3 alone was an artifact. The problem is not that the model lacks headroom to improve; the problem is that the geometric adaptation mechanism itself fundamentally fails under severe corruption.

### Strategic Reset
With the prior estimation pivot dead and the geometric gating mechanism proven fragile under severity, future directions must abandon prototype-level geometric pseudo-labeling in favor of either feature-level alignment (e.g., test-time batch normalization) or vastly more robust multi-modal consistency constraints.
