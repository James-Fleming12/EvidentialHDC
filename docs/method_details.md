# Method Details: Evidential HDC for Test-Time Adaptation
**Location:** `EvidentialHDC/docs/method_details.md`
**Last Updated:** July 24, 2026

This document formalizes the Evidential Hyperdimensional Computing (HDC) adaptation framework into four core mathematical pillars. For each pillar, we detail the theoretical formulation, the empirical test results, and the analysis of rejected alternative methods.

---

## 2. The Unified Architecture: End-to-End Adaptation Pipeline

While the subsequent sections detail the mathematical derivations and ablations of each component, the unified Evidential HDC pipeline integrates these mechanisms into a single, cohesive forward and backward pass during online adaptation:

1. **Latent Projection**: A point $x$ is embedded into the $D$-dimensional HDC hypersphere: $z = \frac{f_\theta(x)}{||f_\theta(x)||_2}$.
2. **Inter-Class Calibration (The Boundary Shift)**: The raw cosine similarities against the prototype matrix are adjusted using the source frequency prior $\pi$ ($\tau$-shift) to instantaneously suppress false-positive hallucinations caused by severe semantic corruption.
3. **Network Uncertainty (The Epistemic Veto)**: The unadjusted similarities are passed through a scaled Softplus activation to estimate Dirichlet Epistemic Density. Points exceeding the uncertainty threshold ($u > 0.5$) are flagged as out-of-distribution noise (e.g., fog scatter) and vetoed from updating the prototypes.
4. **Intra-Class Epistemic Scaling (IC4)**: For the points that *survive* the veto, their epistemic uncertainty is used as an active-learning multiplier. Highly ambiguous (but valid) points trigger larger gradient step sizes, explicitly performing Hard-Example Mining to rapidly stretch the prototype geometry toward the target domain.
5. **Temporal Consistency (Bayesian Momentum)**: The computed gradient step is applied to the *unnormalized* class prototypes. The growing magnitude of the prototypes provides intrinsic geometric inertia, naturally decaying the angular learning rate to protect majority classes from confirmation bias while allowing rare classes to remain agile.

---

## 3. Measuring Uncertainty

### 3.1 The Winning Method: Epistemic (Dirichlet) Density
To prevent out-of-distribution (OOD) unstructured noise (e.g., fog scatter) from permanently degrading the prototype geometries, we require a mathematically rigorous gate to assess point validity. We map raw HDC cosine similarities into Dirichlet Evidence via a source-anchored scaled Softplus activation:
$$ e_c = \text{Softplus}(\gamma \cdot (S(z, \tilde{w}_c) - \mu_c) / \sigma_c) $$
where $\mu_c, \sigma_c$ are the geometric statistics of the class on the clean source domain. The total evidence $E = \sum (e_c + 1)$ yields the **Epistemic Uncertainty** $u = \frac{C}{E}$. 
Points with $u > 0.5$ trigger the **Epistemic Veto**, which scales down or zeros out the gradients for that instance.

**Test Results:** In isolation, Epistemic Density proved to be an exceptionally strong universal gate. It yielded significant improvements across structured corruptions (e.g., **+8.53% mIoU on Snow** over the frozen baseline) while uniquely providing robust protection against chaotic noise (+0.99% on Fog). By actively measuring Epistemic Density rather than just entropy, Evidential HDC completely shatters standard SOTA architectures (like D3CTTA) which suffer from mode collapse via confirmation bias.

### 3.2 Rejected Method: HDC Latent Geometric Density (Free Energy)
HDC provides a fixed $D$-dimensional hypersphere where semantic relationships are physically encoded by angular distances. Geometric uncertainty attempts to measure the dispersion of a point within this space using Free Energy:
$$ F(z) = -T \log \sum_{c=1}^C \exp\left(\frac{S(z, \tilde{w}_c)}{T}\right) $$
**Why it Failed (The Ensemble Paradox):** We originally hypothesized that ensembling orthogonal uncertainty metrics (Network Epistemic + HDC Geometric/Free Energy) into a single logical AND gate would yield the ultimate robust filter. However, over-gating triggered severe **Representation Shrinkage**. The intersection of the two strict filters dropped the gradient admission rate from ~70% to ~2%. Because the model vetoed all diverse, edge-case, and heavily deformed examples, the prototypes shrank into hyper-dense, trivial geometric cores, severely starving the adaptation loops and degrading performance on complex corruptions. A single, well-calibrated relative metric (Epistemic Density) proved mathematically superior.

---

## 4. Inter/Intra Class Balance

### 4.1 The Method: Explicit Calibrated Prior & IC4
**Inter-Class Calibration:**
Under severe corruption, uncalibrated argmax operations scatter pseudo-labels randomly, drowning rare classes in hundreds of thousands of false positive hallucinations (The Precision Paradigm). We solve this by directly injecting the source frequency prior $\pi$ into the decision boundary:
$$ L_c = \kappa \cdot S(z, \tilde{w}_c) - \tau \log(\pi_c) $$
**Intra-Class Epistemic Scaling (IC4):**
Once false positives are suppressed by the $\tau$ boundary shift, we apply an active-learning "Hard-Example Miner" by dynamically scaling the gradient step size $\eta$ using the network's epistemic uncertainty: $\eta_{dynamic} = \eta_0 \cdot u$. This forces the model to take larger steps toward the most ambiguous, deformed points that survived the veto, rapidly stretching the prototype to encompass the target domain geometry.

### 4.2 Test Results
* **Inter-Class Calibration ($\tau$ prior):** The 2D Inter-Class sweep identified a golden ratio of $\tau/\kappa \approx -0.06$ (e.g., $\tau=-1.0, \kappa=15.0$). Because the argmax is scale-invariant, this ratio perfectly penalizes the tail to eliminate false positives without destroying true positive recall. This single mathematical shift launched Snow Tail IoU from **0.05 up to 0.26** without any adaptation.
* **Intra-Class Epistemic Scaling (IC4):** Ablations in the Phase 2 3-Chunk protocol confirmed that IC4 acts as an essential **Step Dilution Guard**. By scaling step sizes by epistemic uncertainty, IC4 reduced the average update magnitude (`UpdateMag`) from `0.0062` down to `0.0046` (~26% step dilution). This protective dilution prevented over-adaptation to noisy pseudo-labels and improved Final Frozen mIoU from `0.4135` to `0.4142`.

### 4.3 Rejected Methods
* **Supervised Long-Tail Intuition:** Standard long-tail logic attempts to *boost* tail logits to solve a recall problem. In TTA semantic corruption, the problem is a *precision* failure. Boosting the tail geometrically amplifies the hallucinations, zeroing out performance.
* **IC1 (Hard Rotation Budget):** Attempted to restrict updates to a hard $5^\circ$ angular cap per chunk. It failed to change performance because Bayesian Momentum (see Section 5) inherently constrained rotations to $<4.5^\circ$ organically.
* **XC2 (Geometric Sub-clustering):** Evaluated on both uncalibrated and calibrated data, XC2 proved to be a complete dud. While an early single-seed run appeared promising, multi-seed evaluation confirmed it performs no better than (and slightly below) baseline variance. Under a precision failure, equal-weight-per-subcluster is mathematically the wrong operator because it grants diffuse false-positive noise clouds the same influence as real object cores, corrupting the gradients.

---

## 5. Temporal Consistency

### 5.1 The Method: Bayesian Momentum
Rather than explicitly managing moving averages, we leave the prototype matrix $W$ unnormalized during continuous gradient accumulation:
$$ w_c^{(t+1)} = w_c^{(t)} + \eta \cdot \Delta w $$
As a class is updated frequently, its vector magnitude $||w_c||$ grows. Because the gradient step $\Delta w$ has a fixed magnitude, vector addition against a massive $w_c$ results in an increasingly smaller angular rotation. This provides an intrinsic, per-class geometric inertia (Learning Rate Decay). 

### 5.2 Test Results
We ablated standard unnormalized TTA (Bayesian Momentum) against Normalized TTA (where prototypes are continually reset to length 1.0). The Bayesian Momentum model gained **+2.58% mIoU** on Wet Ground, while the Normalized model gained only **+0.67%**. Without the geometric inertia of the growing weights, the uncalibrated model swung wildly into hallucinations and failed to adapt.

### 5.3 Rejected Methods
* **Momentum Veto (Temporal Uncertainty):** Attempted to gate points based on frame-to-frame trajectory consistency. While highly effective at filtering chaotic noise (like fog), its scale-invariance made it vulnerable to confirmation bias in structured corruptions, actively degrading performance on Wet Ground and Motion Blur.
* **Global LR Schedules:** Attempting to decay the learning rate via explicit global schedules (e.g., $1/t$ or Cosine Annealing) failed catastrophically. A global schedule decays all classes equally, meaning tail classes (which may not appear until halfway through a sequence) encounter a near-zero learning rate and remain permanently frozen. Bayesian Momentum solves this inherently because it is a *per-class* geometric decay.

---

## 6. Multi-View Consensus (Test-Time Augmentation)

To prevent over-rotation and stabilize semantic decision boundaries under heavy corruption, the adaptation pipeline subjects the input 360-degree LiDAR range image $X \in \mathbb{R}^{5 \times H \times W}$ to $M=3$ spatial transformations:
1. $X_{base}$: The original projection.
2. $X_{yaw}$: A yaw-shifted projection, computed by rolling the panorama horizontally (tested across 11°, 22°, and 90° shifts).
3. $X_{scale}$: A depth-scaled projection ($X_{base} \times 0.95$).

These views are passed through the encoder $f_\theta$ and aggregated to enforce spatial and topological consistency.

### 6.1 The Winning Methods: Feature Bundling and Prediction-Path Consensus
We identified two distinct, highly effective consensus mechanisms depending on the aggregation layer:

* **Feature-Space Latent Bundling (`bundle`)**: Averaging the latent encodings across all views before normalizing and passing to the classification head:
  $$ \bar{Z} = \frac{Z_{base} + Z_{yaw} + Z_{scale}}{\|Z_{base} + Z_{yaw} + Z_{scale}\|_2} $$
  *Result:* In Phase 2 diagnostics on Snow-3 ($\tau=0.0$), latent bundling consistently outperformed the unaugmented baseline on both initial zero-shot mIoU (`0.4100` vs. `0.4078`) and final frozen mIoU (`0.4159` vs. `0.4135`). Crucially, sweeping rotation angles revealed that **larger yaw rotations provide superior view diversity**: 90° yaw bundling achieved the highest mIoU (+0.24% over baseline) and raised the update firing rate from 48.1% to 52.2%, proving that averaging diverse, uncorrelated spatial views effectively cancels out corruption artifacts in latent space.

* **Prediction-Path Probability Consensus (`conf_pred`)**: Averaging softmax probabilities across views before taking the argmax prediction:
  $$ P_{consensus} = \frac{1}{M} \sum_{m=1}^M \text{Softmax}(L_m) $$
  *Result:* Probability confidence averaging (`conf_pred`) consistently surpasses discrete majority voting (`vote_pred`). Under calibrated inference ($\tau=-1.0$), `conf_pred` boosted overall mIoU by **+0.27%** (`0.5492` vs. `0.5465`) and lifted Tail mIoU by **+0.42%** over baseline (`0.4140` vs. `0.4098`), demonstrating powerful synergy between multi-view probability consensus and prior-calibrated boundaries.

### 6.2 The MV-2 Hypothesis: View Disagreement as an Unsupervised Precision Filter
To test whether view disagreement between spatial augmentations can serve as an unsupervised signal to identify false-positive pseudo-labels, we tracked the empirical precision of agreeing vs. disagreeing predictions across the 3 views.
* **Global Precision (All Classes, $\tau=-1.0$):** When the 3 views agree, precision is **87.8% – 89.3%** (~525M points). When views disagree, precision drops to **38.1% – 43.0%** (~8M–12M points).
* **Tail Class Precision (Class 10: Truck, $\tau=-1.0$):** Agreeing points achieve **70.7% – 75.4% precision**, whereas disagreeing points plunge to **13.1% – 22.3% precision** (representing ~22k to 35k false positives).

*Takeaway:* When the three views disagree on tail classes, **78% to 87% of those pseudo-labels are False Positives**. This empirically validates view disagreement as an exceptionally strong unsupervised precision filter that can be used to veto or dampen noisy gradient updates during online adaptation.

### 6.3 Rejected Methods
* **Uncertainty Scaling Veto (`min_uncert`, `mean_uncert`):** Scaling adaptation step sizes by taking the minimum or mean Dirichlet uncertainty across views successfully reduced update firing rates (from 47.1% to 43.6%), proving the consensus veto worked mechanically. However, it provided zero structural improvement to final mIoU because the samples it vetoed were already low-impact, leaving centroid adjustments identical to baseline.
* **Discrete Majority Voting (`vote_pred`):** Taking a discrete majority vote per point across views performed worse than probability confidence averaging (`conf_pred`) because argmax voting discards relative confidence distributions and struggles with 3-way tie resolution in ambiguous boundary regions.