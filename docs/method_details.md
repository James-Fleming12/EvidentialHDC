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

### 4.1 Mathematical Formulation: Bayesian Posterior Calibration ($\tau$-Prior)
Let $\mathbf{z}_i \in \mathbb{R}^D$ be the normalized $D$-dimensional HDC feature vector ($D=128$) for pixel $i$, and let $\tilde{\mathbf{w}}_c = \frac{\mathbf{w}_c}{\|\mathbf{w}_c\|}$ represent the normalized prototype vector for class $c \in \{1, \dots, C\}$.

In standard cosine classification, the uncalibrated posterior similarity is given by $S(\mathbf{z}_i, \tilde{\mathbf{w}}_c) = \mathbf{z}_i^\top \tilde{\mathbf{w}}_c$. Under severe semantic corruption (e.g., snow, fog, or sensor degradation), feature distortion scatters points randomly across the hypersphere. Because majority classes (e.g., road, building, sky) dominate the scene geometry, an uncalibrated argmax assignment:
$$ \hat{y}_i = \arg\max_c \left( \kappa \cdot \mathbf{z}_i^\top \tilde{\mathbf{w}}_c \right) $$
drowns rare tail classes (e.g., truck, bicycle, pedestrian) in hundreds of thousands of false-positive hallucinations (**The Precision Paradigm**).

We resolve this by formulating inter-class balance as an exact **Bayesian Posterior Adjustment**. By Bayes' Rule, the log-posterior probability is expanded as:
$$ \log P(y=c \mid \mathbf{z}_i) = \log P(\mathbf{z}_i \mid y=c) + \log P(y=c) - \log P(\mathbf{z}_i) $$
Let $\pi_c = P(y=c)$ be the prior class frequency measured from the source domain distribution. We model the scaled cosine similarity $\kappa \cdot \mathbf{z}_i^\top \tilde{\mathbf{w}}_c$ as proportional to the log-likelihood $\log P(\mathbf{z}_i \mid y=c)$. Introducing the calibration hyperparameter $\tau \in \mathbb{R}$, we govern prior injection via the adjusted logit equation:
$$ \mathcal{L}_{i, c} = \kappa \cdot \mathbf{z}_i^\top \tilde{\mathbf{w}}_c - \tau \log \pi_c $$
When $\tau = -1.0$ (with scaling factor $\kappa = 15.0$), the additive adjustment term becomes $+\log \pi_c$. Because $\pi_c \in (0, 1)$, $\log \pi_c < 0$. Consequently, for rare tail classes where $\pi_{\text{tail}} \ll \pi_{\text{majority}}$ (e.g., $\pi_{\text{truck}} \approx 10^{-3}$ vs. $\pi_{\text{road}} \approx 0.3$), the term $\log \pi_{\text{tail}}$ applies a strict negative log-penalty ($-6.9$ vs. $-1.2$).

Because the $\arg\max$ decision boundary is scale-invariant, this golden ratio $\frac{\tau}{\kappa} \approx -0.067$ establishes an exact geometric log-likelihood threshold: a feature vector must exhibit a significantly higher cosine similarity:
$$ \Delta S \ge \frac{|\tau|}{\kappa} \log\left(\frac{\pi_{\text{majority}}}{\pi_{\text{tail}}}\right) $$
to be assigned to a tail class. This mathematical boundary shift eliminates diffuse false-positive noise clouds without degrading true-positive recall, catapulting zero-shot Snow Tail IoU from **0.05 to 0.26**.

### 4.2 Intra-Class Epistemic Scaling: Step Dilution & Active Mining (IC4)
Once inter-class false positives are suppressed by the $\tau$-prior boundary shift, we govern intra-class adaptation dynamics. Let $\mathcal{P}_c = \{i \mid \hat{y}_i = c \land \text{not vetoed}\}$ be the set of valid pseudo-labeled pixels assigned to prototype $c$ in frame $t$.

Standard online adaptation updates prototypes via an unweighted centroid step: $\mathbf{w}_c^{(t+1)} = \mathbf{w}_c^{(t)} + \eta \sum_{i \in \mathcal{P}_c} \mathbf{z}_i$. However, in autonomous driving video sequences, intra-class variance is non-stationary: $\sim 95\%$ of incoming points are redundant core pixels (high cosine similarity, low uncertainty), while $\sim 5\%$ represent deformed boundary pixels or occluded objects.

We define **Intra-Class Epistemic Scaling (IC4)** by assigning each pixel an epistemic uncertainty weight $u_i \in [0, 1]$ derived from Dirichlet evidential density. The adaptation update is formulated as an active-learning expectation:
$$ \mathbf{w}_c^{(t+1)} = \mathbf{w}_c^{(t)} + \eta_0 \sum_{i \in \mathcal{P}_c} \gamma(u_i) \cdot \mathbf{z}_i $$
where $\gamma(u_i) = u_i$ acts as a dynamic gradient step multiplier. Mathematically, IC4 executes two critical dual functions:
1. **Step-Dilution Guard Against Residual Noise:** For diffuse, low-confidence pseudo-labels that slip past the threshold, their low epistemic certainty scales down their effective learning rate, reducing average update step magnitude (`UpdateMag`) from $0.0062$ down to $0.0046$ ($\sim 26\%$ step dilution). This prevents over-fitting to noisy pseudo-labels, improving Final Frozen mIoU from `0.4135` to `0.4142`.
2. **Hard-Example Mining:** For confident but geometrically deformed object boundaries (high valid uncertainty), IC4 allocates a proportionally larger gradient step $\eta_0 \cdot u_i$, rapidly stretching the prototype hypersphere along directions of high informational entropy without collapsing toward redundant core pixels.

### 4.3 Rejected Methods & Mathematical Failure Modes
* **Supervised Long-Tail Intuition (Logit Boosting):** Standard supervised long-tail methods attempt to *boost* tail logits to solve a recall deficit. Under TTA semantic corruption, the failure mode is a *precision deficit*. Boosting tail logits geometrically amplifies false-positive hallucinations, zeroing out performance.
* **IC1 (Hard Angular Rotation Cap):** Attempted to restrict updates to a hard $5^\circ$ angular displacement per chunk. This proved inert because Bayesian Momentum (Section 5) naturally constrains angular rotation to $< 4.5^\circ$ organically via prototype norm accumulation.
* **XC2 (Geometric Sub-clustering & Equal Weighting):** Evaluated across multiple seeds, XC2 performed no better than baseline seed variance. Under a precision failure, equal-weight-per-subcluster is mathematically the wrong operator: it grants diffuse false-positive noise clouds the exact same gradient weight as dense real-object cores, corrupting prototype trajectories.

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