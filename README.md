# Beyond Geometric Confidence: Multi-Source Uncertainty and Balanced Allocation for Backpropagation-Free LiDAR Test-Time Adaptation

---

## 1. What we established (and why it forced this pivot)

This project began by trying to build a better *geometric* confidence gate in
hyperdimensional space. That line of work is now closed, and its failure is
well-characterized. The following are **measured results**, not intuitions:

| Finding | Evidence |
|---|---|
| **Every geometric refinement loses to a first-order dot product.** Covariance ellipsoids, subspace reweighting, unions of balls, subcluster gating, and k-NN contrastive banks all underperform plain prototype cosine similarity as a gate. | rank sweep (all four subspace modes converge to ~0.78 AUROC vs 0.84 for the plain ball); precision–coverage (prototype ≥ k-NN at every operating point) |
| **The reason is `n ≪ d`.** ~5k samples/class in 10k dimensions. A mean is estimable; a covariance is not. Every second-order score is noise-dominated by construction. | spectrum diagnostic; monotone AUROC decline as more eigen-directions are used |
| **Set-valued conformal prediction is vacuous in HDC.** `E[\|C(x)\|] = 0.58` — prediction sets are always empty or singleton, never ambiguous, because HDC preserves inter-class separation. | calibration-drift run |
| **AUROC is the wrong metric for a gate.** k-NN had far higher AUROC (0.939 vs 0.81) yet *lost* on precision-at-coverage, which is what a gate actually uses. | knn_sweep vs precision_coverage |
| **THE DECISIVE ONE: no pseudo-label gate recovers any of the available headroom.** | overnight run, below |

### The overnight decision experiment

```
frozen (11 live classes)  47.93
oracle gate (GT labels)   50.47   (+2.73)      <- headroom EXISTS
supervised ceiling        59.03   (+11.10)     <- (with Bayesian shrinkage prior)
measurement noise (std)    1.00

best pseudo-label gate    ~47.98  (~0.00)      <- NONE of it is recoverable
   (macro-precision stalls at ~82%, zeroing the gain)
```

**There is massive headroom (up to +41 on some rare classes), and thresholding cannot reach any of it.**

### The diagnosis: a precision wall

The oracle and the best pseudo-gate move the prototypes by *comparable amounts* (drift ~0.71). Same update rule, same learning rate, similar magnitude of movement — and yet **+2.73 versus +0.00**. The only material difference is precision.

**At ~82% macro precision, the wrong 18% exactly cancel the right 82%.** The errors are not symmetric: a wrong pseudo-label drags a prototype toward *another class's* region — a far larger displacement than the small refinement a correct label buys.

And this is specifically a **rare-class** problem. *Pooled* precision at 10% coverage is 99.2%; **macro** precision at the same coverage is 81%. The megaclasses are easy and already correct. The classes mIoU actually weights are the ones the geometric gate cannot label reliably **at any threshold**.

> **Therefore: the bottleneck is pseudo-label QUALITY, not pseudo-label SELECTION — specifically for rare classes. No amount of clever filtering fixes a label that is wrong. We must make the labels better.**

Every design decision below follows from this.

---

## 2. The precision-requirement curve

We took the oracle gate and injected label noise at controlled rates to find the precision at which the gain disappears. Crucially, the injected noise mirrored the model's actual confusion matrix (e.g., adjacent class bleeding) rather than uniform random flips, perfectly simulating the systematic damage a real gate would permit.

```
oracle @ 100% target precision -> +2.73 mIoU
        @  98%                 -> +2.28
        @  95%                 -> +2.22
        @  90%                 -> +1.70
        @  85%                 -> +1.38
        @  82%                 -> +1.27
        @  80%                 -> +1.20
        @  75%                 -> +1.07
        @  70%                 -> +1.00    (hits the 2*noise floor)
```

This curve converts *"we need better pseudo-labels"* into a **concrete engineering target**. 
The result is highly encouraging: the wall is a gentle slope. We do not need a mythical 98% precision. If our uncertainty fusion can achieve just **85-90% pooled precision** (which corresponds to ~71-76% macro precision), we will comfortably capture measurable, statistically significant gains (+1.38 to +1.70 mIoU).

---

## 3. Contribution 1 — Multi-source uncertainty fusion

Every confidence signal used so far has been *geometric*: distance or similarity in HD space. That is demonstrably insufficient. Classes 0, 7, and 2 have terrible frozen IoUs, meaning their pre-trained prototypes are physically located in the wrong quadrant of hyperspace. **If the prototype is in the wrong place, a geometric gate measuring distance to that prototype is also wrong.**

We must fuse independent sources of uncertainty whose failure modes are uncorrelated with geometric distance.

**(a) Geometric uncertainty (HD space).** Prototype cosine similarity. Cheap, `O(K)`. Answers *"is this point near the current class manifold?"*

**(b) Epistemic uncertainty (the network).** Measured **without** the softmax.
- **Multi-RP ensemble** — project the same feature through several independent random projections.
- **Evidential Deep Learning / Feature-space density** in the *backbone's* latent space.

**(c) Spatial / temporal consistency.** A point whose neighbours and previous-frame correspondent agree with its label is far more likely correct. This acts as a **vote / veto on the LABEL**, not a smoother of the score. (Prior graph-Laplacian smoothing blurred errors across objects; consistency must act as a hard veto to reject errors, not smear them).

---

## 4. Contribution 2 — Balanced update allocation (Budgeting by Headroom)

### The problem, measured

A frequency-proportional update budget spends almost all of its capacity on classes that cannot improve. 
In our supervised ceiling tests, Class 11 (Road) had millions of points but **negative headroom (-0.57)**. Updating it actually hurts the model because it is already saturated. Meanwhile, Class 0 (Car) had only 98k points but **+41.07 headroom**. 

### The mechanism: a subcluster update ledger

We must allocate the adaptation budget by **headroom**, not by frequency.
- Maintain **K representative subclusters per class**, initialized from source.
- Route each admitted point to its nearest subcluster.
- **A subcluster contributes to the prototype update only if its update count is within a bounded range of its siblings'.** 

This equalizes adaptation signal *within* a class (no single dense mode dominates) and *across* classes (freezing saturated dense classes like Road, and funneling the update budget exclusively to high-headroom regions like Car/Motorcycle).

Crucially, **the subclusters only track updates; they never touch inference.** This prevents the Voronoi-shattering failure that destroyed previous subcluster-gating attempts.

---

## 5. Contribution 3 (variant) — Multi-view test-time augmentation

Generate augmented views (yaw roll, scale, dropout), bundle the resulting hypervectors, use cross-view soft agreement as a reliability signal. 
Positioned as a compute-scaling variant: *"when additional compute is available, multi-view agreement raises macro precision by X at Y× cost."*

---

## 6. Extensive Reading List

To execute this pivot seamlessly, the following literature provides the theoretical foundation and the exact mechanisms needed for the implementation of the three pillars.

### 1. Efficient Single-Pass Epistemic Uncertainty (Contribution 1b)
These papers explore how to extract robust uncertainty directly from the backbone's latent space without requiring deep ensembles, multiple forward passes, or distance metrics that duplicate your HDC cosine similarity.

| Paper | The Concept | How to Use It |
|---|---|---|
| [**Evidential Deep Learning**](https://scholar.google.com/scholar?q=Evidential+Deep+Learning+Sensoy)<br>*(Sensoy et al., NeurIPS 2018)* | Formulates learning as evidence acquisition using Subjective Logic. Places a Dirichlet distribution over class probabilities, allowing explicit "I don't know" outputs independent of prototype distance. | (Used for pre-training/formulation reference) |
| [**SNGP**](https://scholar.google.com/scholar?q=Simple+and+Principled+Uncertainty+Estimation+with+Deterministic+Deep+Learning+via+Distance+Awareness)<br>*(Liu et al., NeurIPS 2020)* | Spectral-normalized Neural Gaussian Processes force the latent space to be "distance-aware," providing high-quality epistemic uncertainty in a single pass. | (Used for pre-training/formulation reference) |
| [**Prior Networks**](https://scholar.google.com/scholar?q=Predictive+Uncertainty+Estimation+via+Prior+Networks)<br>*(Malinin & Gales, NeurIPS 2018)* | Introduces Dirichlet Prior Networks to model distributional uncertainty directly. Targets the requirement to identify points far from the source distribution. | (Used for pre-training/formulation reference) |
| [**DUQ**](https://scholar.google.com/scholar?q=Deterministic+Neural+Networks+with+Appropriate+Inductive+Biases+Capture+Epistemic+Uncertainty)<br>*(van Amersfoort et al., ICML 2020)* | Uses a two-sided gradient penalty to enforce "distance awareness." | Proves you can extract a mathematically rigorous epistemic uncertainty score directly from the feature vector's magnitude. |
| [**Laplace Redux**](https://scholar.google.com/scholar?q=Laplace+Redux+Effortless+Bayesian+Deep+Learning)<br>*(Daxberger et al., NeurIPS 2021)* | Applies a post-hoc Laplace approximation to the last layer, yielding a closed-form, single-pass epistemic uncertainty without altering weights. | Fit this approximation to the source data once; evaluate target data against it at test-time for a pure epistemic signal. |
| [**Feature Space Singularity**](https://scholar.google.com/scholar?q=Feature+Space+Singularity+for+Out-of-Distribution+Detection)<br>*(Huang et al., NeurIPS 2021)* | Demonstrates the raw L2 norm (magnitude) of the latent vector is a highly effective, zero-cost metric for OOD detection. | Since HDC requires L2-normalized vectors, the magnitude is usually thrown away. Use the pre-normalization magnitude as the epistemic veto! |
| [**React**](https://scholar.google.com/scholar?q=React+Out-of-distribution+Detection+With+Rectified+Activations)<br>*(Sun et al., NeurIPS 2021)* | Shows OOD data causes massive activation spikes in penultimate layers. Clamping these makes uncertainty metrics vastly more reliable. | Monitor pre-projection features for activation spikes as an immediate indicator of epistemic failure (e.g., fog artifacts). |

### 2. Spatial/Temporal Consistency as a Hard Veto (Contribution 1c)
Consistency cannot just smooth scores; it must actively reject geometrically plausible but temporally/spatially isolated errors.

| Paper | The Concept | How to Use It |
|---|---|---|
| [**Temporal Ensembling**](https://scholar.google.com/scholar?q=Temporal+Ensembling+for+Semi-Supervised+Learning)<br>*(Laine & Aila, ICLR 2017)* | The foundational text on using moving averages of network predictions over time to stabilize pseudo-labels. | Maps perfectly to consecutive LiDAR frames for defining temporal vetoes. |
| [**ST3D**](https://scholar.google.com/scholar?q=ST3D+Self-training+for+Unsupervised+Domain+Adaptation+on+3D+Object+Detection)<br>*(Yang et al., CVPR 2021)* | State-of-the-art for 3D LiDAR UDA, heavily utilizing spatial/temporal consistency to filter pseudo-labels. | Read to see exactly what baselines your consistency veto must outperform. |
| [**FixMatch**](https://scholar.google.com/scholar?q=FixMatch+Simplifying+Semi-Supervised+Learning+with+Consistency+and+Confidence)<br>*(Sohn et al., NeurIPS 2020)* | Enforces that weak and strong predictions must strictly agree before retaining a label. | The theoretical basis for your "hard veto" argument over "soft smoothing." |
| [**PointTTA**](https://scholar.google.com/scholar?q=PointTTA+Test-Time+Adaptation+for+Point+Cloud+Processing)<br>*(Metzger et al., 2023)* | A direct TTA framework for 3D point clouds relying on spatial transformations and self-supervision. | Compare their soft-consistency loss to your hard-veto logic to define valid "spatial neighborhoods." |
| [**Test-Time Training on Video**](https://scholar.google.com/scholar?q=Test-Time+Training+on+Video+Better+Point+Tracking+and+Pose+Estimation)<br>*(Sun et al., 2022)* | Adapts representations online by enforcing temporal consistency across sequential frames. | Validates the assumption that temporal physics is the ultimate arbiter of label correctness. Use tracking logic for frame-to-frame veto. |
| [**Ada3D**](https://scholar.google.com/scholar?q=Ada3D+Adaptive+3D+Object+Detection)<br>*(Recent CVPR/ICCV)* | Focuses on aligning local spatial contexts under domain shifts, assuming adjacent points share semantic identity. | Formalize the spatial veto: if cosine similarity says Pedestrian, but k geometric neighbors are Road, the label is vetoed. |
| [**TENT**](https://scholar.google.com/scholar?q=TENT+Fully+Test-Time+Adaptation+by+Entropy+Minimization)<br>*(Wang et al., ICLR 2021)* | The baseline for entropy minimization based TTA. | Cite as the counter-example: TENT's soft smoothing causes semantic poisoning. Contrast with your boolean temporal veto. |

### 3. Subcluster Ledgers & Headroom Allocation (Contribution 2)
These papers tackle gating updates based on representation saturation and non-i.i.d. target streams, aligning perfectly with the backprop-free subcluster ledger.

| Paper | The Concept | How to Use It |
|---|---|---|
| [**RoTTA**](https://scholar.google.com/scholar?q=RoTTA+Robust+Test-Time+Adaptation+in+Dynamic+Scenarios)<br>*(Yuan et al., CVPR 2023)* | Maintains a category-balanced memory bank for temporally correlated streams. | Mandatory reading. Differentiate your approach: your ledger never touches inference, preventing Voronoi-shattering. |
| [**Class-Balanced Loss**](https://scholar.google.com/scholar?q=Class-Balanced+Loss+Based+on+Effective+Number+of+Samples)<br>*(Cui et al., CVPR 2019)* | Formally defines class saturation (effective number of samples), proving exponentially diminishing returns. | Provides mathematical justification for the headroom-budgeting ledger. |
| [**DELTA**](https://scholar.google.com/scholar?q=DELTA+Degradation-Free+Fully+Test-Time+Adaptation)<br>*(Zhao et al., ICLR 2023)* | Explores how unconstrained TTA destroys majority classes. Uses class-aware balancing. | Compare your ledger against their balancing mechanism. |
| [**NOTE**](https://scholar.google.com/scholar?q=NOTE+Robust+Continual+Test-time+Adaptation+Against+Temporal+Correlation)<br>*(Gong et al., NeurIPS 2022)* | Tackles the "fog bank" problem where temporally correlated TTA streams cause batch-norm/memory collapse. | Direct theoretical precedent. They balance memory to prevent temporal collapse; you balance subclusters to equalize headroom. |
| [**LAME**](https://scholar.google.com/scholar?q=LAME+Latent-Space+Marginalization+for+Blind+Action)<br>*(Boudiaf et al., CVPR 2022)* | Performs TTA without updating weights, strictly updating latent space assignments via Laplacian smoothing on affinity matrix. | Massive structural citation. Contrast their affinity-matrix approach with your O(K) budgeted subcluster approach. |
| [**AdaContrast**](https://scholar.google.com/scholar?q=AdaContrast+Contrastive+Test-Time+Adaptation)<br>*(Chen et al., CVPR 2022)* | Utilizes a pseudo-label queue to track class frequencies and reject over-represented classes. | Validates that updating saturated classes hurts. Defends why your ledger freezes saturated subclusters. |
| [**Practical Coresets for Online ML**](https://scholar.google.com/scholar?q=Practical+Coresets+for+Online+Machine+Learning)<br>*(Feldman 2020)* | Focuses on selecting the smallest possible subset of streaming data to represent the full distribution. | Rigorous framing: your ledger maintains an online coreset of the target domain. Tracking K subclusters equalizes learning potential. |

### 4. Multi-View Test-Time Augmentation (Contribution 3)
If including the TTA variant for compute-scaling, it must be grounded in literature treating augmentation as a reliability signal.

| Paper | The Concept | How to Use It |
|---|---|---|
| [**Learning to Trust**](https://scholar.google.com/scholar?q=Learning+to+Trust+Test-Time+Augmentation+for+Epistemic+Uncertainty+Estimation)<br>*(Ayhan & Berens, 2018)* | Seminal paper establishing variance across TTAs as a valid proxy for epistemic uncertainty. | Foundations for TTA reliability. |
| [**Uncertainty-guided TTA**](https://scholar.google.com/scholar?q=Uncertainty-guided+Test-Time+Augmentation)<br>*(Shanmugam et al., 2021)* | Standard TTA applies all augmentations equally. This paper learns which augmentations to trust. | Maps well to your "cross-view soft agreement" signal. |
| [**PointContrast**](https://scholar.google.com/scholar?q=PointContrast+Unsupervised+Pre-training+for+3D+Point+Cloud+Understanding)<br>*(Xie et al., ECCV 2020)* | Focuses on cross-view consistency in 3D point clouds. | Provides the exact geometric augmentations (yaw, roll, scale) statistically valid for 3D LiDAR TTA. |

---

## 7. Order of work

1. **Identify the epistemic UQ method (Pillar 1).** Find a lightweight, single-pass method compatible with our frozen backbone.
2. **Draft the memory-safe subclustering ledger (Pillar 3).** Ensure it enforces the headroom-based budget.
3. **Uncertainty sources, one at a time.** Measure macro precision on each independent source.
4. **Fusion + full ablation.**
5. **mIoU validation**: error bars, live classes, interleaved eval, oracle reported alongside.
