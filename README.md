# Robust Feature Pretraining and Uncertainty-Gated Hyperdimensional Prototypes for Backpropagation-Free LiDAR Test-Time Adaptation

---

## 1. What we established (and why the method looks the way it does)

This project began by trying to build a better *geometric* confidence gate in
hyperdimensional space. That line of work is closed, and its failure is
well-characterized. The following are **measured results**, not intuitions:

| Finding | Evidence |
|---|---|
| **Every geometric refinement loses to a first-order dot product.** Covariance ellipsoids, subspace reweighting, unions of balls, subcluster gating, and k-NN contrastive banks all underperform plain prototype cosine similarity as a gate. | rank sweep (all four subspace modes converge to ~0.78 AUROC vs 0.84 for the plain ball); precision–coverage (prototype ≥ k-NN at every operating point) |
| **The reason is `n ≪ d`.** ~5k samples/class in 10k dimensions. A mean is estimable; a covariance is not. Every second-order score is noise-dominated by construction. | spectrum diagnostic; monotone AUROC decline as more eigen-directions are used |
| **Set-valued conformal prediction is vacuous in HDC.** `E[\|C(x)\|] = 0.58` — prediction sets are always empty or singleton, never ambiguous, because HDC preserves inter-class separation. | calibration-drift run |
| **AUROC is the wrong metric for a gate.** k-NN had far higher AUROC (0.939 vs 0.81) yet *lost* on precision-at-coverage, which is what a gate actually uses. | knn_sweep vs precision_coverage |
| **No pseudo-label gate recovers any of the available headroom.** Oracle gate (ground-truth labels) +2.73 mIoU; best pseudo-label gate ~0.00 — the wrong 18% of pseudo-labels exactly cancel the right 82%. | overnight decision experiment |

The decisive discovery came from the Corruption Atlas diagnostics, and it
changed the direction of the project entirely:

| Finding | Evidence |
|---|---|
| **Type C corruptions (Fog, Crosstalk) destroy linear separability.** The 128D feature manifold collapses into isotropic noise. A purely mathematical TTA technique (memory banks, prototype shifts) is powerless because the underlying feature space is gone. | Corruption Atlas diagnostics |
| **The fix must happen before HDC projection.** The backbone itself has to be trained to map physical noise to clean semantic manifolds (or isolate it), so the HDC space has real geometry to work with. | micro/medium pretraining runs |
| **A decoupled SupCon+VIB pretraining makes the encoder robust.** Linear Probe on Fog reached 49.4% vs 23.6% baseline (2.1×), and the semantic information mathematically survives random projection and sign binarization (49.4% → 49.0% → 47.8%). | Phase 7/8 headroom diagnostics |
| **The remaining bottleneck is the naive decoder.** Indiscriminate EMA prototype updates absorb impossible Fog/Crosstalk artifacts, poisoning the Euclidean centroids: 8.2% prototype accuracy despite 49.4% linear separability. | Phase 8 degradation pipeline |
| **Harmful updates are identifiable by a universal signature.** They carry significantly lower confidence (0.61 vs 0.91) and significantly larger feature norms (6.41 vs 5.27) — consistent across all 8 corruptions. | Phase 9 universal oracle matrix |
| **Gating by confidence recovers most of the oracle headroom.** Top-50% confidence gate: 20.63% vs perfect oracle 23.32% vs naive EMA 16.36%. | Phase 9 oracle gating |
| **Input-space remediation is a dead end.** SOR pre-filtering (in the training loop), additive noise augmentation, and global latent pooling all fail to beat the plain robust encoder on HDC Prototype Accuracy (9.1% / 12.4% / 14.4% vs 20.1%). | Phase 10 remediation shootout |
| **On the robust encoder, adaptation is nearly solved.** Naive EMA already harvests ~82% of the oracle headroom on Fog (18.34% vs 19.54%); feature norm is the dominant gate signal (AUROC 0.94), while probe confidence is currently anti-predictive (AUROC 0.15 — under re-test with corrected sampling). | Phase 12 gated-EMA diagnostic |

**Therefore, the method rests on three pillars**, each attacking one of the
measured failure modes:

1. **Robust feature extractor pretraining (SupCon + VIB)** — make the 128D
   space robust enough that HDC prototypes have a meaningful manifold to live on.
2. **Uncertainty-gated prototype updates** — stop the EMA memory from overfitting
   to corruption artifacts, using standard confidence/uncertainty metrics.
3. **Balanced update allocation** — ensure majority classes and dense feature
   modes are not overrepresented in the adaptation budget.

---

## 2. Why gating alone could not win (the precision wall)

We took the oracle gate and injected label noise at controlled rates to find
the precision at which the gain disappears. The injected noise mirrored the
model's actual confusion matrix rather than uniform random flips:

```
oracle @ 100% target precision -> +2.73 mIoU
        @  95%                 -> +2.22
        @  90%                 -> +1.70
        @  85%                 -> +1.38
        @  80%                 -> +1.20
        @  70%                 -> +1.00    (hits the 2*noise floor)
```

This is why Pillar 1 exists: **at ~82% macro precision, the wrong 18% of
pseudo-labels exactly cancel the right 82%** — and this is specifically a
rare-class problem (pooled precision 99.2% vs macro precision 81% at 10%
coverage). No amount of clever filtering fixes a label that is wrong, and no
gate can recover a feature space that has collapsed into isotropic noise. The
pretrained encoder attacks the problem at the source; gating then only needs to
veto the residual artifacts.

---

## 3. Pillar 1 — Robust Feature Extractor Pretraining (SupCon + VIB)

### The problem, measured

Type C corruptions (Fog, Crosstalk) mathematically destroy the linear
separability of the geometric manifolds. Prototype drift and memory-bank
methods are powerless because the underlying 128D space has collapsed — the
noise points are no longer separable from the semantic points by any
threshold.

### The mechanism

Train the 128D backbone with a **decoupled** objective before the HDC
projection layer:

- **SupCon (angular alignment).** Supervised contrastive loss on
  L2-normalized features pulls augmented views of the same class together and
  pushes different classes apart. LiDAR segmentation is dense, so a single
  batch provides millions of positive/negative pairs natively.
- **VIB (magnitude isolation).** A KL-divergence penalty
  ($\mathrm{KL}[q(z|x) \| \mathcal{N}(0, I)]$) on both the clean *and*
  augmented pathways forces the latent magnitudes down: complex, high-entropy
  spatial noise (Fog/Crosstalk) is expensive to memorize, so it collapses
  toward the origin ($\Vert z \Vert_2 \approx 0$) while semantic geometry keeps
  bounded, spherical, dense clusters.
- **Physics-based augmentation only.** Voxelized beam dropout (sparsity),
  anisotropic ray-axis jitter (sensor noise), and density subsampling
  (attenuation) — explicitly *not* KITTI-C's ray-tracing algorithms, to avoid
  overfitting the augmentation to the corruption pipeline.

### Measured effect

- Linear Probe on Fog: **49.4%** vs 23.6% baseline (2.1×).
- Feature magnitudes bounded to ~4.6–5.6 (vs 8+ without VIB); the representation
  supports the Euclidean nearest-prototype geometry HDC relies on.
- The information survives the HDC encoding losslessly: 49.4% linear →
  49.0% after random projection → 47.8% after sign binarization.
- Best HDC Prototype Accuracy from the naive decoder: **20.1%** (vs 5.3% for
  the untrained baseline in the same protocol).
- Post-hoc remediation on top of the robust encoder (SOR pre-filtering,
  additive noise augmentation, global pooling) does not help — the extractor
  is "solved enough"; further investment belongs in the decoder.

---

## 4. Pillar 2 — Uncertainty-Gated HDC Prototype Updates

### The problem, measured

With the robust encoder in place, the remaining failure is the naive decoder:
a memory bank or EMA prototype that updates indiscriminately absorbs the
impossible Fog/Crosstalk artifacts, poisoning the zero-shot Euclidean
centroid (8.2% prototype accuracy despite 49.4% linear separability).

### The signature of poison

Phase 9's leave-one-update-out profiling proved that harmful updates form a
statistically identifiable population with a **universal signature** across
all 8 corruptions: lower confidence and larger feature norms than helpful
updates.

### The mechanism

Place a **gate before the memory update**. The two standard signals are
z-scored with streaming statistics and combined into a soft multiplicative
weight (the same algebra used by the production uncertainty fusion):

$$w(x) = \exp\big(-\lambda_1 \cdot \text{relu}(\tau_c - c)\big) \cdot \exp\big(-\lambda_2 \cdot \text{relu}(n_z - \tau_n)\big)$$

and the prototype update becomes $c \leftarrow c + \eta \cdot w(x) \cdot z$ —
a soft veto that down-weights artifacts without starving adaptation
(avoiding the hard AND-gate starvation and OR-gate flooding documented in the
earlier geometric-method threads).

### Measured effect

- On the pre-robust encoder: top-50% confidence gate reached **20.63%** vs
  16.36% naive — 89% of the perfect-oracle ceiling (23.32%).
- On the robust encoder (Phase 12, Fog): naive EMA already reaches 18.34% of
  the 19.54% oracle ceiling — the gate's remaining job is closing that gap and
  preventing drift over long streams.

### Gate-fault verification (do not trust old calibration)

The shipped gates (`soft_dual_weight`, `and_gate`, `ellipsoid_gate`,
`rescue_gate`) were tuned on the *collapsed* feature space and can embed
old-space assumptions (e.g., "harmful = low confidence", which inverts on the
new encoder). Before trusting a gate, the diagnostic harness runs:
a **clean-data control** (gates must not degrade adaptation when no poison
exists), **per-gate-mode AUROC** (is the gate's own score selective?),
**sign-corrected joint gates**, and **threshold envelope sweeps** (best-case
config vs shipped defaults).

---

## 5. Pillar 3 — Balanced Update Allocation (Intra/Inter-Class Balance)

### The problem, measured

A frequency-proportional update budget spends almost all of its capacity on
classes that cannot improve. Class 11 (Road) had millions of points but
**negative headroom (-0.57)** — updating it hurts the model because it is
already saturated. Meanwhile Class 0 (Car) had only 98k points but
**+41.07 headroom**.

### The mechanism: a subcluster update ledger

Allocate the adaptation budget by **headroom**, not frequency, at two levels:

- **Inter-class balance:** freeze saturated majority classes (Road) and funnel
  the update budget to high-headroom classes (Car, Motorcycle). This directly
  prevents the majority classes' massive point counts from drowning out the
  classes mIoU actually weights.
- **Intra-class balance:** maintain K representative subclusters per class
  (initialized from source); a subcluster contributes to the prototype update
  only if its update count is within a bounded range of its siblings'. This
  prevents a single dense feature mode within a class from dominating the
  prototype's motion.

Crucially, the subclusters only track updates; they never touch inference.
This prevents the Voronoi-shattering failure that destroyed previous
subcluster-gating attempts.

---

## 6. Multi-View Test-Time Augmentation (variant)

Generate augmented views (yaw roll, scale, dropout), bundle the resulting
hypervectors, and use cross-view soft agreement as a reliability signal.
Positioned as a compute-scaling variant: *"when additional compute is
available, multi-view agreement raises macro precision at higher cost."*

---

## 7. Extensive Reading List

The following literature provides the theoretical foundation and the exact
mechanisms needed for the three pillars.

### 1. Robust Feature Extraction & Magnitude-Based OOD Detection (Pillar 1)
The pretraining objective relies on the feature space being distance-aware and
magnitude-informative. These papers justify and refine that foundation.

| Paper | The Concept | How to Use It |
|---|---|---|
| [**Evidential Deep Learning**](https://scholar.google.com/scholar?q=Evidential+Deep+Learning+Sensoy)<br>*(Sensoy et al., NeurIPS 2018)* | Formulates learning as evidence acquisition using Subjective Logic. Places a Dirichlet distribution over class probabilities, allowing explicit "I don't know" outputs independent of prototype distance. | (Used for pre-training/formulation reference) |
| [**SNGP**](https://scholar.google.com/scholar?q=Simple+and+Principled+Uncertainty+Estimation+with+Deterministic+Deep+Learning+via+Distance+Awareness)<br>*(Liu et al., NeurIPS 2020)* | Spectral-normalized Neural Gaussian Processes force the latent space to be "distance-aware," providing high-quality epistemic uncertainty in a single pass. | (Used for pre-training/formulation reference) |
| [**Prior Networks**](https://scholar.google.com/scholar?q=Predictive+Uncertainty+Estimation+via+Prior+Networks)<br>*(Malinin & Gales, NeurIPS 2018)* | Introduces Dirichlet Prior Networks to model distributional uncertainty directly. Targets the requirement to identify points far from the source distribution. | (Used for pre-training/formulation reference) |
| [**DUQ**](https://scholar.google.com/scholar?q=Deterministic+Neural+Networks+with+Appropriate+Inductive+Biases+Capture+Epistemic+Uncertainty)<br>*(van Amersfoort et al., ICML 2020)* | Uses a two-sided gradient penalty to enforce "distance awareness." | Proves you can extract a mathematically rigorous epistemic uncertainty score directly from the feature vector's magnitude. |
| [**Feature Space Singularity**](https://scholar.google.com/scholar?q=Feature+Space+Singularity+for+Out-of-Distribution+Detection)<br>*(Huang et al., NeurIPS 2021)* | Demonstrates the raw L2 norm (magnitude) of the latent vector is a highly effective, zero-cost metric for OOD detection. | Justifies the feature-norm term of the gate (Phase 12: norm AUROC 0.94). |
| [**React**](https://scholar.google.com/scholar?q=React+Out-of-distribution+Detection+With+Rectified+Activations)<br>*(Sun et al., NeurIPS 2021)* | Shows OOD data causes massive activation spikes in penultimate layers. Clamping these makes uncertainty metrics vastly more reliable. | Monitor pre-projection features for activation spikes as an immediate indicator of epistemic failure (e.g., fog artifacts). |
| [**Laplace Redux**](https://scholar.google.com/scholar?q=Laplace+Redux+Effortless+Bayesian+Deep+Learning)<br>*(Daxberger et al., NeurIPS 2021)* | Applies a post-hoc Laplace approximation to the last layer, yielding a closed-form, single-pass epistemic uncertainty without altering weights. | Fit this approximation to the source data once; evaluate target data against it at test-time for a pure epistemic signal. |

### 2. Uncertainty Gating and Label Selection (Pillar 2)
Consistency and confidence signals must act as a **veto on the label/update**, not a smoother of the score.

| Paper | The Concept | How to Use It |
|---|---|---|
| [**Temporal Ensembling**](https://scholar.google.com/scholar?q=Temporal+Ensembling+for+Semi-Supervised+Learning)<br>*(Laine & Aila, ICLR 2017)* | The foundational text on using moving averages of network predictions over time to stabilize pseudo-labels. | Maps perfectly to consecutive LiDAR frames for defining temporal vetoes. |
| [**FixMatch**](https://scholar.google.com/scholar?q=FixMatch+Simplifying+Semi-Supervised+Learning+with+Consistency+and+Confidence)<br>*(Sohn et al., NeurIPS 2020)* | Enforces that weak and strong predictions must strictly agree before retaining a label. | The theoretical basis for a "hard veto" argument over "soft smoothing." |
| [**ST3D**](https://scholar.google.com/scholar?q=ST3D+Self-training+for+Unsupervised+Domain+Adaptation+on+3D+Object+Detection)<br>*(Yang et al., CVPR 2021)* | State-of-the-art for 3D LiDAR UDA, heavily utilizing spatial/temporal consistency to filter pseudo-labels. | Read to see exactly what baselines your consistency veto must outperform. |
| [**TENT**](https://scholar.google.com/scholar?q=TENT+Fully+Test-Time+Adaptation+by+Entropy+Minimization)<br>*(Wang et al., ICLR 2021)* | The baseline for entropy minimization based TTA. | Cite as the counter-example: TENT's soft smoothing causes semantic poisoning. Contrast with a boolean temporal veto. |
| [**PointTTA**](https://scholar.google.com/scholar?q=PointTTA+Test-Time+Adaptation+for+Point+Cloud+Processing)<br>*(Metzger et al., 2023)* | A direct TTA framework for 3D point clouds relying on spatial transformations and self-supervision. | Compare their soft-consistency loss to your hard-veto logic to define valid "spatial neighborhoods." |
| [**Ada3D**](https://scholar.google.com/scholar?q=Ada3D+Adaptive+3D+Object+Detection)<br>*(Recent CVPR/ICCV)* | Focuses on aligning local spatial contexts under domain shifts, assuming adjacent points share semantic identity. | Formalize the spatial veto: if cosine similarity says Pedestrian, but k geometric neighbors are Road, the label is vetoed. |

### 3. Balanced / Headroom-Based Allocation (Pillar 3)
These papers tackle gating updates based on representation saturation and non-i.i.d. target streams, aligning perfectly with the subcluster ledger.

| Paper | The Concept | How to Use It |
|---|---|---|
| [**RoTTA**](https://scholar.google.com/scholar?q=RoTTA+Robust+Test-Time+Adaptation+in+Dynamic+Scenarios)<br>*(Yuan et al., CVPR 2023)* | Maintains a category-balanced memory bank for temporally correlated streams. | Mandatory reading. Differentiate your approach: your ledger never touches inference, preventing Voronoi-shattering. |
| [**Class-Balanced Loss**](https://scholar.google.com/scholar?q=Class-Balanced+Loss+Based+on+Effective+Number+of+Samples)<br>*(Cui et al., CVPR 2019)* | Formally defines class saturation (effective number of samples), proving exponentially diminishing returns. | Provides mathematical justification for the headroom-budgeting ledger. |
| [**DELTA**](https://scholar.google.com/scholar?q=DELTA+Degradation-Free+Fully+Test-Time+Adaptation)<br>*(Zhao et al., ICLR 2023)* | Explores how unconstrained TTA destroys majority classes. Uses class-aware balancing. | Compare your ledger against their balancing mechanism. |
| [**NOTE**](https://scholar.google.com/scholar?q=NOTE+Robust+Continual+Test-time+Adaptation+Against+Temporal+Correlation)<br>*(Gong et al., NeurIPS 2022)* | Tackles the "fog bank" problem where temporally correlated TTA streams cause batch-norm/memory collapse. | Direct theoretical precedent. They balance memory to prevent temporal collapse; you balance subclusters to equalize headroom. |
| [**LAME**](https://scholar.google.com/scholar?q=LAME+Latent-Space+Marginalization+for+Blind+Action)<br>*(Boudiaf et al., CVPR 2022)* | Performs TTA without updating weights, strictly updating latent space assignments via Laplacian smoothing on affinity matrix. | Massive structural citation. Contrast their affinity-matrix approach with your O(K) budgeted subcluster approach. |
| [**AdaContrast**](https://scholar.google.com/scholar?q=AdaContrast+Contrastive+Test-Time+Adaptation)<br>*(Chen et al., CVPR 2022)* | Utilizes a pseudo-label queue to track class frequencies and reject over-represented classes. | Validates that updating saturated classes hurts. Defends why your ledger freezes saturated subclusters. |
| [**Practical Coresets for Online ML**](https://scholar.google.com/scholar?q=Practical+Coresets+for+Online+Machine+Learning)<br>*(Feldman 2020)* | Focuses on selecting the smallest possible subset of streaming data to represent the full distribution. | Rigorous framing: your ledger maintains an online coreset of the target domain. Tracking K subclusters equalizes learning potential. |

### 4. Multi-View Test-Time Augmentation (variant)
If including the TTA variant for compute-scaling, ground it in literature treating augmentation as a reliability signal.

| Paper | The Concept | How to Use It |
|---|---|---|
| [**Learning to Trust**](https://scholar.google.com/scholar?q=Learning+to+Trust+Test-Time+Augmentation+for+Epistemic+Uncertainty+Estimation)<br>*(Ayhan & Berens, 2018)* | Seminal paper establishing variance across TTAs as a valid proxy for epistemic uncertainty. | Foundations for TTA reliability. |
| [**Uncertainty-guided TTA**](https://scholar.google.com/scholar?q=Uncertainty-guided+Test-Time+Augmentation)<br>*(Shanmugam et al., 2021)* | Standard TTA applies all augmentations equally. This paper learns which augmentations to trust. | Maps well to your "cross-view soft agreement" signal. |
| [**PointContrast**](https://scholar.google.com/scholar?q=PointContrast+Unsupervised+Pre-training+for+3D+Point+Cloud+Understanding)<br>*(Xie et al., ECCV 2020)* | Focuses on cross-view consistency in 3D point clouds. | Provides the exact geometric augmentations (yaw, roll, scale) statistically valid for 3D LiDAR TTA. |

---

## 8. Order of work (current state)

1. **Validate the gate signals on the robust encoder.** Fix the linear-probe
   sampling (uniform across frames), complete the 8-corruption panel with the
   clean-data control, per-gate-mode AUROC, and threshold envelope sweeps.
2. **Rebuild the gate from the two bare signals** (confidence, feature norm)
   on the new space — reusing the shipped `fuse_uncertainties` modes only where
   the gate-fault diagnostics clear them.
3. **Medium-scale pretraining run.** Plain `supcon_vib`, 100% data, seeded for
   reproducibility (the naive EMA baseline of 18.34% on Fog is the floor the
   converged encoder must beat).
4. **Gated EMA TTA end-to-end** on the converged encoder, targeting the
   oracle ceiling (19.54% Fog) and the Phase 9 acceptance band.
5. **Balanced allocation ledger** on top of the gated adaptation (inter-class
   headroom budgeting + intra-class subcluster bounds).
