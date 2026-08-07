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
| **Set-valued conformal prediction is vacuous in HDC.** `E[\|C(x)\|] = 0.58`: prediction sets are always empty or singleton, never ambiguous, because HDC preserves inter-class separation. | calibration-drift run |
| **AUROC is the wrong metric for a gate.** k-NN had far higher AUROC (0.939 vs 0.81) yet *lost* on precision-at-coverage, which is what a gate actually uses. | knn_sweep vs precision_coverage |
| **No pseudo-label gate recovers any of the available headroom.** Oracle gate (ground-truth labels) +2.73 mIoU; best pseudo-label gate ~0.00: the wrong 18% of pseudo-labels exactly cancel the right 82%. | overnight decision experiment |

The decisive discovery came from the Corruption Atlas diagnostics, and it
changed the direction of the project entirely:

| Finding | Evidence |
|---|---|
| **Type C corruptions (Fog, Crosstalk) destroy linear separability.** The 128D feature manifold collapses into isotropic noise. A purely mathematical TTA technique (memory banks, prototype shifts) is powerless because the underlying feature space is gone. | Corruption Atlas diagnostics |
| **The fix must happen before HDC projection.** The backbone itself has to be trained to map physical noise to clean semantic manifolds (or isolate it), so the HDC space has real geometry to work with. | micro/medium pretraining runs |
| **A decoupled SupCon+VIB pretraining makes the encoder robust.** Linear Probe on Fog reached 49.4% vs 23.6% baseline (2.1×), and the semantic information mathematically survives random projection and sign binarization (49.4% → 49.0% → 47.8%). | Phase 7/8 headroom diagnostics |
| **The remaining bottleneck is the naive decoder.** Indiscriminate EMA prototype updates absorb impossible Fog/Crosstalk artifacts, poisoning the Euclidean centroids: 8.2% prototype accuracy despite 49.4% linear separability. | Phase 8 degradation pipeline |
| **Harmful updates are identifiable by a universal signature.** They carry significantly lower confidence (0.61 vs 0.91) and significantly larger feature norms (6.41 vs 5.27), consistent across all 8 corruptions. | Phase 9 universal oracle matrix |
| **Gating by confidence recovers most of the oracle headroom.** Top-50% confidence gate: 20.63% vs perfect oracle 23.32% vs naive EMA 16.36%. | Phase 9 oracle gating |
| **Input-space remediation is a dead end.** SOR pre-filtering (in the training loop), additive noise augmentation, and global latent pooling all fail to beat the plain robust encoder on HDC Prototype Accuracy (9.1% / 12.4% / 14.4% vs 20.1%). | Phase 10 remediation shootout |
| **On the robust encoder, adaptation is nearly solved.** Naive EMA already harvests ~82% of the oracle headroom on Fog (18.34% vs 19.54%); feature norm is the dominant gate signal (AUROC 0.94), while probe confidence is currently anti-predictive (AUROC 0.15, under re-test with corrected sampling). | Phase 12 gated-EMA diagnostic |

**Therefore, the method rests on three pillars**, each attacking one of the
measured failure modes:

1. **Robust feature extractor pretraining (SupCon + VIB):** make the 128D
   space robust enough that HDC prototypes have a meaningful manifold to live on.
2. **Uncertainty-gated prototype updates:** stop the EMA memory from overfitting
   to corruption artifacts, using standard confidence/uncertainty metrics.
3. **Balanced update allocation:** ensure majority classes and dense feature
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
pseudo-labels exactly cancel the right 82%**, and this is specifically a
rare-class problem (pooled precision 99.2% vs macro precision 81% at 10%
coverage). No amount of clever filtering fixes a label that is wrong, and no
gate can recover a feature space that has collapsed into isotropic noise. The
pretrained encoder attacks the problem at the source; gating then only needs to
veto the residual artifacts.

---

## 3. Pillar 1: Robust Feature Extractor Pretraining (SupCon + VIB)

### The problem, measured

Type C corruptions (Fog, Crosstalk) mathematically destroy the linear
separability of the geometric manifolds. Prototype drift and memory-bank
methods are powerless because the underlying 128D space has collapsed: the
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
  (attenuation), explicitly *not* KITTI-C's ray-tracing algorithms, to avoid
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
  additive noise augmentation, global pooling) does not help: the extractor
  is "solved enough"; further investment belongs in the decoder.

---

## 4. Pillar 2: Uncertainty-Gated HDC Prototype Updates

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

and the prototype update becomes $c \leftarrow c + \eta \cdot w(x) \cdot z$, a soft veto that down-weights artifacts without starving adaptation
(avoiding the hard AND-gate starvation and OR-gate flooding documented in the
earlier geometric-method threads).

### Measured effect

- On the pre-robust encoder: top-50% confidence gate reached **20.63%** vs
  16.36% naive, 89% of the perfect-oracle ceiling (23.32%).
- On the robust encoder (Phase 12, Fog): naive EMA already reaches 18.34% of
  the 19.54% oracle ceiling; the gate's remaining job is closing that gap and
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

### Phase separation (design rule)

Prior correction and prototype updates must not share a pathway. The prior
term is an inference-time constant that translates decision boundaries between
prototype cells: it does not move prototypes and does not cause drift by
itself. But if **prior-corrected pseudo-labels** feed the EMA updates (or the
gate's confidence is computed on prior-inflated scores), the bias steers the
updates, the prototypes move to reinforce the bias, and the drift compounds.
Rule: the **prediction pathway** may use the prior-corrected score; the
**adaptation pathway** (admission + update) must use prior-free likelihood
confidence. This is why the frozen-decode + query-gate plan can apply the
prior safely (no updates → no feedback loop), while any prototype-adaptation
follow-up (e.g., the `additive` oracle headroom) must keep the prior out of the
update loop entirely.

---

## 5. Pillar 3: Class Balance (Decision-Level Prior Correction + Update-Level Ledger)

Class balance is required at two independent levels: the **decision rule** and
the **update rule**. They are complementary and must not be conflated.

### 5.1 The problem, measured

A frequency-proportional update budget spends almost all of its capacity on
classes that cannot improve. Class 11 (Road) had millions of points but
**negative headroom (-0.57)**, updating it hurts the model because it is
already saturated. Meanwhile Class 0 (Car) had only 98k points but
**+41.07 headroom**. At the decision level, the pure nearest-prototype rule is
a likelihood-only decoder: under corruption, scattered points land in the
largest Voronoi cells, and majority classes' prototypes occupy the largest
solid angle, "majority prototypes cannibalize the tail classes."

### 5.2 Decision-level balance: source prior correction (inference-time)

The corrected score is `score(q, c) = κ·cos(q, P_c) + τ·log π_c` (τ = −1.0 in
the old configuration). Geometrically, `τ·log π_c` is a per-class constant that
**translates every pairwise boundary** between prototype cells:
`cos(q,P_a) − cos(q,P_b) = (τ/κ)·log(π_b/π_a)`, deflating the majority cells
and inflating the rare cells, so scattered points absorbed by Road/Building can
be re-captured by their true rare class. It is Bayes' rule applied to the
frozen decoder: κ·cos is the (uncalibrated) log-likelihood, τ·log π the
log-prior.

Design notes:
- **Balances frequency, not representation volume.** It keys on the scene prior
  π_c. The old EVT density penalty (√(2 ln N) correction) balanced *angular
  volume* instead; if a majority class is also representation-dense, the prior
  alone under-compensates. Measure which imbalance is dominant before choosing.
- **Moves boundaries, not centers.** It cannot fix intra-class mode imbalance:
  a minority feature mode far from its class centroid stays misclassified
  regardless of the margin. That failure needs the update-level ledger below.
- **mIoU-oriented.** It trades majority-class precision for rare-class recall.
  Report both point accuracy and mIoU when evaluating it, a prior-corrected
  decode can look worse on point accuracy while genuinely improving mIoU.
- **Selective application.** The old oracle-switch found it helps where scatter
  absorption dominates (Wet Ground +11.7 mIoU, Echo) and hurts where scatter is
  minimal (Snow, Incomplete Echo), apply per condition, per the Phase 2 rule.
- **Never feeds the updates** (see the phase-separation rule in Pillar 2).

### 5.3 Update-level balance: the subcluster update ledger

Allocate the adaptation budget by **headroom**, not frequency, at two levels:

- **Inter-class balance:** freeze saturated majority classes (Road) and funnel
  the update budget to high-headroom classes (Car, Motorcycle). This directly
  prevents the majority classes' massive point counts from drowning out the
  classes mIoU actually weights.
- **Intra-class balance:** maintain K representative subclusters per class
  (initialized from source); a subcluster contributes to the prototype update
  only if its update count is within a bounded range of its siblings'. This
  prevents a single dense feature mode within a class from dominating the
  prototype's motion, the "balance different feature representations when one
  is majority over another" mechanism, and the only one that can fix
  cell-center displacement (5.2 cannot).

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

## 7. Previous and Current Results

### 7.1 Problem setting

| Component | Configuration |
|---|---|
| **Source data (pretraining)** | SemanticKITTI training split (sequence 08), clean range-image projections, 17 classes (16 evaluated) |
| **Target data (evaluation)** | SemanticKITTI-C, heavy severity, 8 conditions: fog, snow, wet_ground, motion_blur, beam_missing, crosstalk, incomplete_echo, cross_sensor |
| **Backbone** | SENet-2048p → 128D continuous features (the representation both methods act on) |
| **Pretraining objective (Pillar 1)** | Decoupled supervised contrastive (SupCon, L2-normalized, τ = 0.1) + Variational Information Bottleneck (KL → 𝒩(0, I), weight 0.01, applied to clean *and* augmented pathways) + CE, with physics-based augmentations only (voxelized beam dropout, ray-axis jitter, density subsampling) |
| **HDC encoding** | Seeded random bipolar projection 128D → 10,000D + sign binarization (information-preserving: 49.4% → 49.0% → 47.8% linear) |
| **Prototypes** | Per-class means of the binarized clean features (frozen) |
| **Adaptation (Pillar 2)** | Prototype updates gated by uncertainty signals (confidence + feature norm), *currently being redesigned: Phase 14 showed prototype-level TTA has no headroom on the robust encoder; the gate is moving to the query side* |
| **Balance (Pillar 3)** | Headroom-based update allocation (inter-class: freeze saturated majority classes; intra-class: bounded per-subcluster counts), planned |

### 7.2 Previous performance: the original model per condition

The Corruption Atlas measured the *original* (un-pretrained) model on each condition. Some conditions are nearly untouched; others collapse to the point where even an **oracle prototype** (perfect-label prototype updates) can barely classify anything, the feature space itself is gone, so no decoder can recover it.

| Condition | Cosine Shift | Baseline mIoU (corrupted) | Oracle Prototype (perfect graph) | Corrupted 1-NN Purity |
| :--- | :--- | :--- | :--- | :--- |
| **Incomplete Echo** | 0.070 | **25.5%** | **32.1%** | 95.4% |
| **Snow** | 0.395 | 20.6% | 25.0% | 91.5% |
| **Wet Ground** | 0.557 | 18.8% | 28.1% | 95.6% |
| **Beam Missing** | 0.513 | 15.2% | 25.0% | 92.3% |
| **Motion Blur** | 0.524 | 14.8% | 22.6% | 88.8% |
| **Cross Sensor** | 0.715 | 4.4% | 22.6% | 89.8% |
| **Crosstalk** | 0.767 | 4.7% | 13.3% | 86.6% |
| **Fog** | 0.885 | **1.8%** | **8.7%** | 75.1% |

*The drop is bimodal: five conditions keep ≥ 88% neighborhood purity and ≥ 14.8% mIoU, while Fog, Crosstalk and Cross Sensor collapse, Fog's perfect-graph oracle ceiling is 8.7% mIoU, proving the collapse is in the representation, not the decoder. (Metrics: mIoU for the two Oracle Family columns; point-wise 1-NN purity otherwise.)*

#### The previous best method still showed the same bimodal gap

The best pre-pretraining method (`DualGateModel`, `docs/geometric_method/method_details.md`) improved every condition (roughly doubling the collapsed ones), yet the bimodal structure persisted exactly: five conditions at 80–90% point accuracy, Fog and Crosstalk still barely classifiable.

| Condition | Frozen (no TTA) | DualGateModel (previous best) |
| :--- | :--- | :--- |
| **Incomplete Echo** | 88.2% | 87.7% |
| **Wet Ground** | 89.6% | **90.5%** |
| **Snow** | 86.4% | 87.1% |
| **Motion Blur** | 84.1% | 86.6% |
| **Beam Missing** | 80.0% | 86.0% |
| **Cross Sensor** | 56.6% | 62.2% |
| **Crosstalk** | 22.1% | 53.7% |
| **Fog** | **13.2%** | **31.9%** |
| **Mean** | 65.0% | 73.2% |

*And the memory-bank era showed *why* the gap is structural (Iteration 6, `docs/mem_method/adaptive_iterations.md`): prototype adaptation *improved* the survivable conditions (Snow +8.7 mIoU, Motion Blur +6.1) while *collapsing* Fog further (0.065 → 0.029 mIoU, 89% memory error). The Inlier Paradox: fog noise sits *closer* to the seed centroids than real geometry, so geometric gates admit it at 100% firing rate. Adaptation helps exactly where the representation survives and poisons exactly where it doesn't.*

### 7.3 Current performance: what the new methods change

Representation-level gains (Linear Probe, how separable the 128D space is under corruption) and decode-level gains (HDC prototype accuracy from frozen clean prototypes, Phase 14 v4 harness, 1M-point pool, oracle-calibrated). The medium-pretrained `supcon_vib` encoder (Phase 7) roughly doubled Fog linear separability (23.6% → 49.4% Linear Probe), and the robust micro encoder lifted Fog zero-shot HDC classification from 1.8% mIoU to **25.0%** point accuracy.

| Condition | Old baseline mIoU | New Linear Probe (robust encoder) | New zero-shot HDC (robust encoder) | New perfect-oracle HDC |
| :--- | :--- | :--- | :--- | :--- |
| **Incomplete Echo** | 25.5% | **92.8%** | **73.6%** | 73.4% |
| **Snow** | 20.6% | 82.0% | 63.8% | 62.8% |
| **Wet Ground** | 18.8% | 77.0% | 64.3% | 63.5% |
| **Beam Missing** | 15.2% | 87.2% | 71.4% | 68.7% |
| **Motion Blur** | 14.8% | 74.6% | 66.1% | 65.0% |
| **Cross Sensor** | 4.4% | 73.8% | 57.5% | 56.3% |
| **Crosstalk** | 4.7% | 23.6% | 35.4% | 13.9% |
| **Fog** | **1.8%** | **41.2%** | **25.0%** | 9.5% |

*Three stories in one table: (1) near-SOTA conditions stay near-SOTA (Incomplete Echo: 92.8% linear probe); (2) the collapsed conditions gain 10–20× (Fog 1.8% → 25.0% zero-shot, Crosstalk 4.7% → 35.4%); (3) the perfect-oracle column now sits *below* zero-shot on every condition, the robust encoder's clean prototypes are already the best prototypes available, and prototype-level adaptation has no headroom (Phase 14). Metric families differ across columns (mIoU vs point accuracy vs linear probe), and the Linear Probe column is from the medium-pretrained encoder while the HDC columns are from the micro-pretrained encoder; the within-column comparisons are the meaningful ones.*

#### Intermediate results: the plain medium encoder, frozen, on accuracy and mIoU

The current encoder (plain `supcon_vib`, 26 epochs on 100% data ≈ 83k steps, clean zero-shot 82.7% / mIoU 49.6%) evaluated frozen against the previous baselines on both metrics. **Fog and Crosstalk improve on both axes** (mIoU ×5.6 and ×2.6, accuracy ×2.0 and ×1.5); the geometric corruptions show **semi-equivalent accuracy with substantially better mIoU**.

| Condition | Old acc (frozen) | Ours acc (frozen) | Old mIoU (baseline) | Ours mIoU (frozen) |
| :--- | :--- | :--- | :--- | :--- |
| **Fog** | 13.2% | **26.4%** | 1.8% | **10.1%** |
| **Crosstalk** | 22.1% | **33.5%** | 4.7% | **12.0%** |
| **Snow** | 86.4% | 66.6% | 20.6% | **39.4%** |
| **Wet Ground** | 89.6% | 68.8% | 18.8% | **49.0%** |
| **Motion Blur** | 84.1% | 73.4% | 14.8% | **44.3%** |
| **Beam Missing** | 80.0% | 77.2% | 15.2% | **53.7%** |
| **Incomplete Echo** | 88.2% | 78.8% | 25.5% | **41.2%** |
| **Cross Sensor** | 56.6% | **68.9%** | 4.4% | **41.5%** |
| **Mean (8 conditions)** | 65.0% | 61.7% | 13.2% | **36.4%** |

*Protocols differ across the "Old" columns (DualGate-era: converged clean backbone with adaptation, full-sequence eval; atlas-era: original model, mIoU) and the "Ours" columns (frozen plain medium encoder, oracle-calibrated 100k-point val). The within-column comparisons are the meaningful ones: mIoU improves on every condition (mean ×2.75), and the Fog/Crosstalk gains are robust across both metrics, while the geometric-corruption accuracy is semi-equivalent (within ~10 points, lower on some, higher on Cross Sensor) and their mIoU is substantially higher.*

#### The label-free artifact gate: real performance on every condition

The margin + cosine gate (thresholds on the frozen clean prototypes, Phase 23 sweep) is evaluated label-free on all 8 corruptions. Reported per condition: the best label-free config at usable retention (≥ 50%), mIoU and retention. *Retention is the fraction of points the gate keeps; the reported mIoU is computed on that retained subset only, so a high-retention gate is preferred. The gate does not discard information, it defers low-confidence points to a fallback (leave unlabeled, a second decoder, another sensor) at deployment, and mIoU@retention is that confident branch's operating point. The gain column is thus a selective-prediction gain over the full-coverage baseline, not a coverage-normalized comparison. This acts more as a diagnostic to show potential over a real evaluation*

| Condition | Zero-shot mIoU | Best label-free gate (mIoU @ retention) | Gate gain |
| :--- | :--- | :--- | :--- |
| **Crosstalk** | 12.0% | **23.1% @ 51.6%** | +11.1 |
| **Fog** | 10.1% | **13.0% @ 54.9%** | +2.9 |
| **Snow** | 39.4% | **62.1% @ 58.9%** | +22.7 |
| **Wet Ground** | 49.0% | **54.8% @ 78.4%** | +5.8 |
| **Incomplete Echo** | 41.2% | **55.8% @ 76.8%** | +14.6 |
| **Beam Missing** | 53.7% | **65.0% @ 69.1%** | +11.3 |
| **Motion Blur** | 44.3% | **55.6% @ 70.4%** | +11.3 |
| **Cross Sensor** | 41.5% | **56.9% @ 54.3%** | +15.4 |

*The gate is a general decode-side lever that works on 7 of 8 conditions: it adds 6–23 points of mIoU at 52–78% retention, moving every geometric corruption well clear of its zero-shot level. Crosstalk is the collapsed condition that behaves like a healthy one: the gate nearly doubles its mIoU (12.0% → 23.1% at 51.6% retention), and BN-statistic alignment adds another +3.1 (12.0 → 15.1 raw mIoU). The gate works there because crosstalk's artifacts are sparse, localized wrong-beam returns whose margin/cosine geometry is separable from the correct points, so a large, high-precision subpopulation can be carved off without labels.*

*Fog is the lone exception. Its gate gain is +2.9, the worst of any condition, and every other label-free decode-side lever tested to date (alignment, adaptation, oracle retraining, buffer selection) also fails to move it above ~10% mIoU. The mechanism: fog's errors are confident artifacts (99.96% of fog misclassifications fail even the artifact filters, Phase 22.2), dense scattering inflates the feature magnitude of ~88% of all fog points, and even the correct classifications have collapsed margins (0.29 vs 0.40–0.50 for the geometric conditions). The misclassified points are thus geometrically indistinguishable from correct points without the true label: the oracle-loss bound reaches 55% @ 28% retention, proving the information exists but is not estimable from confidence geometry alone. The collapse is additionally class-conditional (Road, Building, Other-ground, Traffic-sign, Bicycle die while Terrain and Truck survive), and no pretraining regimen tested so far (plain, strong-vib, additive volumetric) has fixed it. The residual sits in the representation: the fog features of the collapsing classes are separable in principle (the plain 128D linear probe reaches 36–49%), but the 10kD binarized decode cannot exploit them. The one thing that *does* move fog full-scene is the true-label prototype oracle (sec 7.4); the missing information is the label, not the geometry.*

### 7.4 The labeled-prototype oracle: the target a TTA method must chase

The frozen clean prototypes were long assumed to be the best decoder (Phase 21), but re-estimating the 10kD prototypes from the corrupted stream with true labels recovers the collapsed conditions on the **full scene, no points removed**, and the result is pool-size-stable (200k / 500k / 1M pools agree; Phase 24.10):

| Condition | Zero-shot mIoU | Full-label oracle mIoU | Gain |
| :--- | :--- | :--- | :--- |
| **Fog** | 10.1% | **16.6%** | +6.5 |
| **Crosstalk** | 12.0% | **26.2%** | +14.2 |
| Wet Ground | 49.0% | 51.2% | +2.2 |
| Cross Sensor | 41.5% | 43.6% | +2.1 |
| Snow | 39.4% | 40.7% | +1.3 |
| Incomplete Echo | 41.2% | 41.2% | 0.0 |
| Beam Missing | 53.7% | 53.6% | −0.1 |
| Motion Blur | 44.3% | 44.8% | +0.5 |

*The prototype decoder is not exhausted: with the right per-point weighting, re-estimated prototypes beat the frozen clean ones on every condition and substantially on the collapsed ones (point accuracy roughly doubles, fog 26% → 51%, crosstalk 34% → 49%). Every label-free TTA variant tested (naive EMA, soft-dual-weight EMA, BN-stat alignment, artifact gating, prior correction, artifact-free prototype estimation) fails to reach it, so the current target is a TTA method that replicates the oracle's weighting without labels: a learned estimator of each point's cos-to-true (the loss-estimator head), used as the per-point weight in a prototype re-estimate. Excluding the confident hallucinations does not get there on its own (artifact-free ≈ full-label), so the problem is not "drop the artifacts," it is "estimate the weights the oracle would assign."*

### 7.5 The kNN reassignment: the best label-free re-estimate, and why local structure is the only source of the missing labels

The best label-free TTA so far is a single-round, recoverability-gated kNN reassignment (Iteration 7, `--iter7_knn_reassign`): detect the recoverable subset with the combined label-free signal, then reassign those points by a confidence-weighted kNN vote over their neighbors' LP labels, and re-estimate the prototypes. Full-scene mIoU:

| Condition | Zero-shot | zs-pseudo re-est | **kNN reassign (best)** | Full-label oracle |
| :--- | :--- | :--- | :--- | :--- |
| **Fog** | 10.1% | 9.3% | **9.8%** | 17.1% |
| **Crosstalk** | 12.0% | 10.7% | **12.7%** | 26.2% |
| Snow | 39.4% | 38.1% | 39.5% | 40.6% |
| Wet Ground | 49.0% | 47.7% | 48.1% | 51.5% |
| Incomplete Echo | 41.2% | 39.5% | 40.5% | 40.9% |
| Beam Missing | 53.7% | 51.6% | 53.0% | 49.4% |
| Motion Blur | 44.3% | 43.0% | 43.9% | 44.8% |
| Cross Sensor | 41.5% | 39.8% | 41.2% | 43.5% |

*The honest framing: BN-stat alignment is still the best single label-free *full-scene* result on crosstalk (15.1%, sec 7.3), and kNN is the best **re-estimate** (the first label-free prototype re-estimate to beat the frozen zero-shot decoder on crosstalk, and the best at not collapsing the geometric conditions, which all stay within 0.5-0.9 pts of zero-shot). The kNN method is cheap (one kNN graph + one prototype re-estimate), which keeps it in the TTA category.*

**Why the kNN captures information the other gates and classifiers do not:**
- **Detection is solvable; assignment is the hard part.** The combined recoverability signal separates fixable from unfixable points (AUROC 0.68-0.80 on fog/crosstalk), but it says *which* points are fixable, not *what class* they belong to. Every other mechanism tested (artifact gates, LP assignment, MVAC consensus, Sinkhorn balance, ReAct clipping, recoverability reweighting) operates on the decode or the existing assignment and cannot change the assignment.
- **The recoverable points' true class is invisible to every global signal.** Their true class sits at rank 3.7-4.8 in the clean-prototype similarity ordering, and the linear probe gets only 5-8% on them (worse than on unfixable points). No global classifier or clean prototype can name them.
- **The local structure can.** The recoverable set clusters by class in the 128D space (kNN true-label agreement 0.76-0.95), and the kNN vote exploits exactly that: the assignment information that global geometry throws away is present locally. This is why gating (which only selects) and reweighting (which only scales) fail, while a local consensus reassignment recovers some of it.

**What a TTA method should learn from the failure modes (full detail in `docs/tta_iterations.md`):**
- **Spend capacity on "what class", not "which points".** Detection is achievable label-free; the binding constraint is naming the fixable points' classes, which requires local structure.
- **The label source feeding the local vote is the ceiling.** The kNN vote is only as good as the labels it reads (the LP, ~35%); the local clusters exist but cannot be exploited beyond the label source. This is why the EM bootstrap diverged (Iteration 8): iterating the vote over the current pseudo-labels adds no new information and compounds error.
- **Only reassign the points you are confident are wrong.** Reassigning correct points (the bootstrap's top-25%-of-everything selection) actively degrades the healthy majority.
- **A better method needs a label source that is locally more accurate than the LP** (e.g., clustering the recoverable set before labeling it, or a per-cluster consensus that does not depend on the probe), which is the direction a learned head should pursue.

---

## 8. Order of work (current state)

1. **Overnight medium-scale pretraining run** (in flight): `supcon_vib_strongvib`,
   26–30 epochs on 100% data (proper full-length cosine LR, checkpoint + optimizer
   + scheduler saved for cheap `--continue_training` continuation), with headroom
   + deep diagnostics baked into `med_pretrain_eval.py`.
2. **Measure the converged encoder** (morning): re-run the v4 oracle ladder +
   deep diagnostics, expect Fog zero-shot > 35% and the benign-condition mean to
   recover toward the old frozen 65% baseline as training converges.
3. **Test decision-level prior correction on the strongvib decode** (cheap, no
   training): static source prior (τ = −1.0) applied *selectively* per condition,
   reporting **both point accuracy and mIoU** (the prior is mIoU-oriented). If the
   benign-condition mIoU recovers toward the DualGateModel-era numbers, reinstate
   it as the decision-level inter-class balance (Pillar 3.2).
4. **Build the query-side gate on the strongvib signals** (confidence + norm,
   direction-calibrated per corruption, fog joint AUROC 0.856 is the reference),
   applied to the frozen prototypes. The prior (if reinstated) lives only in the
   prediction pathway, never in the gate or update (phase-separation rule).
5. **Update-level class balance (Pillar 3.3)**: subcluster ledger on top of the
   gated adaptation, inter-class headroom budgeting + intra-class per-subcluster
   bounds, once prototype adaptation is re-engaged (e.g., the `additive` oracle
   headroom follow-up).
6. **Prototype-adaptation path (follow-up)**: study why `additive`'s Fog means
   are usable (+19.2 oracle headroom) and whether its weak gate signals can be
   sharpened, if so, gated prototype adaptation becomes viable again, with the
   prior strictly excluded from the update loop.
