# Robust Feature Pretraining and Adaptive Hyperdimensional Prototypes for LiDAR Test-Time Adaptation

---

## 1. What we established

This project started by trying to build a better geometric confidence gate in
hyperdimensional space. That line of work is closed, and its failure is well
understood. Everything below is a measured result, not an intuition.

### The geometric-gate line is closed

| Finding | Evidence |
|---|---|
| Every geometric refinement loses to a plain prototype similarity score. Covariance ellipsoids, subspace reweighting, and k-NN contrastive banks all do worse. | rank sweep |
| The reason is that there are far too few samples per class for the number of dimensions. A mean is estimable; a covariance is not. | spectrum diagnostic |
| No pseudo-label gate recovers the available headroom. A gate with perfect labels gains +2.73 mIoU; the best pseudo-label gate gains about nothing, because the wrong 18% of pseudo-labels cancel the right 82%. | overnight decision experiment |

### The decisive discovery: corruption destroys the representation

The Corruption Atlas diagnostics changed the direction of the project:

| Finding | Evidence |
|---|---|
| Fog and Crosstalk destroy the linear separability of the feature space. The 128D manifold collapses into noise, so no purely mathematical TTA technique can work: the feature space itself is gone. | Corruption Atlas |
| The fix must happen before the HDC projection. The encoder itself has to be trained to map physical noise to clean semantic manifolds, so the HDC space has real geometry to work with. | micro / medium pretraining runs |
| A contrastive + information-bottleneck pretraining makes the encoder robust. Linear separability on Fog jumped from 23.6% to 49.4%, and that information survives random projection and sign binarization (49.4% to 49.0% to 47.8%). | Phase 7/8 headroom diagnostics |
| Input-space remediation is a dead end. Pre-filtering, additive noise augmentation, and global pooling all fail to beat the plain robust encoder. | Phase 10 remediation shootout |

### The 7-class setting: a simpler problem, not a fix

We compared our 17-class setting against the 7-class setting that D3CTTA and GIPSO
use. The 7-class map folds the fragile rare classes (bicycle, truck, traffic-sign)
into background, so its numbers are naturally higher. That is a confound when
comparing to those papers, and it matters for how we report results. But it is
**not** a robustness fix:

| Finding | Evidence |
|---|---|
| The higher absolute mIoU is a label-space effect, not a feature improvement. The 7-class and 17-class encoders have nearly identical feature geometry: both sit at a participation ratio of about 4 (out of 128), both have roughly 1% dead HDC coordinates, and both have the same code diversity. The 7-class clean mIoU is 0.63 vs 0.44 for the same reason a 7-way problem is easier than a 17-way one. | seven-class diagnostics |
| The collapse is intrinsic to the features, not to the label granularity. Under fog and crosstalk, the true class of a wrongly-classified point is found in the top-3 clean prototypes only 8-13% of the time, at or below the ~19% random baseline. The information to name those points is simply absent from the features, regardless of how the labels are grouped. | recoverability check |
| The class-conditional collapse survives the coarse map. The one rare class kept in the 7-class map (pedestrian) is 0.10 on clean and 0.00 under both corruptions, exactly like its counterpart in the 17-class map. The 7-class mIoU is carried by road and the partial survivors; the dead classes are hidden, not fixed. | seven-class per-class diagnostics |
| The recoverable ceiling is set by the features, not the encoder or the map. Under the 7-class map, even the perfect-label oracle reaches only ~0.15 on fog and ~0.27 on crosstalk, and every encoder we have (the plain one, the hard-negative one, and the one trained on 7 classes) lands at the same ceiling. | class-paradigm triage |
| D3CTTA's adaptation mechanism does not transfer to our features. Its confident pseudo-label selection is only about 10% accurate on our fog and crosstalk, so its ridge adaptation collapses the decode to nearly zero on every condition. Its reported gains came from its backbone and pretraining, not the mechanism. | D3CTTA mechanism diagnostic |

The point of the 7-class setting is therefore twofold: it is the fair comparison
space against the thirdparty papers, and it gives higher absolute numbers because
the problem is simpler. It does **not** solve the failure modes we care about,
because those live in the features.

### The three pillars

The method rests on three pillars, each attacking one measured failure mode:

1. **Robust feature extractor pretraining.** Make the 128D space survive fog and
   crosstalk before the HDC projection, so prototypes have a real manifold to live
   on. This is framed against the DGLSS / DGLSS++ generalization frameworks by the
   isotropy hypothesis (Section 3).
2. **Adaptive prototype updates when necessary.** At deployment, decide from the
   distance-to-clean-prototype uncertainty which points update the prototypes,
   then re-decode. The measurements bound what this can achieve: the label-free
   update is flat in full-coverage mIoU, and the perfect-label oracle is the wall.
3. **Balanced update allocation.** Ensure the majority classes do not consume the
   whole adaptation budget.

---

## 2. The (potential) adaptive learning framework

The framework has two threads, now evaluated in the 7-class setting.

The first thread is **robust feature-extractor training**. The goal is a feature
space whose fog and crosstalk recoverable ceiling approaches clean. The working
hypothesis is angular isotropy: our contrastive training keeps the classes spread
evenly over the unit hypersphere, which stops the space from collapsing into a
low-rank anisotropic manifold that would saturate the HDC sign-projection. This is
being tested directly against the DGLSS and DGLSS++ generalization frameworks
(Section 3).

The second thread is **adaptive prototype updates when necessary**. At deployment,
we use the distance to the nearest clean prototype to decide which points may
update the prototypes, then re-decode. The measurements bound this thread tightly:
the label-free update does not move full-coverage mIoU above zero-shot, and the
perfect-label oracle ceiling is only ~0.15 on fog and ~0.27 on crosstalk. That
ceiling is the same for every encoder, so only the first thread can move it.

The key measurement that motivated this framing: the oracle ceiling is set by the
map and by the corrupted feature structure, not by the gating mechanism and not by
the encoder. Two consequences follow. First, we use the 7-class map as the
evaluation space with our existing 17-class-trained encoders, which already decode
better under it than an encoder trained on 7 classes, so no retraining is needed.
Second, the standing target is the encoder: whatever training regime produces a
feature space whose fog and crosstalk ceiling approaches clean is the win.

---

## 3. Pillar 1: Robust Feature Extractor Pretraining

### The problem, measured

Fog and Crosstalk mathematically destroy the linear separability of the feature
manifold. The noise points become indistinguishable from the semantic points, so
prototype drift and memory-bank methods have nothing to work with. The only fix is
to train the encoder so the corruption maps to clean semantics.

### The mechanism

We train the 128D backbone with a decoupled objective before the HDC projection
layer:

- **Contrastive alignment.** A supervised contrastive loss pulls augmented views of
  the same class together and pushes different classes apart. LiDAR segmentation is
  dense, so a single batch provides millions of positive and negative pairs
  natively.
- **Magnitude isolation.** A variational information-bottleneck penalty forces the
  latent magnitudes down. Complex, high-entropy spatial noise is expensive to
  memorize, so it collapses toward the origin, while the semantic geometry keeps
  bounded, spherical, dense clusters.
- **Physics-based augmentation only.** We use voxelized beam dropout, ray-axis
  jitter, and density subsampling, and explicitly avoid KITTI-C's own ray-tracing
  algorithms, so the augmentation does not overfit to the corruption pipeline.

### Measured effect

- Linear separability on Fog reached **49.4%**, vs 23.6% for the baseline (2.1x).
- The information survives the HDC encoding losslessly: 49.4% linear, then 49.0%
  after random projection, then 47.8% after sign binarization.
- Feature magnitudes stay bounded (about 4.6-5.6), which is what the Euclidean
  nearest-prototype geometry of HDC needs.
- Best HDC prototype accuracy from the naive decoder: **20.1%**, vs 5.3% for the
  untrained baseline in the same protocol.
- Post-hoc remediation on top of the robust encoder does not help. The extractor
  is solved enough; further investment belongs elsewhere.

### Isotropy in comparison to other LiDAR generalization frameworks

The two closest single-source domain-generalization frameworks are DGLSS and
DGLSS++. Both train a robust encoder with correlation-consistency constraints
(the SCC and LSCC losses). When we adapted them to our architecture they collapsed
the HDC decode, and the working hypothesis is the anisotropy mechanism.

The concern is this. A constraint that only asks the class-correlation matrix to be
consistent can be satisfied by collapsing the embedding into a low-rank subspace.
A low-rank space with a strong shared direction saturates the HDC sign-projection:
most of the 10,000 binarized coordinates become near-constant across points, so
the prototypes collapse. Our contrastive term is the opposite: it is a uniformity
objective that is only satisfied when the classes spread evenly over the sphere, so
every random projection sees a balanced mix of points.

The baseline measurement is important context. Our own contrastive class-count
models sit at a participation ratio of about 4 in both the 7-class and 17-class
regimes, with roughly 1% dead HDC coordinates and the same code diversity. The HDC
pathway is healthy and the collapse mechanism is not triggered at the baseline. So
the isotropic-vs-anisotropic claim must be tested against a model that is
genuinely more anisotropic than ours. The isotropy diagnostic trains DGLSS, DGLSS++,
and our method at equal budget and measures the participation ratio, the top-5
variance share, the condition number, and the HDC dead-coordinate fraction on clean
and corrupted features. That will decide whether the DGLSS regimes push the space
below our baseline, and whether any regime moves the fog and crosstalk ceiling.

---

## 4. Pillar 2: Adaptive Prototype Updates When Necessary (TTA)

### The problem, measured

With the robust encoder in place, the remaining failure is the naive decoder:
updates that accept every point absorb the fog and crosstalk artifacts and poison
the zero-shot centroid. On the pre-robust encoder, even the perfect-label prototype
ceiling was only 8.7% mIoU on fog, which proved the collapse is in the
representation, not the decoder.

### The uncertainty signal

We use the distance to the nearest clean class prototype, measured as a scale-
invariant cosine similarity. On 7-class fog this separates correct from wrong
points with AUROC 0.71-0.82.

### What the measurements bound (full-coverage mIoU)

- The label-free gated update is flat. Re-estimating the centroids from the
  distance-confident points reproduces the clean centroids, because the confident
  points already decode correctly and the far points that would move the centroids
  are exactly the ones the gate excludes. This is the assignment wall from the TTA
  iterations: a label-free signal can say which points are wrong, but not what
  class they belong to.
- The only full-coverage gains come from true labels. The perfect-label oracle
  ceiling is ~0.15 on fog and ~0.27 on crosstalk. The distance signal ranks the
  points well, but it cannot harvest that ceiling label-free on the full scene.

### The target

The oracle ceiling is the wall. The TTA thread's job is to approach it; only the
encoder thread can raise it.

### A design rule

Prior correction and prototype updates must not share a pathway. The prior is an
inference-time constant that shifts decision boundaries; it does not move
prototypes by itself. But if prior-corrected pseudo-labels feed the updates, the
bias steers the prototypes and the drift compounds. The prediction pathway may use
the prior-corrected score; the adaptation pathway must not.

---

## 5. Previous and Current Results

### 5.1 Problem setting

| Component | Configuration |
|---|---|
| **Source data (pretraining)** | SemanticKITTI training split (sequence 08), clean range-image projections, 17 classes (16 evaluated) |
| **Target data (evaluation)** | SemanticKITTI-C, heavy severity, 8 conditions: fog, snow, wet_ground, motion_blur, beam_missing, crosstalk, incomplete_echo, cross_sensor |
| **Backbone** | SENet-2048p, 128D continuous features (the representation both methods act on) |
| **Pretraining objective (Pillar 1)** | Decoupled supervised contrastive + variational information bottleneck + cross-entropy, with physics-based augmentations only |
| **HDC encoding** | Seeded random bipolar projection 128D to 10,000D, then sign binarization (information-preserving: 49.4% to 49.0% to 47.8%) |
| **Prototypes** | Per-class means of the binarized clean features (frozen) |
| **Adaptation (Pillar 2)** | Adaptive prototype updates when necessary, gated by distance-to-clean-prototype, bounded by the measured label-free ceiling |
| **Balance (Pillar 3)** | Headroom-based update allocation (freeze saturated majority classes; bound per-subcluster counts), planned |

### 5.2 Previous performance: the original model per condition

The Corruption Atlas measured the original, un-pretrained model on each condition.
Some conditions are nearly untouched; others collapse to the point where even an
oracle prototype can barely classify anything. The feature space itself is gone,
so no decoder can recover it.

| Condition | Cosine Shift | Baseline mIoU (corrupted) | Oracle Prototype | Corrupted 1-NN Purity |
| :--- | :--- | :--- | :--- | :--- |
| **Incomplete Echo** | 0.070 | **25.5%** | **32.1%** | 95.4% |
| **Snow** | 0.395 | 20.6% | 25.0% | 91.5% |
| **Wet Ground** | 0.557 | 18.8% | 28.1% | 95.6% |
| **Beam Missing** | 0.513 | 15.2% | 25.0% | 92.3% |
| **Motion Blur** | 0.524 | 14.8% | 22.6% | 88.8% |
| **Cross Sensor** | 0.715 | 4.4% | 22.6% | 89.8% |
| **Crosstalk** | 0.767 | 4.7% | 13.3% | 86.6% |
| **Fog** | 0.885 | **1.8%** | **8.7%** | 75.1% |

The drop is bimodal. Five conditions keep high neighborhood purity and at least
14.8% mIoU, while Fog, Crosstalk and Cross Sensor collapse. Fog's perfect-label
oracle ceiling is 8.7% mIoU, which proves the collapse is in the representation,
not the decoder.

The previous best method (a dual-gate model) improved every condition and roughly
doubled the collapsed ones, yet the bimodal structure persisted exactly. The memory-
bank era showed why it is structural: prototype adaptation improved the survivable
conditions while collapsing Fog further, because fog noise sits closer to the seed
centroids than real geometry does. Adaptation helps exactly where the
representation survives, and poisons exactly where it does not.

### 5.3 Current performance: what the new methods change

The robust encoder roughly doubled Fog linear separability (23.6% to 49.4% linear
probe) and lifted Fog zero-shot HDC point accuracy from 13.2% to 25.0% (the
full-coverage mIoU equivalents are in sections 5.5 and 5.6).

| Condition | Old baseline | New Linear Probe | New zero-shot HDC | New perfect-oracle HDC |
| :--- | :--- | :--- | :--- | :--- |
| **Incomplete Echo** | 88.2% | **92.8%** | **73.6%** | 73.4% |
| **Snow** | 86.4% | 82.0% | 63.8% | 62.8% |
| **Wet Ground** | 89.6% | 77.0% | 64.3% | 63.5% |
| **Beam Missing** | 80.0% | 87.2% | 71.4% | 68.7% |
| **Motion Blur** | 84.1% | 74.6% | 66.1% | 65.0% |
| **Cross Sensor** | 56.6% | 73.8% | 57.5% | 56.3% |
| **Crosstalk** | 22.1% | 23.6% | 35.4% | 13.9% |
| **Fog** | **13.2%** | **41.2%** | **25.0%** | 9.5% |

All four columns are point accuracy, so they are directly comparable. The
full-coverage mIoU equivalents are in sections 5.5 and 5.6 (for example, Fog
zero-shot mIoU is 10.1% and Crosstalk 12.0%). The accuracy-to-mIoU gap is large
because mIoU averages per-class IoU and is dominated by the rare classes, which
collapse under corruption.

The perfect-oracle column sits below zero-shot on every condition because this is a
naive perfect-label re-estimation: re-estimating the prototypes from the degraded
corrupted features is worse than keeping the frozen clean prototypes. This is the
"prototype-level adaptation has no headroom" finding. The weighted oracle in
section 5.5, which uses the right per-point weighting, is what actually beats
zero-shot (Fog 16.6%, Crosstalk 26.2% full-scene mIoU).

The current encoder, evaluated frozen against the previous baselines, improves mIoU
on every condition (mean 13.2% to 36.4%), and the Fog and Crosstalk gains are
robust across both accuracy and mIoU.

### 5.4 The uncertainty signal, current measurement (7-class setting)

The distance-to-clean-prototype signal ranks correct from wrong points well (AUROC
0.71-0.82), but in full-coverage mIoU the label-free prototype update is flat: it
does not beat zero-shot. The perfect-label oracle ceiling is ~0.15 on fog and ~0.27
on crosstalk. The uncertainty signal is sound; the ceiling is the feature
structure, which is the encoder thread's target.

### 5.5 The labeled-prototype oracle: the target a TTA method must chase

Re-estimating the prototypes from the corrupted stream with true labels recovers
the collapsed conditions on the full scene, with no points removed, and the result
is stable across pool sizes. The old baseline mIoU (the original, un-pretrained
model, section 5.2) is included as the reference the robust encoder starts from:

| Condition | Old baseline mIoU | Zero-shot mIoU | Full-label oracle mIoU | Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Fog** | 1.8% | 10.1% | **16.6%** | +6.5 |
| **Crosstalk** | 4.7% | 12.0% | **26.2%** | +14.2 |
| Wet Ground | 18.8% | 49.0% | 51.2% | +2.2 |
| Cross Sensor | 4.4% | 41.5% | 43.6% | +2.1 |
| Snow | 20.6% | 39.4% | 40.7% | +1.3 |
| Incomplete Echo | 25.5% | 41.2% | 41.2% | 0.0 |
| Beam Missing | 15.2% | 53.7% | 53.6% | -0.1 |
| Motion Blur | 14.8% | 44.3% | 44.8% | +0.5 |

The prototype decoder is not exhausted: with the right per-point weighting,
re-estimated prototypes beat the frozen clean ones on every condition, and
substantially on the collapsed ones. Every label-free TTA variant we tried fails to
reach it. The problem is not "drop the artifacts"; it is "estimate the weights the
oracle would assign" without labels.

### 5.6 The kNN reassignment: the best label-free re-estimate

The best label-free TTA so far is a single-round kNN reassignment: detect the
recoverable subset, reassign those points by a confidence-weighted vote over their
neighbors, and re-estimate the prototypes. Full-scene mIoU:

| Condition | Zero-shot | zs-pseudo re-est | **kNN reassign** | Full-label oracle |
| :--- | :--- | :--- | :--- | :--- |
| **Fog** | 10.1% | 9.3% | **9.8%** | 17.1% |
| **Crosstalk** | 12.0% | 10.7% | **12.7%** | 26.2% |
| Snow | 39.4% | 38.1% | 39.5% | 40.6% |
| Wet Ground | 49.0% | 47.7% | 48.1% | 51.5% |
| Incomplete Echo | 41.2% | 39.5% | 40.5% | 40.9% |
| Beam Missing | 53.7% | 51.6% | 53.0% | 49.4% |
| Motion Blur | 44.3% | 43.0% | 43.9% | 44.8% |
| Cross Sensor | 41.5% | 39.8% | 41.2% | 43.5% |

BN-statistic alignment remains the best single label-free full-scene result on
crosstalk (15.1%), and the kNN method is the best re-estimate: the first label-free
prototype re-estimate to beat the frozen zero-shot decoder on crosstalk, and the
best at not collapsing the geometric conditions.

Why the kNN captures information the other gates and classifiers do not:

- **Detection is solvable; assignment is the hard part.** A label-free signal can
  separate fixable from unfixable points, but it says which points are fixable, not
  what class they belong to.
- **The recoverable points' true class is invisible to every global signal.** The
  true class sits at rank 3.7-4.8 in the clean-prototype ordering, and a linear
  probe gets only 5-8% on those points. No global classifier can name them.
- **The local structure can.** The recoverable set clusters by class in the 128D
  space (kNN true-label agreement 0.76-0.95). The kNN vote exploits that local
  structure, which is why gating (which only selects) and reweighting (which only
  scales) fail while a local consensus reassignment recovers some of it.

What a TTA method should learn from the failure modes:

- Spend capacity on "what class", not "which points".
- The label source feeding the local vote is the ceiling. The kNN vote is only as
  good as the labels it reads, which is why iterating the vote over the current
  pseudo-labels compounds error instead of adding information.
- Only reassign the points you are confident are wrong; reassigning correct points
  degrades the healthy majority.
- A better method needs a label source that is locally more accurate than the
  linear probe.

---

## 6. Order of work (current state)

1. The 7-class evaluation map is adopted, using the existing 17-class-trained
   encoders (no retraining needed). The 14-class middle ground is closed: it is
   strictly worse because it keeps the fragile classes in the metric. Background
   results are tracked in the seven-class iterations doc.
2. The encoder thread is the standing target. The fog and crosstalk oracle ceiling
   under the 7-class map is ~0.15 / ~0.27 and is encoder-independent. The isotropy
   comparison trains DGLSS, DGLSS++ and our method at equal budget to measure
   whether any regime pushes the space below our baseline or moves the ceiling.
3. The TTA thread is bounded. The label-free gated update is flat in full-coverage
   mIoU, and the oracle ceiling is the wall only true labels cross. Any remaining
   TTA work is subordinated to the ceiling the encoder sets.
4. Balanced update allocation remains planned, to be engaged if a regime produces
   headroom the TTA thread can harvest.
5. Prototype adaptation becomes viable only if the encoder thread raises the
   ceiling.
