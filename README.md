# Robust Feature Pretraining for Adaptive Hyperdimensional Prototypes: from Active Learning to Label-Free LiDAR Test-Time Adaptation

---

## 1. What we established

The ultimate goal is **label-free adaptation for prototype updates**: at
deployment, the decoder should update its prototypes from the corrupted stream
without any labels. That goal is currently bounded by a measured wall (a
label-free signal can say which points are wrong, but not what class they belong
to). The current target is therefore one step short of it: a feature extractor
robust enough to work really well inside an **active learning framework**, where
the uncertainty signal picks a small set of the hard-condition points, a small
budget of labels is spent on exactly those, and the prototypes are updated from
them. The robust encoder is what makes both the label budget tiny and the
resulting update ceiling high.

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

### The three pillars (placeholders, to be filled in)

The method is intended to rest on three pillars. Each is a placeholder at this
stage: the general direction is set, the concrete design is still being measured
and will be filled in as the diagnostics land.

1. **Robust feature extractor pretraining** (intended to improve on DGLSS++).
   [to be filled: the pretraining objective and architecture. The goal is a 128D
   feature space that survives fog and crosstalk before the HDC projection, with a
   higher recoverable ceiling than the DGLSS++ baseline currently measured.]

2. **Test-time adaptation, with an adaptively activated active-learning fallback**
   for the particularly bad scenarios. [to be filled: the form of the TTA. The
   measurements so far bound it: the label-free update is flat in full-coverage
   mIoU, and the labeled ceiling is the wall, so a small label budget, activated
   only in the worst conditions, is the current direction.]

3. **Balanced update allocation** (inter- and intra-class balance). [to be filled:
   how the adaptation budget is split across classes and subclusters so the
   majority classes do not consume it all.]

---

## 2. Pillar 1: Robust feature extractor pretraining

[To be filled: the pretraining objective and architecture.] The goal is a 128D
feature space that survives fog and crosstalk before the HDC projection, with a
higher recoverable ceiling than the DGLSS++ baseline currently measured. The
isotropy comparison (DGLSS / DGLSS++ / supcon_vib) and the recoverability
diagnostics are tracked in the robust-iterations doc.

---

## 3. Pillar 2: Test-time adaptation, with an active-learning fallback

The framework has two threads, now evaluated in the 7-class setting. Together they
support a path from the current target (active learning) to the ultimate goal
(label-free prototype adaptation).

The first thread is robust feature-extractor training, and it is the current focus
(Pillar 1). The second thread is adaptive prototype updates when necessary: at
deployment, the distance to the nearest clean prototype decides which points may
update the prototypes, and the decoder re-decodes.

What the measurements bound so far (full-coverage mIoU):

- The label-free gated update is flat. Re-estimating the centroids from the
  distance-confident points reproduces the clean centroids, because the confident
  points already decode correctly and the far points that would move the centroids
  are exactly the ones the gate excludes. This is the assignment wall from the TTA
  iterations: a label-free signal can say which points are wrong, but not what
  class they belong to.
- The only full-coverage gains come from true labels. The perfect-label oracle
  ceiling is about 0.15 on fog and 0.27 on crosstalk, and it is the same for every
  encoder, so only the encoder thread can move it.

The near-term bridge between the threads is active learning. The measured facts
line up for it: the recoverable set of points is identifiable label-free, the
recoverable points cluster by class in the local feature structure, and the
full-label oracle is well above what any label-free update reaches. A small budget
of labels, spent on exactly the ranked hard points, should update the prototypes
far more effectively than any label-free signal, activated only in the
particularly bad scenarios. The robust encoder is what makes the selection signal
informative and the ceiling worth reaching.

[To be filled: the form of the TTA, and how the active-learning fallback is
activated.]

A design rule: prior correction and prototype updates must not share a pathway.
The prior is an inference-time constant that shifts decision boundaries; it does
not move prototypes by itself. But if prior-corrected pseudo-labels feed the
updates, the bias steers the prototypes and the drift compounds. The prediction
pathway may use the prior-corrected score; the adaptation pathway must not.

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

### 5.6 Test-time adaptation and the labeled ceiling, per feature extractor

The three feature extractors (supcon_vib, the DGLSS arm, the DGLSS++ arm), frozen,
with the full-coverage mIoU on the target conditions for zero-shot, the naive EMA
prototype update, the best label-free TTA found (BN-statistic alignment), and the
full-label oracle (the ceiling). The assignment wall holds on all three: a
label-free signal can separate which points are wrong, but not what class they
belong to.

| extractor | condition | zero-shot | naive EMA | best label-free | label ceiling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib | fog | 8.2% | 9.2% | **9.6%** | 11.0% |
| supcon_vib | crosstalk | 11.9% | 13.6% | **14.7%** | 24.3% |
| supcon_vib_dglss | fog | 7.6% | 10.6% | **10.8%** | 11.6% |
| supcon_vib_dglss | crosstalk | 13.0% | 16.4% | **17.9%** | 21.1% |
| supcon_vib_dglsspp | fog | 10.1% | 12.9% | **13.1%** | 12.7% |
| supcon_vib_dglsspp | crosstalk | 12.4% | 16.1% | **18.6%** | 22.2% |

What this shows:

- **Naive EMA works on every extractor**, not just one: it closes about a third of
  the fog gap on supcon_vib, three quarters on the DGLSS arm, and reaches the fog
  ceiling on the DGLSS++ arm. The label-free update is not flat at this scale; it
  was the weak weight signals (confidence, distance) that were flat, not the
  update itself.
- **The best label-free TTA is BN-statistic alignment on every extractor**, closing
  most of the fog gap and 20-40% of the crosstalk gap. The recent density gate
  (supcon_vib) and norm gate (DGLSS arms), which rank correct from wrong points at
  AUROC 0.84-0.91, are comparable principled weights but do not beat naive EMA on
  the update objective.
- **Fog is largely closable label-free at this scale; crosstalk is the wall.** The
  fog label-free numbers approach or reach the ceiling, while crosstalk label-free
  stays 6-10 points below its ceiling on every extractor, consistent with the
  assignment wall being the binding constraint on crosstalk.

**At medium scale (`--med`, medium DGLSS++ 24 ep/100% data, medium supcon_vib
pretrain, DGLSS arm still micro):** the same battery (`tta_ceiling_diag.py --med`)
gives

| extractor | condition | zero-shot | naive EMA | best label-free | label ceiling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib (med) | fog | 10.1% | 8.4% | **10.7%** | 16.4% |
| supcon_vib (med) | crosstalk | 12.0% | 9.8% | **15.1%** | 26.2% |
| supcon_vib_dglss (micro) | fog | 7.6% | 10.6% | **10.8%** | 11.6% |
| supcon_vib_dglss (micro) | crosstalk | 13.0% | 16.5% | **17.9%** | 21.1% |
| supcon_vib_dglsspp (med) | fog | 9.2% | 11.4% | **12.7%** | 20.0% |
| supcon_vib_dglsspp (med) | crosstalk | 14.1% | 15.0% | **17.4%** | 25.0% |

The wall holds at scale: the DGLSS++ fog rec@3 stays below the random baseline
(0.14 vs ~0.19) and the oracle-vs-LP assignment gap is ~0.01. Medium DGLSS++
beats medium supcon_vib on the frozen labeled ceilings (HDC oracle mean 0.399 vs
0.380), and the norm gate from the small-scale run does not scale its gap fraction
(0.20 fog vs 0.58 micro) — at scale it matches naive EMA and stays below BN
alignment. Full tables and interpretation in the iterations doc (Iteration 4).

---

## 6. Order of work (current state)

1. The 7-class evaluation map is adopted, using the existing 17-class-trained
   encoders (no retraining needed). The 14-class middle ground is closed: it is
   strictly worse because it keeps the fragile classes in the metric. Background
   results are tracked in the seven-class iterations doc.
2. The encoder thread is the current target. The fog and crosstalk oracle ceiling
   under the 7-class map is ~0.15 / ~0.27 and is encoder-independent. The isotropy
   comparison trains DGLSS, DGLSS++ and our method at equal budget to measure
   whether any regime pushes the space below our baseline or moves the ceiling.
3. The active-learning path is next: use the distance signal to rank the hard
   points, spend a small label budget on the ranked recoverable set, and update
   the prototypes from those labels. This is the near-term mechanism that should
   close most of the oracle gap (Fog 16.6, Crosstalk 26.2 in the 17-class metric).
4. The label-free TTA thread is the long-term goal and is bounded: the label-free
   gated update is flat in full-coverage mIoU, and the oracle ceiling is the wall
   only labels cross. It becomes reachable only if the encoder thread moves the
   ceiling.
5. Balanced update allocation remains planned, to be engaged once the encoder or
   the active-learning updates produce headroom to harvest.
5. Prototype adaptation becomes viable only if the encoder thread raises the
   ceiling.
