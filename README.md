# Robust Feature Pretraining for Adaptive Hyperdimensional Prototypes: from Active Learning to Label-Free LiDAR Test-Time Adaptation

---

## 1. What we established

The ultimate goal is **label-free adaptation for prototype updates**: at
deployment, the decoder should update its prototypes from the corrupted stream
without any labels. The current method is **cov-shift DGLSS++**, a feature
extractor that fixes the corruption collapse at the source, paired with a
**linear-probe decoder on the HDC code** that recovers the structure the
distance-to-prototype rule throws away (Section 2). The label-free path then
covers the healthy conditions and crosstalk, and the remaining conditions fall
back to an **active learning framework**: the uncertainty signal picks a small set
of the hard-condition points, a small budget of labels is spent on exactly those,
and the prototypes are updated from them. The robust encoder is what makes both
the label budget tiny and the resulting update ceiling high.

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

The method rests on three pillars, presented in the order the paper builds them:
the robust extractor (Pillar 1) produces a feature space that survives corruption;
the label-free TTA method (Pillar 2) raises mIoU on the conditions where that
representation survives; and where the corruption shifts the features too far for
any label-free update to recover (fog), a detection mechanism hands off to the
backprop-free active learning framework (Pillar 3), which spends a small label
budget to convert the surviving cluster structure into prototypes.

1. **Robust feature extractor pretraining: cov-shift DGLSS++** (Section 2).
   The DGLSS++ consistency framework with a covariate-shift normalization that
   fixes the collapsed conditions (fog/crosstalk) at the source, and a
   linear-probe decoder on the HDC code that recovers the structure the
   distance-to-prototype rule throws away.

2. **Label-free test-time adaptation** (raises mIoU on the healthy conditions,
   Section 3). A prototype update that works wherever the representation
   survives: on the healthy conditions and crosstalk it reaches the labeled
   ceiling with no labels at all.

3. **Backprop-free active learning** (the fill-in for the conditions TTA cannot
   recover, Section 4). When a mechanism detects that the corruption has shifted
   the features too far for the label-free update to close the gap to the
   supervised ceiling, query one point per dense per-class cluster under a very
   strict label-or-don't gate and re-estimate the prototypes from the labeled
   cluster representatives. The balanced allocation of the label budget across
   classes is part of this pillar.

This is the pivot from the earlier framing: the paper's narrative is that TTA is
the method that works where it works (the healthy conditions and crosstalk), and
active learning is the completion that handles the conditions TTA cannot (engaged
by a detection signal, not bolted on as a variant). The deployment consequence is
that the label budget is preserved for exactly the conditions that need it.

---

## 2. Pillar 1: Robust feature extractor pretraining

The current method has two parts: the **cov-shift DGLSS++ encoder** (fixes the
corruption collapse at the source) and the **linear-probe decoder** on the HDC code
(recovers the structure the nearest-centroid rule throws away).

### The encoder: cov-shift DGLSS++

The extractor is the DGLSS++ consistency framework adapted to a
corruption-robustness target, with the cov-shift normalization layered on top. The
DGLSS++ core has three pieces, each doing a distinct job:

- **GMSIFC + LSCC consistency stack**: GMSIFC aligns cross-view features and LSCC
  enforces per-cell class-correlation consistency. Ablations show GMSIFC is what
  makes label-free TTA work, and LSCC is what keeps the representation decodable.
- **Corruption-targeted augmented view**: replaces the sparse beam-drop view with a
  fog/crosstalk-targeted one (depth jitter + density sparsity + fake-return
  injection), so the constraints learn the corruptions that collapse the minority
  classes, not just sensor sparsity.
- **Decoupled supervised-contrastive pull (SupCon)**: pulls the corrupted view's
  points toward their clean class anchors, inverting the majority-class polarization
  plain DGLSS++ develops at scale (rho(freq, feat_cos) +0.48 -> -0.49).

**The cov-shift normalization** (the piece that closes the collapsed conditions)
resolves the anchoring trade-off without anchoring at all: instead of pulling
corrupted features toward clean (which raised crosstalk TTA but erased fog's
recoverable ceiling), it fixes the covariate-shift statistics directly via per-scan
input normalization restricted to the statistics-shifted channels (range/remission)
plus internal InstanceNorm. It is the first extractor to raise BOTH the labeled
ceiling and the label-free TTA on BOTH fog and crosstalk.

**Zero-shot HDC-zs per condition** (isotropy pipeline, cov-shift ep-10/ep-21 vs the
DGLSS++ reference):

| condition | DGLSS++ | Cov-shift (ep-10) | Cov-shift (ep-21) |
| :--- | :--- | :--- | :--- |
| fog | 6.8% | **20.1%** | 18.5% |
| crosstalk | 11.5% | 39.5% | **41.9%** |
| snow | **39.6%** | 37.7% | 38.6% |
| wet_ground | **48.3%** | 35.8% | 33.3% |
| incomplete_echo | **44.9%** | 40.6% | 40.0% |
| beam_missing | **50.6%** | 44.3% | 44.5% |
| motion_blur | **50.2%** | 44.2% | 44.6% |
| cross_sensor | **43.4%** | 36.1% | 38.8% |
| **mean (8 corrupted)** | 36.9% | **37.3%** | **37.5%** |

The cov-shift model has the best fog and crosstalk zero-shot of any extractor (fog
20.1% / 18.5% vs DGLSS++ 6.8%; crosstalk 39.5% / 41.9% vs 11.5%) and the best
8-condition mean (37.3% / 37.5% vs 36.9%), at the cost of the healthy conditions
sitting 2-15 points below DGLSS++ (wet_ground 35.8% / 33.3% vs 48.3%): the
normalization trades some healthy-condition headroom for the large fog/crosstalk
gains. ep-10 is the better fog checkpoint (20.1% vs 18.5%), ep-21 the better
crosstalk (41.9% vs 39.5%). Clean HDC mIoU: DGLSS++ 53.0%, cov-shift 47.2%.

**Labeled ceiling (HDC-oracle) per condition** (frozen-ceiling harness, the
recoverable bound from re-estimating prototypes with true labels):

| condition | DGLSS++ | Cov-shift (ep-10) | Cov-shift (ep-21) |
| :--- | :--- | :--- | :--- |
| fog | 15.1% | **21.4%** | 20.2% |
| crosstalk | 21.4% | 38.9% | **39.8%** |
| snow | **41.0%** | 38.8% | 39.8% |
| wet_ground | **51.4%** | 40.5% | 36.7% |
| incomplete_echo | **44.8%** | 40.1% | 39.3% |
| beam_missing | **50.6%** | 44.2% | 44.8% |
| motion_blur | **50.3%** | 44.0% | 44.6% |
| cross_sensor | **45.1%** | 38.5% | 39.4% |

The cov-shift ceiling is the highest on fog and crosstalk (21.4% / 39.8% vs DGLSS++
15.1% / 21.4%) and the lowest on the healthy conditions. The healthy-condition
ceiling loss is the cov-shift trade: normalizing the input statistics lifts the
collapsed conditions but slightly compresses the healthy ones.

### The decoder: linear probe on the HDC code (C10)

The distance-to-prototype rule is the binding constraint, not the extractor. On the
frozen cov-shift features, replacing nearest-centroid cosine ("R1") with a linear
probe fit on the HDC code itself ("R4") recovers the healthy-condition ceiling the
centroid rule throws away (C10 decision-rule diagnostic):

| condition | cov-shift zs R1 | cov-shift zs R4 | cov-shift ceil R1 | cov-shift ceil R4 | DGLSS++ zs R4 | DGLSS++ ceil R4 | Robust zs R4 | Robust ceil R4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog (ep-10) | 29.2% | **32.2%** | 30.1% | **36.9%** | 12.9% | 17.0% | 13.7% | 17.3% |
| crosstalk (ep-10) | 39.2% | **47.7%** | 39.4% | **49.1%** | 19.0% | 22.4% | 19.3% | 23.2% |
| snow (ep-10) | 39.1% | **48.2%** | 39.2% | **49.5%** | 18.9% | 19.9% | 20.8% | 21.5% |
| wet_ground (ep-10) | 24.5% | **29.7%** | 25.4% | **41.9%** | 12.4% | 19.9% | 11.7% | 22.3% |
| fog (ep-21) | 27.4% | **31.3%** | 28.1% | **35.5%** | — | — | — | — |
| crosstalk (ep-21) | 39.3% | **45.1%** | 39.5% | **46.2%** | — | — | — | — |
| snow (ep-21) | 37.9% | **45.7%** | 38.1% | **46.5%** | — | — | — | — |
| wet_ground (ep-21) | 25.4% | **30.0%** | 26.5% | **41.1%** | — | — | — | — |

The cov-shift columns are the FULL-DATASET numbers: every point of every frame
of KITTI seq 08 (~4k frames, ~300M points/condition) streamed through the frozen
extractor, with the zero-shot fit on a 200k clean reservoir and the ceiling on a
400k corrupted-pool reservoir (spectral-exact ridge; `al_full_dataset_diag.py`).
The DGLSS++ and Robust DGLSS++ columns are the same full harness on their
checkpoints (single full runs, no ep-10/ep-21 split): the cov-shift extractor
beats both by 0.17-0.30 on fog/crosstalk and 0.24-0.30 on the healthy conditions
at the R4 ceiling. The earlier 100-frame harness over-estimated the headroom:
the full-scale gaps are fog +4.7, crosstalk +1.4, snow +1.3, wet_ground +12.2
(ep-10).

The linear-probe decoder raises the ceiling over distance-to-prototype on every
condition (fog 30.1->36.9% ep-10; wet_ground 25.4->41.9%). The zero-shot gain
is small because the frozen clean-fit probe and the frozen prototypes both start
from clean structure; the ceiling gain is the recoverable structure the centroid
rule throws away when the prototypes are re-estimated on the corrupted pool. The R4
probe is fit on labeled data (clean for zero-shot, pool for the ceiling), so the
label-free version is the frozen clean-fit probe (R4-zs), which at 0.45-0.48 healthy
(crosstalk 47.7, snow 48.2) already beats the R1 oracle (39-40) at zero-shot. The
direction: keep the cov-shift extractor and make the decoder a learned boundary on
the code.

**Efficiency.** The probe's training (fit/refit) is the cost: ~40-50x slower than
building prototypes (77-97s vs 2s per fit; a per-condition pool refit is 112-176s vs
0.02-0.04s), but inference is essentially equal (0.22-0.23s vs 0.18s per 100k
points). Improving the probe's training efficiency (e.g. the closed-form
accumulate-and-solve update in
[`docs/lin_probe_updates/tta_iterations.md`](docs/lin_probe_updates/tta_iterations.md))
is a later step.

The cov-shift extractor's development, the healthy-condition ceiling-loss
diagnostic, and the decision-rule finding are tracked in
[`docs/cov_shift/cov_shift_iterations.md`](docs/cov_shift/cov_shift_iterations.md).

---

## 3. Pillar 2: label-free test-time adaptation

Test-time adaptation re-estimates the decoder from the corrupted stream with no
labels. The story has three chapters: what the previous (prototype) method could
do, what the linear-probe decoder changes, and why the label-free route is
fundamentally closed in this space (Section 3.2), which is what pushes the
recoverable headroom to the active-learning handoff (Pillar 3).

### 3.0 What the previous method saw

The original decoder was **distance to the class-mean prototype** (nearest-centroid
cosine). Under that decoder the label-free picture was:

- **On the healthy conditions, label-free TTA is sufficient.** Snow, wet_ground,
  motion_blur, beam_missing, incomplete_echo and cross_sensor sit at or near their
  *prototype* ceiling with the label-free update, so no labels are needed there.
- **Crosstalk is closed at the frozen level** (zero-shot ~= prototype ceiling ~=
  naive): the cov-shift extractor, not TTA, fixed it.
- **Fog is the remaining gap** (naive 21.0% vs prototype ceiling 26.1%): the
  prototype's recoverable ceiling is low and label-free TTA reaches most of it.

But the *prototype* ceiling is low on fog/crosstalk (26.1% / 46.1%): the centroid
rule cannot express the rotation that corruption induces. This is the assignment
wall: a label-free signal can say *which* points are wrong, but not *what class*
they belong to. A learned decoder reopens this as large recoverable headroom.

A design rule: prior correction and prototype updates must not share a pathway.
The prior is an inference-time constant that shifts decision boundaries; it does
not move prototypes by itself. But if prior-corrected pseudo-labels feed the
updates, the bias steers the prototypes and the drift compounds. The prediction
pathway may use the prior-corrected score; the adaptation pathway must not.

### 3.1 What the linear probe in HDC space adds

The linear probe (fit on the HDC code) raises the labeled ceiling far above the
prototype rule on every condition, because it learns per-coordinate weights and a
boundary rotation that the centroid rule cannot express. The probe is the ridge
solution on the binarized code $X \in \{-1,+1\}^{n \times d}$:

$$
W = (S + \lambda I)^{-1} T, \qquad S = X^{\top} X, \qquad T = X^{\top} Y,
$$

where $Y$ is the one-hot (or soft) label matrix of the pool. It is fit with the
**Nystrom-warm-started matrix-free CG** update (see the efficiency section below),
and the decoder is cosine to the learned $W$.

**The efficiency <-> accuracy table** (wet_ground ep-10, ceiling mIoU vs update
speed), for the baseline and the probe-update variants:

| method | update pts/s | ceiling mIoU | note |
| :--- | :--- | :--- | :--- |
| R1 prototype (baseline) | 1.9M | 42.4% | no probe gain; the old decoder |
| full matrix probe (LR) | ~1-2k | 67.1% | the ceiling; update too slow |
| Nystrom sketch (m=1000) | 2.0M | 54.7% | fast, lower ceiling |
| matrix-free CG-20 | ~0.64M | 62.0% | accurate but 20 iters |
| **Nystrom warm-start + CG-8** | **~1.5M** | **61.6%** | **best accuracy/efficiency** |

The winner is **Nystrom warm-start + matrix-free CG**: the Nystrom sketch (m=1000)
provides a cheap approximate second-order geometry, then CG-8 finishes in the full
10000-d space via the matrix-free matvec

$$
Sv = X^{\top}(X v),
$$

never building the $d \times d$ matrix $S$. CG-8 from the Nystrom
start reaches 0.62 wet_ground / 0.38 fog (ep10), essentially the plain CG-20
accuracy, in ~8 iterations instead of 20 (0.034s, ~1.5M pts/s). The warm start is
the key: CG-5 from Nystrom already matches CG-20 from scratch. The prototype is NOT
a good warm start; the Nystrom sketch is. The full sweep and the failed alternatives
(first-order separators, low-margin coresets, sparse covariance) are in
`docs/lin_probe_updates/tta_iterations.md` (Iterations 4-7).

**Inference (decode) speed is unchanged** by the probe: cosine to the learned W
runs at the prototype's decode rate (~0.20-0.29M pts/s vs 0.29-0.31M). The
efficiency cost of the probe is in the UPDATE, not the deployed decode.

### 3.2 Why unsupervised TTA is so hard in this space

The Iteration 9-12 closure is not a failure of specific update rules; it is forced
by measurable properties of the binarized code space and the frozen probe. Each
property below is measured, and each kills a family of label-free methods:

1. **The frozen probe's errors are systematic, not noise.** The wrong
   pseudo-labels contribute ~equal magnitude to the update (||W_wrong||/||W_correct||
   ~= 0.8-1.1) and ANTI-align with the oracle rotation (cos(W_wrong, W_oracle) < 0
   on every condition). Any supervised refit, gated, weighted, soft, or
   two-stage, inherits the frozen probe's own mistakes. Confirmation bias is
   structural here, not a gate deficiency (Iterations 9-11).
2. **The rotation is in the decision rule, not the geometry.** The top eigenspaces
   of S_clean and S_target coincide (spectral overlap 0.995-1.000) and the spectra
   barely change, yet cos(W_zs, W_oracle) is small (0.05-0.19 fog, 0.42-0.59
   crosstalk). The corruption does not rotate the pool's second-order statistics;
   the boundary change lives in how the same geometry is labeled. This is why
   Procrustes, CORAL, and whitening, which realign S, have nothing to align
   (Iteration 12).
3. **Confidence is anti-correlated with what the update needs.** The points with
   the highest influence on the oracle rotation are the LOW-confidence ones
   (Spearman -0.40 to -0.64); wrong points carry ~2x the influence of correct
   ones; and the max softmax is < 0.5 for every corrupted-pool point, so no
   absolute confidence threshold even exists. Confidence-gated supervision
   structurally selects the least useful points (Iteration 11).
4. **Per-class reliability is wildly uneven.** Pseudo-label precision spans
   0.08-0.83 across classes (0.18-0.97 at the top confidence quantile); no global
   threshold is right at the class level, and class-conditional gates fail too
   because of properties 1 and 5 (Iteration 11).
5. **Purity cannot buy coverage.** Even a perfect-purity gate (correct-only
   labels, oracle-gate precision 1.0) reaches only 0.29-0.53 vs the oracle's
   0.38-0.61: the correct points are a biased, low-coverage subset of the pool.
   The rotation needs labels spread over the whole geometry; clean-but-sparse
   supervision is structurally insufficient (Iteration 11).
6. **The second-order solve amplifies the contaminated half.** The update
   $$
   W = (S + \lambda I)^{-1} X^{\top} Y
   $$
   applies the inverse geometry to the label statistics; with $S$ fixed and
   correct, the wrong-label contribution is amplified by the same inverse that
   would recover the rotation. S=all, T=gated never beats S=all, T=all at scale
   because the geometry faithfully represents the pool, and the pool's
   pseudo-labels are what is wrong (Iteration 11).
7. **The binarized code fixes the norm, making every shift angular.**
   Sign-binarization removes scale; the covariance's dominant directions are
   shared between clean and corrupted (property 2), so corruption expresses as a
   change in the class-conditional means and angles, exactly the part of the
   statistics a covariance-only method ignores (Iterations 10, 12).

The only supervision that resolves all seven properties is a true label on a
covering point (Pillar 3): it fixes property 1 (no pseudo-label), 3-4 (query by
influence, not confidence), 5 (labels spread per cluster), and 6-7 (the labeled
ridge is the oracle construction itself).

### 3.3 Label-free TTA is bounded at the frozen level

The linear probe's LABELED ceiling is high relative to zero-shot (36.9% fog,
41.9% wet_ground at full scale), but its
label-free ceiling is the FROZEN decoder: Iterations 9-10 showed that neither
gating nor weighting pseudo-labels lets the probe update beat the frozen decode,
because the 33-55% wrong pseudo-labels contaminate any supervised refit, and
Iterations 11-12 showed the same closure for the S/T decomposition and for the
pure-geometry routes (Section 3.2). So under
the linear probe, **label-free TTA does not reach the (higher) probe ceiling on any
condition: it is bounded at what the frozen decoder already achieves. This is a
different statement from the prototype story (where label-free TTA reached the
prototype's lower ceiling on the healthy conditions).

Zero-shot (frozen decoder), the current label-free TTA, and the labeled ceiling for
the current setup (cov-shift ep-10, FULL-dataset harness: every point of every
frame of seq 08). TTA is the FROZEN DECODER: the Iteration
9-12 finding is that no label-free update beats it, so the label-free column
equals zero-shot, and the recoverable headroom is what the AL handoff closes:

| condition | zero-shot (frozen) | label-free TTA | label ceiling (probe) | AL-closeable gap |
| :--- | :--- | :--- | :--- | :--- |
| fog | 32.2% | = zero-shot | **36.9%** | +4.7 |
| crosstalk | 47.7% | = zero-shot | **49.1%** | +1.4 |
| snow | 48.2% | = zero-shot | **49.5%** | +1.3 |
| wet_ground | 29.7% | = zero-shot | **41.9%** | +12.2 |

The gap (label ceiling - frozen) is the recoverable headroom that only true labels
close, which is the active-learning handoff (Pillar 3): a small true-label budget
(one label per dense cluster) re-estimates the probe from labeled points and
converts the headroom into prototypes. At full scale the gaps are small on every
condition except wet_ground and fog: the extractor and the R4 decoder already
close most of the recoverable headroom, and the AL budget is only worth spending
where the gap is real (Section 4.6).

The cov-shift method's per-condition source harnesses are in
`docs/cov_shift/cov_shift_iterations.md` (Iterations C1-C5, C11); the label-free
probe-update closure, the S/T decomposition, and the geometric closure are in
`docs/lin_probe_updates/tta_iterations.md` (Iterations 3-12).

### 3.4 The detection signal that triggers the handoff

The detection signal that ranks correct from wrong points (density / norm / fusion)
decides when the label-free path is insufficient: if the label-free update's
gap-closed is below a threshold on a condition or a cluster, i.e. the label-free
update cannot close most of the gap to the labeled ceiling, then the condition falls
back to the active-learning framework. The TTA machinery is kept as the efficiency
lever, not the whole answer.

---

## 4. Pillar 3 (primary): backprop-free active learning (the fill-in)

The active-learning framework closes the residual gap on the conditions the
label-free TTA cannot handle. The design is built entirely from measured facts: it
exists because a label-free signal can say *which* points are wrong but not *what
class* they are, and it is viable because the corrupted points are still densely
packed in per-class clusters, so one label per cluster can label the cluster.

The measured facts line up for it: the recoverable set of points is identifiable
label-free, the recoverable points cluster by class in the local feature structure,
and the full-label oracle is well above what any label-free update reaches. A small
budget of labels, spent on exactly the ranked hard points, updates the prototypes
far more effectively than any label-free signal. The robust encoder is what makes
the selection signal informative and the ceiling worth reaching.

### 4.1 The structure that makes it cheap: dense per-class clusters

The corrupted points are not a noise floor; they are still clustered by class.
The evidence:

- Even on the worst conditions, a corrupted point's nearest neighbor in the
  corrupted stream is the same class 75-87% of the time (Corruption Atlas
  corrupted 1-NN purity: fog 75.1%, crosstalk 86.6%, on the *un-pretrained*
  model).
- The labeled oracle (re-estimating prototypes from corrupted points with true
  labels) recovers far more than any label-free update (full-scene mIoU Fog
  +6.5, Crosstalk +14.2 over zero-shot; section 5.3). The label ceiling is
  11-25% per extractor on fog/crosstalk, and the best label-free TTA sits 2-9
  points below it (section 5.4).
- The failure is the *assignment* (which cluster is which class), not the
  *packing* (that the clusters exist). This is exactly why the labeled ceiling is
  much higher than the TTA ceiling: labels convert the surviving cluster structure
  into prototypes; label-free signals cannot, because they cannot name the
  clusters.

### 4.2 The mechanism: query one point per cluster, strictly

Backprop-free (no extractor fine-tuning, only prototype re-estimation):

1. **Cluster.** Group the corrupted points into tight per-class clusters in the
   128D space (the packing that the 1-NN purity measures).
2. **Rank.** Use the detection signal (density / norm / fusion) to rank clusters by
   how much their labels would move the decode (the clusters in the uncertain
   regions where zero-shot is wrong).
3. **Query strictly.** A very strict label-or-don't gate decides whether a query
   is worth the budget: a stronger version of the update/don't-update gate from
   the TTA thread. Query only when the cluster is tight (the point is a
   representative centroid), the detection signal is confident the cluster is
   wrong, and the cluster is not already decodable label-free. The gate must
   waste almost no labels.
4. **Label the cluster.** One label on the representative point assigns the class
   to the rest of the cluster (the packing guarantees this holds for most of it).
5. **Re-estimate.** Update the prototypes from the labeled cluster representatives
   with the per-point weighting that we know beats zero-shot (the oracle
   operator). Optionally freeze the saturated majority classes and bound the
    budget per class (the balanced-allocation rule from Pillar 3).

The leverage is that the label budget scales with the number of clusters, not the
number of points: each queried cluster is worth many correct labels, so a small
budget recovers most of the oracle gap.

### 4.3 When it activates

Active learning is the fill-in, engaged by a detection mechanism (Section 3.4):
run the extractor, attempt the label-free TTA path, and activate active learning on
exactly the conditions/clusters where the label-free gap-closed is below a
threshold (or where the TTA-to-supervised gap is not >90% closed). The measurements
say this is the default on the collapsed conditions (fog, crosstalk), where the
assignment wall caps any label-free update well below the labeled ceiling, while
the healthy conditions stay on the label-free path (Section 3). It spends the small
label budget on exactly the clusters the label-free thread cannot name, and converts
the surviving cluster structure into prototypes. It is the only path that closes the
residual gap, because it is the only one that supplies the missing class labels.

### 4.4 Why the label budget is not free (measured)

The cluster packing is real (1-NN purity 0.51-0.77, intra vs inter cosine
0.62-0.70 vs 0.004-0.055), but the update makes the label cost scale with the
MASS, not the clusters. Three measured mechanisms:

1. **The gap is the missing mass.** The probe update is a sum over labeled
   points: the labeled solution differs from the oracle exactly by the
   contribution of the points NOT labeled,
   $$
   W_{\mathrm{labeled}} = W_{\mathrm{oracle}} -
   (S + \lambda I)^{-1} \sum_{i \notin L} x_i y_i^{\top}.
   $$
   Scaling $W$ does not change the decode, but DIRECTION does, and
   a partial sum's direction is only right when the labeled subset's per-class
   sums are proportional to the full per-class sums. The labeled ceiling is
   only reached by labeling ~40-60% of the pool (measured across every code
   dimension from 128-d to 10k-d and every expansion scheme tried).
2. **Influence selection picks the boundary, not the bulk.** The query signal
   that best ranks the points worth labeling (influence) anti-correlates with
   confidence (-0.40 to -0.64) and the wrong points carry 2x the influence --
   it concentrates labels on outlier/boundary points, whose direction is not
   the class-mean direction. Reaching the mean direction requires the bulk of
   each class, and the pool is class-imbalanced ~40x.
3. **The boundary is pathologically sensitive while the means barely move.**
   The corruption shifts the class means only ~5-10 degrees (unlabeled-cos
   0.92-0.99), yet cos(W_frozen, W_oracle) is 0.05-0.19 on fog. The classes
   are fat blobs (intra-cos 0.62-0.70) whose means are ~89 degrees apart, so a
   small mean shift flips which side of the boundary the bulk falls on. Only
   the full-mass oracle T gets the mass-weighted boundary right.

So the label cost is not a property of a specific mechanism; it is a property
of the ridge's missing-mass term, which is dimension-independent (the class-mean
estimation SNR is set by the 128-d intrinsic geometry, not the code dimension).
This mass requirement applies to the FULL-SPACE probe refit ($W_{pseudo}$ on the
labeled points); the low-rank residual (Section 4.6) sidesteps it by restricting
the update to the $r=8$ $U$ subspace, which is why $56+500$ labels now close
23-91% of the closeable gap where the full probe collapsed ($-0.22$ to $-0.48$,
Iterations C30-C31).
The open question is whether the shift structure can synthesize the missing
mass: the corrupted class means ARE predictable from the clean means plus a
per-class shift (partially shared across classes, pairwise-cos 0.2-0.37,
estimated from 2-4 labeled classes), so the cheap path may be to estimate the
shift from a few labels and fit the probe on the shift-corrected means rather
than label the mass.

### 4.5 The mechanism that works: a sensitivity-bounded residual update

The AL search measured, in order: the packing is real but label propagation
fails (Iterations 0-2), the missing-mass argument is real but the class means
ARE estimable (8-32 points per class reach cos 0.95-0.99, Iteration 7), the
T-synthesis chain is exact, and the soft-mass counts/assignments are both
wrong (8D WEAK, Iteration 8). The final bottleneck is the DECODER UPDATE
itself: the inverse covariance $(S + \lambda I)^{-1}$ amplifies small
unavoidable $T$ errors into large $W$ errors (the ridge-relevant error is
4-6x; the ordinary Euclidean quality of $T$ stops mattering once it is past
~0.7 cosine).

The fix is a sensitivity-bounded parameterization: keep the update a
LOW-RANK residual on the frozen decoder instead of a full-space refit. The
current deployed form (Iterations C30-C31, Section 4.6) is
$W_{\mathrm{new}} = W_0 + U_r C$ with $r=8$, where $U_r$ is the top-$r$
left singular subspace of $R = W^* - W_0$ (oracle $U$) and
$C = (U^{\top} X^{\top} X U)^{-1} U^{\top} X^{\top} (Y - X W_0)$ is fit on the
$56+500$ labeled points. The rank restriction replaces the full inverse: it is
never worse than frozen ($C = 0$ reproduces $W_0$ exactly), it is stable where
the full probe collapses ($-0.22$ to $-0.48 \rightarrow +0.01$ to $+0.12$,
Iteration C31), and it recovers the oracle at $r = 8$ on every extractor
(Iteration C20). The earlier Iter-10 form was the fractional-residual
$W_{\beta} = (S + \lambda I)^{-\beta} T_{\mathrm{hat}}$ with $\beta \approx
0.75$ and $\eta \approx 0.1$, which achieved the same sensitivity bound with
64-72 labels; C30 showed the fractional gain shaping is superseded by the
low-rank restriction ($\gamma \approx 0$, $\eta = 1.0$ best), so the rank
restriction is the current mechanism and the class-mean labels are replaced by
the $56+500$ bank (Section 4.6).

The frozen vs labeled-ceiling vs method table (ep10; the labeled ceiling is
the SPECTRAL-exact oracle, the matrix-free CG-8 approximation previously
used under-converges and underestimates the true ceiling by 0.04-0.06):

| condition | frozen (zero labels) | method (64-72 labels) | labeled ceiling |
| :--- | :--- | :--- | :--- |
| fog | 20.1% | 26.2% | 41.6% |
| crosstalk | 39.5% | 50.3% | 58.7% |
| snow | 37.7% | 44.2% | 51.1% |
| wet_ground | 35.8% | **44.7%** | 67.0% |

The method beats frozen on wet_ground (+8.9) and ep21-fog (+1.6) and moves
substantially toward the ceiling on every condition, the first label-
efficient update in the AL thread to do so. The update is a spectral solve
(~4s) plus matrix-free decode; the remaining deployment step is replacing the
oracle-informed class counts with the source-count prior (Iteration 8F).

This table is the fractional-residual recipe measured on its own harness
(older 100k-cap clean fit; the frozen column is the under-fit zero-shot).
The current random-memory-bank baseline on the README-accurate harness is in
Section 4.6.

**Efficiency and the path to a cheaper solve.** The low-rank residual update
itself is cheap: $C$ is an $r \times r$ solve ($r = 8$), the decode is the
matrix-free linear probe, and inference is still the single $W_{res}$ (no
bank at test time). The cost is in the $U$-basis: the oracle $U$ used here
comes from the SVD of $R = W^* - W_0$, which requires the full pool ridge fit
(a spectral solve, ~4s per condition). Three measured facts say this is
avoidable, and the eventual cheaper method must preserve them:

1. **The residual is low-rank.** Effective rank of $R = W^* - W_0$ is 4-5 and
   $mIoU(W_0 + R_8) = $ oracle on every extractor (Iteration C20), so a
   truncated SVD over the top ~8 directions is the full mechanism.
2. **The oracle $U$ is the bottleneck, not the rank.** Oracle $U$ recovers
   the ceiling ($+0.05$ to $+0.14$ at $k=8$), but every estimated $U$
   (est-basis SVD of $W_{sub} - W_0$, pool-covariance, code-shift) collapses
   ($-0.21$ to $-0.61$, Iterations C21-C22): the cheap $U$ is the open
   problem, and the bank's $G$-quality (not its 1-NN mIoU) is the lever to
   improve it.
3. **The per-condition cost is dominated by the pool fit for oracle $U$ and
   the $56+500$ bank feature extraction.** The bank's labels are the only
   supervision; the deployment step is a $U$ estimator that reaches the
   oracle basis without the full labeled pool (the next step, Section 6.1).

### 4.6 Current AL baseline: the random memory bank ($W_{res}$) on the conditions with a gap to close

The current deployed AL baseline on the FULL-dataset $R4$ harness (every point of
every frame of seq 08, spectral-exact solve, $200$k-point clean $W_0$ fit, $400$k
pool, $56$ true labels $k=8$ per class $+$ $500$ random bank points, low-rank
residual $W_{res} = W_0 + U_r C$ with $r=8$ oracle $U$; `run_al_full_dataset.sh`).
Only the conditions with a measurable closeable gap are shown; beam_missing
($-0.001$) and motion_blur ($+0.006$) have nothing to close:

| condition | zero-shot $W_0$ | AL $W_{res}$ ($56+500$ random) | $\Delta$ | ceiling $W^*$ | closeable gap | gap closed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 32.2% | 32.7% | +0.5 | 36.9% | +4.7 | 11% |
| wet_ground | 29.7% | 34.1% | +4.3 | 41.9% | +12.2 | 35% |
| cross_sensor | 42.0% | 43.0% | +1.0 | 44.6% | +2.6 | 38% |
| crosstalk | 47.7% | 48.9% | +1.1 | 49.1% | +1.4 | 79% |
| snow | 48.2% | 46.6% | -1.6 | 49.5% | +1.3 | 0% (noise) |

The random bank $W_{res}$ is positive on every condition with a gap to close
except snow (inside noise), and closes 11-79% of the closeable gap. The
$U$-basis estimation, not the harness, is the bottleneck (oracle $U$ recovers
the ceiling, estimated $U$ collapses, Iterations C21-C22). The next steps
(Section 6.1): (1) improve the gap closed by AL on fog, wet_ground,
cross_sensor, crosstalk, snow; (2) build a label-free gauge of whether a
condition has anything to close, so the budget is not spent on beam_missing /
motion_blur / snow-borderline.

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
| **Adaptation (Pillar 2)** | Label-free gated prototype updates, used where the label-free path is sufficient (the healthy conditions); engaged unless the detection signal (Section 3.4) says active learning is needed | 
| **Active learning (Pillar 3)** | Backprop-free fill-in: query one point per dense per-class cluster under a strict label-or-don't gate, re-estimate prototypes from the labeled cluster representatives (Section 4) |

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

### 5.3 The labeled-prototype oracle: the target a TTA method must chase

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

The oracle that recovers the gap is the *weighted* one. A naive perfect-label
re-estimation (re-estimating the prototypes from the degraded corrupted features
with unit weights) sits below zero-shot on every condition, worse than keeping the
frozen clean prototypes (the "prototype-level adaptation has no headroom" finding).
The table above uses the right per-point weighting, which is what actually beats
zero-shot (Fog 16.6%, Crosstalk 26.2% full-scene mIoU). The prototype decoder is
therefore not exhausted: with the right per-point weighting, re-estimated
prototypes beat the frozen clean ones on every condition, and substantially on the
collapsed ones. Every label-free TTA variant we tried fails to reach it. The
problem is not "drop the artifacts"; it is "estimate the weights the oracle would
assign" without labels.

### 5.4 Test-time adaptation and the labeled ceiling, per feature extractor

The robust encoder roughly doubled Fog linear separability (23.6% to 49.4% linear
probe) and, evaluated frozen against the previous baselines, improves mIoU on every
condition (mean 13.2% to 36.4%); the Fog and Crosstalk gains hold in both point
accuracy and mIoU. The accuracy-to-mIoU gap is large because mIoU averages
per-class IoU and is dominated by the rare classes, which collapse under
corruption.

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
(0.20 fog vs 0.58 micro); at scale it matches naive EMA and stays below BN
alignment. Full tables and interpretation in the iterations doc (Iteration 4).

## 6. Order of work (current state)

The narrative: extractor -> label-free TTA (healthy conditions + crosstalk) -> the
residual gap (fog) -> active learning (the fill-in the detection signal engages).
The order reflects that narrative and the deployment consequence that the label
budget is preserved for exactly the conditions that need it.

1. The 7-class evaluation map is adopted, using the existing 17-class-trained
   encoders (no retraining needed). The 14-class middle ground is closed: it is
   strictly worse because it keeps the fragile classes in the metric. Background
   results are tracked in the seven-class iterations doc.
2. The encoder thread is **cov-shift DGLSS++**: it fixes the collapsed conditions
   (fog/crosstalk) at the source via per-scan covariate-shift normalization, the
   first extractor to raise both the ceiling and the label-free TTA on both
   conditions (Section 2). The decoder is the **linear-probe on the HDC code**,
   which raises the ceiling 1.2-1.8x over distance-to-prototype on every condition
   (C10). The next encoder step is a convergence metric to confirm the ep-10 peak
   is stable.
 3. Label-free TTA (Pillar 2, Section 3) is the method: the label-free update
    reaches the ceiling on the healthy conditions and crosstalk, with the
    detection signal (Section 3.4) deciding where it is sufficient. The TTA
    machinery is the efficiency lever, not the whole answer.
 4. Backprop-free active learning (Pillar 3, Section 4) is the fill-in for the
    conditions the detection signal routes to it: rank the dense per-class clusters,
    query one point per cluster under a strict label-or-don't gate, and re-estimate
    the prototypes from the labeled cluster representatives. This closes the residual
    gap on the conditions that have one (wet_ground, fog at full scale; Section 6.1),
    because it supplies the missing class labels. The AL update itself (the
    $U$-basis estimate) and its online form are open todos (Section 6.1).
 5. Balanced allocation of the label budget across classes is folded into Pillar 3,
    to be engaged once the active-learning updates produce headroom to harvest.

### 6.1 Current state and the next steps (2026-08-19)

The full-dataset baseline (every point of every frame of seq 08, R4 probe,
$200$k clean fit / $400$k pool, `run_al_full_dataset.sh`) is the paper table:
the closeable gaps are fog $+0.047$, crosstalk $+0.014$, snow $+0.013$,
wet_ground $+0.122$ (ep-10). The $56+500$ random bank $W_{res}$ closes $11\%$
of the fog gap and $35\%$ of the wet gap, and is inside noise on the
small-gap conditions (snow $-0.016$). The open todos, in priority order:

1. **Figure out the AL method: estimate $U$ (or a fast, low-label
   alternative).** The $W_{res} = W_0 + U_r C$ machinery works with oracle $U$
   (recovers the ceiling) but every estimated $U$ collapses (C21-C22: est-basis
   SVD, pool-covariance, code-shift). The lever is a $U$-basis estimate from
   unlabeled structure (pool covariance / code-shift / bank-gradient geometry),
   or another update form that is fast and low-label and does not need the
   oracle basis. This subsumes the earlier "improve the gap closed by AL" item:
   the bank's $G$-quality and the $U$-basis estimation are the same lever.
2. **Make the AL update online.** Current AL is batch (pool + bank). Find an
   incremental/streaming form of the low-rank residual update (accumulate the
   top-$r$ residual subspace over time, bounded memory, no stored pool) at a
   very low price in either accuracy or labels required.
3. **Explore the HDC projection $R$.** The code is $\operatorname{sign}(zR)$
   with a fixed random projection; it is a free parameter. Test a better
   projection (learned, class-discriminative, minority-aware) or a dynamic
   projection-pruning mechanism, measured by the linear classifier's $mIoU$.
4. **Get a method that tells whether there is something to close at all.**
   A label-free gauge of the closeable gap (residual norm
   $\|W^* - W_0\|$ or feature-shift strength vs frozen-cosine stability) so
   the label budget is spent only on conditions where the gap is significant:
   beam_missing ($-0.001$) and motion_blur ($+0.006$) have nothing to close,
   and snow's $+0.013$ is borderline; labels should not be spent there.