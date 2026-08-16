# Robust Feature Pretraining for Adaptive Hyperdimensional Prototypes: from Active Learning to Label-Free LiDAR Test-Time Adaptation

---

## 1. What we established

The ultimate goal is **label-free adaptation for prototype updates**: at
deployment, the decoder should update its prototypes from the corrupted stream
without any labels. The current method is **cov-shift DGLSS++** — a feature
extractor that fixes the corruption collapse at the source — paired with a
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
recoverable ceiling), it fixes the covariate-shift statistics directly — per-scan
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
sitting 2-15 points below DGLSS++ (wet_ground 35.8% / 33.3% vs 48.3%) — the
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

| condition | zs R1 | zs R4 | ceiling R1 | ceiling R4 |
| :--- | :--- | :--- | :--- | :--- |
| fog (ep-10) | 23.9% | 23.5% | 26.1% | **43.3%** |
| crosstalk (ep-10) | 46.0% | 49.9% | 46.1% | **59.4%** |
| snow (ep-10) | 39.8% | 43.2% | 40.8% | **51.0%** |
| wet_ground (ep-10) | 40.2% | 41.3% | 42.5% | **68.3%** |
| fog (ep-21) | 20.8% | 20.4% | 21.9% | **38.7%** |
| crosstalk (ep-21) | 45.1% | 49.1% | 45.1% | **58.6%** |
| snow (ep-21) | 38.4% | 43.7% | 39.5% | **49.1%** |
| wet_ground (ep-21) | 37.6% | 41.4% | 40.5% | **66.8%** |

The linear-probe decoder raises the ceiling 1.2-1.8x over distance-to-prototype on
every condition (fog 26.1->43.3% ep-10; wet_ground 42.5->68.3%). The zero-shot gain
is small because the frozen clean-fit probe and the frozen prototypes both start
from clean structure; the ceiling gain is the recoverable structure the centroid
rule throws away when the prototypes are re-estimated on the corrupted pool. The R4
probe is fit on labeled data (clean for zero-shot, pool for the ceiling), so the
label-free version is the frozen clean-fit probe (R4-zs), which at 0.43-0.50 healthy
already beats the R1 oracle (0.40-0.43) at zero-shot. The direction: keep the
cov-shift extractor and make the decoder a learned boundary on the code.

The cov-shift extractor's development, the healthy-condition ceiling-loss
diagnostic, and the decision-rule finding are tracked in
[`docs/cov_shift/cov_shift_iterations.md`](docs/cov_shift/cov_shift_iterations.md).

---

## 3. Pillar 2: label-free test-time adaptation (raises mIoU on the healthy conditions)

The label-free TTA path: on the conditions where the representation survives,
re-estimating the decoder from the corrupted stream (with no labels at all) raises
mIoU, and on the healthy conditions it reaches the ceiling. The current story is
about the **cov-shift extractor** (Section 2): where the frozen decoder already
holds, and where the decode rule leaves room.

For the cov-shift extractor, the label-free prototype update is essentially free —
the naive re-estimation reaches the ceiling on every condition:

- **On the healthy conditions the label-free path is sufficient.** Snow,
  wet_ground, motion_blur, beam_missing, incomplete_echo and cross_sensor sit at
  or near their labeled ceiling with the label-free update, so no labels are needed
  there at all.
- **Crosstalk is closed at the frozen level** (zero-shot ~= ceiling ~= naive): the
  extractor, not TTA, fixed it.
- **Fog is the remaining label-free gap**, small (naive 21.0% vs ceiling 23.5%)
  relative to every prior extractor (6-10 points) but still present.

A design rule: prior correction and prototype updates must not share a pathway.
The prior is an inference-time constant that shifts decision boundaries; it does
not move prototypes by itself. But if prior-corrected pseudo-labels feed the
updates, the bias steers the prototypes and the drift compounds. The prediction
pathway may use the prior-corrected score; the adaptation pathway must not.

**Zero-shot, labeled ceiling, and label-free TTA for the cov-shift extractor
(ep-10, the paper's model).** The zs column is the frozen-prototype HDC decode
(isotropy pipeline), the ceiling is the full-label oracle (re-estimating the
prototypes with true labels, frozen-ceiling harness), and the naive column is the
label-free prototype re-estimation on the corrupted pool (extractor-diff harness):

| condition | zero-shot | labeled ceiling | label-free naive TTA |
| :--- | :--- | :--- | :--- |
| fog | 20.1% | 23.5% | 21.0% |
| crosstalk | 39.5% | 39.4% | 38.6% |
| snow | 37.7% | 38.8% | ~ceiling |
| wet_ground | 35.8% | 40.5% | ~ceiling |
| incomplete_echo | 40.6% | 40.1% | ~ceiling |
| beam_missing | 44.3% | 44.2% | ~ceiling |
| motion_blur | 44.2% | 44.0% | ~ceiling |
| cross_sensor | 36.1% | 38.5% | ~ceiling |
| **mean (8 corrupted)** | **37.3%** | — | — |

The two columns that matter for TTA:
- **Crosstalk is closed at the frozen level**: zero-shot 39.5% ~= ceiling 39.4%, and
  the label-free update holds it (naive 38.6%). No assignment wall remains — the
  extractor, not TTA, closed it.
- **The healthy conditions sit at/near their ceiling label-free** (~ceiling rows):
  the label-free update reaches the labeled bound, so no labels are needed there.
- **Fog is the remaining label-free gap** (naive 21.0% vs ceiling 23.5%), small
  relative to every prior extractor (6-10 points) but still present.

The cov-shift method's zero-shot and ceiling tables and the per-condition source
harnesses are in `docs/cov_shift/cov_shift_iterations.md` (Iterations C1-C5, C11).

**The recoverable gap depends on the decoder rule (background).** With the
distance-to-prototype rule (R1), the ceiling sits almost exactly at zero-shot
(gap ~0-2 points), which is why the label-free thread looked closed: there seemed
to be no recoverable structure left. With the learned HDC-code decoder (R4, Section
2), the same features show a large ceiling above zero-shot -- the recoverable
structure was there all along, the centroid rule just could not express it:

| condition (cov-shift ep-10) | R1 gap (ceiling-zs) | R4 gap (ceiling-zs) |
| :--- | :--- | :--- |
| fog | +2.2 | **+19.8** |
| crosstalk | +0.1 | **+9.5** |
| snow | +1.0 | **+7.8** |
| wet_ground | +2.3 | **+27.0** |

The R1 gap ~0 is the assignment-wall signature; the R4 gap reopens it as decoder
headroom. This reframes "where TTA stops": the limit is the decode rule, not the
features.

[To be filled: the form of the TTA (the update/don't-update gate, the support
threshold, and where the label-free path is engaged).]

---

### 3.1 Where the distance-to-prototype update is limited

The distance-to-prototype (nearest-centroid) update cannot recover fog and crosstalk:
re-estimating the centroids from the distance-confident points reproduces the clean
centroids, because the confident points already decode correctly and the far points
that would move the centroids are exactly the ones the gate excludes. This is the
assignment wall: a label-free signal can say *which* points are wrong, but not *what
class* they belong to.

The C10 decision-rule finding (Section 2) sharpens this: the wall is a property of
the **decode rule**, not the features. A learned decoder on the same code reopens a
large ceiling gap (R4 gap +19.8 on fog, +27.0 on wet_ground vs R1's ~0-2). So the
label-free limit has two distinct causes:

- **For the healthy conditions, the wall is gone** — the cov-shift extractor's frozen
  decoder holds, and the label-free update reaches the ceiling.
- **For fog/crosstalk, the wall is the centroid rule.** The features retain the
  recoverable structure (the R4 ceiling is high); the nearest-centroid update cannot
  express it. A learned decoder (fit on a labeled pool, or re-estimated label-free on
  the stream) is the lever.

#### The detection signal that triggers the handoff

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
  +6.5, Crosstalk +14.2 over zero-shot; section 6.3). The label ceiling is
  11-25% per extractor on fog/crosstalk, and the best label-free TTA sits 2-9
  points below it (section 6.4).
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

Active learning is the fill-in, engaged by a detection mechanism (Section 3.1):
run the extractor, attempt the label-free TTA path, and activate active learning on
exactly the conditions/clusters where the label-free gap-closed is below a
threshold (or where the TTA-to-supervised gap is not >90% closed). The measurements
say this is the default on the collapsed conditions (fog, crosstalk), where the
assignment wall caps any label-free update well below the labeled ceiling, while
the healthy conditions stay on the label-free path (Section 3). It spends the small
label budget on exactly the clusters the label-free thread cannot name, and converts
the surviving cluster structure into prototypes. It is the only path that closes the
residual gap, because it is the only one that supplies the missing class labels.

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
| **Adaptation (Pillar 2)** | Label-free gated prototype updates, used where the label-free path is sufficient (the healthy conditions); engaged unless the detection signal (Section 3.1) says active learning is needed | 
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
    detection signal (Section 3.1) deciding where it is sufficient. The TTA
    machinery is the efficiency lever, not the whole answer.
 4. Backprop-free active learning (Pillar 3, Section 4) is the fill-in for the
   conditions the detection signal routes to it: rank the dense per-class clusters,
   query one point per cluster under a strict label-or-don't gate, and re-estimate
   the prototypes from the labeled cluster representatives. This closes the residual
   gap on fog/crosstalk, because it supplies the missing class labels.
5. Balanced allocation of the label budget across classes is folded into Pillar 3,
   to be engaged once the active-learning updates produce headroom to harvest.
