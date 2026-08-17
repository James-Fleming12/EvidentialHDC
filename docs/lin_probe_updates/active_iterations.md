# Active-learning iterations (Pillar 3)

The active-learning framework closes the residual gap the label-free TTA cannot:
a small budget of TRUE labels, spent on exactly the ranked hard points, converts
the recoverable headroom into prototypes. This doc tracks the iterations of that
framework.

## Background

### The ceilings: feature extractor and linear probe

The current setup (cov-shift ep-10; fog / crosstalk / snow / wet_ground), from
the README Pillar-2 tables:

| condition | zero-shot (frozen) | label ceiling (probe) | AL-closeable gap |
| :--- | :--- | :--- | :--- |
| fog | 20.1% | **43.3%** | +23.2 |
| crosstalk | 39.5% | **59.4%** | +19.9 |
| snow | 37.7% | **51.0%** | +13.3 |
| wet_ground | 35.8% | **68.3%** | +32.5 |

- The feature-extractor ceiling is what the FROZEN decoder (zero-shot) already
  achieves: crosstalk 39.5% and snow 37.7% mean the extractor survives those
  corruptions; fog 20.1% is where the extractor's structure is destroyed.
- The linear-probe ceiling is the LABELED oracle (S_all, T_oracle): re-estimating
  the probe from corrupted points with true labels. It is far above zero-shot on
  every condition (fog +23.2, wet_ground +32.5), so the recoverable structure
  survives -- the label budget buys the rotation the frozen decoder cannot
  express.
- The gap (label ceiling - frozen) is the AL-closeable headroom: what only true
  labels recover. Fog and wet_ground are where the budget buys the most.

### What the update looks like, and its efficiency

The update is the Nystrom-warm-started matrix-free CG ridge (the established
probe update):

- Solve W = (X^T D X + lI)^-1 X^T D Y by CG with matrix-free (X^T D X) v =
  X^T (D (X v)); never builds the 10000 x 10000 S.
- Nystrom warm start (m = 1000) + CG-8: ~1.5M pts/s update, ~0.034s per fit;
  CG-8 from the warm start matches plain CG-20 from scratch.
- Decode is cosine to the learned W at the prototype's decode rate
  (~0.20-0.29M pts/s); the efficiency cost is in the UPDATE, not the deployed
  decode.
- Weighted ridge verified exact: (X^T D X + lI)^-1 X^T D Y with D = diag(weights)
  matches the dense solve to ~6e-6; soft targets (probabilities in Y) are
  supported by the same machinery.

For active learning this means: a small labeled pool (1k-10k points) refits the
probe in well under a second, so the AL budget fits the gradient-free,
accumulate-and-solve framework with no new machinery.

### Why label-free TTA failed (Iterations 0-12 of the TTA doc)

The label-free route is exhaustively closed; the probe's label-free ceiling IS
the frozen decoder:

- Iterations 9-10: pseudo-label gating and weighting fail -- neither hard gates
  nor confidence weights let the probe update beat the frozen decode.
- Iteration 11 (S/T decomposition): with S and T decoupled, every S=all, T-gated
  variant is worse than no_gate; S=gated is catastrophic; even a PERFECT-purity
  T (correct-only labels) cannot reproduce the oracle -- the gap is label
  COVERAGE, not label noise. Wrong pseudo-labels ANTI-align with the oracle
  rotation (cos(W_wrong, W_oracle) < 0); influence is anti-correlated with
  confidence (Spearman -0.40 to -0.64).
- Iteration 12 (geometric): the corruption does NOT rotate the pool's covariance
  (spectral overlap 0.995-1.000 between S_clean and S_target eigenspaces), so
  Procrustes / CORAL / whitening have nothing to align; pseudo-anchored label
  diffusion poisons itself. The one thing that moves the probe toward the oracle
  is diffusing TRUE labels (oracle-anchored: cos to oracle 0.59-0.83) -- the
  geometry can carry a sparse set of true labels, but cannot manufacture the
  supervision.

The 7 properties that caused this (README Section 3.2): systematic probe errors
(not noise), the rotation lives in the decision rule not the geometry, confidence
anti-correlates with update-usefulness, wildly uneven per-class reliability,
purity cannot buy coverage, the second-order solve amplifies the contaminated
half, and the binarized code fixes the norm so shifts are angular.

Net: the recoverable headroom needs TRUE labels on covering points. The AL
question is now purely: how few labels, and how do we pick them.

## Iteration 0: the cluster packing and label-budget diagnostics (2026-08-17)

The Pillar-3 leverage is that the budget scales with the number of clusters, not
points: one queried point per tight per-class cluster grounds the whole cluster
by distance, so a small budget recovers most of the oracle gap. The prior
evidence for the dense per-class packing (README 4.1: corrupted 1-NN purity
75-87% fog/crosstalk) was measured on the UN-PRETRAINED model. This iteration
re-verifies the packing on the CURRENT cov-shift extractor (ep10 + ep21, all 4
conditions) and measures how few labels the space actually needs
(`al_cluster_grounding_diag.py`):

- A. Packing: per-class 1-NN / k-NN same-class purity on the corrupted pool AND a
  clean reference; k-means cluster purity at K = #classes; intra vs inter class
  cosine separation.
- B. One-label-per-cluster grounding: for K in {17, 34, 68, 136, 272} clusters,
  representative = point nearest the centroid (the single queried point),
  label = its true label. Reports the budget-to-coverage curve (fraction of the
  pool correctly grounded by K labels) and the DISTANCE-GATED coverage (coverage
  restricted to points within distance quantile q of their representative -- the
  "label if close, else ask" operating curve), plus the radius needed to cover
  90% of each cluster's grounded points.
- C. Label-reduction properties: per-class shift alignment (are the
  corrupted-vs-clean class-mean shifts near-global, so a few classes' labels
  estimate the shift for all?); confidence-representativeness (does the frozen
  probe's confidence pick the centroid-near points, i.e. can it self-select the
  query point?); within-class multi-modality (how many subclusters per class are
  needed for ~90% purity -- the real per-class label budget); pseudo-label
  agreement vs distance to centroid (the probe is more right near centroids ->
  its own confidence ranks the query candidates).

Result (ep10; ep21 identical pattern, fog slightly worse; full synthesis per
condition in the JSONs):

**A. The packing survives, degraded exactly where the label budget is needed.**
Corrupted-pool 1-NN purity vs the clean reference (nn1 0.771 all conditions):

| condition | pool nn1 | pool nnk | intra / inter cosine | k-means purity K=#classes |
| :--- | :--- | :--- | :--- | :--- |
| fog | 0.513 | 0.380 | 0.628 / 0.015 | 0.649 |
| crosstalk | 0.688 | 0.594 | 0.670 / 0.055 | 0.652 |
| snow | 0.586 | 0.525 | 0.704 / 0.039 | 0.644 |
| wet_ground | 0.770 | 0.639 | 0.621 / 0.004 | 0.635 |

The clusters are strongly separated (inter-class cosine ~0.01-0.06 vs intra
~0.6-0.7) and ~65% of the pool falls in a same-class cluster at K = #classes.
The per-class picture is the important one: on fog, class 15 (nn1 0.341) and
class 7 (0.434) are badly loosened while class 11 is tight (0.849); on
wet_ground, classes 7/15 are again the weak ones (0.62-0.63) while 4/11/14/16
are tight (0.83-0.92). The weakly-packed classes are rare (n = 50-300 in the
pool) -- they are hard to ground AND they are exactly the classes the frozen
probe gets wrong.

**B. The label budget is small and the curve is flat.** One label per cluster
(representative = centroid-nearest point, label = its true label):

| condition | coverage K=17 | K=34 | K=68 | K=136 | K=272 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.669 | 0.646 | 0.649 | 0.679 | 0.692 |
| crosstalk | 0.506 | 0.641 | 0.674 | 0.690 | 0.683 |
| snow | 0.640 | 0.618 | 0.667 | 0.696 | 0.695 |
| wet_ground | 0.495 | 0.623 | 0.625 | 0.657 | 0.655 |

Coverage = fraction of the pool correctly grounded by K labels. The marginal
gain from K=68 to K=272 is +0.01-0.04: ~34-68 labels capture essentially all
the grounding the cluster structure offers. Per-class grounding at K=68 shows
the same weak classes as A (fog class 7 grounded 0.000, class 15 0.054, class 16
0.258, class 14 0.286; class 11 0.813).

The distance-gated operating curve works as the mechanism predicts -- coverage
is higher for points close to their representative:

| condition | q=0.5 | q=0.75 | q=0.9 | radius q90 (128-d units) |
| :--- | :--- | :--- | :--- | :--- |
| fog | 0.707 | 0.677 | 0.660 | 1.92 |
| crosstalk | 0.781 | 0.731 | 0.699 | 1.96 |
| snow | 0.819 | 0.734 | 0.693 | 2.06 |
| wet_ground | 0.702 | 0.661 | 0.642 | 2.81 |

The "label if close, else ask" gate: points within the q=0.5 radius are grounded
at 0.70-0.82, and the radius for 90% coverage is ~1.9-2.1 units (2.81 for
wet_ground -- its clusters are the loosest, which matches it having the largest
AL-closeable gap, +32.5).

**C. Label-reduction properties -- all three hold.**

- **Classes are near-mono-modal**: within-class k-means dominant fraction at
  K_c = 4 is ~1.000 on every condition and checkpoint. The per-class clusters
  are single tight modes, so the label budget scales with CLASSES, not modes:
  ~4 labels per class fully cover it (and K = #classes clusters already ground
  ~65%).
- **The frozen probe self-selects the query points**: corr(confidence, distance
  to centroid) is negative everywhere (fog -0.15, crosstalk -0.36/-0.40, snow
  -0.42/-0.46, wet_ground -0.15/-0.18): the high-confidence points ARE the
  centroid-near representatives, so the probe's own confidence ranks the query
  candidates for free (and the pseudo-accuracy is higher near centroids:
  fog 0.61 nearest-quartile vs 0.47 outer-quartile in the dry run).
- **Partial shift carry-over**: corrupted-vs-clean class-mean shift vectors are
  only moderately aligned across classes (pairwise cosine 0.20-0.37, wet_ground
  highest at 0.37). The corruption shift is partially shared (some cross-class
  carry-over possible) but not a near-global transform; a few classes' labels
  estimate the shift for the rest only partially.

**Net for the framework design**: ~34-68 labels (2-4 per class) ground most of
the pool correctly; the classes that escape grounding are the rare, badly-loosened
ones (7, 15 on fog/wet_ground) -- the query rule must spend its budget on them
first (influence-based ranking, per the TTA Iteration-11 findings), not on the
tight majority classes. The distance gate (label if within ~1.9 units, else ask)
and the confidence self-selection are both validated.

## Iteration 1: the query-rule comparison and grounding simulation (2026-08-17)

The Iteration-0 design is tested end-to-end (oracle-simulated,
`al_query_rule_diag.py`, ep10 + ep21, all 4 conditions): cluster the pool (K in
{17, 68, 136}), query ONE point per cluster (the representative = centroid-nearest
point), label it TRUE, ground the cluster by distance (points within the gate
radius inherit the representative's label; beyond it they are not grounded). The
probe update is the established ridge with S = ALL pool points and T = grounded
points only. Four query rules (budget -> mIoU curves) and two grounding gates:

- **R1 influence**: rank clusters by J_c = sum of per-point influence I_i
  (the exact magnitude of each point's W contribution, Iteration-11 signal).
- **R2 confidence**: rank by representative confidence, ascending (uncertainty
  sampling, the free baseline).
- **R3/R4**: the same with the prototype-vs-probe disagreement gate (only
  disagreeing clusters are eligible).
- Grounding gate: agreement-gated (propagate only where the frozen probe predicts
  the rep's class -- the default) vs distance-only control (the `*_nodistgate`
  runs).

**The budget -> mIoU table (ep10, agreement-gated; frozen / oracle per
condition; K=68 shown, K=17/136 in the JSONs):**

| condition | rule | b=17 | b=34 | b=68 (grounded_all) | frozen | oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | influence | 0.055 | 0.116 | 0.182 | 0.259 | 0.377 |
| fog | confidence | 0.048 | 0.066 | 0.182 | 0.259 | 0.377 |
| crosstalk | influence | 0.215 | 0.275 | 0.335 | 0.524 | 0.554 |
| crosstalk | confidence | 0.055 | 0.159 | 0.335 | 0.524 | 0.554 |
| snow | influence | 0.150 | 0.217 | 0.289 | 0.457 | 0.493 |
| snow | confidence | 0.059 | 0.113 | 0.289 | 0.457 | 0.493 |
| wet_ground | influence | 0.151 | 0.263 | 0.311 | 0.427 | 0.614 |
| wet_ground | confidence | 0.069 | 0.107 | 0.311 | 0.427 | 0.614 |

ep21 is the identical pattern (fog frozen 0.231 / oracle 0.332; crosstalk
0.504/0.534; snow 0.444/0.474; wet_ground 0.426/0.579). The distance-only
grounding control is within +/-0.02 of the agreement-gated numbers everywhere
(grounded_all fog 0.185 vs 0.182, wet_ground 0.320 vs 0.311): the agreement gate
grounds fewer points (17k vs 30k of 50k) at the same mIoU -- it does not change
the result.

**Efficiency (per condition, K=68):** one 50k-pool ridge fit = 0.056s
(~0.89M pts/s update); k-means = 0.32s; ~260 pool points grounded per label
(K=68: 17737 grounded / 68 labels). The AL loop's total compute is the k-means
plus one fit -- under half a second.

**Result: the query-rule ranking is confirmed, but the grounding mechanism does
NOT beat the frozen decoder -- and the reason is now measured.**

1. **Influence > confidence at every budget** (fog b34: 0.116 vs 0.066;
   wet_ground b34: 0.263 vs 0.107; crosstalk b17: 0.215 vs 0.055). The
   influence-ranked query spends each label where it moves the probe most; the
   free confidence baseline selects the tight, already-correct clusters. The
   rule comparison works as designed.
2. **The disagreement-gated rules are useless as queries**: only 3-27 of K
   clusters have a disagreeing representative, so they saturate at ~4-27 labels
   with mIoU stuck at 0.03-0.09. The disagreement gate is almost always closed
   on these conditions; it is a HANDOFF signal, not a query rule.
3. **grounded_all < frozen on EVERY condition** (fog 0.182 vs 0.259, crosstalk
   0.335 vs 0.524, snow 0.289 vs 0.457, wet_ground 0.311 vs 0.427; ep21
   identical). The distance grounding propagates labels through clusters that
   are only ~65% class-pure (Iteration 0), so T receives ~35% WRONG propagated
   labels. Per README Section 3.2 property 1, systematic label errors poison the
   ridge below the frozen ceiling. The agreement gate does not fix this: it
   gates on the frozen probe's prediction, which is itself only 55-79% accurate
   on these pools, so it removes points without removing the error.
4. **The class-balance observation**: even the influence rule's low budgets are
   far below oracle because the grounded T is dominated by the few queried
   classes; the oracle's T has all classes. The gap from grounded_all to oracle
   (fog +0.19, wet_ground +0.30) is the cost of grounding ~35% of the pool
   through impure clusters.

**Net: the mechanism that works is NOT "propagate through mixed clusters".**
Iteration 0 showed the per-class structure is mono-modal; the problem is that
K-means at K ~ #classes MIXES classes (65% purity). The fix directions for the
next iteration, in order of expected value:
- (a) CLASS-CONDITIONAL clustering: cluster within each class's points (the
  probe's pseudo-label as the class prior), so clusters are ~pure by
  construction -- one label per class-subcluster, no distance propagation of
  cross-class error.
- (b) TIGHT-radius grounding: propagate only to the nearest fraction of the
  cluster (Iteration 0's q0.5-gated coverage was 0.70-0.82) and leave the rest
  unlabeled, instead of the q0.6 radius used here.
- (c) Hybrid: label K representatives, use their labels DIRECTLY in T (no
  propagation), and accept the smaller T -- the coverage loss vs the
  contamination gain must be measured.

## Iteration 2: graph diffusion with queried anchors + the shift structure (2026-08-17)

Iteration 1's cluster + hard-distance propagation failed because 65%-pure clusters
propagate ~35% wrong labels. Two replacements tested
(`al_diffusion_shift_diag.py`, ep10 + ep21, all 4 conditions):

**A. Graph diffusion with QUERIED anchors** (no clustering, no kNN): pick K
anchors (influence_floor: one per class by max influence, then fill by
influence; plus pure influence / confidence / random controls), label them TRUE
(simulated oracle), diffuse through the HDC Hamming point graph
(Y_diff = (I - a G)^-1 Y_sparse, matrix-free CG, implicit all-pairs), then the
ridge S=all, T=Y_diff. Budget K in {8, 17, 34, 68}, a in {0.5, 0.9}.

| condition | frozen | oracle | diffusion best (any K, a) | Iteration-1 grounded_all |
| :--- | :--- | :--- | :--- | :--- |
| fog | 0.258 | 0.375 | 0.050 (K17 a0.5) | 0.182 |
| crosstalk | 0.524 | 0.553 | 0.054 (K8 a0.9) | 0.335 |
| snow | 0.456 | 0.493 | 0.051 (K8 a0.9) | 0.289 |
| wet_ground | 0.427 | 0.615 | 0.064 (K8 a0.9) | 0.311 |

**Diffusion with 8-68 queried anchors does NOT work.** The mIoUs sit at
0.03-0.06, far below frozen and below even the (failed) cluster grounding. The
Iteration-12 oracle-anchored diffusion that beat frozen used ~50% of the pool as
anchors; with a handful of anchors the diffusion has too little signal. The
anchor rules barely differentiate and RANDOM is often best (fog random 0.059 vs
influence_floor 0.042) -- the differences at these tiny mIoUs are noise. The
efficiency claim held (diffuse 0.084s + fit 0.057s = 0.14s per loop, no
clustering) but efficiency is moot when the mechanism does not recover the
geometry. The anchor density, not the propagation rule, is the binding
constraint -- and a queried budget of 8-68 anchors cannot supply it.

**B. The shift structure is robust but weak** (prototype decode; frozen / oracle
per condition):

| condition | proto frozen | k=2 carry_over | k=4 carry_over | all-labeled | oracle_shift | probe frozen | probe oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.237 | 0.234 | 0.247 | 0.251 | 0.250 | 0.258 | 0.375 |
| crosstalk | 0.463 | 0.461 | 0.462 | 0.464 | 0.464 | 0.524 | 0.553 |
| snow | 0.402 | 0.401 | 0.402 | 0.408 | 0.409 | 0.456 | 0.493 |
| wet_ground | 0.390 | 0.401 | 0.404 | 0.422 | 0.421 | 0.427 | 0.615 |

ep21 identical. The labeled classes are always [4, 7] then [11, 13] (the
influence ranking). Pairwise shift cosine 0.16-0.30 (wet_ground highest, fog
lowest).

**carry_over ~= per_class_only ~= oracle_shift at every k: the global shift from
2-4 labeled classes corrects ALL classes' prototypes to essentially the
oracle-shift ceiling.** With k=2, fog carry 0.234 vs oracle 0.250; crosstalk
0.461 vs 0.464. The label-skipping works -- the structure is robust. BUT the
shift-corrected PROTOTYPE decode (0.23-0.46) still sits BELOW the frozen probe
decode (0.26-0.52): correcting prototypes to their ceiling does not beat the
learned probe. The unlabeled-cos numbers explain why: before 0.92-0.99 -> after
0.84-0.99, i.e. the global shift barely moves the unlabeled prototypes (they
were already near their corrupted means -- the corruption shift is small in the
code space, and the pairwise alignment 0.16-0.30 is modest).

**Net: two mechanisms closed, two properties established.**
1. Diffusion needs dense anchors (50% of the pool), so a queried budget of
   8-68 labels cannot drive it -- the anchor density is the constraint.
2. The shift structure is robust (2-4 labels reach the ceiling) but its ceiling
   is the prototype decoder, which the probe already beats -- it is a free
   add-on for the prototype path, not a path past the frozen probe.
3. The query rule (influence) and the class-floor selection remain the validated
   pieces; what is still missing is a mechanism that turns ~tens of labels into
   a T that beats frozen.

## Iteration 3: the one-label information content (2026-08-17)

The decisive reframe after Iteration 2: the bottleneck is anchor density, not the
propagation rule. So instead of trying to expand labels, this iteration asks what
ONE true label tells us about the decision rule (`al_label_information_diag.py`,
ep10 + ep21, all 4 conditions). The A/B/C experiment compares three ways of
turning K labels (1 per class, influence-selected) into T -- nearest-anchor
propagation (A), class-centroid cosine (B), decision correction (C:
(frozen_pred, margin_bin) -> true label) -- plus the direct-sparse baseline and
the soft-confusion family:

| condition | frozen | oracle | direct_sparse | A_best | B_best | C_best | C_soft_est | C_soft_ORACLE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.259 | 0.375 | 0.033 | 0.068 | 0.068 | 0.062 | 0.062 | 0.218 |
| crosstalk | 0.524 | 0.554 | 0.054 | 0.080 | 0.080 | 0.087 | 0.133 | 0.508 |
| snow | 0.456 | 0.493 | 0.045 | 0.122 | 0.122 | 0.014 | 0.017 | 0.414 |
| wet_ground | 0.428 | 0.614 | 0.067 | 0.035 | 0.035 | 0.071 | 0.070 | 0.412 |

**Result: every expansion fails at 1 label/class, and the confusion family has a
real ceiling.** A and B are identical (1 anchor = the centroid) and useless -- the
HDC code similarities saturate (all cosines ~0.98), so the nearest-anchor/centroid
gate cannot differentiate. C's lookup is too noisy from 8-9 labels (anchor acc
0.11-0.38). C_soft_est <= 0.133 everywhere. BUT **C_soft_ORACLE (the confusion
matrix from ALL pool labels) nearly reaches frozen on crosstalk (0.508 vs 0.524)**
-- the systematic error structure exists (top pairs: 13->14 at 0.25-0.62,
15->16, 7->15 -- the same pairs as every prior diagnostic), it just cannot be
estimated from 1 label/class (est-vs-oracle row-cos 0.18-0.58). direct_sparse is
the floor everywhere (0.03-0.07).

**Verdict: the confusion structure exists but 1 label/class cannot estimate it.**
The `labels_per_class=2` variant (32 labels) is the immediate follow-up; if
C_soft_est climbs toward C_soft_ORACLE on crosstalk, the correction family is the
mechanism with a measured budget.

## Iteration 4: the 128-d geometry promises (2026-08-17)

The A/B failure in Iteration 3 had a systematic cause: the expansions computed
similarity in the SATURATED 10k-d HDC code space, while the packing we keep
citing (1-NN purity, intra vs inter cosine) was measured in the 128-d features.
This iteration measures the intrinsic information content of each geometric
property IN THE 128-d SPACE, with robustness: mean +- std over R=10 repeated
RANDOM anchor draws (no selection rule can hide a weak property, no lucky draw
can fake a strong one) (`al_geometry_promise_diag.py`, ep10 + ep21, all 4
conditions).

| condition | A nearest t0.9 | B oracle-centroid t0.9 | C min-agreement t0.9 | spatial P4 | per-class nn1 mean |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.56 @ cov 0.16 | 0.57 @ cov 0.39 | **0.73 @ cov 0.07** | 0.810 | 0.638 |
| crosstalk | 0.60 @ cov 0.29 | **0.82 @ cov 0.55** | 0.61 @ cov 0.12 | 0.929 | 0.787 |
| snow | 0.78 @ cov 0.31 | **0.82 @ cov 0.53** | **0.87 @ cov 0.15** | 0.879 | 0.774 |
| wet_ground | 0.72 @ cov 0.21 | 0.73 @ cov 0.39 | **0.78 @ cov 0.07** | 0.931 | 0.846 |

**The 128-d space carries real signal where the HDC space was saturated:**
- **B_centroid_oracle is the best feature-space promise**: crosstalk/snow reach
  0.82 precision at 0.5+ coverage -- class-level geometry DOES carry labels at
  0.8+ precision. But B_centroid_1 == A (1 anchor is the centroid), so the
  centroid needs k >= 2-4 anchors/class (32-64 labels) to approach the oracle.
- **C min-agreement is the contamination-free certificate**: precision climbs
  with the agreement gate (snow 0.87 @ cov 0.15, fog 0.73 @ cov 0.07) -- the
  operating curve the mechanism needs.
- **SPATIAL adjacency is the standout**: P4 = P(same class | projection
  neighbor) is 0.81 (fog) / 0.93 (crosstalk) / 0.88 (snow) / 0.93 (wet_ground).
  The sensor's own geometry -- label one point, its projection neighbors
  inherit -- is more reliable than every feature-space signal, and it survives
  fog where the feature packing collapses (nn1 0.51). Caveat: the min per-class
  coherence is 0.075-0.77 (fog classes 15/7 are spatially scattered), so the
  budget must target the incoherent classes.
- **Nearest-anchor alone is too weak** (0.44-0.78 at t0.9): single-anchor
  propagation cannot beat frozen; the geometry only pays when aggregated
  (centroid), certified (agreement), or spatial (superpixel).
- F (confidence-conditioned packing): the frozen probe's ABSOLUTE confidence is
  < 0.3 on every corrupted pool point (the Iteration-11 calibration finding), so
  absolute gates are vacuous; the relative-quantile version is in the rerun.

**Net: two live mechanisms** -- spatial superpixels (new, strongest, sensor
geometry) and k-anchor class centroids + min-agreement (feature geometry, needs
2-4 labels/class). They are complementary: superpixels expand a label into many
points cheaply; centroids turn those expanded points into a decode.

## Iteration 5: the hybrid superpixel + centroid + gate ablation ladder (2026-08-17)

The compound mechanism and its ablation ladder (`al_hybrid_grounding_diag.py`,
ep10 + ep21, all 4 conditions): connected components on the projection mask
(CCL, no labels), query the top-B components by influence, ground the component
with the rep's TRUE label, then five T constructions per budget:
S0_direct (reps only) / S1_spatial (whole component) / S2_centroid (128-d
centroid decode, cosine gate) / S3_hybrid_AND (spatial AND feature agree) /
S4_union (both expansions).

| condition | frozen | oracle | superpixels | CCL time |
| :--- | :--- | :--- | :--- | :--- |
| fog | 0.209 | 0.375 | 288k (mean size 5) | 4.6s |
| crosstalk | 0.414 | 0.553 | 224k (mean size 6) | 5.0s |
| snow | 0.350 | 0.493 | 241k (mean size 6) | 4.7s |
| wet_ground | 0.362 | 0.614 | 95k (mean size 8) | 3.2s |

Full-budget ladder (every component queried; ep10):

| condition | S0 direct | S1 spatial | S2 centroid | S3 AND | S4 union |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.211 (cov 0.18) | 0.129 (cov 1.0) | 0.197 | 0.166 (prec 0.889) | 0.129 |
| crosstalk | 0.286 (cov 0.17) | 0.215 | 0.378 | 0.308 (prec 0.913) | 0.215 |
| snow | 0.277 (cov 0.16) | 0.166 | 0.219 | 0.192 (prec 0.891) | 0.166 |
| wet_ground | 0.310 (cov 0.12) | 0.213 | 0.287 | 0.278 (prec 0.828) | 0.213 |

**Result: the spatial hypothesis is real but the superpixels are tiny, and
nothing beats frozen.**

1. **The superpixels are small (mean 5-8 points)**: the range-view projection is
   fragmented into islands, not large coherent regions. The P4=0.81-0.93 promise
   does not translate into large pure components. S1's propagated labels are
   only 0.16-0.48 precise -- spatial grounding alone is weak.
2. **S0_direct beats everything at small budgets and at full budget**: the
   influence-ranked reps are valuable AS THEMSELVES (prec 1.0 by construction),
   and propagation dilutes them. At full budget S0 reaches 0.21-0.31 -- the best
   rung -- but still sits at or below frozen.
3. **The AND gate (S3) is the only clean-T path** (prec 0.79-0.92 at full
   budget) but coverage collapses to 0.11-0.24, so there are not enough agreed
   points to move the probe; mIoU stays below frozen.
4. **The ladder's verdict: no rung beats frozen at any budget.** The AL
   mechanism -- in all four forms tried (cluster+propagation, diffusion, sparse,
   spatial/centroid/gated) -- cannot beat the frozen probe with the tested
   budgets, because the budgets that would work (thousands of labels) are no
   longer ultra-cheap, and the small budgets do not provide enough clean T.

**The synthesis: the AL-closeable gap exists (the oracle proves it), but no
label expansion mechanism converts a small budget into it.** The ladder's
S0-curve (still climbing at 9k labels) points to the honest question: how many
DIRECT labels (no expansion) are needed to cross frozen -- the cost of the gap.

## Iteration 6: the dimension test -- a smaller code space does NOT make labels
cheaper (2026-08-17)

The hypothesis: in the binarized code space, the per-class contribution to T is
a sum of +-1 vectors, so the class-centroid estimate has noise ~ 1/sqrt(n_c)
per coordinate while the signal is tiny after sign-binarization spreads the
128-d signal over d dims -- relative estimation error scales as sqrt(d / n_c),
so labels per class should scale with d. If true, a smaller code dim would
dramatically cheapen the S0/direct-label budget curve. The test
(`al_dimension_budget_diag.py`, ep10 + ep21, all 4 conditions): the S0 budget
curve {100, 300, 1k, 3k, 10k, 30k, 50k} across code dims {128 real-valued,
512, 1k, 2k, 5k, 10k binarized}, with the frozen/oracle refs per dim.

Key numbers (ep10; the budget at which the curve CROSSES the frozen probe):

| condition | dim 128 | dim 1000 | dim 10000 | oracle 128 | oracle 1000 | oracle 10000 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 30000 | 30000 | 30000 | 0.279 | 0.314 | 0.375 |
| crosstalk | 30000 | 30000 | 30000 | 0.506 | 0.529 | 0.553 |
| snow | 10000 | 30000 | 30000 | 0.442 | 0.474 | 0.493 |
| wet_ground | 10000 | 30000 | 30000 | 0.534 | 0.570 | 0.615 |

ep21 identical (all conditions cross at 10000-30000 regardless of dim).

**Result: the dimension hypothesis is FALSIFIED, and the data is decisive.**

1. **Lowering dim does NOT make labels cheaper -- the curves are parallel
   across all six dims.** Every dim, every condition: the S0 curve crosses
   frozen at ~30k labels (60% of the pool). The estimation argument predicted
   128-d should cross at ~2k labels (n_c ~ d); it needs 30k. The bottleneck is
   NOT the code dimension -- it is the coverage required by the selection,
   which is dim-independent. At equal budget, 128-d is only marginally better
   (fog b3000: 0.160 vs 0.124 at 10k), never 10-100x.
2. **Shrinking dim trades away ceiling for nothing.** The oracle-per-dim GROWS
   with dim (fog 0.279 at 128-d -> 0.375 at 10k-d): the AL-closeable gap lives
   in the high-d space, and at 128-d there is almost no gap to close. The C8
   "encoding changes lose equally" claim holds for the FROZEN decode, but the
   LABELED oracle clearly prefers the 10k-d space.
3. **The S0 knee is the honest cost of the gap: ~40-60% of the pool must be
   labeled, in every space tested.** The bottleneck is the selection/coverage
   structure, not the estimator's dimensionality.

**Why the space resists cheap labels (the mechanism, all measured):**

- **The gap is the missing mass.** W_labeled = W_oracle - (S + lI)^-1
  (sum over UNLABELED points of x_i y_i^T). Scaling W does not change the
  decode, but DIRECTION does -- and the direction of a partial sum is only
  right if the labeled subset's per-class sums are proportional to the full
  per-class sums. The AL gap is exactly the missing-mass term.
- **Influence selection picks the boundary, not the bulk.** Influence
  anti-correlates with confidence (-0.40 to -0.64) and wrong points carry 2x
  the influence: the ranked labels concentrate on outlier/boundary points, so
  T's class columns point at the boundary direction, not the mean direction.
  Reaching the mean direction requires the bulk of each class -- and the pool
  is class-imbalanced ~40x, so the majority classes need thousands of labels.
- **The boundary is pathologically sensitive: the means barely move but the
  probe rotates ~90 degrees.** The corruption shift moves the class means only
  ~5-10 degrees (unlabeled-cos 0.92-0.99), yet cos(W_frozen, W_oracle) is
  0.05-0.19 on fog. The classes are fat blobs (intra-cos 0.62-0.70, points
  45-50 deg from their mean) whose means are ~89 deg apart; a small mean shift
  flips which side of the boundary the BULK falls on. The correct boundary is
  a mass-weighted function of the means, and only the full-mass oracle T gets
  it right.
- **The update is stable; the sensitivity is in T, not the solve.** The ridge
  is exact and well-conditioned; the inverse covariance amplifies T errors
  along low-variance directions (README 3.2 property 6), but the dominant term
  is the missing-mass bias itself.

**Net for the paper**: the cheap-label path is not a smaller code space. The
one untested combination the data points at: the corrupted means ARE
predictable (clean mean + shift, shift partially shared across classes at
pairwise-cos 0.2-0.37, estimated from 2-4 labeled classes per Iteration 2), and
the expensive part is that the ceiling DECODER is the probe, not the prototype.
The candidate cheap mechanism: **synthesize the corrupted T from a few labels
via the shift model (clean + predicted shift per class), then fit the probe on
the synthesized T** -- using the shift structure to estimate the mass-weighted
means without labeling the mass.

## Next: Iteration 7: the shift-synthesized T probe

The synthesis from Iterations 2-6: the corrupted class means are predictable
from the clean means plus a per-class shift (Iteration 2: 2-4 labeled classes
reach the prototype ceiling via carry-over; shift pairwise-cos 0.2-0.37), and
the gap to the ceiling is the PROBE decoder needing mass-weighted means
(Iterations 5-6: the missing-mass term). Iteration 7 combines them:
- estimate the per-class shifts from a few labeled classes (k in {2, 4, 8}),
- synthesize the corrupted T: T_c = (clean class mean + predicted shift_c)
  weighted by the class proportions,
- fit the probe on the synthesized T (S = the real pool covariance),
- measure mIoU vs frozen/oracle and vs the label budget.

Verdict rule: if the shift-synthesized probe closes most of the AL gap with
~10-100 labels (vs the ~30k the direct S0 curve needs), the Pillar-3 mechanism
is the shift-model probe, and the label budget story becomes: a few labels per
class estimate the corruption's effect on the class means, and the probe does
the rest at ~0.9M pts/s.
