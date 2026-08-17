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

## Next: Iteration 2: class-conditional clusters and tight-radius grounding

Iteration 1 showed the query rule works (influence > confidence everywhere) but
the grounding mechanism is poisoned by ~65%-pure K-means clusters. Iteration 2
tests the three fixes:
- class-conditional clustering (within pseudo-class clusters, K_c per class from
  the Iteration-0 mono-modality measurement);
- tight-radius grounding (ground only the nearest fraction, q in {0.25, 0.5});
- labels-direct-in-T (no propagation) as the contamination-free control.

Verdict rule: if class-conditional clusters + tight radius lift the budget curve
above frozen toward the oracle (closing part of the +13 to +32 AL-closeable gap),
the Pillar-3 mechanism is confirmed; if even pure per-class clusters cannot beat
frozen, the AL framework's leverage must be re-thought (the labeled points'
direct contribution to T, not propagation).
