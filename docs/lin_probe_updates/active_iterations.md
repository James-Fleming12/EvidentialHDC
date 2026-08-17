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

Result: (filled after the run; the synthesis per condition is in the JSON).

## Next: Iteration 1: the one-label-per-cluster query simulation

If the packing holds (high corrupted-pool NN purity, high cluster purity at
K = #classes), the next iteration simulates the actual AL loop:
- query the K representatives (by influence / disagreement, per the Iteration-11
  findings), label them TRUE (simulated oracle),
- re-estimate the probe from the labeled representatives with the standard ridge
  (S keeps the full pool),
- measure the mIoU vs the oracle ceiling and vs the label budget curve --
  closing the AL-closeable gap table above.

Verdict rule: if ~K labels (K ~ #clusters, 17-100) close most of the +13 to +32
AL-closeable gap, the Pillar-3 mechanism is confirmed and the paper's budget
story is grounded.
