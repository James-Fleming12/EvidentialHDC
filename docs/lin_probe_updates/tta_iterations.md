# Linear-Probe Decoder Updates: TTA Iterations

This tracks the decoder-side line of work: replacing the distance-to-prototype
(nearest-centroid) decode with a **linear probe fit on the HDC code**, and how to
update that probe at test time while staying gradient-free and efficient. The
extractor is **cov-shift DGLSS++**; this doc covers what we saw, why the two
improvements (cov-shift encoder, linear-probe decoder) work individually, and how
the probe can be updated label-free without backprop.

Companion docs: `docs/cov_shift/cov_shift_iterations.md` (the encoder),
`docs/lin_probe_updates/active_iterations.md` (the active-learning fill-in).

## Background: what we saw

The frozen-extractor diagnostics (C8/C10, cov-shift doc) established two measured
facts:

1. **The healthy-condition ceiling loss survives every encoding change.** Sign,
   bias, zscore, and Fourier encodings all lose the healthy conditions equally
   (C8). The loss is in the continuous features, not the binarization.
2. **The HDC code is linearly separable in a way nearest-centroid misses.**
   Fitting a logistic probe on the binarized 10k-d code (R4) beats the
   distance-to-prototype cosine (R1) on every condition, 1.24-1.77x on the ceiling
   (C10).

The README (Pillar 1 / Pillar 2) now carries the headline tables. This doc records
the decoder-side mechanism and the update path.

## Why performance improved: the two improvements individually

### 1. The cov-shift encoder (feature side)

The cov-shift extractor fixes the collapsed conditions at the source: per-scan
input normalization restricted to the statistics-shifted channels (range/remission)
plus internal InstanceNorm. It is the first extractor to raise BOTH the labeled
ceiling AND the label-free TTA on BOTH fog and crosstalk, and it closes the
assignment wall on crosstalk (zero-shot ~= ceiling ~= naive).

- On fog/crosstalk it works because it normalizes the shifted statistics instead of
  erasing the recoverable direction (the anchoring trade-off that capped prior
  variants).
- Its trade-off: the healthy-condition ceilings sit 2-15 points below DGLSS++ (the
  cov-shift normalization slightly compresses the healthy conditions).

### 2. The linear-probe decoder (decision-rule side)

The nearest-centroid rule decodes each point by cosine to the closest prototype
mean. That rule is a rigid, unweighted decision: every coordinate of the 10k-d code
contributes equally, and the boundary is a Voronoi tessellation around class means.
A linear probe learns a per-coordinate weighting and per-class offset, so it can
express decision boundaries the centroid rule cannot.

The measured effect (C10, ep-10 cov-shift):

| condition | R1-ceiling | R4-ceiling | R4/R1 |
| :--- | :--- | :--- | :--- |
| snow | 0.408 | 0.510 | 1.25 |
| wet_ground | 0.425 | 0.683 | 1.61 |
| fog | 0.261 | 0.433 | 1.66 |
| crosstalk | 0.461 | 0.594 | 1.29 |

Why R4 does so much better than R1 on the ceiling: the oracle re-estimates the
decoder from the corrupted pool. For prototypes, re-estimation just moves the class
means — it cannot re-weight the coordinates or change the boundary shape. For the
probe, re-estimation re-fits the full linear decision boundary on the corrupted
points, so it recovers structure the centroid rule cannot express even with perfect
labels.

The zero-shot (frozen clean-fit) comparison is smaller and mixed — R4-zs beats R1-zs
on healthy + crosstalk but is slightly worse on fog (-0.004). The big R4 gain is on
the oracle (pool-refit), which is the ceiling the label-free update chases.

## How we can stay gradient-free and efficient

The existing HDC TTA is gradient-free because the decoder update is statistics
accumulation: `weighted_mean_update` maintains per-class sums/counts of the sign
codes and re-normalizes. A linear probe `y = argmax(W @ code + b)` has the same
structure, so updating it does NOT require gradients. Three options, in increasing
sophistication:

### Option 1: bias/intercept-only update (prior correction)

Freeze `W` from the clean-fit probe; update only `b` on the corrupted stream. This
is the probe analog of the BN-statistic alignment lever (the best label-free TTA on
prior extractors), and it is the "prior correction" pathway the README already
separates from the weight pathway. Cost: O(classes), fully gradient-free.
Caveat: it shifts the boundary offset but does not reorient it, so it may be too
weak where the corruption changes the boundary direction.

### Option 2: closed-form ridge / least-squares with accumulated statistics

A least-squares classifier `min ||XW - Y||^2 + lambda ||W||^2` has the closed form

    W = (X^T X + lambda I)^{-1} X^T Y

Updated exactly like prototypes: accumulate the sufficient statistics `S = X^T X`
(10k x 10k, ~400MB float32, ~100MB as int32 counts since the codes are +/-1) and
`T = X^T Y` point-by-point from the corrupted stream (pseudo-labels), then do ONE
matrix solve. No gradient iterations. This is the natural generalization of the
prototype mean: prototypes accumulate first moments (class sums); this accumulates
second moments and inverts. Same "accumulate-and-solve" backprop-free philosophy.

### Option 3: FLDA / spread-aware linear discriminant

The diagnostics already compute per-class means and within-class scatter
(`corr_tight`). A Fisher linear discriminant `w = S_w^{-1} (mu_1 - mu_2)` is built
entirely from those class means + within-class scatter — no gradients — and it
re-tightens exactly the spread the C6 packing loss measured. It directly targets
the per-class packing that nearest-centroid and even a plain probe can miss.

### Design constraint (from the README)

Prior correction (option 1) and weight updates (options 2/3) must not share a
pathway. The prior is an inference-time constant that shifts decision boundaries; it
does not move the weights by itself. If prior-corrected pseudo-labels feed the
weight updates, the bias steers the weights and the drift compounds. Prediction may
use the prior-corrected score; adaptation must not.

### Why gradient-free matters

The gradient version (online SGD fine-tune of `W`) works but breaks the
"backprop-free" claim that is the point of the HDC prototype pipeline and Pillar 3.
Keeping the decoder update gradient-free preserves: (a) the efficiency (one matmul
decode, accumulate-and-solve update), (b) the label-free story (pseudo-label
statistics, no loss/backward), and (c) consistency with the active-learning
framework, which is also backprop-free.

## Iteration 0: the probe-gap diagnostic (2026-08-16)

Before designing the TTA/AL update, `probe_gap_diag.py` characterizes WHAT the
zero-shot -> labeled gap is. Results (cov-shift ep10/ep21, 4 conditions):

| cond (ep10) | zs | oracle | gap | cos(W_zs,W_or) | bias-only share | pool 1k / 10k / 100k |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| snow | 0.432 | 0.509 | +0.078 | 0.068 | 1% | 0.427 / 0.493 / 0.509 |
| wet_ground | 0.415 | 0.691 | +0.276 | 0.088 | 3% | 0.532 / 0.626 / 0.691 |
| fog | 0.234 | 0.433 | +0.199 | 0.028 | -1% | 0.309 / 0.401 / 0.433 |
| crosstalk | 0.497 | 0.588 | +0.091 | 0.112 | 0% | 0.486 / 0.555 / 0.588 |

The verdict has three parts, and it rules out the cheapest option:

1. **The gap is a ROTATION, not a translation.** `cos(W_zs, W_oracle)` is 0.03-0.12
   everywhere — the clean-fit probe's decision boundary and the pool-refit
   boundary point in almost completely different directions. A bias/intercept-only
   update (option 1) closes **0-4% of the gap** (bias-only share ~0-3%): freezing W
   and re-centering the intercept does essentially nothing. **Option 1 is dead as a
   TTA mechanism.** The weights must move.
2. **The oracle-fixed points are LOW-margin.** Frozen-probe margin: zs-correct
   ~12-16, zs-wrong ~7-9, oracle-fixed ~6-8. The points the labeled decoder fixes
   are exactly the confident-but-wrong boundary points — low margin but on the wrong
   side. So a margin gate CAN identify the points TTA needs to fix (the signal is
   there), but it cannot fix them: they need the boundary to move (weights), not a
   veto. This is the probe-space analog of the assignment wall, sharpened: the
   wall is a boundary-rotation problem, not a detection problem.
3. **The norm axis is DEGENERATE in the sign code.** Every point's L2 norm is
   sqrt(10000) = 100 (all entries are +-1), so zs-correct / zs-wrong / oracle-fixed
   all report 100.0. Outlier analysis must use the CONTINUOUS 128-d norm, not the
   binarized code — the diagnostic's norm axis is not meaningful as-is. This does
   not affect the other axes.

Per-class: the oracle gain is concentrated in a few classes per condition (fog: 0
+0.53, 7 +0.24, 11 +0.22, 14 +0.17; wet_ground: 14 +0.38, 2 +0.25, 7 +0.20, 15
+0.20) — those are the TTA/AL target classes. (Class 0 is the ignore class; the
others are real targets.)

Pool curve (the AL budget): a SMALL labeled pool closes most of the gap. 1k points
already capture 40-77% of the oracle gain (wet_ground 1k 0.532 vs 100k 0.691; fog
1k 0.309 vs 100k 0.433), and 10k points are near-saturated. The active-learning
budget to close the probe gap is small — consistent with Pillar 3's "one label per
cluster" structure.

**Implication for the update design.** The gap needs a WEIGHT update (rotation), so
the TTA/AL mechanism is option 2 (closed-form ridge / accumulate-and-solve on the
stream or a small labeled pool) or option 3 (FLDA from class scatter) — NOT option
1. The margin signal identifies which points drive the rotation (low-margin wrong
points), and the pool curve says a small labeled (or pseudo-labeled) pool suffices.
The open question becomes whether a label-free accumulate-and-solve refit (option 2
with pseudo-labels) closes most of the R4-oracle gap the way the pool curve does.

## Next: the label-free probe-update test

The open question is whether a label-free probe update climbs toward the R4-oracle
ceiling the way naive prototype TTA reaches the R1 ceiling. Add to
`hdc_rule_diag.py`:
- **naive probe-refit**: option 2 with pseudo-labels (accumulate S/T on the pool,
  one solve) — the label-free analog of the R4 oracle. Iteration 0 says this is the
  required mechanism (the gap is a rotation, weights must move).
- **bias-only**: option 1 (freeze W, update b from the pool class proportions) — the
  prior-correction baseline. Iteration 0 says this is NEGATIVE (0-4% of gap), so it
  is a control, not a candidate.
- **pool-curve at label-free sizes**: does 1k-10k PSEUDO-labeled points close the
  gap as well as 1k-10k TRUE-labeled points? (The iteration-0 curve is the labeled
  budget; the label-free question is how much pseudo-labels degrade it.)
- Efficiency columns (fit/decode/refit wall-clock, peak RSS) via
  `hdc_rule_bench.py` (the R1 vs R4 efficiency benchmark).

Verdict rule: if naive probe-refit recovers most of the R4-oracle ceiling on the
healthy conditions and crosstalk without hurting fog, the linear-probe decoder with
a gradient-free refit is the Pillar 2 update mechanism.
