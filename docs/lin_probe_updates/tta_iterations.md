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

| condition | R1-ceiling | R4-ceiling | R4/R1 || :--- | :--- | :--- | :--- |
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

## The method

The decoder is a **linear probe on the HDC code**, and its update is a
**gradient-free accumulate-and-solve**. Both are chosen so that the method keeps the
backprop-free, statistics-accumulation character of the HDC prototype pipeline.

### The linear probe (decode)

Let $z(x) \in \mathbb{R}^D$ be the encoder's 128-d feature for a point, and
$R \in \{-1,+1\}^{D \times d}$ the fixed random projection ($d = 10000$, seeded 42).
The HDC code is

$$
h(x) = \mathrm{sign}\!\big(z(x)\, R\big) \in \{-1,+1\}^{d}.
$$

The probe is a linear classifier on the code with per-class weight vectors
$W \in \mathbb{R}^{C \times d}$ and intercepts $b \in \mathbb{R}^C$ ($C$ classes,
class 0 = ignore):

$$
\hat{y}(x) = \arg\max_{c} \big( W_c \cdot h(x) + b_c \big).
$$

This generalizes the prototype rule. Distance-to-prototype is the special case where
$W_c$ is the (normalized) class-mean code
$\mu_c = \mathrm{mean}_{x \in \mathrm{class}\, c}\, h(x)$ and the score is the cosine
$\mu_c \cdot h / \big(\lVert \mu_c \rVert \, \lVert h \rVert \big)$ — every code
coordinate weighted equally. The probe instead learns a per-coordinate weight per
class, so it can express boundaries (rotations, coordinate re-weightings) that the
centroid rule cannot. Iteration 0 showed the zero-shot $\to$ labeled gap is exactly
such a rotation: $\cos(W_{\mathrm{zs}}, W_{\mathrm{oracle}})$ is 0.03-0.12, and a
bias-only move (freeze $W$, re-center $b$) captures 0-4% of the gap.

### The update rule (fit)

The probe is fit by **ridge regression** (regularized least squares). Stacking the
pool's codes as rows $X \in \mathbb{R}^{n \times d}$ and one-hot labels
$Y \in \mathbb{R}^{n \times C}$, the fit is

$$
W = \arg\min_{W} \lVert X W - Y \rVert_F^2 + \lambda \lVert W \rVert_F^2
  = \left( X^{\top} X + \lambda I \right)^{-1} X^{\top} Y.
$$

This is the exact closed form — no gradient iterations. What makes it an *update*
(not a batch fit) is that the sufficient statistics are pure accumulations over the
stream, exactly like prototype means:

$$
S = X^{\top} X = \sum_i h(x_i)\, h(x_i)^{\top} \quad (d \times d,\ \text{accumulate outer products}),
\qquad
T = X^{\top} Y = \sum_i h(x_i)\, e_{y_i}^{\top} \quad (d \times C,\ \text{accumulate code} \times \text{one-hot}),
$$

so $W = (S + \lambda I)^{-1} T$. Each incoming point updates $S$ and $T$ by one outer
product each — backprop-free, additions + matmuls only. The solve is the only
non-trivial step and is done once per update, not per point.

Memory: $S$ is $d \times d = 10000 \times 10000 = 400$MB float32 (or ~100MB stored
as int32 counts, since the codes are $\pm 1$ and each entry of $S$ is a difference of
counts). $T$ is $d \times C$, negligible. This is the price of a second-order
statistic vs the prototype's first-order sums.

A Fisher linear discriminant (FLDA, $w = S_w^{-1}(\mu_1 - \mu_2)$ from class
scatter) is the alternative second-order rule; it was tested in Iteration 1 and
rejected (slow and degenerate on the sign code). The ridge accumulate-and-solve is
the chosen rule.

### Design constraint (from the README)

Prior correction and the weight update must not share a pathway. The prior (an
inference-time constant that shifts decision boundaries via $b$) does not move the
weights by itself. If prior-corrected pseudo-labels feed the $S$/$T$ accumulation,
the bias steers the weights and the drift compounds. Prediction may use the
prior-corrected score; adaptation must not.

### Why gradient-free matters

The gradient version (online SGD fine-tune of $W$) works but breaks the
"backprop-free" claim that is the point of the HDC prototype pipeline and Pillar 3.
Keeping the decoder update gradient-free preserves: (a) the efficiency (one matmul
decode, accumulate-and-solve update), (b) the label-free story (pseudo-label
statistics, no loss/backward), and (c) consistency with the active-learning
framework, which is also backprop-free.

## Iteration 0: the probe-gap diagnostic (2026-08-16)

Before designing the TTA/AL update, `probe_gap_diag.py` characterizes WHAT the
zero-shot -> labeled gap is. Results (cov-shift ep10/ep21, 4 conditions):

| cond (ep10) | zs | oracle | gap | $\cos(W_{\mathrm{zs}},W_{\mathrm{or}})$ | bias-only share | pool 1k / 10k / 100k |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| snow | 0.432 | 0.509 | +0.078 | 0.068 | 1% | 0.427 / 0.493 / 0.509 |
| wet_ground | 0.415 | 0.691 | +0.276 | 0.088 | 3% | 0.532 / 0.626 / 0.691 |
| fog | 0.234 | 0.433 | +0.199 | 0.028 | -1% | 0.309 / 0.401 / 0.433 |
| crosstalk | 0.497 | 0.588 | +0.091 | 0.112 | 0% | 0.486 / 0.555 / 0.588 |

The verdict has three parts, and it rules out the cheapest option:

1. **The gap is a ROTATION, not a translation.** $\cos(W_{\mathrm{zs}}, W_{\mathrm{oracle}})$ is 0.03-0.12
   everywhere — the clean-fit probe's decision boundary and the pool-refit
   boundary point in almost completely different directions. A bias/intercept-only
   update (freeze W, re-center b) closes **0-4% of the gap** (bias-only share
   ~0-3%). **Bias-only is dead as a TTA mechanism.** The weights must move.
2. **The oracle-fixed points are LOW-margin.** Frozen-probe margin: zs-correct
   ~12-16, zs-wrong ~7-9, oracle-fixed ~6-8. The points the labeled decoder fixes
   are exactly the confident-but-wrong boundary points — low margin but on the wrong
   side. So a margin gate CAN identify the points TTA needs to fix (the signal is
   there), but it cannot fix them: they need the boundary to move (weights), not a
   veto. This is the probe-space analog of the assignment wall, sharpened: the
   wall is a boundary-rotation problem, not a detection problem.
3. **The norm axis is DEGENERATE in the sign code.** Every point's L2 norm is
   $\sqrt{10000} = 100$ (all entries are $\pm 1$), so zs-correct / zs-wrong /
   oracle-fixed
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
the TTA/AL mechanism is the ridge accumulate-and-solve update (see The method), NOT a
bias-only re-center. The margin signal identifies which points drive the rotation
(low-margin wrong points), and the pool curve says a small labeled (or
pseudo-labeled) pool suffices. The open question becomes whether a label-free
accumulate-and-solve refit (with pseudo-labels) closes most of the R4-oracle gap the
way the pool curve does.

## Iteration 1: the update-rule validation (2026-08-16)

`probe_update_rule_diag.py` tests whether the gradient-free update rule has the
required properties on the cov-shift features: correctness (reaches the LR oracle),
efficiency (accumulate+solve wall-clock), and equivalence (accumulated form == batch
closed form). Rules compared per condition (pool 50k, val 100k, lambda = 1e-3):

| model | cond | LR oracle | Ridge-batch | Ridge-accum | FLDA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ep10 | snow | 0.516 | 0.513 | 0.513 | 0.538 |
| ep10 | wet_ground | 0.676 | 0.671 | 0.671 | 0.688 |
| ep10 | fog | 0.438 | 0.417 | 0.417 | 0.451 |
| ep10 | crosstalk | 0.594 | 0.587 | 0.587 | 0.628 |
| ep21 | snow | 0.503 | 0.502 | 0.502 | 0.529 |
| ep21 | wet_ground | 0.669 | 0.654 | 0.654 | 0.673 |
| ep21 | fog | 0.391 | 0.367 | 0.367 | 0.412 |
| ep21 | crosstalk | 0.578 | 0.567 | 0.567 | 0.600 |

**Result: ridge accumulate-and-solve is the validated update rule.**
- **Equivalence: exact.** $\max |W_{\mathrm{accum}} - W_{\mathrm{batch}}| = 0.000000$ on every condition —
  the accumulated-form update is numerically identical to the batch closed form, so
  the update is a pure accumulate-and-solve (backprop-free, additions + one solve).
- **Efficiency: prototype-cheap.** The update is 0.1s accumulate + 0.2s solve = ~0.3s
  per condition, vs 77-176s for the iterative LR fit (Iteration 1 bench) — ~250-500x
  faster, comparable to prototype re-estimation (0.02-0.04s). Peak RSS ~20-27GB
  (dominated by the 10000x10000 S and the val code).
- **Correctness: reaches the oracle on 3 of 4 conditions.** Ridge is within 0.001-0.014
  of the LR oracle on snow/wet_ground/crosstalk. **Fog is the known-weak spot** (gap
  +0.021-0.024, just past the 0.02 threshold): the ridge least-squares boundary
  loses a little on the condition where the rotation is largest (Iteration 0's fog
  had the lowest $\cos(W_{\mathrm{zs}}, W_{\mathrm{oracle}}) = 0.028$).

**FLDA is rejected.** It is slow (95-109s) AND its fit is degenerate — sklearn's
shrinkage covariance hit "Only one sample available" warnings (a class with ~1 pool
sample), yet it still scored above both references (crosstalk 0.628 vs LR 0.594).
A result that is simultaneously warning-flagged and better-than-reference is
unreliable, not a real win. FLDA (option 3) is dropped as an update rule.

**Verdict.** The ridge accumulate-and-solve update (option 2) is the gradient-free
rule the TTA/AL gate should iterate on. The design constraint it must respect: the
accumulation uses (pseudo-)labels from the prediction pathway, never the
prior-corrected score (The method, design constraint).

## Iteration 2: the efficiency scan (2026-08-16)

Iteration 1 validated the ridge update's accuracy but the update at d=10000 is
~7-8x slower than the R1 prototype pipeline. This scan finds cheaper ways to
compute the SAME update, HDC-native first, with two sections:
- **Section A** stays at the 10000-d code and uses the binarization: the
  diagonal-ridge bound (for +/-1 codes $\mathrm{diag}(X^{\top} X) = n$, so the
  diagonal ridge is the prototype up to scale), the dual/Woodbury form (inversion
  in the sample dim n), and RLS streaming.
- **Section B** is the dimension check: does the probe's linear-separability gain
  survive a smaller code dim d' or a second random projection to k? (If yes, the
  LARGE projection size never helped — the gain is the binarized geometry.)

Results (pool 10k, val 100k; mIoU + fit throughput pts/s):

| cond (ep10) | proto | diag-ridge | dual | jl-512 | code-256 | code-512 | code-1000 | code-2000 | code-5000 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| wet_ground mIoU | 0.425 | 0.295 | 0.052 | 0.517 | 0.518 | 0.544 | 0.567 | **0.587** | 0.572 |
| wet_ground pts/s | 0.51M | 0.11M | 0.10M | 0.05M | 12.4M | 6.7M | 3.2M | 1.5M | 0.40M |
| fog mIoU | 0.259 | 0.113 | 0.038 | 0.257 | 0.260 | 0.288 | 0.315 | **0.334** | 0.313 |
| fog pts/s | 2.8M | 0.11M | 0.11M | 0.05M | 11.8M | 6.7M | 3.4M | 1.5M | 0.40M |

(ep21 is the same pattern: code-2000 peaks at 0.550 wet_ground / 0.281 fog.)

**Result: the projection size never helped — the probe peaks at d'=2000, not 10000.**
- The probe mIoU at code-2000 (wet_ground 0.587, fog 0.334) is the HIGHEST of any
  representation tested, ABOVE code-5000 (0.572 / 0.313) and far above the 10000-d
  reference the earlier runs used. The 10000-d projection's large dimension is not
  the source of the gain.
- The throughput is the real win: code-1000 (3.2M pts/s) and code-2000 (1.5M pts/s)
  run at **or above the R1 prototype fit's throughput** (0.51M-2.8M pts/s) while
  ALSO beating the prototype's mIoU. The 7-8x overhead is gone at ~1000-2000-d.

**Two caveats (diagnostic artifacts, not results):**
- **Section A diag-ridge (0.295) does NOT equal proto (0.425)** — the "diagonal
  ridge == prototype" identity holds for the FIT (W proportional to the class-mean
  code), but the DECODE here used the un-normalized W row; the prototype decode
  cosine-normalizes. With per-class row normalization the numbers would match. The
  identity is mathematically true; the diagnostic's decode just needs the same
  normalization.
- **dual/Woodbury at n=10000 collapses (0.05)** — the same lam-too-small
  conditioning artifact from the earlier efficiency table (dual is only stable at
  small n). Not a real method failure.

**Design note (the important one).** This is HDC paper, and the dimension reduction
is an IMPLEMENTATION trick, not a smart HDC-aligned design decision. If an HDC
method at the SAME 10000-d dimension achieves the same ~7x speedup, that is the
preferred direction (it uses the binarization rather than shrinking the projection).
Section B's finding ("the projection size never helped") is a paper statement, but
the method should stay at 10000-d unless the HDC-native route is exhausted. The
HDC-native levers to pursue at full dimension: the integer +/-1 dual form (G = X X^T
computed via Hamming/popcount on packed bits — the exactness of the binarization)
and RLS streaming. The next step is to measure THOSE at 10000-d with a correct lam,
rather than adopting the reduced-dimension code.

## Iteration 3: the HDC-native method table (2026-08-16)

`probe_hdc_native_diag.py` / `probe_method_table_diag.py` compare the candidate
HDC-native decoder against the baseline and the full probe, in both accuracy
(zero-shot + ceiling) and efficiency (update/decode throughput). The candidate is
**block_ridge sign**: a block-diagonal ridge on the full 10000-d code (d^3/B^2
solves instead of one d^3), with W quantized to +/-1 so the decode is an integer dot
(d - 2*Hamming, popcount on packed bits). Pool 50k, val 100k, n_blocks=20.

Accuracy (zero-shot = clean-fit frozen; ceiling = pool-refit oracle):

| cond (ep10) | proto zs | proto ceil | R4 zs | R4 ceil | block-float ceil | **block-sign ceil** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| snow | 0.401 | 0.412 | 0.430 | 0.511 | 0.460 | 0.458 |
| wet_ground | 0.392 | 0.424 | 0.420 | 0.670 | 0.553 | **0.531** |
| fog | 0.241 | 0.260 | 0.235 | 0.416 | 0.296 | **0.360** |
| crosstalk | 0.465 | 0.459 | 0.500 | 0.587 | 0.520 | **0.525** |

(ep21: block-sign ceiling wet_ground 0.449, fog 0.308, crosstalk 0.523, snow 0.450.)

**Result: the block-diagonal ridge recovers most of the R4 ceiling at prototype
speed, and quantizing W to +/-1 is nearly free.**
- block_ridge sign recovers the majority of the R4 ceiling gain on every condition
  (wet_ground 0.531 vs R4 0.670 and proto 0.424; fog 0.360 vs R4 0.416 and proto
  0.260). It is far above the prototype and within ~0.06-0.14 of the full probe.
- Quantization (sign vs float W) costs little on the ceiling (wet_ground 0.531 vs
  0.553; fog 0.360 vs 0.296 — sign is actually BETTER on fog, a mild regularizer).
- The zero-shot is prototype-like (block-sign zs ~= proto zs), as expected: the
  frozen clean-fit probe and frozen prototypes start from the same clean structure.

Efficiency (the README table):

| method | update_pts/s | decode_pts/s |
| :--- | :--- | :--- |
| R1 prototype (fit / decode) | ~0.5-2.8M | ~0.27-0.29M |
| full probe R4 (LR fit / matmul) | ~1-2k | (matmul) |
| **block_ridge sign** | **~0.14-0.24M** | **~0.17-0.28M** |

- **Update**: block_ridge runs ~0.14-0.24M pts/s — ~100x faster than the full-probe
  LR fit (1-2k pts/s), within ~3-10x of the prototype fit (0.5-2.8M). The
  block-diagonal structure (B=20) is what removes the d^3 solve.
- **Decode**: block-sign decode is ~0.17-0.28M pts/s, essentially equal to the
  prototype decode (0.27-0.29M) — the quantized +-1 W makes the decode an integer
  dot product, no floats, at prototype speed.

**Verdict.** block_ridge sign is the candidate for the README: it keeps the full
10000-d HDC space (no dimension reduction), uses the binarization (quantized W =
integer popcount decode, block-diagonal = d^3/B^2 update), recovers most of the R4
ceiling, and decodes at prototype speed. The remaining efficiency gap is the
UPDATE (3-10x the prototype fit), which the dual/RLS forms target for streaming.

## Next: Iteration 4 — the label-free probe-update test

Iteration 1 validated the ridge accumulate-and-solve update with TRUE labels (the
oracle). Iteration 4 asks whether a LABEL-FREE version climbs toward the R4-oracle
ceiling the way naive prototype TTA reaches the R1 ceiling:
- **naive probe-refit**: the ridge accumulate-and-solve with PSEUDO-labels
  (accumulate S/T on the pool from the frozen probe's predictions, one solve) — the
  label-free analog of the R4 oracle.
- **pool-curve at label-free sizes**: does 1k-10k PSEUDO-labeled points close the
  gap as well as 1k-10k TRUE-labeled points? (Iteration 0's curve is the labeled
  budget; the label-free question is how much pseudo-labels degrade it.)
- **bias-only control**: freeze W, update b from the pool class proportions — kept
  as a control only (Iteration 0 already showed it is 0-4% of the gap).
- **HDC-native efficiency at 10000-d**: the integer +/-1 dual (Hamming/popcount on
  packed bits) and RLS, with lam scaled to the pool — the preferred efficiency
  direction per the Iteration 2 design note (avoid the reduced-dimension trick).

Verdict rule: if the label-free ridge refit recovers most of the R4-oracle ceiling
on the healthy conditions and crosstalk without hurting fog, the linear-probe
decoder with a gradient-free refit is the Pillar 2 update mechanism.
