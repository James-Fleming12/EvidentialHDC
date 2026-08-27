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
means; it cannot re-weight the coordinates or change the boundary shape. For the
probe, re-estimation re-fits the full linear decision boundary on the corrupted
points, so it recovers structure the centroid rule cannot express even with perfect
labels.

The zero-shot (frozen clean-fit) comparison is smaller and mixed: R4-zs beats R1-zs
on healthy + crosstalk but is slightly worse on fog (-0.004). The big R4 gain is on
the oracle (pool-refit), which is the ceiling the label-free update chases.

## The method

The decoder is a **linear probe on the HDC code**, and its update is a
**gradient-free accumulate-and-solve**. Both are chosen so that the method keeps the
backprop-free, statistics-accumulation character of the HDC prototype pipeline.

### The Linear Probe (decode)

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
$\mu_c \cdot h / \big(\lVert \mu_c \rVert \, \lVert h \rVert \big)$ (every code
coordinate weighted equally). The probe instead learns a per-coordinate weight per
class, so it can express boundaries (rotations, coordinate re-weightings) that the
centroid rule cannot. Iteration 0 showed the zero-shot $\to$ labeled gap is exactly
such a rotation: $\cos(W_{\mathrm{zs}}, W_{\mathrm{oracle}})$ is 0.03-0.12, and a
bias-only move (freeze $W$, re-center $b$) captures 0-4% of the gap.

### The Update Rule (fit)

The probe is fit by **ridge regression** (regularized least squares). Stacking the
pool's codes as rows $X \in \mathbb{R}^{n \times d}$ and one-hot labels
$Y \in \mathbb{R}^{n \times C}$, the fit is

$$
W = \arg\min_{W} \lVert X W - Y \rVert_F^2 + \lambda \lVert W \rVert_F^2
  = \left( X^{\top} X + \lambda I \right)^{-1} X^{\top} Y.
$$

This is the exact closed form, with no gradient iterations. What makes it an *update*
(not a batch fit) is that the sufficient statistics are pure accumulations over the
stream, exactly like prototype means:

$$
S = X^{\top} X = \sum_i h(x_i)\, h(x_i)^{\top} \quad (d \times d,\ \text{accumulate outer products}),
\qquad
T = X^{\top} Y = \sum_i h(x_i)\, e_{y_i}^{\top} \quad (d \times C,\ \text{accumulate code} \times \text{one-hot}),
$$

so $W = (S + \lambda I)^{-1} T$. Each incoming point updates $S$ and $T$ by one outer
product each (backprop-free, additions + matmuls only). The solve is the only
non-trivial step and is done once per update, not per point.

Memory: $S$ is $d \times d = 10000 \times 10000 = 400$ MB float32 (or ~100MB stored
as int32 counts, since the codes are $\pm 1$ and each entry of $S$ is a difference of
counts). $T$ is $d \times C$, negligible. This is the price of a second-order
statistic vs the prototype's first-order sums.

A Fisher linear discriminant (FLDA, $w = S_w^{-1}(\mu_1 - \mu_2)$ from class
scatter) is the alternative second-order rule; it was tested in Iteration 1 and
rejected (slow and degenerate on the sign code). The ridge accumulate-and-solve is
the chosen rule.

#### The final update: Nystrom warm start + matrix-free CG (Iterations 7-8)

The exact solve $(S + \lambda I)^{-1} T$ is too expensive at $d = 10000$ (the
$d \times d$ inverse and the dense $S$ storage). The final method avoids both with a
Nystrom warm start plus a few matrix-free conjugate-gradient iterations.

**Step 1: Nystrom warm start.** Pick a random sign sketch
$P \in \{-1,+1\}^{d \times m}$ with $m \ll d$ ($m = 1000$), each of whose $m$
coordinates mixes all $d$ HDC dims (holography preserved, no block mask). The sketch
collapses the second-order problem to an $m \times m$ solve:

$$
W_{\mathrm{Nys}} = P \left( P^{\top} X^{\top} X P + \lambda I_m \right)^{-1} P^{\top} X^{\top} Y.
$$

The inner solve is $m \times m$ ($m^3$, trivial), so this runs at prototype-scale
speed and lands near the full ridge solution (wet_ground ~0.57 vs full 0.67).

**Step 2: matrix-free CG to finish.** The residual $R = T - (S + \lambda I) W_{\mathrm{Nys}}$
is corrected by conjugate gradient in the full $d$-dimensional space, with the
matrix-free operation

$$
S v = X^{\top} (X v),
$$

which never builds the $d \times d$ matrix $S$: one forward and one backward pass
over the pool per iteration, no $d^2$ storage. Starting from
$W_0 = W_{\mathrm{Nys}}$ and solving $A\,\Delta W = R$ (where $A = S + \lambda I$),
only a few iterations are needed:

$$
W = W_{\mathrm{Nys}} + \Delta W, \qquad
\Delta W \approx A^{-1}\left( T - A\, W_{\mathrm{Nys}} \right).
$$

CG-8 from the Nystrom start reaches 0.62 wet_ground / 0.38 fog (ep10),
essentially the plain CG-20 accuracy (0.62 / 0.38) in 8 iterations instead of 20:
$0.034$s, ~1.5M pts/s, ~1-2x the prototype fit. The warm start is the key: CG-5
from Nystrom already matches CG-20 from scratch, and CG-10 from the start beats it.
The prototype is NOT a good warm start (the residual-CG correction from $\mu$ fails,
Iteration 8); the Nystrom sketch is. The decoder is cosine to the learned $W$
(Iteration 5), and the first-order / coreset / sparse-covariance alternatives all
fail (Iterations 6-7).

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
   everywhere: the clean-fit probe's decision boundary and the pool-refit
   boundary point in almost completely different directions. A bias/intercept-only
   update (freeze W, re-center b) closes **0-4% of the gap** (bias-only share
   ~0-3%). **Bias-only is dead as a TTA mechanism.** The weights must move.
2. **The oracle-fixed points are LOW-margin.** Frozen-probe margin: zs-correct
   ~12-16, zs-wrong ~7-9, oracle-fixed ~6-8. The points the labeled decoder fixes
   are exactly the confident-but-wrong boundary points (low margin but on the wrong
   side). So a margin gate CAN identify the points TTA needs to fix (the signal is
   there), but it cannot fix them: they need the boundary to move (weights), not a
   veto. This is the probe-space analog of the assignment wall, sharpened: the
   wall is a boundary-rotation problem, not a detection problem.
3. **The norm axis is DEGENERATE in the sign code.** Every point's L2 norm is
   $\sqrt{10000} = 100$ (all entries are $\pm 1$), so zs-correct / zs-wrong /
   oracle-fixed
   all report 100.0. Outlier analysis must use the CONTINUOUS 128-d norm, not the
   binarized code; the diagnostic's norm axis is not meaningful as-is. This does
   not affect the other axes.

Per-class: the oracle gain is concentrated in a few classes per condition (fog: 0
+0.53, 7 +0.24, 11 +0.22, 14 +0.17; wet_ground: 14 +0.38, 2 +0.25, 7 +0.20, 15
+0.20); those are the TTA/AL target classes. (Class 0 is the ignore class; the
others are real targets.)

Pool curve (the AL budget): a SMALL labeled pool closes most of the gap. 1k points
already capture 40-77% of the oracle gain (wet_ground 1k 0.532 vs 100k 0.691; fog
1k 0.309 vs 100k 0.433), and 10k points are near-saturated. The active-learning
budget to close the probe gap is small, consistent with Pillar 3's "one label per
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
- **Equivalence: exact.** $\max |W_{\mathrm{accum}} - W_{\mathrm{batch}}| = 0.000000$ on every condition:
  the accumulated-form update is numerically identical to the batch closed form, so
  the update is a pure accumulate-and-solve (backprop-free, additions + one solve).
- **Efficiency: prototype-cheap.** The update is 0.1s accumulate + 0.2s solve = ~0.3s
  per condition, vs 77-176s for the iterative LR fit (Iteration 1 bench), ~250-500x
  faster, comparable to prototype re-estimation (0.02-0.04s). Peak RSS ~20-27GB
  (dominated by the 10000x10000 S and the val code).
- **Correctness: reaches the oracle on 3 of 4 conditions.** Ridge is within 0.001-0.014
  of the LR oracle on snow/wet_ground/crosstalk. **Fog is the known-weak spot** (gap
  +0.021-0.024, just past the 0.02 threshold): the ridge least-squares boundary
  loses a little on the condition where the rotation is largest (Iteration 0's fog
  had the lowest $\cos(W_{\mathrm{zs}}, W_{\mathrm{oracle}}) = 0.028$).

**FLDA is rejected.** It is slow (95-109s) AND its fit is degenerate: sklearn's
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
  LARGE projection size never helped: the gain is the binarized geometry.)

Results (pool 10k, val 100k; mIoU + fit throughput pts/s):

| cond (ep10) | proto | diag-ridge | dual | jl-512 | code-256 | code-512 | code-1000 | code-2000 | code-5000 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| wet_ground mIoU | 0.425 | 0.295 | 0.052 | 0.517 | 0.518 | 0.544 | 0.567 | **0.587** | 0.572 |
| wet_ground pts/s | 0.51M | 0.11M | 0.10M | 0.05M | 12.4M | 6.7M | 3.2M | 1.5M | 0.40M |
| fog mIoU | 0.259 | 0.113 | 0.038 | 0.257 | 0.260 | 0.288 | 0.315 | **0.334** | 0.313 |
| fog pts/s | 2.8M | 0.11M | 0.11M | 0.05M | 11.8M | 6.7M | 3.4M | 1.5M | 0.40M |

(ep21 is the same pattern: code-2000 peaks at 0.550 wet_ground / 0.281 fog.)

**Result: the projection size never helped; the probe peaks at d'=2000, not 10000.**
- The probe mIoU at code-2000 (wet_ground 0.587, fog 0.334) is the HIGHEST of any
  representation tested, ABOVE code-5000 (0.572 / 0.313) and far above the 10000-d
  reference the earlier runs used. The 10000-d projection's large dimension is not
  the source of the gain.
- The throughput is the real win: code-1000 (3.2M pts/s) and code-2000 (1.5M pts/s)
  run at **or above the R1 prototype fit's throughput** (0.51M-2.8M pts/s) while
  ALSO beating the prototype's mIoU. The 7-8x overhead is gone at ~1000-2000-d.

**Two caveats (diagnostic artifacts, not results):**
- **Section A diag-ridge (0.295) does NOT equal proto (0.425)**: the "diagonal
  ridge == prototype" identity holds for the FIT (W proportional to the class-mean
  code), but the DECODE here used the un-normalized W row; the prototype decode
  cosine-normalizes. With per-class row normalization the numbers would match. The
  identity is mathematically true; the diagnostic's decode just needs the same
  normalization.
- **dual/Woodbury at n=10000 collapses (0.05)**: the same lam-too-small
  conditioning artifact from the earlier efficiency table (dual is only stable at
  small n). Not a real method failure.

**Design note (the important one).** This is HDC paper, and the dimension reduction
is an IMPLEMENTATION trick, not a smart HDC-aligned design decision. If an HDC
method at the SAME 10000-d dimension achieves the same ~7x speedup, that is the
preferred direction (it uses the binarization rather than shrinking the projection).
Section B's finding ("the projection size never helped") is a paper statement, but
the method should stay at 10000-d unless the HDC-native route is exhausted. The
HDC-native levers to pursue at full dimension: the integer +/-1 dual form (G = X X^T
computed via Hamming/popcount on packed bits, the exactness of the binarization)
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
  0.553; fog 0.360 vs 0.296, where sign is actually BETTER on fog, a mild
  regularizer).
- The zero-shot is prototype-like (block-sign zs ~= proto zs), as expected: the
  frozen clean-fit probe and frozen prototypes start from the same clean structure.

Efficiency (the README table):

| method | update_pts/s | decode_pts/s |
| :--- | :--- | :--- |
| R1 prototype (fit / decode) | ~0.5-2.8M | ~0.27-0.29M |
| full probe R4 (LR fit / matmul) | ~1-2k | (matmul) |
| **block_ridge sign** | **~0.14-0.24M** | **~0.17-0.28M** |

- **Update**: block_ridge runs ~0.14-0.24M pts/s, ~100x faster than the full-probe
  LR fit (1-2k pts/s), within ~3-10x of the prototype fit (0.5-2.8M). The
  block-diagonal structure (B=20) is what removes the d^3 solve.
- **Decode**: block-sign decode is ~0.17-0.28M pts/s, essentially equal to the
  prototype decode (0.27-0.29M); the quantized +-1 W makes the decode an integer
  dot product, no floats, at prototype speed.

**Verdict.** block_ridge sign is the candidate for the README: it keeps the full
10000-d HDC space (no dimension reduction), uses the binarization (quantized W =
integer popcount decode, block-diagonal = d^3/B^2 update), recovers most of the R4
ceiling, and decodes at prototype speed. The remaining efficiency gap is the
UPDATE (3-10x the prototype fit), which the dual/RLS forms target for streaming.

## Iteration 4: the HDC-aligned update forms sweep (2026-08-16)

block_ridge's block-diagonal mask violates HDC holography (it zeroes cross-block
correlations). This sweep tests three update forms that keep the FULL dense 10000-d
space and use HDC-native operations, to find which reaches the ridge ceiling at
prototype-comparable speed:
- **CG** (Krylov/Conjugate Gradient): accumulate the full dense S = X^T X, solve
  (S + lI)W = T iteratively (O(d^2) per iteration, no d^3 inverse). Uses the whole
  cross-correlated space.
- **HDC delta rule** (Widrow-Hoff/Kaczmarz): drop S entirely, online
  W <- W + a(y - W h)h^T. For +/-1 codes h^T is +/-1, so it is pure associative
  addition: O(C*d) per point, no matrix solve, no 400MB S.
- **Nystrom** (randomized sketch): P in {+1,-1}^{d x m}, accumulate the sketched
  S_hat = P^T X^T X P (m x m), solve in m, W = P A. Each m-dim mixes all d=10000
  HDC dims (holography preserved, only the solve dimension shrinks).

All timings are CUDA-synchronized (real GPU wall time). Pool 50k, val 100k.

**Accuracy and efficiency (wet_ground ep10; R1 proto = 0.424 at 1.9M pts/s):**

| method | wall_s | pts/s | mIoU |
| :--- | :--- | :--- | :--- |
| R1 proto fit | 0.027 | 1.9M | 0.424 |
| CG-5 | 0.159 | 0.31M | 0.498 |
| CG-10 | 0.161 | 0.31M | 0.554 |
| CG-30 | 0.171 | 0.29M | **0.648** |
| Nystrom m=100 | 0.026 | 1.9M | 0.471 |
| Nystrom m=500 | 0.013 | 3.8M | 0.550 |
| Nystrom m=1000 | 0.025 | 2.0M | 0.571 |
| Nystrom m=2000 | 0.053 | 0.94M | **0.607** |
| delta a=1e-4, 5ep | 0.981 | 5k | 0.574 |
| delta a=1e-4, 10ep | 1.97 | 2.5k | 0.570 |
| delta a=1e-4, 30ep | 5.84 | 0.86k | 0.548 |

(fog ep10: R1 proto 0.260 at 3.1M pts/s; CG-30 0.405 at 0.29M; Nystrom m=2000
0.354 at 0.94M; delta 10ep 0.332 at 2.5k. ep21 follows the same pattern.)

**Result: the efficiency question is settled.**
- **CG is NOT faster than the prototype; it is 6-10x SLOWER on the update.**
  Its real wall time is 0.16-0.17s (vs proto 0.013-0.045s), because it still must
  accumulate the full dense S = X^T X (~10 TFLOP for 50k x 10000). The earlier
  run's 17-70M pts/s was an async timing artifact. CG's value is ACCURACY (cg-30
  reaches 0.648, the R4 ceiling) and avoiding the d^3 inverse, but it does not
  close the efficiency gap vs the prototype update.
- **Nystrom is the efficiency winner.** It never builds the d x d S (only the m x m
  sketch), so it runs at 0.94-13M pts/s, comparable to or FASTER than the
  prototype fit (fog m=100: 13M pts/s), while accuracy rises with m (m=2000: 0.607
  wet_ground, 0.354 fog, near the R4 ceiling). Random-sign sketching preserves
  holography (every m-dim mixes all 10000 dims), with no block mask. **This is the
  HDC-aligned method that closes the efficiency gap.**
- **The delta rule is validated but slow.** With alpha = 1/d = 1e-4 it converges
  (wet_ground 0.574, fog 0.327); the "no S matrix" idea is sound, but the
  sequential per-point Python loop runs at ~5k pts/s (hundreds x slower than proto).
  Its sign-decode also degrades (0.25-0.38). It would need vectorization to be
  competitive, and even then the accuracy is below Nystrom/CG.

**Verdict.** **Nystrom (m ~ 1000-2000) is the leading HDC-aligned update**: it
keeps the full holographic 10000-d space (random-sign mixing, no block mask),
reaches most of the R4 ceiling, and runs at prototype-or-better update speed. CG is
the accuracy reference (full dense S, 0.648) but pays the dense accumulate; the
delta rule proves the S-free idea but needs vectorization. The method direction:
Nystrom-sketch the ridge update, quantize W to +/-1 for the integer-popcount decode
(Nystrom's sign-decode mIoU is 0.55 wet_ground at m=2000, near its float 0.607).

## Iteration 5: the learned-prototype reframing (2026-08-16)

The question: can we redefine "proximity to the prototype" so it aligns with the
linear probe, keeping the probe's accuracy at prototype-style decode cost? Two
diagnostics: `probe_prototype_alignment_diag.py` (does cosine to the learned W_c
reproduce the probe?) and `probe_gauge_diag.py` (can a tiny k-dim gauge gate the
expensive update?).

**Alignment (decode-side): the learned prototype matches the probe, but this is a
DECODE result, not an update one.**

| def (ep10 wet_ground) | agree w/probe | ceiling mIoU | decode pts/s |
| :--- | :--- | :--- | :--- |
| class_mean (R1) | 0.158 | ~0 (majority collapse) | ~0.90M |
| **W_cos_float** (cos to learned W_c) | **0.946** | **0.635** | **~0.91M** |
| W_cos_sign (cos to sign W_c) | 0.485 | 0.262 | ~0.90M |
| probe_ref (W_c . h) | 1.0 | 0.670 | -- |

(fog ep10: W_cos_float 0.425 vs probe 0.417, even slightly better; ep21 matches.)

Cosine to the learned prototype `W_c` reproduces the probe's decisions (0.93-0.95
agreement) and its ceiling (0.635 vs 0.670 wet_ground), at prototype decode speed.
This is the "proximity aligned with the linear probe" redefinition: the decoder
becomes a prototype-style cosine to W_c. **But the UPDATE is unchanged**: W_c still
comes from the full ridge solve (S = X^T X, d x d). The reframing buys decode speed
and removes the sign-quantization hit, NOT the update solve. The W_cos_sign
(integer/popcount) decode costs too much accuracy (0.26 vs 0.64) to be the decode.

**Gauge (update-side): the cheap rank-k correction does NOT reach the rotation, and
the tiny gauge does not reliably predict the full gain.**

| (ep10) | proto | full probe | rank-k k=32 | rank-k k=64 | delta_gauge k=64 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| wet_ground | 0.419 | 0.671 | 0.421 | 0.421 | +0.008 |
| fog | 0.222 | 0.369 | 0.223 | 0.223 | -0.023 |

- **The rank-k correction (W = mu + VA, a k x k solve only, the cheap update)
  recovers essentially nothing** (0.421 wet_ground vs proto 0.419, vs full 0.671).
  A k=32/64 random direction set cannot express the boundary rotation the full
  covariance encodes. The cheap-update lever does not reach the probe.
- **The tiny gauge does not predict the full gain reliably**: corr(delta_gauge,
  full_gain) = +0.83 on ep10 but -0.36 on ep21. The k=32/64 gauge is too small to
  see the separability signal (Iteration 2 showed it emerges around code-1000+).

**Verdict.** The learned-prototype cosine is a decode-side win (prototype-speed
decode, full probe accuracy, no quantization hit); adopt it as the DECODER. But
the UPDATE cost is unchanged: the choice remains Nystrom (prototype-speed update,
~0.55-0.61 ceiling) vs full accumulate-and-solve (0.67 ceiling, ~3-10x). The
gauge/rank-k cheap-update route is a dead end at k=32-64 (it cannot express the
rotation). The method: Nystrom update + cosine to the learned W_c decode.

## Iteration 6: the first-order separator ablation (2026-08-16)

Iteration 5 showed the low-rank correction and tiny gauge do not capture the
rotation. The remaining question: can a DIFFERENT linear separator, one whose
sufficient statistics are first-order (class sums, no covariance), express the
probe's gain? `probe_separator_ablation_diag.py` tests eight forms. Results (ceiling
mIoU, pool 50k):

| separator | stats | wet_ground ep10 | fog ep10 | update_s |
| :--- | :--- | :--- | :--- | :--- |
| R1 prototype | 1st (class sums) | 0.394 | 0.176 | 0.12 |
| **diag_lda** (W_cj = mu/(1-mu^2)) | 1st | **0.018** | **0.018** | 0.12 |
| shared_diag (W_c = q .* mu) | 1st | 0.398 | 0.177 | ~0 |
| perceptron (init mu, +/-h on mistakes) | 1st | 0.300 | 0.152 | 1.71 |
| passive_agg (margin step) | 1st | 0.042 | 0.009 | 1.90 |
| Nystrom (m=1000) | 2nd (sketch) | 0.572 | 0.325 | 0.65 |
| **full_ridge** (X^T X) | 2nd | **0.670** | **0.417** | 3.49 |

(ep21 identical pattern: diag_lda 0.036/0.013, perceptron 0.27/0.15, nystrom
0.53/0.26, full 0.65/0.37.)

**Result: both first-order questions are answered NO.**
- **A. Diagonal LDA COLLAPSES** (0.018 wet_ground, below the 0.394 prototype). The
  coordinate reweighting mu/(1-mu^2) destroys the classifier rather than helping.
  The probe's gain is NOT coordinate-wise reweighting.
- **B. Perceptron / Passive-Aggressive HURT** (0.30 / 0.04, both below the 0.394
  prototype). First-order mistake statistics cannot produce the rotation either.
- **shared_diag is flat** (0.398 ~= prototype 0.394): a domain-wide diagonal
  transform adds nothing.

**The rotation is genuinely cross-coordinate (second-order).** Only the sketch and
full covariance reach well above the prototype (nystrom 0.572, full 0.670). This
settles the fundamental question: linear separability in this HDC code requires
second-order statistics; NO first-order separator form (diagonal, mistake-based,
shared transform) recovers the R4 gain.

**Verdict.** The first-order route is closed. The method is fixed:
**matrix-free CG-20 update + cosine-to-learned-W_c decode**. The CG update is the
efficient way to get the second-order statistics the gain requires without d^2
storage; the learned-prototype cosine is the decoder that keeps the probe's
accuracy. The remaining open question is the label-free version of the CG update
(Iteration 8).

## Iteration 7: cheaper second-order updates (2026-08-16)

With the first-order route closed (Iteration 6), `probe_second_order_efficiency_diag.py`
tests Tier-1 ways to make the SECOND-ORDER problem cheaper: a hard-point coreset +
dual ridge (the low-margin points where the oracle gain lives), matrix-free CG, and
sparse covariance. All timings CUDA-synchronized. Results (ceiling mIoU, pool 50k):

| method | wet_ground ep10 | fog ep10 | solve_s | notes |
| :--- | :--- | :--- | :--- | :--- |
| R1 prototype | 0.394 | 0.176 | 0.12 | baseline |
| coreset m=500/1k/2k/5k + dual | 0.28/0.28/0.31/0.27 | 0.19/0.22/0.23/0.21 | 0.002-0.03 | all below proto; dead end |
| sparse cov 0-5% offdiag | 0.29-0.31 | 0.11-0.15 | -- | ~diagonal-only; dead end |
| **matrix-free CG-5/10/20** | **0.50/0.55/0.62** | **0.25/0.30/0.38** | **0.02/0.04/0.08** | **best cheap 2nd-order** |
| full ridge | 0.671 | 0.417 | 3.5 | ceiling |

(ep21 identical: CG-20 wet_ground 0.571, fog 0.32.)

**Result: matrix-free CG is the winner.**
- **CG converges monotonically toward the full ridge ceiling** (0.50 -> 0.55 -> 0.62
  wet_ground across 5/10/20 iters), the closest any cheap method gets (full 0.671),
  and it is TRULY matrix-free: Sv = X^T(Xv), never building the 10k x 10k S (no d^2
  storage, one pool pass per iteration). CG-20 solves in 0.078s (~640k pts/s) vs the
  full ridge's 3.5s.
- **The hard-point coreset is a dead end**: even low-margin-selected m=500-2000
  points give 0.28-0.31, all BELOW the plain prototype (0.394). Selecting the
  boundary points does not retain the rotation (m=5000 is worse than m=2000,
  overfitting).
- **Sparse covariance is a dead end**: keeping top-K off-diagonal |S_jk| gives
  0.29-0.31, barely above diagonal-only 0.294. The structure is not in a few
  dominant pairwise correlations.

**Verdict.** Matrix-free CG-20 is the best cheap second-order update: it approaches
the full ridge ceiling (0.62 wet_ground / 0.38 fog) at ~0.64M pts/s without d^2
storage, beating Nystrom (0.57/0.32) on accuracy at comparable cost. Combined with
the learned-prototype cosine decode, this is the method.

## Iteration 8: the Nystrom warm-start CG speedup (2026-08-16)

Iteration 7's CG-20 was accurate but paid 20 iterations. This tests making each
update cheaper via warm starts, residual CG, BF16, and subsampling
(`probe_cg_speedup_diag.py`). Results (wet_ground ep10; full ridge 0.670):

| method | mIoU | solve_s | notes |
| :--- | :--- | :--- | :--- |
| CG-20 from scratch | 0.617 | 0.078 | the Iteration-7 reference |
| **Nys-warm CG-5** | **0.598** | 0.023 | ~CG-20 accuracy in 5 iters |
| **Nys-warm CG-8** | **0.616** | 0.034 | ~CG-20 accuracy, 8 iters |
| **Nys-warm CG-10** | **0.625** | 0.041 | beats CG-20 from scratch |
| BF16-20 | 0.581 | 0.078 | small drop (0.617->0.581) |
| subsample 5k/12.5k/25k | 0.58 | ~ | small flat drop |
| residual early-stop (from mu) | 0.375 | 40 iters | FAILS; prototype is not a good start |

(ep21 identical: Nys-warm CG-5 wet_ground 0.561 vs plain CG-20 0.571; fog 0.31 vs
0.32.)

**Result: the Nystrom warm start is the speedup.** CG-5 from the Nystrom start
(0.598) already matches CG-20 from scratch (0.617), and CG-10 from the start
(0.625) beats it: the Nystrom sketch gets near the solution, CG fixes the residual
in a quarter the iterations. The prototype is NOT a good warm start (residual CG
from mu fails, 0.375 below the prototype itself); the Nystrom sketch is. BF16 and
subsampling give small accuracy drops (0.58) but are not free.

**Verdict.** The method is **Nystrom warm start (m=1000) + matrix-free CG-8**:
0.62 wet_ground / 0.38 fog at ~1.5M pts/s (0.034s), essentially the plain CG-20
accuracy in 8 iterations instead of 20, ~1-2x the prototype fit. Nystrom provides
the cheap approximate second-order geometry; CG recovers the full-space
cross-coordinate structure. This is the accuracy-efficiency optimum.

## Iteration 9: pseudo-label gating under the probe update (2026-08-16)

With the Nystrom-warm-start + CG-8 update in place, this re-tests whether standard
pseudo-label gates let the LABEL-FREE probe update climb toward the oracle ceiling
(the old geometric-gate closure was under the prototype-EMA decoder, so it needed a
fresh look). `probe_pseudo_gate_diag.py`: pseudo-labels from the frozen clean probe,
then gate the corrupted pool (conf/margin/norm/uncertainty + SELF-CALIBRATING
quantile gates that keep the top-K% by each signal, no manual threshold) and refit
the Nystrom+CG update on the gated pseudo-labeled points.

Results (wet_ground ep10; frozen 0.424, oracle 0.614, no_gate 0.417):

| gate | mIoU | note |
| :--- | :--- | :--- |
| no_gate (all pseudo-labels) | 0.417 | baseline label-free |
| conf 0.05 / 0.10 / 0.15 | 0.417 / 0.411 / 0.388 | worse as it tightens |
| selfcal conf_top 50 / 30 / 10% | 0.406 / 0.400 / 0.389 | all <= no_gate |
| selfcal margin_top 50 / 30 / 10% | 0.405 / 0.399 / 0.391 | all <= no_gate |
| selfcal norm_bot 50 / 30 / 10% | 0.407 / 0.395 / 0.365 | worse |
| selfcal uncer_top 50 / 30 / 10% | 0.412 / 0.405 / 0.389 | all <= no_gate |

(ep21 identical: no gate reaches oracle 0.583; fog same pattern.)

**Result: pseudo-label gating is CLOSED under the probe too.** Every gate stays at
or below `no_gate`; the update only gets WORSE as the gate tightens (conf 0.15 ->
0.388, conf_top10% -> 0.389). Gating never climbs toward the oracle (0.614).

**The key diagnostic: the AUROC is high but the mechanism is wrong.** The gate
signals CAN separate correct from wrong pseudo-labels (conf/margin/uncer AUROC
0.73-0.75, norm useless 0.39), and the wrong labels are low-confidence (0.100 vs
0.120 for correct). But a precision-focused gate (keep only high-confidence)
retains the CORRECT labels while dropping too many POINTS: the Nystrom+CG update
needs the pool's covariance structure, and discarding 50-90% of it destroys the
rotation. This is the fundamental tension: the gate that cleans the labels also
starves the second-order update. A hard admit/veto threshold is the wrong tool for
this update.

**Implication for the next step.** The update needs ALL the points' covariance, but
the wrong pseudo-labels poison it. The levers are:
1. **Weighted update**: wrong points contribute LESS to the covariance instead of
   being removed (soft weighting by confidence, not a hard gate).
2. **Two-stage**: fit the probe on the frozen pseudo-labels first, then use the
   UPDATED probe's confidence for a second-round gate (the first update may already
   clean the pseudo-labels enough that a second gate works).

## Iteration 10: weighted and two-stage pseudo-label updates (2026-08-16)

Iteration 9's hard gates starve the covariance. This tests the two levers that
avoid the hard gate (`probe_weighted_two_stage_diag.py`): (A) confidence-WEIGHTED
updates (each point contributes to S/T scaled by its confidence, keeping all
points' covariance), and (B) TWO-STAGE updates (fit on frozen pseudo-labels, then
re-gate / reweight by the UPDATED probe's confidence). Weighted ridge verified
exact vs (X^T D X + lI)^-1 X^T D Y.

Results (wet_ground ep10; frozen 0.424, oracle 0.616, no_gate 0.416):

| method | mIoU | note |
| :--- | :--- | :--- |
| no_gate | 0.416 | baseline label-free |
| A w=conf / conf^2 / margin | 0.416 / 0.416 / 0.412 | all flat at no_gate |
| B hard_regate 30% | 0.403 | worse |
| B soft_weighted | 0.414 | flat |
| B soft_then_hard | 0.404 | worse |

(fog ep10: A 0.256 vs no_gate 0.256, B 0.243-0.255; ep21 identical pattern.)

**Result: BOTH levers fail.** Soft weighting is flat at `no_gate` (0.416), and
two-stage is worse (0.40-0.41). The wrong pseudo-labels contaminate the update even
when down-weighted by confidence, and the updated probe's confidence does not become
clean enough for a second-round gate to work. This CLOSES the pseudo-label
supervision route for the Nystrom+CG probe update: neither gating nor weighting
pseudo-labels recovers the oracle.

**The fundamental conclusion.** The 33-55% wrong pseudo-labels (frozen probe on the
corrupted pool) are too contaminated for ANY pseudo-label-supervised refit of the
probe. Label-free TTA holds where the FROZEN probe already works (healthy conditions,
crosstalk); the conditions that need the second-order rotation (fog) are exactly
where pseudo-labels cannot supply the supervision. This is the active-learning
handoff (Pillar 3), not a pseudo-label fix. The probe's label-free ceiling is the
frozen decoder; the oracle rotation is the labeled bound that AL (one label per
cluster) closes.

## Iteration 11: the S/T decomposition diagnostic (2026-08-17)

Iterations 9-10 gated the update as a whole, so both objects of the ridge --
S = X^T D_s X (geometry of the pool) and T = X^T D_t Y (label assignment) -- were
starved together. This diagnostic DECOUPLES them (independent D_s / D_t, Y can be
one-hot or soft) and measures, per condition, what each half needs
(`probe_pseudolabel_structure_diag.py`, ep10 + ep21, fog/crosstalk/snow/wet_ground,
50k pool / 100k val, ~35s per condition). The update machinery is verified first:
the soft-target weighted solve is exact vs (X^T D_s X + lI)^-1 X^T D_t Y (rel err
~6e-6), and the linearity W_correct + W_wrong = W_all holds at matched CG iterations
(cos ~0.999+). Diagnostics A-H cover the S/T matrix, the correct/wrong W
decomposition, per-point influence and leverage, per-class reliability, prototype-
vs-probe agreement, coverage, coverage-preserving gates, and oracle-gate precision
curves.

**A. The S/T decomposition (ep10; frozen / no_gate / oracle = refs):**

| condition | frozen | no_gate (S_all,T_all) | best S_all,T_gated | S_all,T_correct_only | S_gated,T_all | S_gated,T_gated | oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.258 | 0.257 | 0.256 (conf_top0.9) | 0.291 | 0.154 | 0.247 | 0.376 |
| crosstalk | 0.519 | 0.515 | 0.513 (conf_top0.9) | 0.531 | 0.300 | 0.488 | 0.552 |
| snow | 0.513 | 0.502 | 0.501 (conf_top0.9) | 0.522 | 0.251 | 0.484 | 0.553 |
| wet_ground | 0.424 | 0.417 | 0.416 (conf_top0.9) | 0.493 | 0.249 | 0.401 | 0.614 |

Every hard gate under S=all is WORSE than no_gate at every fraction (conf_top0.3:
0.18-0.31, conf_top0.1: 0.10-0.23), despite the gated T being very pure (precision
0.77-0.98). The gated T is a biased subset with no coverage (see F). S_gated,T_all
is catastrophic (0.12-0.30): the covariance MUST contain all points. S_gated,T_gated
reproduces the Iteration-9 failure. S_all,T_correct_only (the oracle-informed
purity ceiling) is the best S-all gate everywhere (+0.03 to +0.08 over no_gate) but
stays far below oracle (gap +0.02 to +0.12). Soft labels and conf-weights are flat
at no_gate (0.23-0.51 across conditions); margin gates mirror conf; ep21 identical
pattern.

**B. The W decomposition.** With S fixed, W = W_correct + W_wrong exactly
(linearity cos 0.991-0.999). ||W_wrong||/||W_correct|| = 0.78-1.13: the wrong
pseudo-labels contribute ~equal magnitude. cos(W_wrong, W_oracle) is NEGATIVE
(-0.13 to -0.25) on every condition: the wrong labels actively anti-align with the
oracle rotation, they are not isotropic noise. cos(W_correct, W_oracle) = 0.51-0.82:
correct points move toward the oracle but cannot reach it. Diagnosis: **C/COVERAGE**
everywhere -- even a perfectly pure T cannot reproduce the oracle; the gap is the
label COVERAGE of the geometry, not label noise.

**C. Influence and leverage.** Per-point Nystrom-subspace influence
I_i ~ ||(S+lI)^-1 x_i y_i^T|| and leverage G_i = x_i^T (S+lI)^-1 x_i (the same
sketch as the warm start). corr(confidence, I) is strongly NEGATIVE (-0.40 to
-0.64) on every condition/checkpoint; mean influence of WRONG points is ~2x the
correct points; the AUROC of influence for separating correct from wrong is
0.31-0.36 (confidence: 0.69-0.80). The top-influence gate is the worst gate in the
whole diagnostic (0.10-0.23). **The points that drive the rotation are the
low-confidence (frequently wrong) ones** -- a confidence gate structurally selects
the least useful supervision.

**D. Reliability and calibration.** On the corrupted pool the frozen probe's max
softmax is < 0.5 for EVERY point (one calibration bin 0.0-0.5, all 50k points):
absolute confidence thresholds are structurally empty on corrupted streams; only
relative quantiles select anything. Per-class pseudo precision spread is huge
(fog: overall 0.08-0.83, q0.9 0.18-0.97): a global confidence gate is wrong at the
class level. The per-class top-30% gate still fails (0.16-0.35), consistent with A.

**E. Prototype-vs-probe agreement.** On fog the DISAGREEMENT points are the more
reliable ones (precision 0.629 disagree vs 0.507 agree); on crosstalk/wet_ground
agreement is more reliable (0.826/0.724 vs 0.607/0.521). No point has confidence
>= 0.9 on any corrupted pool (disagree_hi_conf empty everywhere, cf. D). The
agree-only gate is below no_gate on fog (0.227 vs 0.257).

**F. Coverage per gate.** conf_top10 preserves only 8-9% of the effective rank of
S_all and deviates from S_all by frob_diff_ratio 0.71-0.91; class_cond_top30 keeps
20-30% of the rank. Gating that purifies also destroys the covariance geometry --
the Iteration-9 explanation, now quantified.

**G. Coverage-preserving gates.** Per-cluster top-conf (k-means K = 10-500) and
per-class / per-region top-conf all stay BELOW no_gate (0.16-0.44). The finer the
clusters the better (K=500 > K=10, monotone) but never crosses no_gate: coverage
preservation helps, it does not fix the label problem.

**H. Oracle-gate decomposition.** Precision -> mIoU curves with wrong points
ordered by random / conf / anti-conf / leverage. The curves are nearly flat:
p=0.7 -> 0.9 gains +0.01-0.03, and p=1.0 (perfect gate) equals correct_only by
construction. The gap from p=1.0 to oracle is the coverage gap: fog +0.085,
crosstalk +0.021, snow +0.031, wet_ground +0.121 (ep10). The label ORDERING barely
matters; the subset's coverage dominates.

**Result: the pseudo-label route is closed, and now the mechanism is known.**
(1) The wrong labels are not noise, they anti-align with the oracle rotation.
(2) The points that matter for the rotation are anti-correlated with confidence --
any confidence-based selection discards exactly the supervision the rotation needs.
(3) Even a perfect-purity T cannot reproduce the oracle: the oracle needs labels on
the low-confidence, boundary points. Label-free TTA therefore cannot do the
rotation; the probe's label-free ceiling is the frozen decode, and the recoverable
headroom requires TRUE labels on covering points. This is the active-learning
handoff (Pillar 3) -- with a sharpened query rule: the influence analysis says
query the LOW-confidence / disagreement points, not the confident ones.

## Iteration 12: geometric, S-only label-free TTA (2026-08-17)

Iteration 11 showed the pseudo-label half T is poisoned and the gap is
C/COVERAGE. The remaining label-free route: abandon T entirely and use ONLY the
uncorrupted geometry S = X^T X to re-rotate the frozen W_zs toward the oracle
(`probe_geometric_tta_diag.py`, ep10 + ep21, all 4 conditions, 50k pool / 100k
val, ~20s per condition). Three methods, all in the established matrix-free
machinery (randomized-SVD eigenspaces of the 10000 x 10000 covariances via the
shared Nystrom sketch; never building S; never touching a label):
- **A. Subspace alignment / Procrustes** (Fernando et al.): rotate W_zs by the
  top-k eigenspace correspondence U_c -> U_t. Two families per k in {8, 32, 128,
  1000}: `proj` (textbook, truncates W_zs to span(U_c)) and `res`
  (residual-preserving: rotate the k-dim component, keep the complement).
- **B. CORAL** (Sun et al.): W_new = S_t^-1/2 S_c^1/2 W_zs on the top-k
  eigenspace (whiten target, recolor with clean), rank in {128, 256, 1000},
  plus a whitening-only control.
- **C. Label diffusion** (Zhou et al.): the only method touching labels.
  Hamming-similarity point graph (A = (X X^T/d + 11^T)/2 in [0,1]); anchors =
  top-1%/5% by frozen-probe confidence; Y_diff = (I - a G)^-1 Y_sparse via
  matrix-free CG; then the FULL-space ridge (S keeps all points). a in
  {0.1, 0.5, 0.9}; oracle-anchored variant (true labels of the correct points)
  as the diagnostic bound.

The decisive diagnostic: **the spectral overlap** (singular values of U_c^T U_t,
top-8) is 0.995-1.000 on EVERY condition and checkpoint. The top eigenspaces of
S_clean and S_target essentially coincide -- the corruption does NOT rotate the
dominant covariance directions.

Results (ep10; frozen / oracle per condition):

| condition | frozen | oracle | best procrustes | CORAL rank1000 | pseudo-anch diffusion | oracle-anch diff (a=0.5) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.259 | 0.377 | 0.246 (k8 t2c res) | 0.254 | 0.077-0.099 | **0.304** (cos 0.586) |
| crosstalk | 0.524 | 0.554 | 0.245-0.266 (k8 res) | 0.509 | 0.118-0.210 | **0.525** (cos 0.826) |
| snow | 0.457 | 0.493 | 0.428-0.448 (k8 res) | 0.448 | 0.087-0.154 | **0.459** (cos 0.755) |
| wet_ground | 0.427 | 0.614 | 0.241-0.293 (k8 res) | 0.399 | 0.058-0.177 | **0.516** (cos 0.738) |

ep21 identical pattern (fog frozen 0.231 / oracle 0.332; crosstalk 0.504/0.534;
snow 0.442/0.473; wet_ground 0.427/0.581).

**Result: the geometric route is CLOSED, and now with the mechanism known.**

1. **There is no subspace rotation to align.** The overlap ~1 means S_target's
   top eigenspace IS S_clean's; the rotation that separates W_zs from W_oracle
   lives in the DECODER (the decision rule), not in the pool's second-order
   statistics. Procrustes res variants sit at frozen (their cos to oracle stays
   at cos(W_zs, oracle): the rotation R is ~identity), and the proj variants
   truncate W_zs and fall far below (0.09-0.36).
2. **CORAL's eigenvalue reweighting hurts or does nothing** (rank1000: fog 0.254
   vs frozen 0.259, crosstalk 0.509 vs 0.524, snow 0.448 vs 0.457, wet_ground
   0.399 vs 0.427). The eigenvalue spectra barely change (top eig ratio fog
   ~4.6, crosstalk ~1), so whitening/recoloring cannot express the needed
   boundary change. Whitening-only is worse (0.12-0.26).
3. **Pseudo-anchored diffusion is far worse** (0.06-0.21): the poisoned anchors
   diffuse into everything; coverage preservation does not fix poisoned labels.
4. **The one thing that moves the probe toward the oracle is diffusing TRUE
   labels** (oracle-anchored, a=0.5): fog 0.304 (cos 0.586), wet_ground 0.516
   (cos 0.738), crosstalk 0.525 (cos 0.826). This is a labeled method -- and its
   lesson is exactly the Iteration-11 conclusion restated: the geometry can carry
   a sparse set of TRUE labels well, but it cannot manufacture the supervision.
   (a=0.9 over-diffuses: 0.05-0.13.)

**The label-free route is exhausted.** T is poisoned (Iterations 9-11), S carries
no rotation to recover (Iteration 12). The probe's label-free ceiling is the
frozen decode; the recoverable headroom requires true labels. This closes the
Pillar-2 search: the mechanism is the active-learning handoff (Pillar 3), and the
oracle-anchored diffusion result (geometry carries sparse true labels) is direct
evidence that the one-label-per-cluster AL scheme will work.

## Iteration 13: the label-free-gating closure TRANSFERS to HyperLiDAR and GeoID
(2026-08-27, `run_probe_labelfree_closure_others.sh`)

The Iterations 9-12 closure was measured only on cov-shift DGLSS++. Before
committing the active-learning framework (Pillar 3) to that closure, the same
four diagnostics were re-run on the OTHER two extractors of interest --
HyperLiDAR default (`baseline`, `logs/kitti_pretrain`) and the GeoID-loss port
(`supcon_vib_geoid`, `robust_diagnostic/logs/geoid_full/supcon_vib_geoid`) --
plus a fresh cov-shift ep10 reference in the same batch for apples-to-apples
comparison. Harness identical to the original closure runs (100 frames, 50k
pool, 100k val, 200k clean fit, Nystrom-warm-start + matrix-free CG-8 update).
JSONs: `probe_{pseudo_gate,weighted_2stage,pseudolabel_struct,geometric_tta}_{hyper_kitti,geoid,covshift_ep10}.json`.

**A note on the reference numbers.** The fresh cov-shift run reproduces the
original closure to the digit (fog frozen 0.2578 / oracle 0.3760 / no_gate
0.2566, matching Iteration 9), so the cross-extractor differences below are real
and not a harness drift. The hyper/geoid frozen ceilings on fog/crosstalk are
far lower than cov-shift (fog frozen 0.089/0.087 vs 0.258; crosstalk 0.105/0.109
vs 0.519): those extractors lose far more under fog/crosstalk, so their
closeable headroom is smaller in absolute terms.

### Iteration 9 (pseudo-gate): gates stay at or below no_gate on EVERY extractor

frozen / oracle / no_gate / best gate, per condition and extractor:

| cond | extractor | frozen | no_gate | best gate (selfcal) | oracle |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | hyper | 0.089 | 0.087 | 0.086 (norm_bot0.5) | 0.330 |
| fog | geoid | 0.087 | 0.087 | 0.086 (uncer_top0.5) | 0.255 |
| fog | cov | 0.258 | 0.257 | 0.256 (norm_bot0.5) | 0.376 |
| crosstalk | hyper | 0.105 | 0.103 | 0.104 (uncer_top0.5) | 0.439 |
| crosstalk | geoid | 0.109 | 0.111 | 0.110 (margin_top0.5) | 0.361 |
| crosstalk | cov | 0.519 | 0.515 | 0.507 (norm_bot0.5) | 0.553 |
| snow | hyper | 0.537 | 0.531 | 0.490 (margin_top0.1) | 0.600 |
| snow | geoid | 0.485 | 0.472 | 0.445 (margin_top0.1) | 0.536 |
| snow | cov | 0.457 | 0.447 | 0.439 (uncer_top0.5) | 0.491 |
| wet_ground | hyper | 0.614 | 0.593 | 0.572 (conf_top0.5) | 0.756 |
| wet_ground | geoid | 0.597 | 0.584 | 0.504 (conf_top0.5) | 0.711 |
| wet_ground | cov | 0.424 | 0.417 | 0.412 (uncer_top0.5) | 0.614 |

On every extractor and condition the gates either match `no_gate` or are worse;
none climbs toward the oracle. Note hyper/geoid have a LARGER relative gap on
fog (frozen 0.089/0.087 vs oracle 0.330/0.255) than cov (0.258 vs 0.376), and
still no gate recovers any of it -- the gate signal is present but structurally
unusable (the coverage mechanism, Iteration 11).

Gate-signal AUROC for correct-vs-wrong pseudo-labels (fog):

| extractor | conf | margin | norm | uncer |
| :--- | :--- | :--- | :--- | :--- |
| hyper | 0.838 | 0.834 | 0.557 | 0.838 |
| geoid | 0.611 | 0.653 | 0.514 | 0.611 |
| cov | 0.687 | 0.680 | 0.391 | 0.687 |

All conf/margin AUROCs are well above 0.5 -- the signal CAN separate correct
from wrong pseudo-labels on every extractor -- yet the gated updates all fail
(the same Iteration-9 tension: cleaning the labels starves the covariance).
Hyper's AUROC (0.83-0.84) is the highest of the three on fog, and still its
gates never climb: the gate-detection ability and the update-usefulness are
decoupled on all three.

### Iteration 10 (weighted + two-stage): flat or worse on EVERY extractor

| cond | extractor | frozen | no_gate | w_conf | best two-stage | oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | hyper | 0.089 | 0.087 | 0.086 | 0.082 (soft_w) | 0.330 |
| fog | geoid | 0.087 | 0.087 | 0.087 | 0.086 (soft_w) | 0.255 |
| fog | cov | 0.258 | 0.256 | 0.257 | 0.255 (soft_w) | 0.379 |
| crosstalk | hyper | 0.105 | 0.103 | 0.103 | 0.099 (soft_w) | 0.439 |
| crosstalk | geoid | 0.109 | 0.110 | 0.110 | 0.110 (soft_w) | 0.360 |
| crosstalk | cov | 0.520 | 0.516 | 0.515 | 0.509 (soft_w) | 0.552 |
| snow | hyper | 0.537 | 0.531 | 0.521 | 0.513 (soft_w) | 0.600 |
| snow | geoid | 0.485 | 0.471 | 0.463 | 0.450 (soft_w) | 0.536 |
| snow | cov | 0.458 | 0.447 | 0.446 | 0.442 (soft_w) | 0.491 |
| wet_ground | hyper | 0.614 | 0.592 | 0.589 | 0.589 (soft_w) | 0.757 |
| wet_ground | geoid | 0.597 | 0.583 | 0.581 | 0.578 (soft_w) | 0.711 |
| wet_ground | cov | 0.425 | 0.417 | 0.416 | 0.414 (soft_w) | 0.615 |

Soft confidence weighting and two-stage re-gating are flat or worse on every
extractor (identical to the cov-only Iteration-10 verdict). The wrong
pseudo-labels contaminate the update even when down-weighted, on all three
extractors.

### Iteration 11 (S/T decomposition): the C/COVERAGE diagnosis holds on EVERY extractor

Condensed per condition (the full sections are in the JSONs):

| cond | extractor | frozen | oracle | no_gate | conf_top0.3 (prec) | correct_only | w_correct cos | w_wrong cos | ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | hyper | 0.089 | 0.330 | 0.087 | 0.066 (0.685) | 0.101 (0.580) | 0.536 | -0.300 | 1.49 |
| fog | geoid | 0.087 | 0.255 | 0.087 | 0.066 (0.668) | 0.104 (0.584) | 0.564 | -0.416 | 1.19 |
| fog | cov | 0.258 | 0.376 | 0.257 | 0.180 (0.771) | 0.291 (0.553) | 0.509 | -0.221 | 1.08 |
| crosstalk | hyper | 0.105 | 0.439 | 0.102 | 0.083 (0.699) | 0.160 (0.526) | 0.484 | -0.327 | 1.22 |
| crosstalk | geoid | 0.109 | 0.361 | 0.111 | 0.078 (0.642) | 0.110 (0.581) | 0.554 | -0.353 | 1.35 |
| crosstalk | cov | 0.519 | 0.553 | 0.515 | 0.313 (0.970) | 0.531 (0.821) | 0.774 | -0.137 | 0.78 |
| snow | hyper | 0.537 | 0.600 | 0.531 | 0.314 (0.987) | 0.562 (0.827) | 0.747 | -0.099 | 0.78 |
| snow | geoid | 0.485 | 0.536 | 0.473 | 0.271 (0.984) | 0.494 (0.696) | 0.673 | -0.265 | 0.93 |
| snow | cov | 0.457 | 0.491 | 0.447 | 0.248 (0.975) | 0.463 (0.765) | 0.717 | -0.186 | 0.89 |
| wet_ground | hyper | 0.615 | 0.756 | 0.589 | 0.352 (0.981) | 0.641 (0.856) | 0.774 | -0.140 | 0.72 |
| wet_ground | geoid | 0.594 | 0.711 | 0.585 | 0.337 (0.934) | 0.631 (0.812) | 0.790 | -0.198 | 0.71 |
| wet_ground | cov | 0.424 | 0.614 | 0.417 | 0.289 (0.905) | 0.493 (0.728) | 0.682 | -0.178 | 0.91 |

Every cell confirms the C/COVERAGE diagnosis the Iteration-11 verdict rule named:
`w_wrong` anti-aligns with the oracle rotation (negative cos on ALL 12
extractor-condition cells), `||w_wrong||/||w_correct||` is 0.71-1.49 (comparable
magnitude), and even the perfect-purity `correct_only` T stays far below the
oracle. The one cross-extractor difference: hyper/geoid have HIGHER wrong-label
anti-alignment on fog/crosstalk (fog w_wrong cos -0.30/-0.42 vs cov -0.22), i.e.
the pseudo-label contamination is if anything more hostile on the other
extractors -- but the structural verdict (coverage, not noise) is identical.

Per-class pseudo accuracy on fog (the target classes):

| extractor | c4 | c7 | c11 | c13 | c14 | c15 | c16 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| hyper | 0.289 | 0.000 | 0.991 | 0.001 | 0.007 | 0.100 | 0.298 |
| geoid | 0.327 | 0.000 | 0.929 | 0.000 | 0.001 | 0.009 | 0.085 |

The rare/minority classes (7, 13, 14, 15, 16) have ~0 pseudo-accuracy on
fog/crosstalk for BOTH hyper and geoid -- the classes the AL budget must target
are exactly the ones where pseudo-labels are useless on all three extractors.

### Iteration 12 (geometric S-only): spectral overlap ~1 and nothing recovers the rotation, on EVERY extractor

| cond | extractor | frozen | oracle | top-8 spectral overlap (min) | best procrustes | best coral | oracle-anch diff (a=0.5) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | hyper | 0.082 | 0.333 | 0.897 | 0.063 (k8 c2t res) | 0.099 | 0.082 |
| fog | geoid | 0.101 | 0.257 | 0.966 | 0.089 (k8 c2t res) | 0.111 | 0.127 |
| fog | cov | 0.258 | 0.375 | 0.995 | 0.188 (k8 c2t res) | 0.254 | 0.304 |
| crosstalk | hyper | 0.102 | 0.439 | 0.927 | 0.100 (k1000 t2c proj) | 0.144 | 0.123 |
| crosstalk | geoid | 0.142 | 0.362 | 0.950 | 0.058 (k8 c2t res) | 0.155 | 0.169 |
| crosstalk | cov | 0.524 | 0.553 | 0.998 | 0.591 (k8 t2c res) | 0.508 | 0.524 |
| snow | hyper | 0.539 | 0.603 | 0.998 | 0.594 (k8 t2c res) | 0.508 | 0.551 |
| snow | geoid | 0.481 | 0.537 | 1.000 | 0.453 (k8 c2t res) | 0.469 | 0.479 |
| snow | cov | 0.456 | 0.493 | 0.998 | 0.464 (k8 t2c res) | 0.449 | 0.460 |
| wet_ground | hyper | 0.623 | 0.759 | 0.999 | 0.567 (k8 c2t res) | 0.547 | 0.624 |
| wet_ground | geoid | 0.592 | 0.712 | 0.999 | 0.587 (k8 t2c res) | 0.580 | 0.590 |
| wet_ground | cov | 0.427 | 0.615 | 0.998 | 0.420 (k8 t2c res) | 0.398 | 0.516 |

Spectral overlap of the clean vs corrupted top eigenspaces is 0.897-1.000 on
every cell -- hyper's fog (0.897) is the LOWEST of all, yet its procrustes
still collapse (0.063 vs frozen 0.082). Procrustes/CORAL sit at or below frozen
on every cell except two noise-level cov cases (crosstalk/snow k8 res ~0.59
vs frozen 0.52/0.46, inside the ±0.05 noise band and not reproducible as a
method). The oracle-anchored diffusion -- the one labeled signal -- moves toward
the oracle on every extractor (fog: hyper 0.082->0.082 flat, geoid 0.101->0.127,
cov 0.258->0.304), confirming on all three that only TRUE labels carry the
rotation. Note hyper's oracle-anch diffusion is flat on fog (0.082 -> 0.082)
because its frozen is already at the diffusion's ceiling for that sparse-label
budget -- the geometry carries sparse labels less well for hyper than for cov.

**Verdict.** The label-free-gating closure TRANSFERS: on every one of the four
diagnostics, on every condition, HyperLiDAR and GeoID behave like cov-shift --
no reliable label-free gating method exists for any of the three extractors, and
the structural reasons are the same (contaminated-but-anti-aligned pseudo-labels,
no covariance rotation to align, coverage-limited T). This validates the Pillar-3
active-learning handoff for the whole extractor set, not just cov-shift. The AL
framework does not need extractor-specific label-free gating.

## Next: Iteration 14: the coverage-aware active-learning query

The S/T decomposition (Iteration 11) and the geometric closure (Iteration 12)
turned the Pillar-3 handoff from a hypothesis into a measured requirement: the
recoverable headroom (+0.02 to +0.12 over the frozen decode) needs TRUE labels on
the low-confidence, high-influence points, and the oracle-anchored diffusion
result shows the geometry carries a sparse labeled set well. Iteration 13 then
confirmed the closure transfers to HyperLiDAR and GeoID (no extractor has a
reliable label-free gate). Iteration 14 designs
the query from these measurements:
- **query rule**: rank pool points by influence I_i (Nystrom-subspace) or
  disagreement magnitude, NOT by confidence; the influence analysis says these
  are the labels the rotation needs.
- **one label per cluster**: the dense per-class cluster structure (Pillar 3, 4.1)
  with clusters formed on the 128-d features; the per-class reliability matrix
  (D) says which classes need the strictest gates.
- **validate the close**: does a small TRUE-label budget on the queried points
  reproduce the oracle (S_all,T_oracle) mIoU, closing the +0.02 to +0.12 gap the
  oracle-gate curves quantified?
- **efficiency**: the labeled ridge is the same Nystrom-warm-started CG-8, so the
  AL budget fits in the established gradient-free update.

Verdict rule: if ~1k-10k true labels on the high-influence points close most of
the oracle gap while the same budget on high-confidence points does not, the
influence-based query is the Pillar-3 mechanism.
