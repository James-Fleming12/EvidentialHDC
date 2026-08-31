# Class-statistics decoder: the reformulated update mechanism

This doc tracks the diagnostics for the class-statistics decoder reformulation,
the concrete "reformat what we store/update" direction that survived the
W-update and acquisition closures. It is written WHITE-BOX: each test splits the
construction into measured pieces so we can see what works and what needs fixing,
rather than treating the method as a blackbox.

## Method: the PROPAGATED-MEAN decoder (the first thing in the arc that provides
value)

### First implementation (Iteration 10 + validation)

The class-statistics line established that the MEAN decoder
W = Sigma0^-1 M^T P0 (the linear probe IS the whitened class means) is
decision-relevant: W_mean_oracle (oracle means) closes +0.50 to +1.15 gc and
matches oracle decisions on 0.59-0.95 of the error points. Every prior attempt
to estimate the oracle means M* from few labels failed (raw 8-point means at 35x
overstep, pseudo-means +0.05-0.12, confusion correction +0.01-0.05). The first
method that estimates M* well enough to give real positive gc is:

    PIPELINE (the first implementation)
    1. FROZEN probe W0 (clean ridge) and the unlabeled CORRUPTED pool.
    2. ANCHORS: select b points per class at random (RANDOM beats influence for
       mean estimation, V3), label them (the AL query). Budget = 17 * b.
    3. PROPAGATE: assign every pool point to its NEAREST anchor in the 128-d
       feature space (the informative space, packing P ~ 0.81-0.93; the 10000-d
       code space is saturated). This multiplies the effective labeled set
       (D1: anchors alone cos 0.76 -> propagated cos 0.89 at b=1).
    4. MEANS: aggregate the propagated labels into class means M_prop (in the
       10000-d code space) with the propagated counts C_prop.
    5. DECODE: W = Sigma^-1 (M_prop^T P_prop), whitened mean decoder.

    RESULT (DGLSS++ fog/crosstalk, 112-125 labels): +0.045 fog / +0.095
    crosstalk mIoU (gc +0.25 / +0.40), the strongest few-label result since
    Iteration 1. The mechanism is real, but the absolute gain is modest and the
    headroom is UNKNOWN (validation note).

    KNOWN PROPERTIES (measured):
    - random anchors > influence (influence picks boundary points, wrong for
      means, V3);
    - the clean-source bank (label-free propagation from clean labels) does NOT
      work for the mean decoder (V2);
    - per-point propagation precision is LOW (0.28-0.48) yet the aggregated
      mean decoder works; the mean aggregation is robust to per-point label
      noise;
    - class 11 is the tightest (propagates best), {7, 13, 14} the loosest
      (need the budget);
    - the true headroom vs W_mean_oracle is UNMEASURED (the oracle-count
      comparison was count-reference-mismatched).

    This is the method the rest of the doc's diagnostics are improving; the next
    validation step is the clean headroom decomposition (see the validation
    note at the bottom).

## Component status (current)

| Component | Status |
| :--- | :--- |
| W = Sigma^-1 M^T P reformulation (the probe IS the whitened class means) | Strongly validated |
| Whitening is essential (proto_oracle << W_mean_oracle) | Strongly validated |
| Mean-shift M* contains the oracle signal (W_mean_oracle +0.72 to +1.15 gc) | Strongly validated |
| Raw 2-8 label sample-mean -> whiten | Closed |
| Raw few-label shift estimate -> whiten | Closed |
| Naive pseudo-mean + high-dim label bias correction | Closed |
| Generic whitening regularization (lam_w) | Closed (no effect at these scales) |
| Arbitrary raw W-update | Closed / deprioritized |
| Label-free pool pseudo-mean (hard) | Real but weak positive (+0.05-0.12) |
| **Nearest-anchor PROPAGATED mean decoder** | **POSITIVE, reproducible (+0.045 fog / +0.095 crosstalk mIoU at ~112-125 labels), the strongest few-label result since Iteration 1 (Iteration 10); headroom UNKNOWN (count-reference mismatch muddles the ceiling claim, validation note)** |
| Structure multiplies anchors (prop > anchors alone on mean cos) | Validated (D1: b1 0.76->0.89, b2 0.87->0.95) |
| Clean-source bank (memory-bank idea, label-free) | Better label set than nearest-anchor on hard conditions (dglsspp fog 0.63 vs 0.31), worth combining with the mean decoder |
| Agreement-gated propagation | Raises per-point precision (0.50-0.66) but drops coverage (0.26-0.47), the Iteration-5 tradeoff |
| Boundary as a label source | Closed (C1: precision INCREASES with margin; the bulk transfers, not the boundary) |
| Pool directions v_c point at the true shift (raw-space) | Validated (align +0.83-0.92 raw) |
| Pool direction SURVIVES decoder geometry | Validated (Iteration 6: per-class align ~0.9-1.0; H1 rejected) |
| Mean direction robust to ~50% corruption | Partially misleading (corrupted whitened diff) but direction IS robust per-class |
| Confusion-matrix correction (Q, C x C) | Works, SATURATES at +0.05 (Iteration 4) |
| Soft / TTA pseudo-mean | Same alignment as hard (Iteration 6); scheme not the issue |
| Pool basis + scalar coefficients (labels estimate alpha in R^K) | FAILS but NOT because the direction is wrong: the per-class mean shift is 2-117x too large and orthogonal to the class's OWN residual column (Iteration 6 G) |
| Class-prior correction (P* vs P0) | Closed (negligible, same-scan P*~P0) |
| Scalar/coefficient estimation | Closed as an object for the PER-CLASS mean shift; the mean shift is not the decision-relevant object |
| Whitening at any rank/fractional power | Not the culprit (Iteration 6 D/E: alignment high, gc negative at all ranks) |
| Pairwise decision decomposition (mean/prior/cov) | Measured (Iteration 7 A): 99.98% of the measured pairwise residual is covariance; mean ~0.2% (PRIOR-WEIGHTED combination cancels, not individual shifts) |
| Mean-only decoder vs oracle (dec_agree) | DISAGREES on the big-gap conditions (fog: 0.509-0.639 on errors): mean correction is NOT the decision correction |
| Pairwise pseudo-mean direction d_ab = Sigma^-1(v_a-v_b) | Closed (degenerate: V_a-V_b ~ 0); closes the v_a-v_b CONSTRUCTION, not pairwise pool stats in general |
| Boundary-local mean displacement | Closed (align 0.01-0.13); boundary-local covariance/density untested |
| TTA decision-space displacement | Inconsistent (not a stable directed estimate); TTA still useful for uncertainty/acquisition |
| Pairwise affine LOGIT correction (alpha,beta) | ONLY tested positive mechanism (oracle +0.02-0.06, few-label sometimes reproduces), a small signal that needs a ceiling test |
| Covariance-only DECODER ceiling | REQUIRED before closing (the vector decomposition is residually defined; needs W_cov,oracle = W0 + R_cov to reproduce the oracle classifier) |
| Full-class (not top-K) mean correction | OPEN: H shows the correction must cover all classes, not the 3-5 largest raw shifts |

The headline: the direction IS correct and survives whitening (Iteration 6), but
the per-class MEAN SHIFT is not the class's decision-relevant object; it is
2-117x too large and orthogonal to the class's own residual column (which is
covariance-dominated), and the decision-relevant correction is distributed across
ALL classes and lives in the pairwise differences. H1 rejected, H2 confirmed, H3
rejected. The line is NOT closed: the next step is the pairwise / covariance-
change direction (d_ab = Sigma^-1(v_a - v_b) and R_cov), not another per-class
mean estimator.

## Background: why a reformulation, and what we are reformulating

**The closure that motivated it.** Iterations 3b-7 of `new_iters.md` closed the
few-label landscape: the label span cannot capture the full residual R = W* - W0
(3b), the unlabeled pool cannot provide its basis (4), the pair-repair loop fails
even with oracle pairs (5), sequential acquisition does not beat one-shot (6), and
even the ORACLE acquisition curve is flat; the label budget is not the binding
constraint (7). The first-order + oracle-U step delivers a fixed ~0.3-0.5 gc that
no acquisition scheme improves and no budget extension grows, because its ceiling
is set by the update form (r=2) and by needing oracle U.

**The positive that opened a new direction (Iteration 8).** The error-
predictability diagnostic showed the frozen representation DOES expose the errors.
On the primary AL target (dglsspp fog/crosstalk) the strongest single statistic
was `proto_dist`: distance from a point to its nearest CLEAN class-mean prototype
(enrichment 1.70 fog / 1.49 crosstalk). The interpretation: the corrupted class
means move away from their clean positions, and points far from their clean class
mean are the errors. This is the class-MEAN-SHIFT story.

**The reformulation.** The linear probe is not an arbitrary matrix: for the ridge
fit with a one-hot target Y (columns = classes), w_c = (X^T X + lam I)^-1 X^T y_c,
and the c-th column of X^T Y is sum over points of class c of x_i = n_c mu_c.
Writing M as the C x d matrix with rows mu_c (class means) and
P = diag(n_c / N), this is EXACTLY

    W = Sigma^-1 M^T P          (M is C x d, rows = mu_c; M^T is d x C)

- M = the class-mean matrix (rows mu_c), "where each class sits"
- Sigma^-1 = the inverse code covariance (whitening), "the shape of the data"
- P = diag(n_c / N), the class priors

(LDA form; exact for the ridge fit up to the sketch/CG approximation. The doc
used the shorthand W = Sigma^-1 P M, which only reads correctly if M is taken as
d x C with columns = mu_c; the orientation above is the one the code computes.)

The residual R then splits. HOLDING the priors fixed at P = P0 (as the
diagnostics do: both W0 and W_mean_oracle use the clean counts C0), the
two-term expansion is

    R = W* - W0
      = Sigma0^-1 M*^T (P* - P0)              (prior term, ZERO when P = P0)
      + Sigma0^-1 (M*^T - M0^T) P0            (mean-shift term)
      + (Sigma*^-1 - Sigma0^-1) M*^T P*       (covariance term)

The diagnostics hold P = P0, so the prior term is absent BY CONSTRUCTION; the
mean-shift vs covariance split is then exact. The class-prior correction (P*
vs P0) is itself an untested cheap mechanism (the corrupted class counts are
directly observable from the unlabeled pool, so P* needs no labels).

The bet: the mean-shift term is LOCAL and LABEL-ESTIMABLE (the mean of a few
labeled points of class c directly estimates mu_c), unlike the full residual R
which the earlier closures showed is a global object. The update object is the M
block; the pool supplies the Sigma^-1 geometry (unlabeled, abundant).

Two honest caveats that every diagnostic must check:
- This is NOT reopening R1 (the prototype decoder). The pure prototype drops the
  whitening and was R1 < R4. Here we keep the full Sigma^-1 M^T P form (it IS W
  and only re-estimate the M block.
- The covariance term may be large; if so, mean-only updates cap out. The first
  thing to measure is how much of R the mean-shift term accounts for.

## The diagnostic (al_class_stats_diag.py): white-box pieces

The construction is split into measured parts, so each arm shows WHAT is working /
WHAT needs fixing:

- **A. The decomposition:** `R_mean_frac` (fraction of R the mean-shift accounts
  for under the pool whitening) and the decoder ladder W0 / proto_oracle (cosine
  to oracle means, NO whitening) / W_mean_oracle (WHITENED mean decoder, oracle
  means) / W* (full oracle). This separates: does the mean shift matter (proto vs
  W0), does the whitening matter (proto vs W_mean_oracle), is the covariance part
  big (W_mean_oracle vs W*).
- **B. Few-label mean estimation:** W_est = Sigma0^-1 P0 M_hat with M_hat the
  shrunk sample mean of b labeled points per class: sweep b per class {2,4,8} x
  shrinkage alpha {0,2,8} (the memory knob) x selection {random, proto_dist},
  reporting gc AND the per-class mean estimation error ||M_hat - M*||/||M*||.
- **C. Selection ablation:** random / proto_dist / entropy / oracle_error at a
  fixed budget: WHICH points of class c to label for the mean estimate.
- **D. Update details:** softmax temperature (step-size-like knob) and top-K
  update scope (sparse update: only the K most-moved classes).

## Iteration 1 result: the DECOMPOSITION is correct, the whitened class-means
capture the residual, but the few-label MEAN ESTIMATOR is the bottleneck
(2026-08-30, `al_class_stats_diag.py`)

Both DGLSS++ and cov-shift, all 4 conds. JSONs verified clean (`al_class_stats_
{dglsspp,covshift_ep10}.json`).

### A. The decomposition is correct (the positive)

The decoder ladder (gc):

| | dglsspp fog | dglsspp crosstalk | covshift crosstalk | covshift snow |
| :--- | :--- | :--- | :--- | :--- |
| W0 (frozen) | +0.00 | +0.00 | +0.00 | +0.00 |
| proto_oracle (cosine, NO whitening) | +0.47 | +0.41 | -1.75 | -0.90 |
| **W_mean_oracle (WHITENED, oracle means)** | **+0.72** | **+0.99** | **+1.04** | **+1.15** |
| W* (full oracle) | +1.00 | +1.00 | +1.00 | +1.00 |

The whitened mean decoder with oracle means closes **+0.72 to +1.15 gc, matching
or even EXCEEDING W\* on crosstalk/snow**. This confirms the reformulation's
ceiling is the FULL residual, and that **whitening is essential**: proto_oracle
(no whitening) is far weaker (+0.41-0.47) or negative (-1.75, -0.90). This is
exactly why the R1 prototype decoder closed (R1 < R4 without whitening) but the
whitened class-mean decoder need not.

`R_mean_frac` reads > 1 (1.3-2.7), a measurement artifact of mixing clean
counts with pool whitening, not a meaningful bound; the ladder is the reliable
number.

### B. The few-label mean estimator is the bottleneck (the negative)

W_est is NEGATIVE at every operating point (best over alpha shown):

| est arm | dglsspp fog | dglsspp crosstalk | covshift fog | covshift wet_ground |
| :--- | :--- | :--- | :--- | :--- |
| random | -0.19 (b8 a2) | -0.23 (b8 a2) | -1.32 (b8 a8) | -1.59 (b8 a8) |
| proto_dist | -0.29 (b2 a8) | -0.32 (b2 a8) | -1.47 (b8 a8) | -1.70 (b8 a8) |

- The mean estimation error ||M_hat - M*||/||M*|| is **0.46-0.65 even at b=8/class
  with alpha=2**: 8 labels cannot estimate a 10000-d class mean, and the
  whitening Sigma^-1 AMPLIFIES this high-dim noise into a catastrophic step.
- **proto_dist selection is WORSE than random for mean estimation** (error 0.87-
  1.25 vs 0.46-0.65): the Iteration-8 error *predictor* (points far from the
  clean mean) selects class OUTLIERS, which are exactly the wrong points for
  estimating a *mean*. Error-prediction points and mean-estimation points are
  DIFFERENT objects.
- Shrinkage helps (dglsspp fog alpha=8 -0.30 vs alpha=0 -0.62) but cannot rescue
  it: the estimate is noise-dominated before whitening.

### C/D. Update details

- Temperature is a no-op on argmax (T 0.5/1/2 identical, as expected).
- Top-K scope: updating only the K=4 most-moved classes hurts LESS than all
  (dglsspp snow K4 -8.56 vs K17 -13.06; covshift wet_ground K4 -1.64 vs K17
  -1.92), confirming the shift is concentrated in a few classes (top_moved on
  fog: c14/c13/c4/c7/c16 for dglsspp, c2/c7/c11/c4 for covshift) but still
  negative because those few mean estimates are also noise.

### Verdict and the fix path

The reformulation is VIABLE in structure but the ESTIMATOR is broken. The ceiling
is real (W_mean_oracle ~ W*), so the fix is not the decoder: it is HOW the
shifted means are obtained. The raw few-label sample mean fails because a 10000-d
mean needs far more than 8 points and the whitening amplifies the variance. The
concrete fix paths this diagnostic isolates:

1. **Pool-pseudo-label means + few-label bias correction.** Estimate the shifted
   means from the UNLABELED POOL via pseudo-labels (low-variance, biased) and use
   the few labels to correct the BIAS (the mean SHIFT, a concentrated low-rank
   object) rather than estimate the mean from scratch.
2. **Estimate only the mean SHIFT** Delta_mu_c = M*_c - M0_c (concentrated in a
   few classes per top_moved) under the clean whitening, with strong shrinkage,
   not the absolute mean.
3. **Regularize the whitening** so it does not amplify the noise floor.

Iteration 2 (planned): path 1: pseudo-label the pool for low-variance shifted
means, use the few labels to correct the class-mean bias.

## Iteration 2 result: all three fix paths FAIL to rescue the few-label estimator:
the pool-pseudo-means alone are the only positive, and the update-norm
diagnostic shows the whitening amplifies the few-label noise 35x
(2026-08-30, `al_class_stats_fix_diag.py`)

Validated the three fix paths independently, all against the same references
(W0 = 0, W_mean_oracle = the ceiling, W* = 1). JSONs verified clean
(`al_class_stats_fix_{dglsspp,covshift_ep10}.json`).

**FIX 1 (pool pseudo-means + bias correction): the pool-pseudo-means ALONE are
positive, the bias correction HURTS.** M_pseudo_c = mean of the pool codes the
frozen probe pseudo-labels as class c. At alpha_d = 0 (no bias correction, the
pure label-free pool pseudo-mean decoder) the gc is the ONLY positive in the
whole test: dglsspp fog +0.05, crosstalk +0.11. But adding the few-label bias
correction (alpha_d > 0) drives it negative (dglsspp fog -0.26, crosstalk -0.26):
the bias estimate M_lab_c from 2-8 points is itself noisy, so correcting a
low-variance pool estimate with a high-variance label estimate makes it worse.
The pool provides low-variance means (+0.05 to +0.11, label-free); the labels
cannot improve them.

**FIX 2 (shift-only + strong shrinkage): fails, all negative.** M_shift_c =
M0_c + s*(M_lab_c - M0_c) with s in {0.1, 0.3, 1.0}: best is s=0.1 at -0.31
(dglsspp fog/crosstalk). The shift M_lab_c - M0_c is estimated from the SAME
2-8 noisy points, so it is exactly as noisy as the absolute mean. Shrinking the
step does not fix a wrong direction.

**FIX 3 (regularized whitening): the lambda_w sweep has NO effect, rank
truncation gives a small positive on crosstalk.** Using the same noisy estimated
means (b=4, alpha=2) and varying ONLY the whitening:
- lambda_whitening {0.001..1.0}: update_norm IDENTICAL (35.73) at every value,
  gc identical (-0.216 fog, -0.249 crosstalk). The ridge values are negligible
  against the +-1-code covariance scale (~N ~ 20000), so they cannot damp the
  noise directions.
- rank truncation {128, 512, 2048}: dglsspp crosstalk rk128 +0.12, fog +0.01;
  covshift negative (-0.15 to -0.17). A small positive on the primary AL target
  but far below the +0.99 ceiling.

**The decisive diagnostic: update_norm = ||W_est - W0|| / ||R||.** The whitened
few-label mean estimate produces a step ~35x the residual norm (and 800-2200x
when rank-truncated to 128-2048). This is the same overstepping signature as
Iteration 1, now quantified.

**IMPORTANT: what this does and does NOT establish.** It establishes that the
ESTIMATOR "few-label raw sample mean, then whitened" is catastrophically noisy. It
does NOT establish that "few labels cannot estimate M*". Two things were never
measured and must be, before closing the estimator family:
- the WHITENED mean error ||Sigma^-1 (M_hat - M*)|| / ||Sigma^-1 M*||, a 46-65%
  raw mean error could be much less damaging if most of it lies in directions the
  decoder suppresses (and conversely a 10% error could be disastrous in high-gain
  directions of Sigma^-1);
- the oracle-relevant projection <Sigma^-1 (M_hat - M0), R> / ||R||^2: whether
  the noise is along R (damaging) or orthogonal to it (benign).

### Verdict: close the raw mean estimator, NOT the class-statistics program

The decomposition is strongly validated (W_mean_oracle +0.72 to +1.15 gc, the
particular statistic M* really contains most/all of the information the linear
probe needs). The reformulation is therefore not "another prototype heuristic."
The narrow, well-supported closure is:

> "2-8 RAW LABELED samples cannot estimate a 10000-d class mean accurately enough
> for the subsequent whitening", under the current class-code distributions and
> the raw-sample-mean-then-whiten estimator.

The wider statement "the few-label class-mean program is a no-go" is NOT yet
supported. The pool provides low-variance class statistics (M_pseudo, +0.05 to
+0.11 label-free positive); the labels have only been asked to do the ONE thing
they cannot (estimate a 10000-d vector). The genuinely open formulations all use
the labels to estimate a LOW-DIMENSIONAL object, not the mean:

1. **Pool estimates the class statistics; few labels estimate a small correction
   to them**, the "labels select a correction, not supply its 10000-d
   direction" idea. Forms: (a) pseudo-label CONFUSION matrix Q (C x C, not
   C x 10000): M_tilde_c ~= sum_j Q_cj mu*_j, labels estimate Q; (b) scalar
   coefficients on pool-derived shift directions: mu*_c ~= M_tilde_c + alpha_c
   v_c, where v_c is pool-derived (pseudo-vs-clean mean, high-confidence subset
   mean, TTA displacement, density-core displacement) and labels estimate only
   alpha in R^K with K = 3-5 suspicious classes. This is the class-statistics
   analogue of the pool-basis idea, but at the M level (K scalars, not the
   full residual), fundamentally different from the closed rank-1-per-label
   construction (which used label CE-gradient directions).
2. **Improve the label-free pool pseudo-mean decoder** (the +0.05-0.11 positive):
   SOFT pseudo-means (weight by p_i(c), TTA-averaged, sharpen only when
   justified) instead of hard argmax pseudo-labels, testing whether the positive
   is limited by hard-label contamination.
3. **Robust/active mean estimation** (cheap, untested): coordinatewise median /
   trimmed / Huber / distance-trimmed / nearest-to-CLEAN-prototype (note: proto_
   dist SELECTED far points was bad for means; the OPPOSITE selection (points
   NEAR the clean class core) is the untested direction), and k-center among
   high-confidence class-c points (multi-modal classes).
4. **Measure pseudo-mean quality independently of decoder gc**: per-class
   ||M_tilde_c - mu*_c||, the whitened error, and the residual-relevant error
   e_c^R = |<Sigma^-1 (M_tilde_c - mu*_c), R_c>|, which classes account for the
   +0.05-0.11 to +0.72-1.15 gap (probably NOT all 17).
5. **The class-prior correction** (from the corrected residual expansion): P* vs
   P0 is directly observable from the UNLABELED pool (class counts need no
   labels), a zero-label mechanism never tested.

### Iteration 3 (planned): a three-arm decisive test, not "one more then close"

The final test should answer three different hypotheses, NOT just step-size
calibration (which is an ORACLE diagnostic, not a deployable fix: scaling a
35x noise vector down turns a big wrong step into a small wrong step; run it
only as the oracle-scale control, separated from any data-driven scale):

- **Arm A: pure pool baseline (label-free ceiling of the construction).**
  W_pseudo = Sigma0^-1 M_pseudo^T P0 with HARD / SOFT / TTA pseudo labels. If
  soft/TTA pushes +0.05-0.11 toward +0.72-1.15, the label-free route is the
  strong one.
- **Arm B: pool basis + scalar/class calibration.** Identify the 3-5 shifted
  classes (pool evidence), estimate only alpha in R^K on pool-derived shift
  directions. Labels estimate K scalars, not a 10000-d vector.
- **Arm C: pseudo-label confusion correction.** Few labels estimate a small
  class-confusion/calibration matrix Q (C x C, restricted to the K x K
  suspicious-class block, pair-concentration evidence supports this), correct
  M_pseudo with it.

Read: Arm A > B,C  -> the label-free route dominates (labels not needed); B or C
> A -> the labels ARE useful for low-dimensional correction, which is the
reformulation's actual claim. Also run the corruption control (corrupt D_oracle =
W_mean_oracle - W0 with additive noise D_rho = sqrt(1-rho^2) D + rho N, measure
gc(rho)) to translate "35x noise" into the required estimation precision: if
even 80% directional corruption retains performance, the estimator problem is
solvable; if 5% destroys it, the route is brittle.

Standing caution (from Iteration 7): the permutation result showed the labels'
content barely mattered for the ONLY working first-order mechanism: the gain
was geometric (U-leverage of selected points). That is a risk for B/C specifically
(the label CONTENT is what drives the class correction there) and is exactly what
this three-arm test decides.

## Iteration 3 result: the three-arm decisive test: Arm C (confusion correction)
is the first few-label signal that works, and the corruption control shows the
direction is robust, NOT brittle (2026-08-30, `al_class_stats_iter3_diag.py`)

Verified against clean JSONs (`al_class_stats_iter3_{dglsspp,covshift_ep10}.json`).

**ARM A: hard pseudo-means are the only label-free positive; soft/TTA are WORSE.**

| | dglsspp fog | dglsspp crosstalk | covshift fog |
| :--- | :--- | :--- | :--- |
| hard | **+0.05** (we 3.63) | **+0.12** (we 1.56) | **+0.02** (we 1.17) |
| soft | -0.34 (we 1.00) | -0.34 (we 1.00) | -1.11 (we 1.00) |
| tta | -0.35 (we 1.00) | -0.37 (we 1.00) | -1.13 (we 1.00) |

The softer estimators are CLOSER to M* in raw whitened norm (we ~1.00) yet give
worse gc. The label-free ceiling of the construction remains the HARD pseudo-mean
decoder at +0.05 to +0.12 (reproducing Iteration 2's positive). Soft/TTA
averaging does not push it toward the +0.73-0.99 ceiling.

**ARM C: confusion correction is the WINNER among few-label arms.** The only
positive few-label arm, on the primary AL target:
- dglsspp fog K3 +0.04, K5 +0.01 (at ALL budgets 2/4/8); covshift fog K5 +0.04.
- And it is the ONLY arm that is not catastrophic on snow/wet_ground (-0.6 to
  -3.4 vs Arm B's -9 to -15 and Arm A's -5 to -7).
Labels estimating a C x C class-confusion matrix (NOT a 10000-d vector) is the
first few-label mechanism with a positive signal in this reformulation.

**ARM B: scalar gamma on pool directions FAILS.** All negative (-0.32 to -0.36
on dglsspp fog/crosstalk, -9 on covshift crosstalk) despite sensible gamma values
(~1.0: "move class c about one v_c"). The pool-derived direction v_c =
M_pseudo_c - M0_c is itself too noisy to be a reliable axis; a scalar on a noisy
axis is still a noisy step.

**CORRUPTION CONTROL: the mean direction is ROBUST, not brittle.** gc(rho) on
the oracle mean direction corrupted with additive noise D_rho:

| rho | dglsspp fog | dglsspp crosstalk | covshift fog | covshift crosstalk |
| :--- | :--- | :--- | :--- | :--- |
| 0.0 | +0.73 | +0.99 | +0.50 | +1.04 |
| 0.3 | +0.71 | +0.91 | +0.54 | +0.96 |
| 0.5 | **+0.70** | **+0.80** | +0.45 | +0.88 |
| 0.8 | +0.48 | +0.34 | +0.33 | +0.51 |
| 1.0 | -0.13 | -0.31 | -0.54 | -1.19 |

Even a 50%-corrupted direction retains 70-88% of its value, and 80% corruption
still keeps +0.33 to +0.51. This REFRAMES the entire line: the estimator problem
is not brittle in principle: a direction estimated to within ~50% noise would
deliver ~0.7-0.8 gc. The raw few-label estimators are simply 30-70x worse than
that tolerance (the 35x update_norm of Iteration 2), so the gap is a precision
gap, not a structural one.

### Verdict

The reformulation is NOT dead. Three things are now established:
1. The direction is robust (corruption control); the required estimation
   precision is achievable-in-principle (~50% noise tolerance).
2. Arm C (confusion correction, C x C) is the first few-label signal that works
   (+0.01 to +0.04, the only non-catastrophic few-label arm).
3. The label-free ceiling stays at the hard pseudo-mean (+0.05 to +0.12); soft/
   TTA do not help.

The path forward, in the order the evidence supports:
1. **Push Arm C further**: it is the only positive few-label mechanism. Refine
   the confusion-correction: pool-regularized Q (shrink the C x C rows toward the
   pool pseudo-label prior to fight the 8-label sample noise), iterated
   (self-training: correct M, re-pseudo-label, re-estimate Q), and soft-weighted
   Q. The +0.04 is small but it is the first label signal and the corruption
   control says the direction can tolerate the noise.
2. **Arm B needs a BETTER direction v_c**, not a better scalar: the failure was
   the noisy v_c = M_pseudo - M0. Test the other pool-derived directions (high-
   confidence pseudo subset, TTA displacement, density-core), and the corruption
   control says a good direction tolerates a bad scalar.
3. **The class-prior correction (P* vs P0)**, untested, label-free, and directly
   observable from the pool.
4. **Iterated self-training on the hard pseudo-mean** (the +0.05-0.12 label-free
   positive): re-pseudo-label with the corrected decoder.

Iteration 4 (planned): refine Arm C (confusion correction with pool-regularized Q
and iteration) + Arm B with the alternative pool-derived directions. The
corruption control gives the clearest green light: the estimator problem is a
precision problem, not an impossibility.

## Iteration 4 result: the direction is RIGHT (align 0.9) but the SCALAR is wrong:
the failure is localized to the coefficient, and confusion-correction
saturates at +0.05 (2026-08-30, `al_class_stats_iter4_diag.py`)

Verified against run output (the pulled JSONs arrived as null-byte files, the
known sftp corruption; the run output was complete).

**PART 1: confusion correction SATURATES; the 8-label Q noise was NOT the
limiter.** C0 (base) is the best form and stays at +0.03 to +0.05 on fog: dglsspp
fog +0.04 to +0.05, covshift fog +0.03 to +0.05. C1 (pool-regularized Q) at
alpha=2 roughly matches C0; larger alpha hurts (alpha=32 -> -0.10 to -0.46). C2
(iterated self-training) does not beat C0. Neither refinement moves +0.05;
**+0.05 appears to be the confusion-correction ceiling.** (Q_err vs the full-pool
oracle Q was captured in the JSON but the null-byte corruption lost it.)

**PART 2: the key finding is that the pool direction is RIGHT and the label-estimated
scalar is WRONG.** Oracle direction alignment cos(v_c, M*_c - M0_c):

| | dglsspp fog | dglsspp crosstalk | covshift fog |
| :--- | :--- | :--- | :--- |
| pseudo | **+0.92** | +0.83 | **+0.90** |
| density | +0.91 | +0.80 | +0.67 |
| highconf | NA | NA | NA |

The pool-derived directions point almost exactly at the true class shift. Yet
EVERY Arm-B variant is negative: dglsspp fog/crosstalk -0.32 to -0.34 (all three
directions), covshift fog -1.1 to -1.5, covshift crosstalk -8 to -9. Since the
direction is aligned ~0.9 and the corruption control (Iteration 3) showed a 50%
noisy direction retains 70-88% gc, the failure is the label-estimated gamma
scalar: gamma_c = <M_lab_c - M0_c, v_c>/||v_c||^2 is itself a noisy 2-8-label
estimate, and the whitening amplifies the resulting mis-scaled step. The
bottleneck is now FULLY localized to the coefficient, not the direction.
(highconf was NA because the confidence threshold 0.5 left too few points per
class; that direction is untested.)

**PART 3: the class-prior term is negligible (confirmed).** P_pseudo and
P_oracle both give -0.01 to -0.03 on dglsspp fog: same-scan corruptions barely
change the class priors, so the prior correction is not a mechanism here (as
expected from the algebra, P* ~ P0 when the scan is the same).

### Verdict

The class-statistics reformulation now has a fully localized failure:
- the DECOMPOSITION is right (W_mean_oracle +0.73-1.15),
- the pool DIRECTIONS are right (align 0.9),
- the DIRECTION is robust to noise (corruption control, Iteration 3),
- the CONFUSION correction works but saturates at +0.05,
- the failure is the SCALAR/COEFFICIENT estimation from 2-8 labels, a noisy
  gamma, mis-scaled, then whitening-amplified.

This means the remaining lever is a GOOD step-size/coefficient estimator: given a
known-good pool direction v_c (align 0.9), estimate the single scalar gamma_c
without the 35x whitening amplification. The corruption control says the direction
tolerates ~50% noise, so the scalar estimator must get within that tolerance.
Candidate coefficient estimators (none yet tested):
1. gamma from the label-vs-pseudo-margin at the boundary (not the raw mean):
   use the few labels to place the a-b boundary shift directly.
2. pool-regularized gamma: shrink the label gamma toward 1 (the pseudo-mean
   default), C1-style regularization on the SCALAR instead of on Q.
3. gamma constrained to keep the update norm at ~1x R (the corruption control's
   operating point), an explicit step-size calibration on the direction.
4. gamma estimated on the CONFIDENT pseudo subset only (highconf direction with a
   lower threshold so it is non-NA).

Iteration 5 (planned): a coefficient-estimation diagnostic: sweep gamma_hat
estimators (raw projection, shrunk-toward-1, boundary-margin, update-norm-
constrained) on the known-good direction, and report the gc(gamma_hat) curve vs
the gamma* oracle optimum. If a practical estimator lands near gamma*, the
reformulation has its mechanism.

## Iteration 5 result: ORACLE scalars on the "good" direction are NEGATIVE; the
direction family is NOT sufficient after whitening (2026-08-30,
`al_class_stats_iter5_diag.py`)

Verified against clean JSONs. The coefficient-estimation diagnostic measured the
gc(gamma) curve on the Iteration-4 "known-good" direction v_c = M_pseudo_c - M0_c,
the per-class ORACLE scalars gamma*_c = <M*_c - M0_c, v_c>/||v_c||^2, and practical
estimators (raw, shrink1, gamma1, normscale, oracle).

**The decisive number: gc(gamma*_perclass) is NEGATIVE on every cell, far below
W_mean_oracle.** Scalar estimation is NOT the whole problem; the direction family
itself is insufficient after whitening.

| | dglsspp fog | dglsspp crosstalk | covshift fog | covshift crosstalk |
| :--- | :--- | :--- | :--- | :--- |
| W_mean_oracle | +0.72 | +0.99 | +0.50 | +1.04 |
| **gamma\*_perclass** | **-0.31** | **-0.34** | **-1.17** | **-5.79** |
| curve peak (best over gamma sweep) | -0.30 | -0.32 | -0.97 | -5.24 |

Even with PERFECT per-class oracle scalars (the projection of the TRUE shift onto
v_c), the update is negative. The gc(gamma) curve is flat-negative across the
whole gamma sweep (0 to 2.0). The oracle gamma values themselves are sensible
(0.8-1.3 on the suspicious classes); the problem is not the scalar.

**Why this reconciles with Iteration 4's "+0.92 direction alignment".** The
Iteration-4 alignment cos(v_c, M*_c - M0_c) ~ 0.9 was measured in RAW code space.
But the decoder applies the non-orthogonal whitening Sigma^-1, which rotates and
re-weights directions. The quantity that matters is whether v_c aligns with
Sigma^-1 (M* - M0) (the WHITENED mean shift), and it does not. upd_norm@gamma1
= 33.7xR (dglsspp fog) confirms: the v-direction, once whitened, is dominated by
amplified noise directions, not the residual. The Iteration-3 corruption control
was equally misleading in the same way: it corrupted D_oracle = W_mean_oracle - W0
which is ALREADY the whitened difference, so it never tested whether a RAW mean
direction survives the whitening.

**Caveat on the gamma=0 reference.** At gamma=0 build_W uses the POOL whitening
while W0 uses the CLEAN whitening, so gamma=0 (-0.31) is not W0 (0); it already
includes a ~-0.31 pool-vs-clean whitening gap. But the flat negative curve means
no gamma recovers anything positive regardless.

**Verdict: the class-statistics reformulation is effectively CLOSED at the
direction level.** The decoder structure is right (W_mean_oracle works), but the
ONLY estimator that reaches the ceiling is the oracle M* (full pool labels). Every
label-free/few-label route to M* (raw means, shift-only, confusion correction
(saturates +0.05), and now the pool pseudo-mean direction v_c with ORACLE scalars)
fails once the whitening is applied. The corruption control's "green light"
was an artifact of testing whitened differences rather than raw directions. The
reformulation's residual value is:
- W_mean_oracle as a diagnostic reference (the labeled ceiling),
- the label-free hard pseudo-mean decoder (+0.05 to +0.12, no labels, no oracle),
- the confusion correction (+0.01 to +0.05, few labels, no oracle).
These are small but real; no further estimator is supported by the evidence. This
doc's line is CLOSED pending a genuinely new information source (e.g. hundreds of
labels, a pretrained pool-derived whitening, or a decoder that does not require
Sigma^-1).

**UPDATE (Iteration 6): the Iteration-5 "direction is dead" conclusion was a
METRIC MISMATCH, not a real closure; the direction DOES survive decoder
geometry, but the per-class mean shift is not the decision-relevant object.** See
the Iteration 6 section below. The line is NOT closed; the failure is now
localized to a different place.

## Iteration 6 result: H2 confirmed: the direction survives whitening (per-class
align ~1.0), but the per-class mean shift is 2-117x too large and nearly
orthogonal to the class's OWN residual column, and the decision-relevant
correction is distributed across ALL classes (2026-08-30,
`al_class_stats_iter6_diag.py`)

Verified against clean JSONs. The covariance-space localization resolved the
three hypotheses and directly answered "did the +0.92 direction survive?"

**A. PER-CLASS DECODER ALIGNMENT ~ 1.0: the direction GENUINELY survives
decoder geometry (H2 confirmed).** cos(Sigma^-1 v_c, Sigma^-1 Delta_mu_c):

| cond | per-class align (5 classes) |
| :--- | :--- |
| dglsspp fog | 1.00 / 0.89 / 1.00 / 0.98 / 0.98 |
| dglsspp crosstalk | 0.94 / 1.00 / 1.00 / 0.97 / 0.99 |
| covshift fog | 1.00 / 0.84 / 0.99 / 1.00 / 0.99 |
| covshift crosstalk | 0.79 / 0.67 / 1.00 / 0.98 / 0.88 |

The pool direction v_c = M_pseudo - M0 is essentially the ORACLE mean shift even
after whitening. The Iteration-5 negative (gamma*_perclass < 0) was NOT "the
whitening destroys the direction"; it was the global-flattened-residual metric
mismatch (resid_rel ~ 0.001 compared a class direction against the aggregate R).
**The user's reframing was correct: H1 (whitening destroys each useful class
direction) is REJECTED.**

**But the direction being right does NOT make it decision-relevant:**

**G. The mean-shift term is 2-117x LARGER than the class's own residual column,
and nearly orthogonal to it.** frac_mean_norm = ||Sigma^-1 Delta_mu_c|| / ||R_c||
is 2-117; align_Rc = cos(Sigma^-1 Delta_mu_c, R_c) is 0.02-0.50 (mostly 0.05-0.4).
The whitening amplifies Delta_mu_c enormously, and the true per-class residual
column R_c is dominated by COVARIANCE effects, not the mean shift. So even the
oracle-correct direction, added at any scale to class c, does not move W_c toward
W*_c; this is WHY the gamma*_perclass scalar test was negative despite the
direction being right.

**B. PAIRWISE decoder alignment is LOW (0.01-0.51).** The decision-relevant
object is w_a - w_b (class competition), and the pairwise differences
Sigma^-1 (v_a - v_b) are NOT aligned with the oracle pairwise shifts; only the
per-class directions are. This matches the earlier confusion-pair results.

**H. Even ORACLE means restricted to the top-5 suspicious classes give gc ~ 0**
(rho=0: dglsspp fog -0.02, crosstalk -0.15; covshift fog -0.87), while W_mean_
oracle (oracle means for ALL 17 classes) is +0.72. **The decision-relevant mean
correction is DISTRIBUTED across all classes, not concentrated in the 3-5
largest raw-shift classes.** The Iteration-1 "top_moved concentration" was an
artifact of raw mean-norm, not decision relevance.

**D/E. Whitening is NOT the culprit at any fractional power or rank.** Fractional
alignment is 0.81-0.96 at beta=1 (and 0.40-0.92 at beta=0); rank-truncated gc is
negative at every r (8-512). H3 (whitening amplifies irrelevant eigendirections)
is also REJECTED at the level that matters: no rank/fractional whitening
recovers a positive gc.

**I. All pseudo-mean schemes (hard/soft/tta) align ~0.9-1.0 per-class; core
~0.7-0.8; highconf NA (too few confident points at tau=0.5).** The direction is
robust to the pseudo-label scheme; scheme choice is not the issue.

### Verdict

The Iteration-5 closure was WRONG in its reasoning (a metric mismatch, exactly as
the user suspected) and the line is NOT closed at the "direction level". The
direction DOES survive decoder geometry (A ~ 1.0). But Iteration 6 localizes the
real failure to two precise places:

1. **The per-class mean shift is not the class's decision-relevant residual.**
   The whitened mean-shift term is 2-117x the class's own residual column and
   mostly orthogonal to it (G); the residual column is covariance-dominated.
   This explains why gamma*_perclass was negative despite align ~1.0: the scalar
   was applied to the wrong object (the mean shift) relative to what the decoder
   needs per class.
2. **The decision-relevant mean correction is distributed across all classes
   (H) and lives in the PAIRWISE differences (B), not the per-class columns.**

What this DOES settle, cleanly:
- H1 rejected: whitening does not destroy the per-class direction.
- H2 confirmed: the +0.92 raw direction genuinely survives; the global residual
  metric was misleading.
- H3 rejected at the level that matters: no rank/fractional whitening rescues gc.

What remains OPEN (the direction is alive but the OBJECT is wrong):
- **Pairwise-boundary corrections** (B): the low pairwise alignment suggests the
  pool directions don't capture w_a - w_b, but the decision object IS pairwise,
  so a pairwise (not per-class) pool direction / correction is the untested form
  (the user's d_ab = Sigma^-1(v_a - v_b) idea).
- **Covariance-aware direction**: since R_c is covariance-dominated (G), a
  direction built from the COVARIANCE change (not the mean shift) may be the
  right object, which connects to the never-measured R_cov term.
- **Full-class (not top-K) corrections**: H says the correction must cover all
  classes; the top-K concentration was an artifact.

Iteration 7 (planned): test the PAIRWISE covariance-space direction
d_ab = Sigma^-1(v_a - v_b) against the oracle pairwise shift, and the
covariance-change direction R_cov, since G shows the per-class residual is
covariance-dominated rather than mean-dominated.

## Iteration 7 result: 99.98% of the measured PAIRWISE oracle residual is
explained by covariance; the prior-weighted pairwise mean combination nearly
cancels; the only tested positive correction mechanism is the tiny pairwise
LOGIT correction (2026-08-30, `al_class_stats_iter7_diag.py`)

Verified against clean JSONs (al_class_stats_iter7_{dglsspp,covshift_ep10}.json).
The exact pairwise decision decomposition resolved what Iteration 6 could only
infer.

**A. 99.98% OF THE MEASURED pairwise decision residual is explained by the
covariance term (with an important qualification).** For every pair on every
condition: n_mean ~ 0.001-0.002, n_prior = 0, n_cov = n_total, cos_cov ~ 1.0.
Global (D): frac_mean 0.0022 (dglsspp fog), frac_prior 0.00004, frac_cov
0.9999, cos_cov_R = 1.0.

**QUALIFICATION: the residual-vs-independent distinction.** The code computes
cov_ab = dw_ab - mean_ab - prior_ab RESIDUALLY, so when mean and prior are tiny,
cov ~ total BY CONSTRUCTION and cos_cov ~ 1.0 is partly tautological. The
strong claim ("covariance explains the residual") requires the INDEPENDENT
covariance expression (Sigma*^-1 - Sigma0^-1) P* M* to reproduce the pairwise
residual on its own, which has NOT yet been measured directly. The residual
decomposition is valid as an accounting identity, but the decisive test is the
covariance-only DECODER ceiling (below, "required before closing").

**A (precise). The mean statement is NOT "individual mean shifts are
common-mode"; it is that their PRIOR-WEIGHTED PAIRWISE COMBINATION nearly
cancels.** Iteration 6 showed the individual whitened mean shifts
Sigma^-1 Delta_mu_c are LARGE and highly aligned with their own oracle mean
shift (align ~1.0). Iteration 7 A shows
Delta_d_ab_mean = Sigma0^-1 (p_a Delta_mu_a - p_b Delta_mu_b) ~ 0.001-0.002.
So: the mean shifts are individually substantial, but their contribution to the
pairwise classifier correction is nearly CANCELED (they move together enough
that the prior-weighted difference is ~0). This is the correct, precise
statement.

**dec_agree (the decisive number): the mean-only decoder DISAGREES with the
oracle on the big-gap conditions.**

| cond | dec_agree (all) | dec_agree (on errors) |
| :--- | :--- | :--- |
| dglsspp fog | 0.687 | **0.639** |
| covshift fog | 0.619 | **0.509** |
| dglsspp crosstalk | 0.967 | 0.956 |
| covshift crosstalk | 0.983 | 0.955 |
| snow / wet_ground | 0.74-0.96 | 0.61-0.88 |

On fog, the mean-only decoder disagrees with the oracle on 36-49% of the points
in the error/oracle-disagreement subset (dec_agree_on_errors 0.509-0.639,
i.e. the mean-only decoder does NOT recover the oracle's corrections on about
half the relevant points on covshift fog). On crosstalk/snow (small gaps) means
already match. The strongest number in the iteration: the decision-relevant
correction is NOT the mean shift.

**B. The PSEUDO-MEAN pair direction is degenerate (all gc ~0.00): this closes
the specific v_a - v_b CONSTRUCTION, not pairwise pool statistics in general.**
The pseudo-means of confused pairs are nearly identical (V_a - V_b ~ 0), so
d_ab = Sigma^-1(v_a - v_b) flips nothing. This is a real cancellation, not a
code issue, and Iteration 7A explains WHY v_a - v_b was doomed: if the true
pairwise correction is covariance, a difference of class MEANS is simply the
wrong statistic. It does NOT close boundary-local covariance, density changes,
TTA disagreement features, or logit-space corrections.

**C. Boundary-local MEAN directions do not recover the oracle pairwise
correction** (align 0.01-0.13, dglsspp fog). This closes the tested
boundary-local mean displacement construction, not boundary-local covariance or
other boundary statistics.

**F. TTA decision-space displacement is INCONSISTENT: Delta z_TTA is not a
stable directed estimate of the oracle correction under the tested formulation**
(dglsspp fog: +0.53, -0.24, -0.45, +0.35, -0.23; mixed signs elsewhere). This
does NOT mean TTA is useless: it may still be extremely useful for uncertainty,
acquisition, identifying boundary points, scalar correction estimation, and
conditional calibration, consistent with the broader AL direction.

**G. The tiny pairwise LOGIT correction is the ONLY TESTED positive correction
mechanism in this diagnostic (not "the only live signal").** Oracle alpha,beta
gives small positives on some pairs: dglsspp fog 13-16 or+0.04, crosstalk 13-16
or+0.06; covshift wet_ground 14-15 or+0.05, 14-16 or+0.02. And the FEW-LABEL fit
sometimes reproduces the oracle: dglsspp 13-16 fl+0.04/0.05, covshift wet_ground
14-16 fl+0.03, 14-15 fl+0.04. IMPORTANT: the oracle gains are only +0.02-0.06,
evidence of a low-dimensional decision-space SIGNAL, but not yet evidence of a
useful adaptation MECHANISM. The few-label matches are encouraging but tiny.

### Verdict (the scientifically strongest version)

The class-mean route is CLOSED as a mechanism for recovering the missing
pairwise decision correction. The exact decomposition shows that, on the tested
conditions, approximately 99.98% of the measured pairwise oracle residual is
explained by the covariance term, the prior contribution is negligible, and the
mean contribution is only ~0.2%. The large per-class mean shifts therefore
survive whitening (Iteration 6) but largely CANCEL when converted into
prior-weighted pairwise decision differences (Iteration 7 A).

This closes the tested class-mean / pseudo-mean routes: global means, pairwise
mean differences, and boundary-local mean displacements do not recover the
missing decision correction. It does NOT establish that covariance itself is
impossible to exploit; it establishes that the tested label-free MEAN
statistics do not encode the required correction.

The remaining positive evidence is the small pairwise affine logit correction.
Its oracle ceiling is modest (+0.02-0.06 on the tested pairs), but the fact
that few-label fits sometimes recover the oracle gain makes it worth one focused
iteration.

**REQUIRED BEFORE FULLY CLOSING the class-statistics line (point 8/9 of the
Iteration-7 review): the covariance-only DECODER ceiling.** The vector
decomposition is residually defined; the decisive test is whether
W_cov,oracle = W0 + R_cov (with R_cov from the INDEPENDENT expression
(Sigma*^-1 - Sigma0^-1) P* M*) reproduces the oracle CLASSIFIER (gc and
dec_agree). Run the ceiling table: W0 / mean-only / prior-only / covariance-
only / mean+prior / mean+cov / full oracle. If covariance-only reaches ~the
oracle, the conclusion is decisive (the remaining problem IS covariance
adaptation). If covariance-only does NOT reproduce the classifier despite the
vector decomposition saying covariance explains R, there is a subtle issue
worth investigating before closing.

### Iteration 8 (planned): the pairwise LOGIT correction, ceiling-first

Per the Iteration-7 review, do NOT immediately build the pairwise logit method;
first establish its ceiling and stability:
1. Pairwise oracle ceiling: z'_a - z'_b = alpha_ab(z_a-z_b) + beta_ab per pair,
   oracle improvement.
2. Pool vs label estimation: fit (alpha,beta) with 1/2/4/8/16 labels vs oracle.
3. Random-pair vs confused-pair selection (sparse = good for AL).
4. Boundary-conditioned fit: fit (alpha,beta) only near |z_a - z_b| < tau.
5. Global vs pairwise correction: z'_c = z_c + b_c vs the O(K^2) pairwise form.
6. MOST IMPORTANT: does ONE SHARED scalar capture most of the oracle correction?
   z'_a - z'_b = alpha(z_a-z_b) + beta with a single global (alpha,beta), a
   tiny parameter count is far more compelling than 17x16 independent pairs.

## Iterations 7.5 + 8 result: cov_only does NOT reproduce the oracle: the
"99.98% covariance" was a VECTOR accounting, not a decision mechanism; and the
logit correction's oracle ceiling is ~0, closing the Iteration-8 premise
(2026-08-30, `al_class_stats_iter78_diag.py`)

Verified against clean JSONs. The two required tests in one run.

**PART A: cov_only = W0 + (W* - W_mean_oracle) is NEGATIVE on every cell. The
"99.98% covariance" was a residual-accounting identity, NOT the decision
mechanism.**

| | dglsspp fog | dglsspp crosstalk | covshift fog | covshift crosstalk |
| :--- | :--- | :--- | :--- | :--- |
| mean_only gc | **+0.72** | **+0.99** | **+0.50** | **+1.02** |
| mean_only dec_agree (errs) | **0.687** | **0.930** | **0.586** | **0.949** |
| cov_only gc | -0.47 | -0.28 | -1.73 | -2.48 |
| cov_only dec_agree (errs) | 0.238 | 0.296 | 0.381 | 0.578 |

The covariance-only decoder is NEGATIVE everywhere, and its decision agreement
is far below mean_only (dglsspp fog: 0.238 vs 0.687). The Iteration-7
"99.98% covariance" statement is now correctly understood: it is a vector
ACCOUNTING identity (cov = R - mean - prior, so when mean/prior are tiny cov ~ R
by construction), but the covariance term is ORTHOGONAL to what actually fixes
decisions. **The MEAN decoder is the one that moves decisions toward the oracle**
(mean_cov = full oracle trivially; cov alone is useless). This is exactly the
"subtle issue" the Iteration-7 review predicted, and it means covariance is NOT
the exploitable mechanism.

**PART B: the pairwise logit correction's ORACLE ceiling is ~0, closing the
Iteration-8 premise.**
- Per-pair oracle (alpha,beta): the best pair is +0.04 to +0.07 (dglsspp 13-14
  fog, 13-16 crosstalk); most pairs are ~0 or negative (covshift all ~0/negative).
- Few-label fits: on the few positive pairs (13-16), b16 tracks the oracle
  (+0.03 to +0.07); elsewhere negative.
- Global bias (17 scalars): +0.06 to +0.08 on dglsspp (weak but the most
  consistent), ~0/-0.5 on covshift.
- **Shared scalar (ONE alpha,beta): NEGATIVE everywhere** (-0.14 to -1.40), and
  the key Iteration-8 test FAILS.

### Verdict: the class-statistics line is now CLOSED with the mechanism fully
measured at every level.

The progression of closures is complete:
- W updates: closed (Iterations 3b/4/7 of new_iters.md).
- Class means as an update object: closed (Iteration 5).
- Class means as a decision object: closed (Iterations 6-7, the mean decoder
  is decision-relevant but not label-estimable).
- Covariance as the explanation: closed (Iteration 7.5, the vector accounting
  over-attributed the residual to covariance; cov_only is negative, the mean
  decoder is the one that matches oracle decisions).
- Pairwise logit correction: closed (Iteration 8, oracle ceiling ~0, shared
  scalar negative).

The one consistent positive across the whole line is the MEAN decoder
(W_mean_oracle = Sigma0^-1 P0 M*), which closes +0.50 to +1.15 gc on the
healthy-capacity conditions and matches oracle decisions on 0.59-0.95 of the
error points, but it requires the ORACLE means M*, which need the full pool
labels. The label-free hard pseudo-mean decoder (+0.05-0.12) and confusion
correction (+0.01-0.05) remain the only label-free/few-label positives, and both
are small.

The class-statistics reformulation does NOT provide a deployable few-label
update. Its value is diagnostic: it conclusively localizes the few-label
information bottleneck to the mismatch between recoverable class statistics
(means, directions) and the decision-relevant correction (which the mean decoder
captures only with oracle means, and which the covariance accounting
misattributes). This is consistent with the broader conclusion from
new_iters.md: the labeled ceiling is only reachable with the full pool labels.

**UPDATE (Iteration 10): the "no further iteration" verdict is REVERSED by the
propagation result: nearest-anchor propagation of few labels produces a
positive mean decoder on the primary AL target.** See below. The line is
REOPENED around the propagated-mean estimator.

## Iteration 10 result: nearest-anchor PROPAGATION of few labels produces a
positive mean decoder on dglsspp fog/crosstalk, the strongest few-label result
since Iteration 1, REOPENING the class-statistics line
(2026-08-30, `al_propagation_potential_diag.py`)

The user's question: can AL labels + feature-space structure approximate the
oracle means M*? The diagnostic separates the ORACLE signal (proximity
precision, mean cos) from the IMPLEMENTATION (nearest-anchor / centroid /
agreement-gated / clean-source propagation), so we can tell "the structure
can't label the pool" from "the estimator is the bottleneck." Verified against
clean JSONs.

**THE HEADLINE: propagation works on the primary AL target.** The propagated
mean decoder (gcP = M_prop x propagated counts) closes REAL positive gc:

| | dglsspp fog | dglsspp crosstalk |
| :--- | :--- | :--- |
| W_mean_oracle ceiling | +0.73 | +0.99 |
| **gcP (propagated mean, prop counts)** | **+0.26 to +0.36** | **+0.19 to +0.46** |
| gcO (prop mean, CLEAN counts) | +0.15 to +0.22 | +0.19 to +0.44 |
| gcC (mass-calibrated) | +0.20 to +0.31 | +0.11 to +0.40 |
| frozen-pseudo baseline (Iteration 2) | +0.05-0.12 | +0.05-0.12 |

From just 1-16 anchors per class, propagation closes +0.26 to +0.45 gc on the
primary target, far above the frozen-pseudo baseline and the confusion
correction (+0.05). This is the first few-label mechanism to reach real positive
gc on dglsspp fog/crosstalk since Iteration 1's oracle-U step. The per-point
propagation precision is LOW (0.28-0.48) yet the aggregated mean decoder works;
the class-mean aggregation is robust to per-point label noise (the mean_cos
0.89-0.98 is in the saturated code space, not diagnostic; the decoder outcome
is what matters).

**The other arms, per the oracle-vs-implementation split:**
- A1 proximity precision (128-d): 0.67-0.84; the ORACLE signal is real and
  high. The structure CAN label the pool at 70-80%+.
- A2 nearest-anchor propagation: 0.28-0.48; the IMPLEMENTATION is noisy at the
  per-point level, but the mean aggregation survives it. (Potential exists;
  the estimator is the bottleneck for per-point labels, but the MEAN is already
  good.)
- A2b centroid anchors: ~same as A2 (0.27-0.71), no gain over nearest-anchor.
- A2c agreement-gated: precision jumps to 0.50-0.66 (dglsspp fog) but coverage
  drops to 0.26-0.47, the Iteration-5 clean-T tradeoff again.
- A4 clean-source bank (the MEMORY-BANK idea): 0.47-0.79 precision, and on the
  hard conditions it beats nearest-anchor (dglsspp fog 0.63 vs 0.31; crosstalk
  0.40-0.75 vs 0.42-0.48), a label-free proxy, worth combining with the mean
  decoder (it gives a better label set to aggregate over).
- C1 boundary: precision INCREASES with margin (low-margin points are WORSE,
  high-margin better), the boundary is NOT where labels transfer; the bulk is.
  This closes the "label the boundary to get labels" idea for means.
- D1: propagation beats anchors alone on mean cos (b1: a0.76 vs p0.89; b2:
  0.87 vs 0.95): structure MULTIPLIES the effective labeled set. The core
  mechanism is validated.

**Failures (same pattern as always):** covshift fog negative everywhere (-0.24
to -0.57); snow/wet_ground negative. Covshift is already near ceiling; the loose
conditions (7/15/14) still poison. The propagated-mean result is specific to the
healthy-capacity conditions on dglsspp, exactly where AL has headroom.

**CAVEAT (code labeling bug):** the "gc_oracle_counts" arm (gcO) actually uses
the CLEAN counts C0, not the oracle counts C_star, so it is "propagated means x
clean counts," not the true oracle-count ceiling. The headline gcP (propagated
counts) is unaffected. The true oracle-count ceiling for propagated means
(gcP with C_star) was NOT measured; it is the first item in the next
iteration.

### Verdict: the line is REOPENED around the propagated-mean estimator

The class-statistics line was closed on the premise that the mean decoder needs
oracle means and few labels cannot estimate them. Iteration 10 shows that is
WRONG: **propagation of a few anchors through the 128-d feature space estimates
the class means well enough that the mean decoder closes +0.26 to +0.45 gc on
the primary AL target, the strongest few-label result since Iteration 1.** The
mechanism is validated end-to-end (D1: structure multiplies the anchors; B1:
the mean decoder works; A4: a label-free clean-source bank gives an even better
label set on the hard conditions).

Next steps, in order of value:
1. **The true oracle-count ceiling for propagated means** (fix the gcO bug: use
   C_star). This tells us how much of the +0.73/+0.99 ceiling the propagated
   MEANS alone capture vs how much is lost to the count error.
2. **Clean-source bank means**: aggregate the A4 label-free clean-source labels
   (0.63-0.79 precision) into the mean decoder; this is label-free, and if it
   works it beats the few-label version entirely.
3. **Agreement-gated propagation for the MEAN (not the per-point label)**: the
   A2c gate raises precision to 0.5-0.66; aggregating the gated points into
   means may preserve the gain without the coverage collapse.
4. **Per-class propagated-mean breakdown**: which classes carry the +0.26-0.45
   (likely the tight majority, with the loose 7/15/14 still failing, so the
   budget must target them).
5. **Propagation in the 128-d space for the means directly** (the current B1
   aggregates in the 10000-d code space; the 128-d space is where the packing
   lives).

The propagated-mean estimator is the direction the whole class-statistics arc
was looking for: it uses AL labels + structure to approximate M* without needing
the full pool labels.

## Validation note (post-Iteration-10, DGLSS++ only): the positive is REAL and
reproducible, but the "not much headroom" conclusion is MUDDLED by a
count-reference mismatch; the true ceiling of the idea is NOT yet established
(2026-08-30, `al_propagation_validate_diag.py`)

DGLSS++ only (the extractor with a ceiling), fog/crosstalk. This is a small
post-iteration validation, NOT a new iteration.

**The naive method's actual gain (112-125 labels, random anchors):**

| | frozen mIoU | prop-mean mIoU | gain | oracle-cnt mIoU | W_mean_oracle mIoU |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.111 | 0.156 | **+0.045** | 0.128 | 0.240 |
| crosstalk | 0.154 | 0.249 | **+0.095** | 0.250 | 0.385 |

The method is real and reproducible (+0.045 fog / +0.095 crosstalk mIoU), the
strongest few-label result since Iteration 1. But the absolute gain is modest.

**The MUDDLE: the "oracle-count ceiling" is not a clean ceiling.** The
validation concluded "the propagated MEANS are the bottleneck, not the counts"
from the oracle-count arm being far below W_mean_oracle. That inference is
CONFOUNDED by a count-reference mismatch:

| decoder | means | counts |
| :--- | :--- | :--- |
| gcP (propagated) | M_prop | C_prop (propagated) |
| gcO (oracle-count) | M_prop | C_star (pool oracle) |
| W_mean_oracle | M_star (true) | C0 (CLEAN counts) |

Three different count references. gcO pairs PROPAGATED means with POOL counts;
W_mean_oracle pairs TRUE means with CLEAN counts. Two things differ at once
(means AND count reference), so it does NOT isolate mean quality.

**The smoking gun that this is broken: on fog, oracle-count gc (+0.09) is BELOW
propagated-count gc (+0.25).** A valid "count ceiling" can never make things
worse. Forcing the true pool counts onto propagated means creates an internal
mismatch (the propagated means were implicitly averaged over the propagated
mass, then re-weighted by the true mass). So "gcO << ceiling" CANNOT be read as
"the means are near their limit."

**A second assumption muddling the headroom:** the propagation signal lives in
the 128-d space (packing P ~ 0.81-0.93), but the means are aggregated in the
10000-d CODE space before the whitened decoder, re-introducing the code-space
saturation/ill-conditioning that motivated the 128-d choice. The "modest
ceiling" may partly be this aggregation choice, not the propagation.

**V2: the clean-source bank (label-free memory-bank hope) does NOT work.** fog
-0.10, crosstalk -0.19 (oracle-count +0.01/-0.05). The clean-source precision
(0.625 fog, 0.394 crosstalk) is not enough for the mean decoder. The label-free
route is closed.

**V3: influence selection is actively WRONG for mean estimation.** Influence
is negative everywhere (-0.07 to -0.45), random positive (+0.07 to +0.40).
Influence picks high-leverage BOUNDARY points (anti-correlated with confidence,
per the prior docs), exactly the wrong points for estimating a MEAN. This
contradicts the cluster-grounding query finding, but that was about moving the
probe, not estimating means. **For means, random is the right selection.**

**V4/V5: class 11 consistently best (tight); {7, 13, 14} worst** (the known
loose classes); count error large on fog (0.76-0.93), smaller on crosstalk.

**Verdict of the validation, corrected: the positive is real, but the
headroom is UNKNOWN, not "modest."** The earlier verdict ("the propagated means
are far from oracle-quality, not the path to the ceiling") was based on a
count-reference-mismatched comparison and should not be trusted. The honest
position:

1. The naive propagation gives a real +0.045/+0.095 mIoU (strongest few-label
   result since Iteration 1).
2. **The ceiling of the idea is NOT established**, and prior ceilings in this
   arc repeatedly failed to predict the successes (the +0.72-1.15 mean-decoder
   ceiling did NOT predict that raw few-label means would fail at 35x, nor that
   propagation would succeed). So a ceiling claim here must be earned by the
   clean decomposition, not assumed.
3. What IS established: random anchors > influence for means; the clean-source
   label-free route is closed; the count error is secondary to the (still
   muddled) mean question.

**NEXT VALIDATION STEP (before any new method): the clean headroom
decomposition.** Diagnostics that resolve the muddle, organized by which part of
the pipeline they improve:

**A. The MEAN ESTIMATOR (the propagated means M_prop):**
- **M_star x C_prop** (TRUE means x PROPAGATED counts) vs M_prop x C_prop:
  holds counts fixed, isolates mean quality cleanly. If true-means-with-
  propagated-counts ~ W_mean_oracle, the propagated COUNTS are fine and the
  means are the (real) bottleneck; if far below, the propagated counts are
  themselves part of the gap.
- **Aggregate the means in the 128-d space** (where the propagation and packing
  live) instead of the 10000-d code space: does the code-space aggregation
  cap the result?
- **Agreement-gated propagation for the MEANS** (not per-point labels): gate
  raises per-point precision to 0.5-0.66; aggregating the gated points into
  means may raise mean quality without the coverage collapse.
- **Per-class anchor budget**: more anchors for the loose classes {7, 13, 14}
  (the tight majority needs fewer). The budget curve b_c per class, not uniform.
- **Weighted propagation**: instead of the hard nearest-anchor assignment, use a
  soft assignment weighted by 128-d similarity (a point contributes to multiple
  class means with weight = softmax(sim/T)), which smooths the boundary contamination.

**B. The AL SELECTION (which points to query):**
- **Confidence selection for means** (the prior docs: corr(confidence, distance
  to centroid) is negative everywhere, so the frozen probe's HIGH-confidence
  points ARE the centroid-near representatives, the free self-selecting query
  rule that influence is NOT). Test confidence vs random anchors.
- **Mass-stratified anchors**: pick anchors in proportion to the class mass, so
  the rare-class means are not starved (the count error V5 was large on fog).
- **Boundary-avoiding anchors**: select points FAR from the boundary (high
  margin) since the bulk transfers (C1: precision increases with margin), the
  opposite of influence.

**C. The UPDATE / DECODER (how the means become a classifier):**
- **Fractional whitening** for the mean decoder: Sigma^-beta with beta in
  {0.25, 0.5}; Iteration 6 showed fractional whitening reduces sensitivity to
  mean error (the 35x amplification); the mean decoder may tolerate the
  propagated-means error better at beta < 1.
- **Update-norm constraint**: scale the propagated update toward ~1x ||R|| (the
  corruption-control operating point); if the propagated means are good in
  direction but noisy in magnitude, the constrained step may realize more of the
  gain.
- **Per-class mean shrinkage toward the pseudo-mean**: M_est = (1-a) M_prop +
  a M_pseudo: blend the propagated mean (low variance, noisy assignment) with
  the frozen-pseudo mean (biased but pool-stable), sweep a.

These are diagnostics about the CURRENT method's problems, in service of
continuing to improve it, not a ceiling assumption to trust blindly.
