# Class-statistics decoder: the reformulated update mechanism

This doc tracks the diagnostics for the class-statistics decoder reformulation --
the concrete "reformat what we store/update" direction that survived the
W-update and acquisition closures. It is written WHITE-BOX: each test splits the
construction into measured pieces so we can see what works and what needs fixing,
rather than treating the method as a blackbox.

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
| Label-free pool pseudo-mean (hard) | Real but weak positive (+0.05-0.11) |
| Soft / TTA pseudo-mean | Open, untested |
| Pool basis + scalar/class coefficients (labels estimate alpha in R^K) | Open, untested |
| Pseudo-label confusion-matrix correction (Q, C x C) | Open, untested |
| Class-prior correction (P* vs P0, label-free) | Open, untested |
| Robust / core-set mean acquisition | Open, untested |
| Whitened / residual-relevant mean-error diagnostic | Open, must run |

The headline: a good low-dimensional-ish target (M*) exists, but the naive
few-label estimator of it is terrible. The open formulations all use the labels
to estimate a LOW-DIMENSIONAL object (scalars, a confusion matrix) on top of
pool-derived class statistics -- not a 10000-d vector. (See Iteration 2 verdict
for the precise closure and the three-arm Iteration-3 plan.)

## Background: why a reformulation, and what we are reformulating

**The closure that motivated it.** Iterations 3b-7 of `new_iters.md` closed the
few-label landscape: the label span cannot capture the full residual R = W* - W0
(3b), the unlabeled pool cannot provide its basis (4), the pair-repair loop fails
even with oracle pairs (5), sequential acquisition does not beat one-shot (6), and
even the ORACLE acquisition curve is flat -- the label budget is not the binding
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

- M = the class-mean matrix (rows mu_c) -- "where each class sits"
- Sigma^-1 = the inverse code covariance (whitening) -- "the shape of the data"
- P = diag(n_c / N) -- class priors

(LDA form; exact for the ridge fit up to the sketch/CG approximation. The doc
used the shorthand W = Sigma^-1 P M, which only reads correctly if M is taken as
d x C with columns = mu_c; the orientation above is the one the code computes.)

The residual R then splits. HOLDING the priors fixed at P = P0 (as the
diagnostics do -- both W0 and W_mean_oracle use the clean counts C0), the
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
  whitening and was R1 < R4. Here we keep the full Sigma^-1 M^T P form -- it IS W
  -- and only re-estimate the M block.
- The covariance term may be large; if so, mean-only updates cap out. The first
  thing to measure is how much of R the mean-shift term accounts for.

## The diagnostic (al_class_stats_diag.py): white-box pieces

The construction is split into measured parts, so each arm shows WHAT is working /
WHAT needs fixing:

- **A. The decomposition** -- `R_mean_frac` (fraction of R the mean-shift accounts
  for under the pool whitening) and the decoder ladder W0 / proto_oracle (cosine
  to oracle means, NO whitening) / W_mean_oracle (WHITENED mean decoder, oracle
  means) / W* (full oracle). This separates: does the mean shift matter (proto vs
  W0), does the whitening matter (proto vs W_mean_oracle), is the covariance part
  big (W_mean_oracle vs W*).
- **B. Few-label mean estimation** -- W_est = Sigma0^-1 P0 M_hat with M_hat the
  shrunk sample mean of b labeled points per class: sweep b per class {2,4,8} x
  shrinkage alpha {0,2,8} (the memory knob) x selection {random, proto_dist},
  reporting gc AND the per-class mean estimation error ||M_hat - M*||/||M*||.
- **C. Selection ablation** -- random / proto_dist / entropy / oracle_error at a
  fixed budget: WHICH points of class c to label for the mean estimate.
- **D. Update details** -- softmax temperature (step-size-like knob) and top-K
  update scope (sparse update: only the K most-moved classes).

## Iteration 1 result: the DECOMPOSITION is correct -- the whitened class-means
capture the residual -- but the few-label MEAN ESTIMATOR is the bottleneck
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

The whitened mean decoder with oracle means closes **+0.72 to +1.15 gc -- matching
or even EXCEEDING W\* on crosstalk/snow**. This confirms the reformulation's
ceiling is the FULL residual, and that **whitening is essential**: proto_oracle
(no whitening) is far weaker (+0.41-0.47) or negative (-1.75, -0.90). This is
exactly why the R1 prototype decoder closed (R1 < R4 without whitening) but the
whitened class-mean decoder need not.

`R_mean_frac` reads > 1 (1.3-2.7) -- a measurement artifact of mixing clean
counts with pool whitening, not a meaningful bound; the ladder is the reliable
number.

### B. The few-label mean estimator is the bottleneck (the negative)

W_est is NEGATIVE at every operating point (best over alpha shown):

| est arm | dglsspp fog | dglsspp crosstalk | covshift fog | covshift wet_ground |
| :--- | :--- | :--- | :--- | :--- |
| random | -0.19 (b8 a2) | -0.23 (b8 a2) | -1.32 (b8 a8) | -1.59 (b8 a8) |
| proto_dist | -0.29 (b2 a8) | -0.32 (b2 a8) | -1.47 (b8 a8) | -1.70 (b8 a8) |

- The mean estimation error ||M_hat - M*||/||M*|| is **0.46-0.65 even at b=8/class
  with alpha=2** -- 8 labels cannot estimate a 10000-d class mean, and the
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
  -1.92) -- confirming the shift is concentrated in a few classes (top_moved on
  fog: c14/c13/c4/c7/c16 for dglsspp, c2/c7/c11/c4 for covshift) but still
  negative because those few mean estimates are also noise.

### Verdict and the fix path

The reformulation is VIABLE in structure but the ESTIMATOR is broken. The ceiling
is real (W_mean_oracle ~ W*), so the fix is not the decoder -- it is HOW the
shifted means are obtained. The raw few-label sample mean fails because a 10000-d
mean needs far more than 8 points and the whitening amplifies the variance. The
concrete fix paths this diagnostic isolates:

1. **Pool-pseudo-label means + few-label bias correction.** Estimate the shifted
   means from the UNLABELED POOL via pseudo-labels (low-variance, biased) and use
   the few labels to correct the BIAS (the mean SHIFT, a concentrated low-rank
   object) rather than estimate the mean from scratch.
2. **Estimate only the mean SHIFT** Delta_mu_c = M*_c - M0_c (concentrated in a
   few classes per top_moved) under the clean whitening, with strong shrinkage --
   not the absolute mean.
3. **Regularize the whitening** so it does not amplify the noise floor.

Iteration 2 (planned): path 1 -- pseudo-label the pool for low-variance shifted
means, use the few labels to correct the class-mean bias.

## Iteration 2 result: all three fix paths FAIL to rescue the few-label estimator
-- the pool-pseudo-means alone are the only positive, and the update-norm
diagnostic shows the whitening amplifies the few-label noise 35x
(2026-08-30, `al_class_stats_fix_diag.py`)

Validated the three fix paths independently, all against the same references
(W0 = 0, W_mean_oracle = the ceiling, W* = 1). JSONs verified clean
(`al_class_stats_fix_{dglsspp,covshift_ep10}.json`).

**FIX 1 (pool pseudo-means + bias correction) -- the pool-pseudo-means ALONE are
positive, the bias correction HURTS.** M_pseudo_c = mean of the pool codes the
frozen probe pseudo-labels as class c. At alpha_d = 0 (no bias correction -- the
pure label-free pool pseudo-mean decoder) the gc is the ONLY positive in the
whole test: dglsspp fog +0.05, crosstalk +0.11. But adding the few-label bias
correction (alpha_d > 0) drives it negative (dglsspp fog -0.26, crosstalk -0.26):
the bias estimate M_lab_c from 2-8 points is itself noisy, so correcting a
low-variance pool estimate with a high-variance label estimate makes it worse.
The pool provides low-variance means (+0.05 to +0.11, label-free); the labels
cannot improve them.

**FIX 2 (shift-only + strong shrinkage) -- fails, all negative.** M_shift_c =
M0_c + s*(M_lab_c - M0_c) with s in {0.1, 0.3, 1.0}: best is s=0.1 at -0.31
(dglsspp fog/crosstalk). The shift M_lab_c - M0_c is estimated from the SAME
2-8 noisy points, so it is exactly as noisy as the absolute mean. Shrinking the
step does not fix a wrong direction.

**FIX 3 (regularized whitening) -- the lambda_w sweep has NO effect, rank
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

**IMPORTANT -- what this does and does NOT establish.** It establishes that the
ESTIMATOR "few-label raw sample mean, then whitened" is catastrophically noisy. It
does NOT establish that "few labels cannot estimate M*". Two things were never
measured and must be, before closing the estimator family:
- the WHITENED mean error ||Sigma^-1 (M_hat - M*)|| / ||Sigma^-1 M*|| -- a 46-65%
  raw mean error could be much less damaging if most of it lies in directions the
  decoder suppresses (and conversely a 10% error could be disastrous in high-gain
  directions of Sigma^-1);
- the oracle-relevant projection <Sigma^-1 (M_hat - M0), R> / ||R||^2 -- whether
  the noise is along R (damaging) or orthogonal to it (benign).

### Verdict: close the raw mean estimator, NOT the class-statistics program

The decomposition is strongly validated (W_mean_oracle +0.72 to +1.15 gc -- the
particular statistic M* really contains most/all of the information the linear
probe needs). The reformulation is therefore not "another prototype heuristic."
The narrow, well-supported closure is:

> "2-8 RAW LABELED samples cannot estimate a 10000-d class mean accurately enough
> for the subsequent whitening" -- under the current class-code distributions and
> the raw-sample-mean-then-whiten estimator.

The wider statement "the few-label class-mean program is a no-go" is NOT yet
supported. The pool provides low-variance class statistics (M_pseudo, +0.05 to
+0.11 label-free positive); the labels have only been asked to do the ONE thing
they cannot (estimate a 10000-d vector). The genuinely open formulations all use
the labels to estimate a LOW-DIMENSIONAL object, not the mean:

1. **Pool estimates the class statistics; few labels estimate a small correction
   to them** -- the "labels select a correction, not supply its 10000-d
   direction" idea. Forms: (a) pseudo-label CONFUSION matrix Q (C x C, not
   C x 10000): M_tilde_c ~= sum_j Q_cj mu*_j, labels estimate Q; (b) scalar
   coefficients on pool-derived shift directions: mu*_c ~= M_tilde_c + alpha_c
   v_c, where v_c is pool-derived (pseudo-vs-clean mean, high-confidence subset
   mean, TTA displacement, density-core displacement) and labels estimate only
   alpha in R^K with K = 3-5 suspicious classes. This is the class-statistics
   analogue of the pool-basis idea, but at the M level (K scalars, not the
   full residual) -- fundamentally different from the closed rank-1-per-label
   construction (which used label CE-gradient directions).
2. **Improve the label-free pool pseudo-mean decoder** (the +0.05-0.11 positive):
   SOFT pseudo-means (weight by p_i(c), TTA-averaged, sharpen only when
   justified) instead of hard argmax pseudo-labels -- tests whether the positive
   is limited by hard-label contamination.
3. **Robust/active mean estimation** (cheap, untested): coordinatewise median /
   trimmed / Huber / distance-trimmed / nearest-to-CLEAN-prototype (note: proto_
   dist SELECTED far points was bad for means; the OPPOSITE selection -- points
   NEAR the clean class core -- is the untested direction), and k-center among
   high-confidence class-c points (multi-modal classes).
4. **Measure pseudo-mean quality independently of decoder gc**: per-class
   ||M_tilde_c - mu*_c||, the whitened error, and the residual-relevant error
   e_c^R = |<Sigma^-1 (M_tilde_c - mu*_c), R_c>| -- which classes account for the
   +0.05-0.11 to +0.72-1.15 gap (probably NOT all 17).
5. **The class-prior correction** (from the corrected residual expansion): P* vs
   P0 is directly observable from the UNLABELED pool (class counts need no
   labels) -- a zero-label mechanism never tested.

### Iteration 3 (planned): a three-arm decisive test, not "one more then close"

The final test should answer three different hypotheses, NOT just step-size
calibration (which is an ORACLE diagnostic, not a deployable fix -- scaling a
35x noise vector down turns a big wrong step into a small wrong step; run it
only as the oracle-scale control, separated from any data-driven scale):

- **Arm A -- pure pool baseline (label-free ceiling of the construction):**
  W_pseudo = Sigma0^-1 M_pseudo^T P0 with HARD / SOFT / TTA pseudo labels. If
  soft/TTA pushes +0.05-0.11 toward +0.72-1.15, the label-free route is the
  strong one.
- **Arm B -- pool basis + scalar/class calibration:** identify the 3-5 shifted
  classes (pool evidence), estimate only alpha in R^K on pool-derived shift
  directions. Labels estimate K scalars, not a 10000-d vector.
- **Arm C -- pseudo-label confusion correction:** few labels estimate a small
  class-confusion/calibration matrix Q (C x C, restricted to the K x K
  suspicious-class block -- pair-concentration evidence supports this), correct
  M_pseudo with it.

Read: Arm A > B,C  -> the label-free route dominates (labels not needed); B or C
> A -> the labels ARE useful for low-dimensional correction, which is the
reformulation's actual claim. Also run the corruption control (corrupt D_oracle =
W_mean_oracle - W0 with additive noise D_rho = sqrt(1-rho^2) D + rho N, measure
gc(rho)) to translate "35x noise" into the required estimation precision: if
even 80% directional corruption retains performance, the estimator problem is
solvable; if 5% destroys it, the route is brittle.

Standing caution (from Iteration 7): the permutation result showed the labels'
content barely mattered for the ONLY working first-order mechanism -- the gain
was geometric (U-leverage of selected points). That is a risk for B/C specifically
(the label CONTENT is what drives the class correction there) and is exactly what
this three-arm test decides.
