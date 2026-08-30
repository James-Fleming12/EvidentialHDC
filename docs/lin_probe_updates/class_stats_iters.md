# Class-statistics decoder: the reformulated update mechanism

This doc tracks the diagnostics for the class-statistics decoder reformulation --
the concrete "reformat what we store/update" direction that survived the
W-update and acquisition closures. It is written WHITE-BOX: each test splits the
construction into measured pieces so we can see what works and what needs fixing,
rather than treating the method as a blackbox.

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
fit, w_c = (X^T X + lam I)^-1 X^T y_c, and X^T y_c / n_c = mu_c (the class mean of
class c's codes). So the probe decomposes EXACTLY as

    W = Sigma^-1 P M

- M = the class-mean matrix (rows mu_c) -- "where each class sits"
- Sigma^-1 = the inverse code covariance (whitening) -- "the shape of the data"
- P = diag(n_c / N) -- class priors

(LDA form; exact for the ridge fit up to the sketch/CG approximation.) The
residual R then splits into a mean-shift term and a covariance term:

    R = Sigma^-1 P (M* - M0)  +  (Sigma*^-1 - Sigma0^-1) P M*

The bet: the mean-shift term is LOCAL and LABEL-ESTIMABLE (the mean of a few
labeled points of class c directly estimates mu_c), unlike the full residual R
which the earlier closures showed is a global object. The update object is the M
block; the pool supplies the Sigma^-1 geometry (unlabeled, abundant).

Two honest caveats that every diagnostic must check:
- This is NOT reopening R1 (the prototype decoder). The pure prototype drops the
  whitening and was R1 < R4. Here we keep the full Sigma^-1 P M form -- it IS W --
  and only re-estimate the M block.
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
Iteration 1, now quantified: the problem is NOT the decoder and NOT the whitening
scale -- it is that a 10000-d class mean CANNOT be estimated from 2-8 labels to
the precision the whitening requires. Every arm that injects few-label means into
the whitening is a ~30x-step disaster.

### Verdict

All three fix paths fail to reach the ceiling. The single positive is the
LABEL-FREE pool pseudo-mean decoder (+0.05 to +0.11 gc on dglsspp) -- the first
positive that needs no oracle U and no labels at all, but far below the +0.72 to
+1.15 W_mean_oracle ceiling. The reformulation's bottleneck is now fully
localized: a few labels CANNOT estimate a 10000-d class mean to whitening
precision (update_norm ~35x). The paths that might still close the gap, in the
order the evidence supports:

1. **Refine the label-free pool pseudo-mean decoder (the +0.05-0.11 positive)**
   -- improve the pseudo-labels (TTA-averaged, iterated/self-training) rather
   than adding noisy label-estimated means. The pool IS the estimator; labels so
   far only add noise.
2. **Estimate the mean-shift in a LOW-RANK subspace** (top_moved showed 3-4
   classes concentrate the shift) using points whose pseudo-class is CONFIDENT
   (high margin), not 2-8 random points -- the shift is a bias, and confident-
   pseudo points may estimate it with lower variance.
3. **Calibrate the whitening step size directly** (shrink the update norm toward
   ~1x R, e.g. rescale W_est - W0) -- the rank-truncated +0.12 on crosstalk shows
   the direction can be right; the 35x magnitude is the killer.

Iteration 3 (planned): the label-free pool pseudo-mean decoder + step-size
calibration -- it is the only arm with a positive, and it needs no labels.
