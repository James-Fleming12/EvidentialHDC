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
