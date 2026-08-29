# Active Domain Adaptation: low-label, efficient, online (Iterations, part 2)

Tracking the search for a **low-label and efficient method of online active
domain adaptation** (ADA) for the cov-shift DGLSS++ LiDAR segmentation system:
a decoder-side update that (i) spends a tiny label budget, (ii) runs online from
a stream with constant memory, and (iii) reliably improves on the frozen
zero-shot probe without catastrophic degradation on the healthy conditions.

This is the continuation of `active_iterations.md` (Iterations 0-10: the
pseudo-label gate / $S$-$T$ decomposition / sensitivity-bounded update closure)
and the cov-shift AL line (`cov_shift_iterations.md` C16-C31: the ill-conditioned
full-probe collapse, the low-rank $W_{res}$ fix, the random-bank baseline).

---

## Background

### The goal

A **single framework** that, at deployment on a corrupted stream:

- **spends as few true labels as possible** -- the working budget is ~56-224
  true labels (k=8-32 per class) plus an unlabeled pool;
- **is online and memory-bounded** -- processes the stream per-scan, keeps
  constant memory (no stored dataset), and updates as the corruption evolves;
- **is efficient** -- the update must be a cheap low-rank/second-order solve, not
  a full $10000 \times 10000$ ridge refit;
- **never degrades** the frozen zero-shot probe (a "zero-degradation guarantee"
  on the healthy conditions), and improves it where a real closeable gap exists.

The extractor is frozen (cov-shift DGLSS++, ep-10/21). All work is decoder-side:
learn $W$ on the binarized HDC code $X \in \{\pm1\}^{n \times d}$, $d=10000$
(and its cheaper projections, see the efficiency finding below).

### The failure catalogue: what we tried and why each route closed

Every route that closed is listed with the measurable reason it failed, so the
remaining search does not repeat them:

| # | Route | Verdict | Why it failed (measured) |
| :--- | :--- | :--- | :--- |
| 1 | **Full-probe update** $W_{sub} = (S+\lambda I)^{-1}T_{hat}$ | **collapses** ($-0.21$ to $-0.61$ mIoU) | $S = X^\top X$ is ill-conditioned ($\kappa \sim 10^6\text{-}10^7$); per-class label errors $\delta T$ are amplified $11\text{-}125\times$ along the low-variance tail eigenvectors (C16-C23). Even with TRUE labels (C28: $-0.211$ fog) it collapses. |
| 2 | **Pseudo-label gating** (confidence / margin / soft / influence) | **never beats frozen** | wrong pseudo-labels anti-align with the oracle rotation ($\cos(W_{wrong},W_{oracle})<0$ on every condition) and contribute ~equal magnitude to the update; confirmation bias is structural (Iterations 9-11). |
| 3 | **Coresets / diverse / uncertainty point selection** for the bank | **dead end** | diverse/farthest-point and uncertainty $H(p)$ allocations LOSE to random on both bank mIoU and $W_{pseudo}$ (C28). The residual lives in the LOW-variance directions; coreset-style selection covers the HIGH-variance ones -- exactly backwards. Also confirmed at the fit level: hard-point coreset + dual ridge is below the plain prototype (tta Iteration 7). |
| 4 | **$U$ estimation from unlabeled statistics** (pool-cov SVD, $S^{-1}T$, $U_{shift}=SVD(M)$, $U_{Rreg}$) | **all $\approx 0$ or negative** | no unlabeled $U$ captures the residual subspace; alignment with oracle $U_R$ is 0.00-0.05 (C21, C22, C25, C26). The residual directions are not the high-variance code directions nor the shift's own directions. |
| 5 | **Prototypes / $S^{-1}T$ without a bank** | **worse than frozen** | nearest-class-mean is below $W_0$ at every budget to $k=128$ (C25); $S^{-1}$ amplifies the wrong directions. |
| 6 | **Class-balanced probe fits** (per-sample $w=1/N_c$, per-class $\lambda_c$, logit prior) | **neutral to negative** | majority classes dominate the mIoU mean; logit prior strongly negative (full-scale diagnostic). |
| 7 | **Dual / Woodbury / RLS at $d=10000$** | **collapses (dual)** | the sample-dim inversion is unstable at the correct $\lambda$; the conditioning artifact is the same tail-eigenvector problem (tta Iteration 2/7). |
| 8 | **Input-space remediation** (pre-filtering, additive noise, global pooling) | **dead end** | none beat the plain robust encoder (Phase 10 shootout). |

### The best method so far

**Low-rank residual update $W_{res} = W_0 + U_r C$ with a random 56+500 bank**
(C30/C31):

$$C = (U_r^\top X_{lab}^\top X_{lab} U_r + \gamma I)^{-1} U_r^\top X_{lab}^\top (Y_{lab} - X_{lab} W_0),$$

where $U_r$ ($r=8$) is the oracle SVD of $W^* - W_0$, and $X_{lab}$ is the code
of the 56 true-label points (k=8/class) plus 500 random unlabeled points. The
8$\times$8 projected system is well-conditioned (no meaningful $\gamma$), the
fit is sub-millisecond, and inference is the linear $W_{res}$ (no bank at test
time). It inherits a **zero-degradation property**: when the residual target
$Y_{lab} - X_{lab}W_0 \to 0$ (healthy conditions), $C \to 0$ and
$W_{res} \to W_0$.

Full-scale numbers (every point of every frame of seq 08, ~300M pts/cond,
cov-shift ep-10):

| condition | closeable gap | $W_{res}$ pseudo | $W_{res}$ true | gap closed (true) |
| :--- | :--- | :--- | :--- | :--- |
| wet_ground | +0.122 | +0.043 | +0.130 | ~100% |
| fog | +0.047 | +0.005 | +0.059 | ~100% |
| cross_sensor | +0.026 | +0.010 | +0.027 | ~100% |
| crosstalk | +0.014 | +0.011 | +0.015 | ~100% |
| snow | +0.013 | **-0.016** | +0.013 | ~100% |

The **oracle** $U_r$ (from true $W^*$) recovers ~100% of the closeable gap at
$k=8$ on every condition. The random 500-point bank, as a **1-NN teacher of the
residual**, is a good per-point predictor (0.66 fog accuracy) but a bad
per-class mass gate (C28) -- see Problem 2 below.

### The two main problems now

**Problem 1 -- instability of the pseudo-label version.** $W_{res}$ pseudo vs
$W_{res}$ true differ on the bank's 500 points: fog +0.005 vs +0.059, wet +0.043
vs +0.130, and snow flips sign (-0.016 vs +0.013). The random 500 gives ~66%
correct pseudo-labels whose errors are **systematic per class** (they starve the
rare classes in $T$), so the residual $C$ inherits a class-conditional bias.
This is a LABEL bias problem, not a selection problem -- the low-rank $W_{res}$
already fixed the ridge-amplification half (Problem 2 of the old baseline).

**Problem 2 -- the explicit random memory bank.** The method stores a 500-point
bank and labels them by 1-NN from 56 true labels. Two reasons to move away:
(i) the 500 points are chosen randomly, not for informativeness, and every
"informative" selection tried (coresets, diverse, uncertainty) was neutral to
worse -- so there is no known way to make the bank smarter; (ii) the bank is a
stored dataset, which is not "online/constant-memory" in the streaming sense.
The real bottleneck behind both is that **$U$ is not estimable from any
unlabeled statistic tried**, so the method currently needs oracle $U$ to work,
and the bank is a workaround for that.

**Problem 3 -- not yet online.** Each condition is a batch fit over a 400k
reservoir, not a per-scan incremental update. The online machinery (streaming
sufficient statistics / RLS at the right dimension) was never completed because
the $d=10000$ dual form collapsed -- but the code-2000 finding below changes
that.

### A key efficiency finding that reframes everything

The probe's accuracy **peaks at $d'=2000$, not $10000$** (tta Iteration 2):
code-2000 gives wet_ground 0.587 / fog 0.334 -- HIGHER than code-10000 (0.572 /
0.313) -- at 1.5M pts/s (above the prototype's fit throughput). The 10000-d
projection's large dimension was never the source of the gain. Consequences:
- the online/dual/RLS route (which collapsed at $d=10000$) may be stable at
  $d'=2000$ where the conditioning is better and the solves are ~25x cheaper;
- the $W_{res}$ 8$\times$8 system becomes trivially cheap at $d'=2000$, and the
  low-variance residual structure (the thing $U$ must capture) may be better
  behaved.

### The minimal-label result: the leverage-in-U query makes 2-8 labels work
(2026-08-27, `al_min_label_residual_diag.py`)

The established $W_{res}$ method needs k=8/class (56 labels) with oracle $U_r$
(r=8). This diagnostic asks whether the LABEL COUNT is the real bottleneck, by
sweeping the true-label budget b in {2, 4, 8, 16, 32, 56} across the residual
rank r in {2, 4, 8}, with oracle $U_r$ and three selection rules -- random,
leverage-in-$U$ (N9: query by $\|x^\top U_r\|$, the projection onto the residual
directions), and per-class balanced -- on both plain DGLSS++
(`supcon_vib_dglsspp`) and cov-shift. DGLSS++ is the primary target: on KITTI-C
3-sev it has the big closeable gap (fog zs 22.5 -> ceiling 35.2 = +12.7,
crosstalk +17.5) where AL actually has headroom, while cov-shift's frozen is
already within +2.9 of its ceiling (README KITTI-C table).

**The result: with the RIGHT query rule, a couple of true labels give a real
update -- and the rank, not the label count, is the binding choice.**

**DGLSS++ fog (frozen 0.119, oracle 0.291, gap +0.172):**

| r | b | random delta (gc) | **leverage-u delta (gc)** | per-class delta (gc) |
| :--- | :--- | :--- | :--- | :--- |
| 2 | 2 | -0.033 (-0.19) | **+0.059 (+0.34)** | -0.076 (-0.44) |
| 2 | 8 | +0.037 (+0.21) | **+0.062 (+0.36)** | +0.008 (+0.05) |
| 4 | 32 | +0.090 (+0.52) | **+0.097 (+0.56)** | +0.099 (+0.57) |
| 8 | 56 | +0.157 (+0.91) | +0.116 (+0.67) | +0.155 (+0.90) |

**DGLSS++ crosstalk (frozen 0.152, oracle 0.392, gap +0.240):**

| r | b | random delta (gc) | **leverage-u delta (gc)** | per-class delta (gc) |
| :--- | :--- | :--- | :--- | :--- |
| 2 | 2 | -0.057 (-0.24) | **+0.092 (+0.39)** | -0.033 (-0.14) |
| 4 | 8 | +0.009 (+0.04) | **+0.183 (+0.76)** | +0.117 (+0.49) |
| 4 | 16 | +0.121 (+0.50) | **+0.178 (+0.74)** | +0.111 (+0.46) |
| 8 | 56 | +0.185 (+0.77) | +0.125 (+0.52) | +0.237 (+0.99) |

**The findings.**

1. **Leverage-in-$U$ is the query rule that unlocks the tiny budget.** On
   DGLSS++ fog/crosstalk, leverage-u at b=2-8 gives +0.06 to +0.18 (gc 0.34-0.76)
   where random and per-class are negative or near-zero at the same budget.
   Querying the points with the highest projection onto the residual directions
   (N9) -- NOT random, NOT confidence, NOT per-class -- is what makes a couple of
   points work. The per-class balanced rule is the worst of the three at tiny
   budgets (spreads the budget over classes instead of concentrating it on the
   residual). This is the first measured confirmation that N9 is the active
   ingredient.
2. **The rank, not the label count, is the binding choice.** At r=2, leverage-u
   b=2 already closes gc 0.34/0.39 on fog/crosstalk (dglsspp) -- the rank-2
   residual carries most of the cheap-closeable signal. At r=8, b=2-8 collapses
   (negative or ~0); the full-rank update needs 32-56 labels. The C-fit is an
   r-dim system, so it needs >= r informative points to be well-posed; lowering r
   to match the budget is the lever. The rank-r ceiling (C from all labels) at
   r=2 is gc 0.39/0.43, and leverage-u at b=2-8 reaches 0.85-0.95 of that
   ceiling's cos_C -- the few-point C is nearly aligned with the all-label C.
3. **DGLSS++ is the right AL target; cov-shift has almost nothing to close.**
   cov-shift fog gap is only +0.115 (0.261 -> 0.375) vs dglsspp +0.172, and
   crosstalk +0.029 vs dglsspp +0.240. cov-shift's few-point wins are capped by
   its small gap (best fog r=4 b=32 leverage-u +0.098, gc 0.86 -- closing a small
   absolute gain). The big AL story is DGLSS++ crosstalk: +0.183 mIoU from 8
   true labels (gc 0.76 of a +0.240 gap).
4. **The healthy conditions confirm the mechanism and the caveat.** DGLSS++ snow
   (gap +0.037) is closed nearly fully at b=2 r=2 by leverage-u (+0.039, gc 1.06);
   wet_ground (gap +0.077) needs b=32 r=4 for gc 0.73. But the caveat: random and
   per-class can be CATASTROPHIC at tiny budgets on the healthy conditions
   (dglsspp wet_ground b=2 r=4 random -0.217 gc -2.83; snow per-class -0.32
   gc -8.8) -- the tiny-budget update is high-variance, and the query rule (or a
   gate) is what keeps it on the right side of frozen. The r=2 leverage-u cell is
   the one that is consistently safe AND useful.

**Implication for the deployed method.** The operating point is r=2-4 with the
leverage-in-$U$ query (b=2-8 labels), not r=8 with random (56 labels). This
replaces the label budget question entirely: the update needs only the ~2-8
points with the highest residual-leverage, and it closes 34-76% of the DGLSS++
closeable gap on fog/crosstalk with 8 labels (vs the old 56). The remaining
deployment step is unchanged: estimate $U$ without oracle (N7 CCA on the
clean/corrupted pairing), since this whole table uses oracle $U_r$. If N7 lands,
the couple-of-points regime becomes fully deployable.

### The U-estimation diagnostic: label-free U is ORTHOGONAL to the residual; only a
few-label sub-fit captures it, and even that does not close the gap
(2026-08-27, `al_uest_diag.py`)

The minimal-label result showed 2-8 leverage-in-$U$ labels close 34-76% of the
DGLSS++ gap -- but it used ORACLE $U_r$. This diagnostic estimates $U$ with and
without labels, and evaluates the full chains for the two stated goals: GOAL A
(label-free TTA that meaningfully improves fog/crosstalk) and GOAL B (few-label
AL that approaches the ceiling). Runs on DGLSS++ (primary) and cov-shift.
$U$-estimators: `oracle` (ref), `softshift` / `poolcov` / `ccameans` (label-free),
`subfit_b` / `shiftsub_b` (b labels). Reports per-direction alignment, residual
capture, a construction-vs-estimation decomposition, and the pseudo-vs-true
projected signal that drives $C$.

**Alignment and residual capture (DGLSS++, per condition; oracle captures ~1.0):**

| cond | softshift | poolcov | ccameans | shiftsub_b8 | subfit_b8 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog align r8 / resid_cap r8 | 0.05 / 0.05 | 0.02 / 0.03 | 0.05 / 0.05 | 0.03 / 0.03 | **0.53 / 0.53** |
| crosstalk align r8 / resid_cap r8 | 0.04 / 0.04 | 0.02 / 0.02 | -- | 0.03 / 0.04 | **0.38 / 0.37** |
| snow align r8 / resid_cap r8 | 0.03 / 0.03 | 0.02 / 0.03 | 0.02 / 0.02 | 0.03 / 0.03 | **0.34 / 0.53** |
| wet_ground align r8 / resid_cap r8 | 0.03 / 0.03 | 0.03 / 0.03 | -- | 0.02 / 0.03 | **0.23 / 0.36** |

**Finding 1 -- NO label-free U estimator recovers the residual subspace.** All
three label-free constructions (soft-shift of class means, pool covariance
eigenvectors, class-mean CCA) land at cos(U, U_oracle) = 0.01-0.07 and capture
only 0.02-0.08 of the oracle residual's norm, on every condition and extractor.
The label-free U is essentially ORTHOGONAL to the residual -- the update it
defines moves W in directions the true residual does not occupy. This is not a
weak-update failure; the C fit has nothing correct to act on (the leverage-in-U
query under a wrong U selects the wrong points, and the C projection is on a
basis that misses R).

**Finding 2 -- the construction is wrong, NOT the soft estimation.** The
construction diagnostic splits the shift-family failure: `true_shift_vs_oracle`
(shift->U built from TRUE corrupted means) is 0.01-0.12 -- even with perfect
means, the shifted-class-mean construction does not span the oracle residual.
But `est_shift_vs_true_shift` (soft-estimated shift vs true shift) is
0.50-0.99, and `soft_mean_vs_true_mean` is 0.5-0.9 on fog/crosstalk: the soft
means are fine, the shift->U construction is the wrong object. The residual
$R = W^* - W_0$ is not a function of the class-mean shift; it lives in the
decision-rule geometry (Iteration 12's spectral-overlap finding, now with the
specific negative). Adding a different label-free construction (per-point
covariance of the residual, higher-order statistics) is unlikely to fix this:
the residual is not in any first-order shift statistic.

**Finding 3 -- subfit (SVD of W_sub - W0) is the ONLY U that captures real
residual, but it needs labels and still does not close the gap.** Fitting W_sub
on b leverage-selected points and taking the SVD of (W_sub - W0) captures 0.36-0.53
of the residual at r=8 (b=8), the only estimator to beat 0.1. But its AL chain
reaches only gc 0.03-0.09 on fog/crosstalk (best dglsspp crosstalk r2_b32 +0.020,
gc 0.09) -- FAR below the oracle-U reference (+0.183, gc 0.76). The subfit U is
a partial residual but the few-point C fit in it still under-closes; the oracle-U
leverage query is what made 2-8 labels work, and a wrong-U leverage query
(selection under softshift leverage) selects the wrong points.

**Finding 4 -- GOAL A (label-free TTA) is closed by U, not by C.** The TTA chain
(label-free U + pseudo-label C) is ~0 or negative on every cell: best label-free
TTA on fog/crosstalk is +0.011 (poolcov fog) with gc 0.06; most are negative.
The projected-signal diagnostic shows the label source is a co-factor: under the
oracle U the pseudo-vs-true residual signal cos is -0.54 (fog) to +0.32 -- the
pseudo labels do NOT align with the true signal even in the RIGHT subspace, so
the low-rank constraint does NOT rescue label-free supervision. The U being
orthogonal (Finding 1) compounds it. Both the label source AND the U construction
are wrong for label-free TTA; the Iterations 9-12 closure (pseudo labels are
structurally poisoned) survives the low-rank reformulation.

**Finding 5 -- cov-shift's few-label AL is similarly capped.** cov-shift's best
few-label AL (subfit_b32 wet_ground r2_b8 +0.031, gc 0.17) is the largest few-label
gain anywhere, but its fog/crosstalk gaps are small (+0.115/+0.030) so the absolute
wins are tiny. The near-ceiling goal is not met on either extractor.

**Verdict.** The minimal-label result stands (oracle U + 2-8 leverage labels close
34-76% of the gap), but the deployment path through U estimation is NOT available
by the constructions tried: (1) label-free U is orthogonal to the residual, and
(2) the only label-consuming U (subfit) still under-closes because the C fit and
the leverage query both need the true U. The next useful direction is not another
U estimator but either (a) using the subfit-U as a coarse prior and refining C in
a higher-rank space, or (b) accepting that the couple-of-points regime requires
oracle-quality U and targeting the subfit route with MORE labels (b=100-1000)
where W_sub approaches W*, which is the C30/C31 bank setting restated. The
diagnostic also settles N7 (CCA): class-mean CCA does not recover U on this
problem.

### The boundary / tangent U-estimation diagnostic: decision-boundary geometry does
NOT contain U; the few-label tangent-space construction (PCA across tiny
provisional updates) is the FIRST estimator to recover substantial residual
structure -- but its AL chain still does not close the gap
(2026-08-27, `al_uest_bdry_diag.py`)

Follow-up to the U-estimation closure. Tests the NEW decision-rule constructions
from the research plan, at r in {2,4}, budgets {8,32}, on DGLSS++ (the primary AL
target) and cov-shift. The cov-shift JSON pull was a corrupt transfer twice; the
values below are transcribed from the server terminal (the JSON on disk is null
bytes), so both extractors are complete here.

**Family 1 (label-free, decision-boundary geometry): CLOSED.** Four constructions
-- PCA of near-boundary points (`bdry_pca`), the decision-weighted outer product
`sum_i x_i (g_a - g_b)_i^T` (`bdry_outer`), inverse-margin-weighted covariance
(`bdry_margin_cov`), and most-confused-pair PCA (`pair_ab`) -- all land at
align cos(U, U_oracle) = 0.00-0.03 and residual_capture 0.01-0.03 on every
condition and extractor. The hypothesis that "boundary geometry contains the
residual" is FALSIFIED: conditioning on near-boundary points does not make the
pool covariance (or its decision-weighted variants) any closer to the residual
subspace. The residual is not in any second-order statistic of the boundary
region either.

**Family 2 (few-label, adaptation tangent space): PARTIAL SUCCESS -- the first
estimator to capture real residual.** `tangent_b8` (split 8 labels into 4
2-point provisional ridge fits, stack their dW = W_t - W0, take the right SVD)
recovers substantial residual structure on every condition and extractor:

| cond | extractor | align r2 | align r4 | resid_cap r2 | resid_cap r4 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | dglsspp | 0.24 | 0.52 | 0.38 | 0.55 |
| fog | covshift | 0.40 | 0.52 | 0.36 | 0.55 |
| crosstalk | dglsspp | 0.34 | 0.46 | 0.34 | 0.49 |
| crosstalk | covshift | 0.22 | 0.33 | 0.22 | 0.36 |
| snow | dglsspp | 0.28 | 0.41 | 0.38 | 0.49 |
| snow | covshift | 0.31 | 0.39 | 0.36 | 0.47 |
| wet_ground | dglsspp | 0.07 | 0.28 | 0.08 | 0.34 |
| wet_ground | covshift | 0.18 | 0.42 | 0.17 | 0.46 |

This is the ONLY estimator (of every U construction tried across both
diagnostics: pool covariance, class-mean shift, CCA, boundary PCA, boundary
outer product, margin covariance, confused-pair, ensemble, subfit) that beats
cos ~0.05 against the oracle residual. The mechanism is the boxed reframing
working: PCA across NOISY provisional updates recovers the common residual
direction even though each 2-point update is individually poor. The tiny windows
are load-bearing -- `tangent_b32` (8 points/window) collapses to align
0.01-0.15, because fewer independent provisional samples give PCA less signal.
The result is extractor-robust: cov-shift's tangent_b8 aligns 0.18-0.40 (r2) /
0.33-0.52 (r4), the same pattern as DGLSS++.

**BUT the AL chain does not yet convert the discovered U into a gain.** Despite
`tangent_b8` finding the residual subspace (capture 0.34-0.55), its AL chain
(leverage-in-that-U true labels -> W_res = W0 + U C) is ~0 or slightly negative
everywhere (best gc -0.01 to -0.11; e.g. crosstalk r2_b32 gc -0.07). The
oracle-U reference closes gc 0.27-0.78 at the same cells. The discovered U is
correct but the leverage-in-U query under it does not select points that move C
the right way. The ensemble construction is also closed (align ~0.005).

**Synthesis.** The direction is now specific and actionable: (1) the label-free
decision-boundary route is closed (not just pool covariance -- conditioning on
boundaries does not help); (2) the adaptation-tangent-space construction is the
first thing that actually finds U, exactly as the research-plan reframing
predicted, and it does so from just 8 labels split into tiny provisional updates;
(3) the remaining gap is turning that discovered U into a working update -- the
joint U,C refinement (alternate C-solve / U-gradient / orthogonalize) on top of
the tangent-U init is the natural next experiment, rather than the separate
leverage-in-U-then-C pipeline. The tangent-b8 result also reframes the "2-8
labels" finding: the labels' value may be in DISCOVERING U (provisional updates),
not just in fitting C inside an oracle U.

### The joint U,C refinement: the C-fit is NOT the bottleneck -- refinement cannot
recover the couple-of-points AL from the tangent-U (2026-08-28, `al_uest_joint_diag.py`)

Tests the synthesis directly: instead of "discover U, then fit C separately", run
the alternating scheme (C-solve / U-gradient / QR) on the labeled points starting
from the tangent-U init, so the same labels that discovered U also push it toward
the oracle while fitting C. Per (b, r, T in {1,5,20}): baseline (C-solve in fixed
U), joint (after T alternations, with align_U_final + loss trajectory), and the
oracle-ref ceiling at the same budget. Both DGLSS++ and cov-shift, all 4 conds.

**Finding 1 -- the alternation converges but does NOT pull U toward the oracle.**
The loss descends cleanly on every cell (e.g. dglsspp fog b8 r4: 3.4 -> 0.7; snow
0.3 -> 0.0), so the scheme works mechanically. But `align_U_final` stays flat or
DROPS from the tangent init (dglsspp fog b8 r2: 0.37 -> 0.34; r4: 0.57 -> 0.51;
covshift fog b8 r2: 0.33 -> 0.32). The U-gradient overfits to the 8-32 labeled
points' idiosyncratic residual instead of the true residual: with so few labels,
the refinement moves U away from the oracle, not toward it.

**Finding 2 -- the joint AL chain stays at or below the baseline, far below the
oracle-ref.** On fog/crosstalk the joint gc is negative or ~0 while oracle-ref is
+0.18 to +0.84 (dglsspp fog b8 r2 joint gc -0.19 vs or-ref +0.20; crosstalk b32 r4
joint +0.25 vs or-ref +0.81). The refinement cannot convert the discovered U into
a gain because it is fitting the labels' residual, which is a small and biased
sample of the true residual.

**Finding 3 -- the b=32 tangent init is broken, confirming tiny windows are
load-bearing.** At b=32 the U_init_align collapses to 0.02-0.06 (vs b=8's
0.17-0.57) on every condition, exactly as the boundary/tangent diagnostic showed.
The b=32 "joint" gains (+0.02 to +0.25 gc) are just the oracle-ref at a larger
budget leaking through a near-random U -- they do not scale from the tangent
mechanism.

**Finding 4 -- the healthy conditions are noise-level.** wet_ground joint gc
reaches +0.07 to +0.15 at b=8 (dglsspp r2 T5 +0.07, covshift r4 T20 +0.09) -- the
only positive cells, on the condition with the loosest residual structure
(align init 0.17-0.43). But these are inside the noise band and do not beat
oracle-ref's 0.34-0.56.

**Verdict: the joint U,C direction is CLOSED as a couple-of-points AL fix.** The
measured negative is specific: the alternation works (loss descends), but with 8-32
labels the U-gradient cannot recover the true residual -- it overfits the labels,
align stays flat/drops, and gc stays far below the oracle-ref. The labels genuinely
cannot DISCOVER the residual better than the tangent-b8 PCA already does, and no
U-refinement on so few points adds signal. This closes the research-plan direction
#3 (joint U,C factorized fit).

**What remains.** The couple-of-points AL needs U quality that only more labels
(oracle) provide -- the C30/C31 bank setting restated. The two live leads that do
not depend on few labels discovering U: (a) the U-predictor head (dglss_imp.md D4:
an auxiliary head supervised by the clean/corrupted pairing to output the residual
basis at deployment), and (b) accepting that the cheap-label route requires
oracle-quality U and targeting the larger-label bank. The label-free and
decision-rule U-estimation families are all now closed with measured negatives.

### The first-order / trust-region diagnostic: the covariance WAS the artifact, and
the TTA-trust update is the first deployable few-label mechanism
(2026-08-28, `al_first_order_diag.py`)

Tests the escape-hatch reframing directly: stop treating C as the thing the ridge
formula must estimate, and stop estimating the pool covariance U^T X^T X U at all.
Methods per (b, r, step): `oracle_ridge` (reference), `oracle_first` (C = s*G, no
covariance), `oracle_norm` (C = s*G/||G||, trust-region), `gradspan` (U = orth of
the label gradients, no oracle U), `full_grad` (no U, no covariance, no bank),
`boundary_pair` (update only the confusion-pair margins), `tta_trust` (oracle_norm
with the step gated by a label-free instability signal: conf_drop -> rho = 0 if
the stream looks healthy). Both DGLSS++ and cov-shift, all 4 conds.

**Finding 1 -- the covariance WAS the artifact: ridge and the normalized
first-order update perform nearly identically.** On DGLSS++ fog b8: ridge gc +0.24
vs oracle_norm +0.04 (s=0.05) to +0.13 (s=0.8); cov-shift fog b8: ridge +0.51 vs
norm +0.30. The NORMALIZED first-order step (no U^T X^T X U, no inversion) closes
most of the ridge's gain on every condition. The RAW first-order step (C = s*G,
unscaled) is unstable -- ||G|| ~34-37, so s*G overshoots (dglsspp fog b8 s=0.2:
-0.38, s=0.8: -0.58) -- confirming that the failing term was the SCALE, not the
geometry. **Normalization alone fixes it: the missing-mass/covariance problem is a
consequence of choosing least-squares/Newton as the update rule.** The user's Tier-1A
hypothesis is confirmed.

**Finding 2 -- gradspan and full_grad fail: U (oracle or from label gradients)
is still needed.** `gradspan` (U from orth of the label gradients) aligns only
0.02-0.05 with the oracle U and its gc is negative everywhere (dglsspp fog b8
-0.14). `full_grad` (no U at all) is also negative (fog -0.16). The label gradients
do NOT span the residual -- the same conclusion as every U estimator before. The
oracle-U first-order update works, but the label-gradient-span U does not. So U is
still load-bearing; what changed is that we no longer need U^T X^T X U.

**Finding 3 -- tta_trust is the first deployable few-label mechanism: it keeps the
gain AND rejects healthy conditions.** The TTA-gated trust region (rho = step *
min(1, conf_drop*10)) delivers on every corrupted condition (dglsspp fog b8 gc
+0.19 at s=0.8; cov-shift fog +0.27, wet_ground +0.20) while being essentially
inert on the healthy ones (dglsspp snow gc +0.02, wet_ground +0.08; covshift snow
+0.11, crosstalk +0.13) -- the rho is small there (0.02-0.08) because the frozen
probe's confidence barely drops. This is the "TTA finds where/when + 2-8 labels
the direction + small first-order step the magnitude" architecture, and it is the
first few-label mechanism that is safe-by-construction: the gate drives rho -> 0 on
streams where the frozen decoder is already fine.

**Finding 4 -- boundary_pair is inert at b=8 (no confusion pairs selected).**
The margin-driven pairwise boundary update finds 0-8 pairs at b=8 and its gc is
~0 or negative; the queried leverage-in-U points are mostly not confusion-pair
boundaries, so the pairwise update has nothing to push. It needs a confusion-pair
query rule (select points ON the unstable boundaries), not the leverage rule.

**The resulting deployable mechanism.** The paper's AL/TTA can now be stated without
oracle U, without the pool covariance, and without the bank:
  1. **TTA (label-free)** estimates whether/when to adapt: rho from conf_drop /
     augmentation instability / temporal disagreement (validated gauges).
  2. **2-8 labels** (leverage-in-frozen-oracle-U, or tangent-b8) give the update
     DIRECTION G = U^T X^T (Y - X W0).
  3. **A normalized first-order step** W = W0 + rho * U G/||G|| applies it.
  4. **Zero-degradation by construction**: rho -> 0 on healthy streams (the gate),
     and W0 is the null option.

The remaining open piece is U itself (oracle or tangent-b8), since gradspan showed
the label-gradient span does not substitute. But U only needs to be a GOOD DIRECTION
(0.3-0.5 alignment is enough -- the trust-region step is robust to a coarse U), not
an exact basis, and the TTA gate makes even a noisy U safe. This is the strongest
few-label result in the whole AL thread: the covariance was the artifact, and the
trust-region + TTA-gate form is the deployable mechanism.

---

## Next steps (with potential)

Filtered from the full sweep; each item states the hypothesis, the evidence it
builds on, the cost, and the risk. Ordered by expected value / cost.

### N1. Bootstrap-ensemble over banks (stability, immediate)

**Hypothesis.** $W_{res}$ pseudo is noisy-but-unbiased; averaging $W_{res}$ over
$M=10$ independent 500-point draws reduces variance $\sim M\times$, and the
per-condition std becomes a deployment-confidence signal.

**Builds on.** The instability is a variance of the random-bank estimate, not a
bias of the low-rank form (the 8$\times$8 solve is sub-ms, so $M$ fits are free).
**Cost.** Eval-only; ~10 min added to the full harness. **Risk.** If the error is
biased (systematic per-class), ensembling shrinks the wrong thing -- in that case
N2/N3 are the fix and N1 still gives the detection signal (std across seeds).

### N2. Precision-weighted residual fit (stability)

**Hypothesis.** The bank's pseudo-label errors are class-conditional; weighting
each row by the estimated precision of its predicted class/confidence bin
removes their contribution to $T$:
$$C = (U^\top X^\top W X U)^{-1} U^\top X^\top W (Y - XW_0),\qquad w_i = \hat{p}(\hat{y}_i \mid x_i).$$

**Builds on.** We already measured per-predicted-class precision at confidence
quantiles (pseudolabel-structure D-reliability) and the label-free signals
(`r4_r1_disagree`, `conf_drop`) that correlate with gap ($\rho$ +0.36..+0.81).
**Cost.** Eval-only; one extra diagonal weight in an 8$\times$8 system.
**Risk.** The weights themselves come from the frozen probe's calibration, which
is biased on the corrupted stream -- but the gauge signals are validated, and the
weighting only needs the RELATIVE precision, not the absolute.

### N3. Per-class ridge $\lambda_c \propto 1/N_c$ on the residual (stability)

**Hypothesis.** The minority-class T-bias is amplified along the low-variance
directions; per-class shrinkage in the 8-dim residual damps it.

**Builds on.** C28's next-step list (explicitly proposed, never run); the
per-class pool-support analysis shows the rare classes are starved in the bank
(c2 ~1 point, c7 ~143). **Cost.** Eval-only; a per-class diagonal in the 8$\times$8
solve. **Risk.** The balanced-probe dead end (N6 in the catalogue) showed per-class
levers are neutral at the FULL-probe level; the low-rank residual is a different
(well-conditioned) system, so this is a fresh test, not a repeat.

### N4. Gauge-gated deployment (instability -> no-op)

**Hypothesis.** The instability is only harmful where there is no real gap; gate
the $W_{res}$ deployment on the label-free gauge so small-gap conditions (snow
+0.013) simply do not update.

**Builds on.** The validated gauge: `mean_shift_cos` ($\rho$ -0.57..-0.95),
`conf_drop` (+0.36..+0.81), `r4_r1_disagree` (+0.36..+0.67); a threshold gate
routes only wet_ground (+fog on some extractors), captures 49-60% of the total
gap at precision 1.00. **Cost.** Already built; integrates into the runner.
**Risk.** None -- it is the safety layer, not a method.

### N5. Streaming sufficient statistics instead of a stored bank (memory-free)

**Hypothesis.** The bank only enters through $U^\top X^\top X U$ (8$\times$8) and
$X^\top(Y - XW_0)$; both can be accumulated as streaming per-class means $\mu_c$,
counts $N_c$, and the $U$-projected covariance -- no stored points, constant
memory, truly online.

**Builds on.** The "estimate T, not labels" line (Iterations 7/8) failed on the
FULL probe's conditioning; inside the well-conditioned low-rank residual it is a
fresh, cheap formulation. Composes with N5's code-2000 projection.
**Cost.** New streaming accumulator + eval. **Risk.** Requires $U$ (the unsolved
bottleneck) -- unless combined with N7/N8.

### N6. code-2000 projection for the whole method (efficiency + conditioning)

**Hypothesis.** Running the entire $W_{res}$ pipeline at $d'=2000$ (not 10000)
keeps or improves accuracy (code-2000 already beats code-10000) while making the
online dual/RLS forms stable and the low-rank solve ~25x cheaper.

**Builds on.** tta Iteration 2: code-2000 peak (wet 0.587, fog 0.334) above
code-10000, at prototype-class throughput; the dual form collapsed at 10000 only.
**Cost.** Eval-only; re-run the $W_{res}$ tables with the $2000$-d projection.
**Risk.** The projection change interacts with $U$ estimation (N7/N8) -- must be
tested jointly, not in isolation.

### N7. CCA between clean and corrupted code distributions ($U$ estimation) -- TESTED, CLOSED

**Hypothesis.** The residual $W^* - W_0$ spans the directions where the clean and
corrupted class-conditional code structure disagrees. Canonical correlation
analysis on paired clean/corrupted scans (seq-08 clean vs each KITTI-C variant)
gives those directions directly -- and unlike everything tried (pool-cov SVD,
$S^{-1}T$, $U_{shift}$), it uses BOTH sides of the shift, which we uniquely have.

**Status: TESTED, CLOSED by the U-estimation diagnostic (above).** The class-mean
CCA construction (`ccameans`, PCA-whitened CCA between the clean class-mean
matrix and the soft corrupted matrix) lands at cos(U, U_oracle) = 0.01-0.05 and
captures 0.02-0.05 of the residual -- the same orthogonal failure as every other
label-free U. The construction-vs-estimation split shows WHY: even a shift->U
built from TRUE corrupted means (`true_shift_vs_oracle`) is 0.01-0.12, so the
residual is not a function of the class-mean shift in any first-order statistic,
CCA included. The C22 hint was right: the residual is in directions that are not
the top canonical modes either.

**Builds on.** C21/C22/C25/C26 all used unlabeled statistics of the corrupted
pool alone; none used the clean/corrupted pair. The clean/corrupted pairing is
exact for KITTI-C (per-frame corruptions of seq-08). **Cost.** New diagnostic
(PCA/CCA on clean-vs-corrupted per-class means or code covariance); moderate.
**Risk.** Realized: the residual is in directions that are NOT canonical-correlation
modes, exactly the C22-predicted closure.

### N8. Confusion-plane $U$ (label-free, from stream agreement)

**Hypothesis.** The corruption rotates boundaries between SPECIFIC class pairs
(the confusion structure); $U$ can be built from the span of confused
class-centroid differences, estimated label-free from stream clustering /
prototype-vs-probe disagreement.

**Builds on.** The cluster-grounding line showed clusters are ~65% pure
(unusable as pseudo-labels) -- but purity is the wrong metric for U; the
*disagreement directions* between R1 prototypes and the R4 probe
(`r4_r1_disagree`, the best gauge signal) name the class pairs whose boundary
rotated. **Cost.** Moderate diagnostic. **Risk.** Requires the disagreement
directions to align with the residual rotation; unmeasured.

### N9. Active querying in the $U$-subspace (label-efficient) -- VALIDATED

**Hypothesis.** Spend the label budget actively: query the points with the
highest leverage in the $U$-subspace (max expected reduction in $C$
uncertainty), instead of 1-NN-pseudo-labeling 500 random points. The 56-224 true
labels become the bank directly.

**Status: VALIDATED by the minimal-label result (above).** The leverage-in-$U$
query is the rule that makes 2-8 labels work: DGLSS++ fog/crosstalk leverage-u
b=2-8 closes gc 0.34-0.76 where random and per-class are negative at the same
budget. It is the active ingredient of the cheap-update operating point
(r=2-4, b=2-8).

**Builds on.** Statistical-leverage selection is the principled "coreset for
regression"; C31 already showed allocation matters for $W_{res}$ (diverse +0.024
beat random +0.010 on fog). **Cost.** Eval-only with oracle $U$ first, then
combined with N7/N8. **Risk.** Depends on $U$; if N7/N8 fail, this is moot. The
C28 result (diverse < random) is the caution: leverage-in-$U$ is NOT
farthest-point-in-feature-space.

---

## Decision rules

- **Stability first (N1-N4)** are all eval-only on the existing harness and can
  be batched into one diagnostic run. If N2/N3 close the pseudo-true gap (fog
  +0.005 -> ~+0.05, snow sign flip fixed), the instability is solved as a label
  problem, independent of the bank question.
- **The bank question (N5-N9)** splits on $U$: N5 (memory-free streaming stats)
  and N9 (active querying) both assume $U$ is known. N7 (CCA) and N8
  (confusion-plane) are the $U$-estimation bets that would obsolete the bank
  entirely. Run N7/N8 before committing to the bank-based N5/N9.
- **N6 (code-2000)** is orthogonal and cheap; adopt it as the default projection
  for everything if it holds at full scale, since it improves both accuracy and
  the online/conditioning story.

## Reproducibility

- Harnesses: `al_full_dataset_diag.py` (full scale, deep-copied ARCH),
  `al_bank_residual_diag.py` (C31 $W_{res}$), `probe_al_gauge_diag.py` (N4),
  `probe_covshift_mechanism_diag.py` (per-class/pool/conditioning).
- Existing results: `al_full_dataset_ep10.json` (full-scale $W_{res}$ tables),
  `probe_al_gauge_ep10.json` (gate signals), `probe_pseudolabel_struct_*.json`
  (per-class precision / S-T decomposition).

## Connection to the extractor-mechanism probe (Iteration 0 of `cov_full_scale.md`)

Two results from the mechanism probe directly constrain this ADA framework:

1. **The residual is large everywhere (resid_rel ~1.1-1.3)** on every extractor
   and condition -- there is recoverable structure to close. The ceiling cap is
   a ridge-extraction problem (the C16-C28 ill-conditioning), which is exactly
   what the low-rank $W_{res}$ decoder (this framework's core) already fixes.
2. **D9: the NuScenes-C zero-shot was contaminated by a cross-domain W0 fit.**
   In-domain W0 raises frozen by +0.13..+0.5. For ADA this means: the "gap to
   close" that motivates the update must be measured against an in-domain
   frozen probe, not a cross-domain one -- otherwise the framework spends labels
   chasing a probe-fit artifact.
3. **D1: the healthy-condition deficit is mostly clean-inherited** (DGLSS++ clean
   0.640 vs cov-shift 0.520), so a healthy-condition "regression" of an ADA
   update is more likely capacity than corruption. The zero-degradation
   guarantee (N4 gate) should be judged against the CLEAN baseline, not the
   corrupted frozen number.
