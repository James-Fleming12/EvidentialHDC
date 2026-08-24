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

### N7. CCA between clean and corrupted code distributions ($U$ estimation)

**Hypothesis.** The residual $W^* - W_0$ spans the directions where the clean and
corrupted class-conditional code structure disagrees. Canonical correlation
analysis on paired clean/corrupted scans (seq-08 clean vs each KITTI-C variant)
gives those directions directly -- and unlike everything tried (pool-cov SVD,
$S^{-1}T$, $U_{shift}$), it uses BOTH sides of the shift, which we uniquely have.

**Builds on.** C21/C22/C25/C26 all used unlabeled statistics of the corrupted
pool alone; none used the clean/corrupted pair. The clean/corrupted pairing is
exact for KITTI-C (per-frame corruptions of seq-08). **Cost.** New diagnostic
(PCA/CCA on clean-vs-corrupted per-class means or code covariance); moderate.
**Risk.** If the residual is in directions that are NOT canonical-correlation
modes (the C22 negative result hints the residual is in low-variance directions
that may not be the top-CCA modes either), this closes quickly -- but it is the
first U idea that uses the pairing, so it is the highest-value untried bet.

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

### N9. Active querying in the $U$-subspace (label-efficient)

**Hypothesis.** Spend the label budget actively: query the points with the
highest leverage in the $U$-subspace (max expected reduction in $C$
uncertainty), instead of 1-NN-pseudo-labeling 500 random points. The 56-224 true
labels become the bank directly.

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
