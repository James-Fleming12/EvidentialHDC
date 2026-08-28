# Active Learning with Trust-Region Updates (Iterations, part 3)

The deployable few-label adaptation method: **TTA controls WHEN to adapt, 2-8
labels control WHERE to move, and a normalized first-order step controls HOW FAR**,
with zero-degradation by construction. This doc tracks the iterations that turn the
first-order/trust-region result (active_iterations_2.md) into the working AL/TTA
mechanism.

Companion docs: `active_iterations_2.md` (the U-estimation closure and the
first-order result this builds on), `tta_iterations.md` (the label-free closure),
`docs/cov_shift/dglss_imp.md` (DGLSS++ extractor improvement).

---

## Background: why the AL thread kept failing, and what finally worked

### The failure catalogue (measured, active_iterations_2.md)

Every decoder-side route to the recoverable headroom closed for a specific reason:

1. **Naive pseudo-label TTA**: never beats frozen (poisoned pseudo-labels, and wrong
   labels ANTI-align with the oracle rotation even under oracle U).
2. **The full-probe ridge update** $W = (S+\lambda I)^{-1}T$: collapses (the inverse
   covariance amplifies label error along low-variance directions, C16-C28).
3. **U estimation (where to move)**: EVERY unlabeled statistic fails -- pool
   covariance, class-mean shift, CCA, boundary PCA, ensemble, joint U,C refinement.
   The residual $R = W^* - W_0$ is a DECISION-RULE object, not a distribution-shift
   object. Only two things recover it: oracle U (full labels) and the few-label
   **tangent-b8** construction (PCA across tiny provisional updates, align 0.3-0.5).
4. **The bank (500 random pseudo-labeled points)**: a workaround for U estimation
   and for the pool-covariance geometry, but not online / not constant-memory.

The two "hard requirements" that seemed to block a cheap AL method were:

$$ \underbrace{U}_{\text{where to move}} \qquad\text{and}\qquad \underbrace{U^\top X^\top XU}_{\text{how much mass/geometry}}. $$

### The escape hatch (the first-order / trust-region diagnostic, active_iterations_2.md)

The decisive reframing: stop treating $C$ as something the RIDGE formula must
estimate. The missing-mass problem applies to estimating the least-squares OPTIMUM;
it does NOT apply to estimating the GRADIENT at $W_0$. So:

- **Don't estimate the covariance** $U^\top X^\top XU$ at all. Replace the Newton step
  $C = (U^\top X^\top XU + \gamma I)^{-1}G$ with a **normalized first-order step**
  $C = \rho \cdot G / \|G\|$ where $G = U^\top X_{\text{lab}}^\top(Y - X_{\text{lab}}W_0)$.
- **Result (measured)**: the covariance WAS the artifact. The normalized first-order
  update closes most of the ridge's gain (DGLSS++ fog b8: ridge gc +0.24 vs norm
  +0.13) with no inversion, no bank. The raw $s \cdot G$ overshoots ($\|G\| \sim 34$);
  normalization fixes the SCALE, which is the only thing that was wrong.
- **TTA controls the trust radius**: $\rho = \text{step} \cdot g(\text{instability})$.
  The gate makes the update INERT on healthy streams ($\rho \to 0$ where the frozen
  decoder is already fine) and active where real corruption exists. This is the first
  few-label mechanism that is safe-by-construction.

### The open pieces this iteration series closes

1. **U is still needed** (gradspan and full_grad fail; label gradients do not span
   the residual). But U only needs to be a GOOD DIRECTION (0.3-0.5 alignment), not an
   exact basis -- the trust-region step is robust to a coarse U, and tangent-b8 is the
   few-label source.
2. **The TTA gate** used only `conf_drop` (a simple monotone of the probe's
   confidence gap). Deployment needs the full validated gauge set (`mean_shift_cos`,
   `r4_r1_disagree`, augmentation/temporal instability) calibrated to a trust radius.
3. **The step size** was a manual sweep in the diagnostic; the deployable method needs
   an automatic $\rho$ from the gauge, plus an accept/reject step.

---

## The method: trust-region adaptation with a TTA gate

The general idea, stated as the deployed pipeline:

1. **TTA (label-free) -- decides whether/when to adapt.**
   Compute a label-free instability gauge on the stream (confidence drop,
   prototype-vs-probe disagreement, augmentation/temporal instability). Map it to a
   trust radius $\rho \in [0, \rho_{\max}]$: healthy/stable stream -> $\rho \approx 0$;
   clearly corrupted -> large $\rho$; uncertain -> intermediate.

2. **2-8 labels -- decide WHERE to move (the direction).**
   Query a few points (leverage-in-U or tangent-b8 provisional fits). Compute
   $G = U^\top X_{\text{lab}}^\top (Y - X_{\text{lab}} W_0)$ in a low-rank basis $U$
   (oracle at eval, tangent-b8 at deployment). This is the gradient of the labeled
   loss at $W_0$ -- a direction estimate that does NOT need the pool mass.

3. **Normalized first-order step -- decide HOW FAR.**
   $W_1 = W_0 + \rho \cdot U \, \frac{G}{\|G\|}$. No covariance, no inversion, no bank.

4. **Zero-degradation by construction.**
   $\rho = 0$ reproduces $W_0$ (the null option), and the gate drives $\rho \to 0$ on
   healthy streams. Optionally accept/reject: keep $W_1$ only if the TTA validation
   score improves; otherwise stay at $W_0$.

The method is the realization of:
$$ \boxed{\text{TTA finds where/when} + \text{2-8 labels the direction} + \text{small first-order step the magnitude}} $$

---

## Iteration 1: the deployable trust-region pipeline with tangent-U and the full TTA gate

**Goal.** Take the trust-region result from the diagnostic and make it deployable:
(1) use the TANGENT-b8 U (the few-label U) instead of oracle U -- does the
trust-region step survive the coarse basis? (2) replace the manual step sweep with a
gauge-driven $\rho$ (conf_drop + mean_shift_cos + r4_r1_disagree); (3) add the
accept/reject step (keep $W_1$ only if the TTA validation improves). Measure per
condition: delta/gap-closed on the corrupted conditions, AND the zero-degradation on
the healthy ones, for BOTH oracle-U and tangent-U, with the gate either on or off.

**Hypotheses.**
- H1 (coarse-U robustness): the normalized trust-region step with tangent-b8 U
  retains most of the oracle-U gain (the step is robust to a 0.3-0.5-aligned basis).
- H2 (gate): the gauge-driven $\rho$ is positive where the frozen probe fails and
  ~0 where it is fine, so the healthy conditions stay at $W_0$.
- H3 (accept/reject): the TTA validation score separates good from bad updates, so
  the reject rule keeps the gain and removes the negatives.

**Verdict rule.** If H1 and H2 both hold (tangent-U + gate closes real gap on
fog/crosstalk and is ~0 on snow/wet_ground), the trust-region AL is deployable.
If H1 fails (tangent-U collapses), the U basis is the remaining bottleneck and the
next iteration targets U refinement.

## Iteration 0: assumption validation -- the coarse tangent-U FAILS in the
trust-region step (2026-08-28, `al_trust_iter0_diag.py`)

Before building the deployable pipeline, validate the three assumptions the method
silently depends on (A1 coarse-U robustness, A2 gate validity, A3 accept/reject),
on both DGLSS++ and cov-shift, all 4 conditions, r=2, b=8, rho sweep
{0.01..0.8}. The trust-region step is `W1 = W0 + rho * U * G/||G||` with
G = U^T X^T (Y - X W0), U in {oracle, tangent-b8, random}.

**A1 -- COARSE-U ROBUSTNESS FAILS: the tangent-b8 U does NOT work in the
trust-region step, despite align 0.3-0.4.**

gc-vs-rho (best gc over the sweep):

| cond | extractor | oracle | tangent | random |
| :--- | :--- | :--- | :--- | :--- |
| fog | dglsspp | +0.18 | **+0.02** | -0.02 |
| fog | covshift | +0.36 | **-0.11** | -0.16 |
| crosstalk | dglsspp | +0.14 | **+0.06** | -0.07 |
| crosstalk | covshift | +0.30 | **-0.57** | -2.11 |
| snow | dglsspp | +0.67 | **-0.07** | -0.12 |
| snow | covshift | +0.37 | **-0.84** | -0.54 |
| wet_ground | dglsspp | +0.41 | **-0.24** | -0.06 |
| wet_ground | covshift | +0.41 | **-0.14** | -0.06 |

The tangent-U (align 0.32-0.39, the ONLY few-label U) produces gc ~0 on
fog/crosstalk and NEGATIVE on snow/wet_ground -- essentially operating at the
random-U level, far below the oracle-U trust-region (fog +0.18 vs tangent +0.02,
covshift fog +0.36 vs -0.11). With r=2, a 0.3-0.4 subspace cosine means the top-2
step directions are only partially aligned, so normalization splits the step
between correct and incorrect directions and most of it goes wrong. **The
trust-region step is NOT robust to a coarse U: it needs oracle-quality U.**
Hypothesis H1 is REJECTED.

**A2 -- GATE VALIDITY IS WEAK/INCONCLUSIVE (n=4).** The gauge rank-correlations
with oracle-best-gc across the 4 conditions are small and sign-inconsistent:
conf_drop -0.80 (dglsspp) vs +0.40 (covshift); mean_shift_cos +0.20 vs -0.40;
r4_r1_disagree -0.80 vs 0.0. The reason is visible in the data: snow has LOW
conf_drop (0.006) yet HIGH oracle-U gain (+0.67) -- the oracle headroom and the
corruption severity are DIFFERENT things. A gauge that ranks "how corrupted is
this stream" does not rank "how much can a correct step recover". H2 is NOT
established.

**A3 -- ACCEPT/REJECT FAILS: the label-free scores do not separate sign(gc).**
The d_conf / d_disagree / comb scores show no consistent relationship to gc. E.g.
dglsspp fog tangent gc +0.02 has comb +0.054, but snow tangent gc -0.09 has comb
+0.026 -- both positive comb despite opposite gc signs. covshift fog tangent gc
-0.11 has comb -0.136, snow tangent gc -0.84 has comb -0.097 -- both negative comb
for negative gc. There is no monotone score that identifies "this update is good".
H3 is REJECTED.

**Verdict.** The deployable trust-region pipeline (Iteration 1 as planned) is NOT
supported. The decisive blocker is A1: the trust-region step is direction-sensitive
and the tangent-b8 U (align 0.3-0.4) is too coarse -- its gc is at the random-U
level. This is consistent with the earlier finding that only oracle-quality U
closes real gap, and it means the "coarse U is enough" hope from the first-order
result does not transfer to the trust-region step. The gate (A2) and accept/reject
(A3) are also not validated, but they are secondary: even a perfect gate and
reject cannot help if the step direction is wrong.

**What this redirects.** The blocking assumption is U quality, again. The next
iteration should either (a) make the trust-region step robust to a coarse U (e.g.
use the tangent-U only to SELECT the step's active coordinates but take the
direction from the labels' own gradient, or combine multiple tangent draws to
average the U), or (b) target U refinement directly (more provisional windows /
iterative U sharpening) since the tangent-b8 U is the bottleneck. Option (a) is
preferred: the trust-region form is otherwise sound (oracle-U gc is large and
monotone in rho on the corrupted conditions), so making the STEP robust to the
coarse basis is a smaller change than re-solving U estimation.

## Iteration 0b: the refinement sweep -- neither step-side fixes nor U-refinement
help; the coarse U is fundamentally insufficient for the trust-region step
(2026-08-28, `al_trust_refine_diag.py`)

Sweeps the two redirect families with explicit efficiency costs (units R =
provisional fit, SVD = svd of rows x 10000, G = label gradient; DEC = val decode
dominates all). Step-side fixes: `A_grad` (full label gradient, no U),
`A_fix` (label gradient projected onto the tangent span), `A_hybrid` (projected +
orthogonal residual). U-refinement: `U_avg` (average 8 tangent draws, 544-row
stack), `U_windows` (16 windows, 136-row stack), `U_sharpen` (3 iterative rounds).
Both DGLSS++ and cov-shift, all 4 conds, r in {2,4}.

**The result is uniformly negative -- every fix fails, and the reason is visible
in the align data.**

**Finding 1 -- U-refinement does NOT improve U: align is flat at the tangent
level.** All three refinement variants land at the SAME align as the baseline
tangent (dglsspp fog: tangent/U_avg/U_windows/U_sharpen = 0.53/0.53/0.55/0.57;
covshift: 0.54 everywhere). More provisional windows, averaged draws, and
iterative leverage-re-selection all add ZERO alignment. The 0.5 align is not a
sampling artifact (more samples don't fix it) -- it is the intrinsic ceiling of
the tangent construction on this problem. Their gc is consequently unchanged
(~0.00-0.06, same as tangent).

**Finding 2 -- the step-side fixes also fail.** `A_grad` (the labels' own full
gradient, no U) is strongly NEGATIVE everywhere (dglsspp fog -0.15, covshift
crosstalk -4.51, snow -3.61): the raw label gradient direction on 8 points is
not a usable update. `A_fix` (project onto tangent span) is ~0, no better than
tangent. `A_hybrid` is ~0, with two weak positives (dglsspp wet_ground r4 +0.14,
covshift wet_ground r2 +0.19 -- inside the noise band).

**Finding 3 -- the oracle gap is not shrinking with any cheap fix.** Oracle-U
gc is 10-30x the best fix on every condition (dglsspp fog oracle +0.18/0.25 vs
best fix +0.02; covshift fog +0.36 vs +0.01). No combination of a few labels,
the tangent U, and the label gradient reaches the oracle.

**Efficiency (the ledger, same on both extractors):**

| method | R | SVD | G | SVD rows |
| :--- | :--- | :--- | :--- | :--- |
| oracle / random | 0 | 0/1 | 0 | -- |
| tangent (baseline) | 4 | 1 | 0 | 68 |
| A_grad | 0 | 0 | 1 | -- |
| A_fix | 4 | 1 | 1 | 68 |
| A_hybrid | 4 | 1 | 2 | 68 |
| U_avg | 32 | 1 | 0 | 544 |
| U_windows | 16 | 1 | 0 | 136 |
| U_sharpen | 16 | 4 | 0 | 68-272 |

Efficiency is NOT the issue: even the most expensive refinement (U_sharpen: 16 R +
4 SVD) is a tiny fraction of one val decode, and it does not help anyway. The
blocker is purely U quality, and it is not fixable by more provisional sampling.

**Verdict.** The trust-region step needs oracle-quality U; every cheap route to U
(step-side fixes, U averaging, more windows, iterative sharpening) is closed with
a measured negative. The align ceiling of 0.5 (tangent) is intrinsic, and a 0.5-
aligned basis is insufficient for the trust-region step regardless of the update
form. This closes the Iteration-0 redirect options. The remaining live leads are:
(a) the U-predictor head (dglss_imp.md D4: an auxiliary head supervised by the
clean/corrupted pairing, the only construction that does NOT depend on a few
labels discovering U), and (b) accepting oracle-quality U requires the larger-label
C30/C31 bank setting. The trust-region form itself is validated (oracle-U gc is
large and monotone); what it cannot tolerate is a coarse basis.
