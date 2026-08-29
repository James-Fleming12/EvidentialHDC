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

## Iteration 0c: the clean->corrupted decision-conditioned U (pair-damage) -- the
counterfactual pairing FAILS too; U* is not inferable from the corrupted side by
any statistic, pairing included (2026-08-28, `al_pair_damage_diag.py`)

The last untested U source: the clean/corrupted PAIRING (KITTI-C is per-frame
corruptions of seq-08, same scan geometry and labels), conditioned on DECISION
DAMAGE. Per paired pixel: dx = code_corr - code_clean; dz = dx^T W0; damage =
(clean correct AND corrupted wrong); loss_gain = CE increase. U constructions:
U_cross (left singulars of sum dx dz^T, the counterfactual cross-moment),
U_dx_all (all-pixel displacement covariance), U_damage (failure-covariance),
U_damage_w (loss-gain-weighted). Both DGLSS++ and cov-shift, fog/crosstalk.

**Pairing sanity (the check that makes this trustworthy):**

| cond | scans | names match | mean range_corr | aligned |
| :--- | :--- | :--- | :--- | :--- |
| fog | 100 | 100/100 | 0.786 | TRUE |
| crosstalk | 100 | 100/100 | 0.684 | TRUE |

The clean and corrupted batches at each index are the SAME physical scan
(scan_names_aligned True), and the range channel at paired pixels correlates
0.68-0.79 -- the pixel-level pairing is sound. (damage_frac 0.63-0.65 is expected,
not a bug: W0 decodes clean scans well and corrupted scans terribly, so most pixels
are clean-right-and-corr-wrong.)

**Result -- the pairing does NOT recover U* either.**

| U | align (fog/crosstalk) | best gc r2 (fog/crosstalk) |
| :--- | :--- | :--- |
| oracle | 1.00 | +0.19 / +0.14 |
| tangent | 0.54 / 0.48 | +0.02 / +0.06 |
| U_cross | 0.01 / 0.02 | -0.29 / -0.01 |
| U_dx_all | 0.01 / 0.01 | -0.62 / -0.10 |
| U_damage | 0.01 / 0.01 | -0.60 / -0.38 |
| U_damage_w | 0.02 / 0.01 | -0.60 / -0.09 |

The paired, decision-conditioned displacements -- the STRONGEST counterfactual
information available (the exact clean->corrupted pairing restricted to decision
failures) -- align 0.01-0.02 with the oracle residual. This is decisive for the
U-predictor head: the head's test-time input is the CORRUPTED STREAM ALONE, and
the pairing is strictly MORE informative than the stream (it includes the clean
side). If even the pairing aligns 0.01 linearly, the head's nonlinear mapping has
little to learn from -- the head is a LOW-expected-value bet. This closes the last
U-estimation family.

**Verdict: the U-estimation head is deprioritized; the efficient bank and the
canonical adapter are the two live routes.** Every route to test-time U inference
is now measured-closed: corrupted-side statistics, tangent-b8 (0.5 intrinsic
ceiling), joint refinement, step-side fixes, U refinement, and the counterfactual
pairing. U* is not inferable at test time from 2-8 labels, from the corrupted
distribution, or from the clean/corrupted pairing. The remaining options are:
(a) make the extractor EXPOSE U offline (the canonical adapter), and
(b) accept oracle-quality U via the efficient bank. Both are training/supervision
routes, not test-time inference routes -- which is where the evidence says the
answer lives.

---

## Why we are moving with the efficient bank

After Iteration 0c, the paper's AL/TTA needs oracle-quality U, and every test-time
route to it is measured-closed. The efficient bank is the route that SOLVES the
problem rather than fighting it, because it supplies -- by construction -- the two
things that no cheap estimator could recover:

### 1. The bank is the only label source that yields oracle-quality U
Every U-estimation failure shared the same root cause: 2-8 labels (or the
corrupted distribution) cannot identify the residual subspace. The bank does not
try to INFER U -- it RE-CONSTRUCTS it. With k labels per class over a
representative sample, W_sub (fit on the bank) approaches W*, so
`U_bank = SVD(W_sub - W0)` approaches the oracle U. This is the C30/C31 result at
scale: oracle U closes ~100% of the closeable gap, and it is exactly the thing
every 2-8-label estimator failed to reach. The bank trades label count for U
quality -- and U quality is the one thing nothing else provided.

### 2. The bank simultaneously provides the pool-geometry (mass) that the
trust-region step proved unnecessary -- but that the ridge form still wants
The first-order diagnostic showed the covariance `U^T X^T X U` is an ARTIFACT of
the ridge estimator (normalized first-order is nearly as good with no inversion).
So the bank's second purpose -- providing the pool mass for the whitening -- is
now optional. But that is a FEATURE, not a cost: with oracle-quality U from the
bank, we can use the cheapest update form (normalized first-order), and the bank
only needs to supply the U direction, not the full covariance. This makes the
bank smaller and cheaper than the original C30/C31 formulation.

### 3. The bank is a supervision route, not an inference route -- which is where
the evidence says the answer lives
Iteration 0c's verdict is that U* is not in the corrupted side (nor the pairing)
at test time. The bank is a TRAINING/supervision object: a stored, representative
labeled set. It does not ask "what can we infer from the stream?" -- it asks
"what can a small but well-chosen labeled set tell us directly?" The 2-8-label
program was an inference problem that the data says is unsolvable; the bank is a
supervision problem that C30/C31 already proved solvable (oracle-U closes ~100% of
the gap at k=8/class).

### 4. The efficiency program makes the label cost acceptable
The original bank used 56 + 500 random points. The efficient-bank program reduces
this cost:
- **Representative points, not random**: select the bank points that maximize
  coverage of the residual subspace (leverage-in-U, gradient diversity) instead of
  500 random draws -- fewer points for the same U quality.
- **Only the necessary information**: the update needs the bank's per-class sums
  and U-projected covariance (streaming sufficient statistics, the N5 idea), not
  the stored points themselves -- constant memory, no explicit dataset at deploy.
- **Compression**: the 10000-d codes can be sketched/projected (code-2000 already
  peaks accuracy) or stored as packed bits (sign codes are 1 bit/dim), cutting the
  bank's storage ~5000x.

### 5. Why this solves what nothing else does -- in one line
The efficient bank is the only route that supplies **oracle-quality U** (the thing
every test-time estimator failed to reach) **and** can be made cheap (representative
selection, streaming statistics, compression). The canonical adapter is the
complementary training-time route (make the extractor expose U); the bank is the
supervision-time route (make a small labeled set provide U). Both are needed for
the paper: the canonical adapter for the extractor story, the efficient bank for
the AL story.

---

## The canonical-adapter training objective (the extractor-side complement)

The canonical adapter makes the DECODER RESIDUAL live in a FIXED, pre-learned
basis, so test-time adaptation never needs to infer U -- it only estimates C in a
known coordinate system.

**The supervision.** During extractor training we HAVE the oracle residual for
every corruption condition c: R_c = W*_c - W0 (the full-label pool-fit decoder
minus the clean decoder). These are exactly the objects we proved are not
inferable at test time.

**The objective.** Learn a shared, low-rank U0 (d x r) such that all training
residuals are well-approximated in its span:

$$ L_{\text{adapt}} = \sum_{c} \| (I - U_0 U_0^\top) R_c \|_F^2
   \;+\; \lambda_{\text{rec}} \sum_{c} \min_{C_c} \| R_c - U_0 C_c \|_F^2
   \;+\; \lambda_{W} \| W_0 \|^2, $$

i.e. U0 is the top-r shared basis of the pooled residuals (equivalently, the
leading singular vectors of the concatenated [R_1 | R_2 | ... | R_N], which is the
canonical-adapter factorization R_c ~ U0 C_c with C_c = U0^T R_c). The extractor is
then trained so that W0 (its own clean decoder) plus the canonical U0 captures the
corruption-induced decision change across ALL conditions.

**At deployment.** U0 is a learned constant -- FREE. Only C is estimated, from
2-4 labels:

$$ C = \rho \cdot \frac{U_0^\top G}{\|U_0^\top G\|}, \qquad G = U_0^\top X_{\text{lab}}^\top (Y - X_{\text{lab}} W_0), $$

using the normalized first-order step (which the diagnostic proved needs no
covariance). The trust-region form is robust to C being coarse, because U0 is
exact. TTA chooses WHICH adapter/severity to activate (the one regression the gate
diagnostic suggested TTA can still do -- classify the mechanism, not the gain).

**Why this is the right training-side bet.** It converts the unsolved TEST-TIME
inference problem (recover U* from the stream) into a SOLVED TRAINING-TIME
supervision problem (learn U0 from the residuals we already have). The evidence
for it: the residual is consistently low-rank (r=2-4 captures most), and the
first-order diagnostic proved the step needs no covariance -- so the ONLY hard part
left is U, and the canonical adapter removes U from the test-time problem entirely.

---

## The Iteration 1-3 roadmap: the problems each category must solve

The two routes (efficient bank, canonical adapter) both depend on the SAME open
questions. Rather than design implementations first, this roadmap isolates the
PROBLEM each category must solve -- the property we need to be true for the method
to work at all -- and tests that property directly. Each category will contain
several concrete iterations; the categories are the organizing ideas.

These are written as "what has to be true," not "how we think we can make it true."

### Iteration 1: do the residuals across conditions share a usable structure?

**The problem.** The canonical adapter assumes ONE low-rank basis U0 serves ALL
corruption conditions: R_c ~ U0 C_c for fog, crosstalk, snow, wet_ground, etc.
The efficient bank similarly assumes that what we learn from labeled points is
transferable across conditions. But we have only ever measured that each condition's
residual is individually low-rank (rank 4-5). We have NOT measured whether the
FOG residual and the CROSSTALK residual live in the SAME directions or in different
ones.

**What has to be true for the method to work.** There is a single low-dimensional
subspace that captures most of the recoverable residual on every condition. If the
conditions share a basis, one learned U0 (or one bank-learned U) serves them all and
the adaptation is a single structure. If each condition's residual is in a
different subspace, then a single adapter cannot work -- we would need
condition-specific structures, which reintroduces the "which one at test time?"
problem before we even get to the bank.

**Why this is the gating test.** If Iteration 1 fails (no shared basis), the
canonical adapter is structurally impossible and the efficient bank must be
re-thought as condition-specific. If it passes, the rest of the roadmap is well-
posed. This is the cheapest test (eval-only on the residuals we already compute)
and the highest-leverage decision.

**The test.** Measure, per condition, how much of R_c lies in the top-r subspace of
the POOLED residuals (concatenate [R_fog | R_crosstalk | R_snow | R_wet], take its
top-r, measure per-condition capture ||U0^T R_c||/||R_c||). Also measure pairwise
subspace agreement (cos of each condition's top-r against each other's). A single
number answers the question: does one low-rank structure explain all conditions, or
does each condition need its own?

### Iteration 2: how much supervision is actually needed to reach a working U?

**The problem.** Every cheap U estimator (2-8 labels, corrupted statistics, the
pairing) failed. The only thing that works is oracle-quality U -- which currently
requires fitting W_sub on a large labeled bank (C30/C31: 56+500 points). The
efficient-bank program claims we can get that U quality with FEWER, better-chosen
points. But we have never measured the curve: at what bank size (and selection
rule) does the bank-learned U become good enough that the trust-region step closes
real gap? We know the 500-point bank works; we do not know the floor.

**What has to be true for the method to work.** There is a bank size, well below the
current 56+500, at which the learned U is good enough to make the trust-region step
close most of the closeable gap (align to oracle > ~0.7, or equivalently gc reaching
most of the oracle-U gc). If the floor is ~500 points, the bank is not cheap and the
"efficient" claim fails; if it is ~50-100 points, the method is viable.

**Why this matters before implementation.** The bank's design (how many points, how
selected, what statistics to keep) is entirely determined by this floor. Building
compression/streaming machinery before knowing the floor risks optimizing the wrong
thing. This is the measurement that turns the bank from "a workaround we know works
at 500 points" into "a mechanism we know works at N points."

**The test.** Sweep bank size (56 + {20, 50, 100, 300, 500}) and selection rule
(random vs leverage-in-U vs gradient-diversity), fit W_sub on each, take U = SVD
(W_sub - W0), and measure (a) align to oracle U and (b) the trust-region step's gc
on fog/crosstalk. The curve answers: what is the cheapest supervision that reaches a
working U?

### Iteration 3: can the stream tell us which adaptation to use?

**The problem.** If Iteration 1 shows conditions need different structures (or even
if a single U0 exists but the stream is a mix), the method must know AT TEST TIME
which adaptation applies. The gate diagnostic showed that a label-free gauge can
say "this stream is corrupted" but NOT "this stream is worth adapting" (snow has
low conf_drop yet huge oracle headroom). We have never checked the weaker, more
useful claim: can the gauge tell WHICH corruption/adaptation is present (fog vs
crosstalk vs snow), even if it cannot predict the gain?

**What has to be true for the method to work.** The unlabeled stream's statistics
(confidence drop, mean shift, prototype-vs-probe disagreement, feature norm shift)
form a separable signature per condition -- enough to classify the active
corruption/adapter without labels. If the signatures overlap, TTA cannot choose the
mechanism and we must either use one universal adapter (needs Iteration 1 to pass)
or query for the mechanism (a label cost we were trying to avoid).

**Why this matters.** This is the difference between "TTA is a useful controller"
(chooses which adapter) and "TTA is decorative" (cannot choose). It determines
whether the deployable method has a free selection step or must spend labels on
mechanism identification.

**The test.** Compute the gauge vector (conf_drop, mean_shift_cos, r4_r1_disagree,
norm_ratio) per condition, and check whether the conditions form separable clusters
in gauge space (e.g. pairwise separability, or a simple classifier's accuracy at
distinguishing fog from crosstalk from snow from wet_ground label-free).

### How the three categories compose

- Iteration 1 decides whether the adaptation is ONE structure or MANY.
- Iteration 2 decides how cheaply we can obtain the structure's U (the bank floor).
- Iteration 3 decides whether the stream can tell us which structure to use.

The composition: if Iteration 1 passes (one shared structure) and Iteration 2 finds
a cheap floor, the method is "learn U0 once (canonical adapter) or from a small bank
(efficient bank), then 2-4 labels estimate C, TTA sets rho." If Iteration 1 fails
(many structures), Iterations 2-3 must be condition-specific, and the whole design
changes. So Iteration 1 is the first test to run, and the roadmap is deliberately
ordered so each category's answer re-frames the next.

## Iteration 1 result: the residuals do NOT share a usable structure -- the
shared-U0 assumption is REJECTED (2026-08-28, `al_shared_basis_diag.py`)

Per condition, measured the top-r subspace of each residual R_c, the top-r of the
POOLED residual [R_fog|R_crosstalk|R_snow|R_wet], and (a) how much of each
condition's own residual the pooled basis captures (ratio vs its own-top-r ceiling)
and (b) pairwise subspace agreement. Both DGLSS++ and cov-shift.

**The three convergent signals:**

1. **The pooled residual is NOT low-rank.** effective_rank(90%) of the pooled
   residual is 16-17 (out of 68 columns = 4 conds x 17 classes): it needs ~4
   directions per condition. Each condition is individually low-rank (own-top-r
   captures 0.92-0.99 at r=4), but the pooled one is effectively full-rank -- the
   conditions' directions do not collapse onto a common few.
2. **Pairwise subspace agreement is low everywhere.** fog~crosstalk cos
   0.37-0.53, and every other pair is 0.21-0.45 (both extractors, all r). No pair
   of conditions shares a strong common subspace.
3. **A clear "left-out" condition.** wet_ground's pooled-ratio is 0.46-0.50
   (dglsspp r=2/4) and 0.66-0.74 (covshift) -- a large fraction of its residual
   is NOT in the shared basis. Even the best pair (fog/crosstalk) leaves ~10-20%
   condition-specific.

**Verdict: the canonical adapter (one U0 for all conditions) is structurally
impossible.** The conditions' residuals live in largely DIFFERENT subspaces, so a
single shared low-rank basis cannot serve them. The canonical-adapter objective
(L_adapt with one U0) is not well-posed and is dropped as a shared-basis idea.

**What survives -- the efficient bank is condition-specific by construction.** The
bank learns U from the TARGET condition's OWN corrupted-pool labels (fit W_sub on
the target stream's labeled points, U = SVD(W_sub - W0)); it never assumed
cross-condition sharing. So this negative does NOT affect the bank route -- in
fact the data now says condition-specific U is exactly what is needed, which is
what the bank provides by design. The bank's U is the target condition's own
residual basis, which is the correct object.

**What changes in the roadmap.** Iteration 1's answer re-frames Iterations 2-3:
- Iteration 2 (the bank floor) is now the PRIMARY route and is well-posed
  unchanged: how much of the target condition's own labeled supervision reaches a
  working U.
- Iteration 3 (can the stream identify the condition) is now LESS critical for the
  bank route, because the bank's labels come from the target stream itself -- there
  is no cross-condition adapter to select. TTA still sets rho (trust), but no longer
  needs to classify WHICH corruption is present.
- The canonical adapter is reduced to a per-condition idea at most (train the
  extractor so ONE CONDITION's residual is more structured), which is a much weaker
  claim and not needed for the bank.

**The efficient bank is now the single live route.** It is the one method that
(a) supplies oracle-quality U by construction (C30/C31), (b) is inherently
condition-specific (which the data says is required), and (c) whose cost can be
reduced by the efficiency program (representative selection, streaming sufficient
statistics, compression). Iteration 2 measures the floor for (c).
