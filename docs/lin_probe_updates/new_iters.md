# New Iterations: Updates to the Linear Classifier (beyond the residual subspace)

This doc records the SEARCH for an update to the linear-probe classifier on the
HDC code that (a) improves fog/crosstalk from the frozen zero-shot, (b) uses a
small label budget, and (c) does not degrade the healthy conditions. It is the
consolidation of what the residual-subspace framing (in `residual_subspace/`)
established -- and the reframing that the closure of that line now forces.

The organizing principle of THIS doc: state what the update must DO (the required
properties), then what was TRIED and WHY each failed. The residual-subspace idea
(U = where to move, C = how far) is now measured-closed; this doc records the
properties any surviving update must satisfy, independent of that formulation.

---

## What we need: the required properties of the classifier update

The update takes the frozen linear probe W0 (fit on clean data, the zero-shot
decoder) and produces W1 on the corrupted stream. It must satisfy:

**P1. Improve the corrupted conditions (fog/crosstalk) from frozen.**
The labeled ceiling (pool-refit oracle W*) is well above frozen on fog/crosstalk
(KITTI-C 3-sev: dglsspp fog 22.5 -> 35.2, crosstalk 11.9 -> 29.4; cov-shift fog
40.7 -> 43.6). The update must move the decoder toward that recoverable headroom.

**P2. Use a small label budget.**
The goal is a few labels (2-56), not the full labeled pool. The labeled ceiling is
reachable with the full pool; the entire search is about how much of it a small
budget can recover.

**P3. Not degrade the healthy conditions (snow/wet_ground/etc.).**
The frozen decoder already works there; the update must be a no-op (or a
zero-degradation guarantee) where there is no closeable gap. This is the
safety-by-construction requirement.

**P4. Be online / constant-memory (preferred, not required).**
The deployment ideal is a streaming, per-scan update with no stored dataset.

**P5. Be efficient.**
The update must be a cheap solve (low-rank / first-order), not a full 10000x10000
ridge refit per scan.

**The framing that has been dropped.** The residual-subspace formulation wrote
W1 = W0 + U C and separated "where to move" (U, the top-r of W* - W0) from "how
far" (C). The measured closure of that line (below) is that U is not obtainable
from any limited-supervision route. THIS doc therefore frames the update directly
by its required properties (P1-P5), and evaluates every candidate update against
them -- NOT against whether it recovers some hypothetical U.

---

## What was tried and why each failed (the measured closure)

Each entry states the route, the property it was meant to satisfy, and the measured
reason it failed. Sources are the docs in `residual_subspace/`.

### R1. Label-free pseudo-label TTA (P1)
**What.** Refit the probe on the corrupted pool using the frozen decoder's
pseudo-labels, gated/weighted/soft/two-stage.
**Why it failed.** The wrong pseudo-labels ANTI-align with the oracle rotation
(cos(W_wrong, W_oracle) < 0 on every condition) and contribute ~equal magnitude to
the update. Confirmation bias is structural: the frozen probe's own mistakes are
selected by every confidence gate, and even a PERFECT-purity T cannot reproduce
the oracle (the gap is label COVERAGE, not label noise). The probe's label-free
ceiling is the frozen decoder. (Iterations 9-12, tta_iterations.md; the closure
transfers to hyper/geoid/cov-shift, Iteration 13.)

### R2. Full-probe ridge update (P1, P5)
**What.** W = (X^T X + lI)^-1 X^T Y on a small labeled set.
**Why it failed.** The inverse covariance amplifies the (unavoidable) label-error
residual along the low-variance directions: a T with t_cos 0.76-0.86 (objectively
good) maps to W with w_cos ~0.05-0.15 and mIoU below frozen. The ill-conditioning
(C16-C28) collapses even with TRUE labels. The full 10000-d solve is also
inefficient. (active_iterations.md Iterations 6-8; the sensitivity-bounded
fractional update, Iteration 9-10, is the only ridge-form that survived, at
64-72 labels.)

### R3. U estimation -- where to move (P1, P2)
**What.** Every construction of "which directions the decoder should move":
   - unlabeled statistics: pool covariance, class-mean shift, CCA, near-boundary
     PCA, boundary outer product, margin-weighted covariance, confused-pair PCA,
     weak-classifier ensemble -- all align 0.00-0.05 with the oracle residual.
   - few-label tangent-b8 (PCA across tiny provisional updates): the ONLY
     non-oracle U that works, but capped at align 0.3-0.5 (intrinsic ceiling; more
     windows/averaging/sharpening add zero).
   - the clean/corrupted pairing (decision-conditioned): aligns 0.01-0.02.
**Why it failed.** The residual R = W* - W0 is a DECISION-RULE object, not a
distribution-shift object: it is only visible by re-fitting the boundary. It is not
in any statistic of the corrupted distribution, not in the pairing, and not
recoverable from 2-8 labels. The trust-region step (R5) is direction-sensitive:
a 0.5-aligned U gives gc ~0, so "good enough" is not attainable. (al_uest_diag,
al_uest_bdry_diag, al_uest_joint_diag, al_trust_iter0, al_pair_damage_diag,
al_trust_refine.)

### R4. The bank as a U-source (P1, P2, P4)
**What.** Fit W_sub on a labeled bank, U = SVD(W_sub - W0); the efficient-bank
program claimed fewer, better-chosen points reach the same U.
**Why it failed.** No bank size (28-556) or selection (random, per-class,
margin-frozen, OR leverage-oracle which uses oracle U) reaches align > 0.55 or a
working gc (max gc ~0.04 vs oracle-U's +0.15-0.30). The curve is FLAT in N. The
reconciliation: C30/C31's oracle U came from the FULL-pool fit; the bank was only
ever a C-source given oracle U, never a U-source. (al_bank_floor_diag.)

### R5. The trust-region / first-order step (P2, P3, P5) -- the one partial success
**What.** Replace the ridge Newton step with a normalized first-order step
W1 = W0 + rho * U * G/||G||, G = U^T X^T (Y - X W0), with rho from a TTA gate.
**What works (measured).** The covariance U^T X^T X U is an ARTIFACT of the ridge
estimator: normalized first-order (no inversion, no bank) closes most of the
ridge's gain (dglsspp fog b8: ridge +0.24 vs norm +0.13). The TTA gate gives
zero-degradation by construction (rho -> 0 on healthy streams). This is the
strongest update form found.
**Why it is not deployable yet.** It requires ORACLE U (the trust-region step is
direction-sensitive; tangent-U gives gc ~0). Without a route to oracle-quality U
from limited supervision, the first-order step is sound but U-unsupplied.
(al_first_order_diag, al_trust_iter0.)

### R6. The canonical adapter / shared basis (P1, P2)
**What.** Train the extractor so the residual lives in a fixed, learned basis U0
(R_c ~ U0 C_c for all conditions), making U free at deployment.
**Why it failed.** The residuals across conditions do NOT share a usable basis:
pooled effective rank 16-17 (needs ~4 directions per condition), pairwise subspace
cos 0.21-0.53, and wet_ground is clearly "left out" (pooled-ratio 0.46-0.50). A
single shared U0 is structurally impossible. (al_shared_basis_diag.)

---

## The synthesis: what the failures collectively establish

1. **The label-free path is closed (R1).** The probe's label-free ceiling is the
   frozen decoder; no pseudo-label gate, weight, or two-stage scheme recovers the
   oracle rotation.
2. **The full ridge is the wrong estimator (R2, R5).** Its covariance amplifies
   unavoidable label error; a normalized first-order step does the same job without
   the inversion. The update must be first-order/trust-region, not Newton.
3. **"Where to move" (U) is not obtainable from limited supervision (R3, R4, R6).**
   Not from the corrupted distribution, not from the pairing, not from a bank, and
   not from training a shared basis. The residual is a decision-rule object visible
   only to the full labeled fit.
4. **The only sound update form is the trust-region step (R5), and its sole missing
   ingredient is oracle-quality U.**

**The consequence.** A small-supervision update to the linear classifier must
either (a) find U outside the supervised-residual route, or (b) accept that U
requires the full labeled fit. The candidate directions that survive:
   - the trust-region step (R5) is the mechanism; the open problem is its U.
   - a per-condition training-time canonicalization (make the extractor expose a
     GIVEN condition's residual structure, since the shared basis failed) -- the
     last training-side bet.
   - accepting the full-pool-label cost (the labeled ceiling as-is, not cheap AL).

This doc will track the iterations that pursue these survivors against the
required properties P1-P5, without re-entering the closed residual-subspace route.

---

## The next question (the shift)

The residual-subspace closure (R3, R4, R6) is unusually consistent: U is not
obtainable from 2-8 labels, tangent-U (0.5 ceiling), refinement, the pairing, or a
556-point bank. This shifts the question from

> "How do I estimate the oracle update direction?"

to

> "How can active learning choose labels that are maximally useful to a classifier
> without needing to reconstruct the oracle residual?"

The reasons this is the right shift, from the measured facts:
- The residual is decision-rule structured and condition-specific (R3, Iteration 1),
  so a GLOBAL W* - W0 is the wrong object to demand from 2-8 labels. But the
  recoverable error is concentrated in a few class-pair boundaries -- which labels
  could identify directly.
- The trust-region step (R5) is sound under oracle U; its only missing piece is a
  usable direction. If active learning finds points that let a LOCAL correction
  (class-pair margin, class bias, prototype) repair the demonstrated errors, we may
  never need a global U.
- TTA's validated role is instability detection, not gain prediction (A2). Using it
  as an ACQUISITION function (where is the classifier unstable) is a much weaker,
  more plausible claim than using it as an oracle.

---

## Potential next steps (the candidate directions)

Organized by the tier the evidence suggests. Each entry states what it is, why it
could work (the measured fact it builds on), which property it targets, and the
key risk.

### Tier 1 -- decision-boundary AL + local corrections (the primary bet)

| # | Direction | What it is | Why it could work | Property | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A1 | **Class-pair boundary sampling** | Query by min top-2 logit margin, with a SEPARATE budget per (a,b) class pair | The residual is decision-rule structured; ~5 pairs may carry ~80% of recoverable error (R3). 8 pair-targeted labels > 8 global labels | P1, P2 | Needs the top-2 class competition to be stable; if error is not pair-concentrated this caps out |
| A2 | **TTA-instability x boundary acquisition** | A(x) = Var[p(y|aug_k(x))] + alpha/|margin| + beta*disagreement(proto,probe) | TTA's validated strength is instability detection (A2); combine with boundary proximity = query the unstable boundary, not the uncertain bulk | P1, P2 | If instability and recoverable error decouple (as conf_drop did for gain), the acquisition may select wrong-but-stable points |
| A3 | **Sequential AL (labels reveal the next query)** | Query 1 -> see the error pair -> focus next query on that boundary -> different region -> next pair | Avoids inferring the whole structure before enough labels; each label sharpens the next (the adaptive loop) | P1, P2 | Slow (sequential), needs the confusion structure to be discoverable in a few steps |
| A4 | **Local class-pair separator updates** | Only update (w_a - w_b) for the implicated pairs, not the full W | A label saying (y=a, pred=b) identifies the relevant margin directly; never asks 8 labels to explain a 17-class residual | P1, P2 | If the needed correction is not a pair-margin (e.g. a shared shift), local updates miss it |
| A5 | **Error-correction AL loop** | 8 labels -> identify recurring (pred,true) errors -> accumulate confusion -> query the uncertain points of those pairs -> repair | Directly targets the demonstrated failures; matches the "decision-rule object" finding | P1, P2 | Same pair-concentration assumption as A1 |

### Tier 2 -- conservative / cheap corrections

| # | Direction | What it is | Why it could work | Property | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| B1 | **Margin / entropy / uncertainty + diversity** | Cluster the top-M uncertain points, query cluster reps (or farthest-point) | Avoids re-querying near-identical points; uncertainty+diversity is strictly stronger than uncertainty alone | P1, P2 | Diversity in feature space may not span the decision-rule structure |
| B2 | **Expected gradient length (local)** | EGL(x) = sum_y p(y|x) ||g(x,y)||, restricted to the top-2 pair | The raw gradient failed as an UPDATE (R5), but is valid as an ACQUISITION score ("how much could this label change the classifier") | P2 | EGL needs the softmax to be meaningful under corruption (weak) |
| B3 | **Class-bias / logit calibration update** | z' = a_c z_c + b_c (per-class temperature/bias) instead of W' = W + dW | A few labels estimate 2C scalars far more easily than a 10000-d residual; zero-cost baseline that may fix confidence/ranking distortion | P1, P3, P5 | Does not fix geometric boundary rotations; may only move confidence, not decisions |
| B4 | **Local prototype correction** | mu_c -> mu_c + (1/|L_c|) sum_{x_i in L_c}(x_i - mu_c), only for classes with evidence + high TTA instability | Very conservative; the earlier prototype failure was a different mechanism, and active prototype correction is untested | P1, P3 | The prototype decoder was below the probe before (R1), so the ceiling is low |

### Tier 3 -- methods to revisit only if Tier 1-2 fail

| # | Direction | What it is | Why it could work | Risk |
| :--- | :--- | :--- | :--- | :--- |
| C1 | More clever U estimators | CCA on class pairs, per-condition basis, etc. | The residual is condition-specific (Iteration 1), so per-condition estimators are the only U-route left | Closed 3x already (R3, R4, R6); low prior |
| C2 | More bank-selection schemes | Conformal scores, entropy-balancing, domain-aware querying | Rigorous selection MIGHT beat leverage/random (which failed) | The bank U-source is closed (R4); selection cannot fix a missing signal |
| C3 | More sophisticated trust-region gating | Richer TTA consistency for rollback | R5's gate is the sound part; better rollback could make aggressive updates safe | A3 showed no score separates good/bad updates; consistency-rollback is a weaker (unproven) claim |
| C4 | Online TTA + micro-updates with rollback | W_{t+1} = W_t + eta_t * Delta_t, keep W_best only if prediction-consistency improves | The only online form; conservative against catastrophic updates | Needs the consistency score to be reliable (A3 caution) |

---

## The relevant literature (mapped to the specific problems)

| Paper | Idea | Maps to | Why it helps |
| :--- | :--- | :--- | :--- |
| Gui, Li & Ji (2024), **ATTA** | Entropy-balanced active selection during streaming TTA | A2, A3 (acquisition + sequential) | A mathematically grounded selection rule that avoids biased low-entropy sampling -- could replace the failing leverage/random heuristics for picking bank points |
| Chen, Luo, Ma & Zhang (2020), **hidden shifting domains** | Domain-aware active sampling, query cost vs regret, scales with the target domain's dim | Iteration-1 result (condition-specific residuals) + A3 | Determines WHEN the stream shifted to a new corruption and queries the labels to build THAT condition's structure -- matches the "no shared basis" finding |
| Wang et al. (2020), **Tent** | Fully test-time entropy minimization on affine params | R1 counterpoint (TTA baseline) | An unsupervised baseline to test whether stream statistics alone can guide linear-classifier adjustment without full-pool labels |
| Shi et al. (2025), **CPATTA** | Conformal scores + top-K certainty for selective querying | C2 (bank selection) | A rigorous, annotation-efficient selection rule -- but only helps if the bank's signal exists (R4 says it does not) |
| Chen et al. (2024), **ARC** | Dynamic classification-layer tuning at test time using output confidence | B3 (logit calibration), C4 | Alternative to the first-order step for continuously shifting corruptions |
| Jang et al. (2022), **NN-prototype TTA** | Prototype/nearest-neighbor consensus for safer pseudo-labels | R1 closure mitigation | NN-prototype pseudo-labels may be less anti-aligned than raw confidence -- worth one gate-level test |
| Yuan et al. (2023), **ATASeg** | Few-click active TTA for dense segmentation | A1-A5 (few-label AL) | Empirical blueprint that a few clicks can bridge heuristic-TTA to supervised -- the strongest external support for the Tier-1 bet |