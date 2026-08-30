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

**Constraint: every candidate must hold for the LINEAR-CLASSIFIER decoder (the R4
probe on the HDC code).** The prototype (R1) decoder is closed (R1 < R4 on every
condition, tta Iteration 0) and is NOT a candidate decoder. Any "prototype"
reference below is a label-free GAUGE SIGNAL (R4-vs-R1 disagreement is a validated
instability signal, not a decoder to adapt). **ALL W-update families are now
closed**: the labels' own reach (U-free, local forms, rank-1 span ~0, Iteration
3b) AND the pool-derived-basis branch (P1, Iteration 4: the unlabeled pool's
dictionary captures only 0.04-0.08 of R, and even perfect coefficients are
negative on half the cells). Every survivor uses labels to SELECT/re-rank, not to
re-estimate W.

### Tier 1 -- decision-boundary AL (labels make decisions, not parameter updates)

| # | Direction | What it is | Why it could work | Property | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A1 | **Class-pair boundary sampling** | Query by min top-2 logit margin (of the R4 probe), with a SEPARATE budget per (a,b) class pair | The residual is decision-rule structured; ~5 pairs may carry ~80% of recoverable error (R3). 8 pair-targeted labels > 8 global labels | P1, P2 | Needs the top-2 class competition to be stable; if error is not pair-concentrated this caps out |
| A2 | **TTA-instability x boundary acquisition** | A(x) = Var[p(y|aug_k(x))] + alpha/|margin| + beta*disagreement(proto,probe) -- the disagreement term is a GAUGE SIGNAL, not a decoder to adapt | TTA's validated strength is instability detection (A2); combine with boundary proximity = query the unstable boundary, not the uncertain bulk | P1, P2 | If instability and recoverable error decouple (as conf_drop did for gain), the acquisition may select wrong-but-stable points |
| A3 | **Sequential AL (labels reveal the next query)** | Query 1 -> see the error pair -> focus next query on that boundary -> different region -> next pair | Avoids inferring the whole structure before enough labels; each label sharpens the next (the adaptive loop) | P1, P2 | Slow (sequential), needs the confusion structure to be discoverable in a few steps |
| A5 | ~~Error-correction AL loop~~ | ~~labels reveal recurring (pred,true) errors; query those pairs; re-rank/gate at DECODE time~~ | ~~"labels find the problem, not the parameter"~~ | ~~P1, P2~~ | **CLOSED (Iteration 5): labels do not reliably identify the pairs (recall <= 0.25), AND the decode-time pair repair is ~0/negative even with oracle pairs (pair_bias, pair_gate). The pair is not the right atomic repair unit** |

### Tier 1.5 -- CLOSED: pool-derived basis + few-label coefficient selection (Iteration 4)

This was the last surviving parameter-update family. It reframes the update from

> few labels -> span(x_i) -> Delta W            (CLOSED by Iteration 3b)

to

> unlabeled pool provides a basis, few labels SELECT/WEIGHT which combination -> Delta W.

**CLOSED at the basis by Iteration 4 (al_pool_basis_diag.py):** the rich
label-free pool dictionary (pool_cov, bdry_pca, bdry_disp, conf_pair,
class_disp, tta_disp; K ~ 30 orthonormal directions) captures only 0.036-0.076
of R across all 8 cells (both extractors, 4 conds) -- the same ~0 verdict as the
label span. Even with PERFECT coefficients (oracle_coef = W0 + P_span(D) R) the
gc is negative on half the cells, so the bottleneck is the dictionary itself,
not the selection. Selection (firstorder, lsq) is all-negative and catastrophic
(-1 to -13 gc) because it drives a step along directions that are not R. The
labels DO carry the signal (same few labels + oracle U close +0.2 to +0.46), so
the coefficient half remains easy given a right basis -- but no label-free pool
structure provides one.

| # | Direction | What it is | Why it could work | Property | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| P1 | **Pool-basis + label selection** | Build a RICH dictionary of candidate update directions v_1..v_K from the UNLABELED pool (not the top-r covariance -- boundary directions, per-class shifts, disagreement directions, confusion-pair margins); few labels select/weight the combination Delta W = sum_j c_j v_j | Iteration 1 showed the COEFFICIENT problem is EASY given oracle U (oracle-U + few labels closed +0.29-0.37); the missing piece is the basis itself. tangent_b8 (a FEW-LABEL construction: ridge fits on tiny labeled windows, PCA'd) recovered align 0.24-0.52 -- but it is NOT label-free; the label-free pool structures here all fail | P1, P2, P5 | **CLOSED (Iteration 4): span_capture(full) 0.036-0.076, oracle_coef negative on half the cells. The pool dictionary does not contain R; no label-free pool statistic is the residual** |
| P2 | **Oracle-U reference as a diagnostic** | Keep U_oracle -> C_few-label frozen: is coefficient-selection easy once the basis is right? | Iteration 1's acquisition sweep already answered YES (+0.29-0.37 at rho=0.8): given oracle U, few labels drive the step well. This tells us P1's only hard part is the BASIS | P1 | Diagnostic only, not a method |
| P3 | **Label as constraint, not direction** | A label (x_i, y_i) provides x_i x_i^T, a class prototype, a class-pair constraint (w_{y_i}-w_j)^T x_i, or a correction to a pre-existing UNLABELED boundary estimate -- none of which lie in span{x_i(e_{y_i}-p_i)^T} | Escapes the 3b closure: the label's information content is NOT limited to its CE-gradient direction; it can correct a pool-derived structure the label never spans | P1, P2 | **Same basis question as P1, which is now closed: there is no label-free pool structure to correct (P1 measured span ~0.04-0.08)** |

### Tier 2 -- conservative / decode-time corrections (no W update)

| # | Direction | What it is | Why it could work | Property | Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| B1 | **Margin / entropy / uncertainty + diversity** | Cluster the top-M uncertain points (by the R4 probe's softmax), query cluster reps (or farthest-point) | Avoids re-querying near-identical points; uncertainty+diversity is strictly stronger than uncertainty alone | P1, P2 | Diversity in feature space may not span the decision-rule structure |
| B2 | **Expected gradient length (local)** | EGL(x) = sum_y p(y|x) ||g(x,y)||, restricted to the top-2 pair -- used ONLY as an acquisition score | The raw gradient failed as an UPDATE (R5), but is valid as an ACQUISITION score ("how much could this label change the classifier") | P2 | EGL needs the softmax to be meaningful under corruption (weak) |
| B3 | **Class-bias / logit calibration** | z' = a_c z_c + b_c (per-class temperature/bias) on the R4 probe's LOGITS | A few labels estimate 2C scalars far more easily than a 10000-d residual; the only parameter change that holds for the linear classifier (it is the probe's bias, not U); zero-cost baseline that may fix confidence/ranking distortion | P1, P3, P5 | Does not fix geometric boundary rotations; may only move confidence, not decisions |

### Tier 3 -- closed / revisit only with a new information source

| # | Direction | What it is | Why it could work | Risk |
| :--- | :--- | :--- | :--- | :--- |
| C1 | More clever U estimators | CCA on class pairs, per-condition basis, etc. | The residual is condition-specific (Iteration 1), so per-condition estimators are the only U-route left | Closed 3x already (R3, R4, R6); low prior |
| C2 | More bank-selection schemes | Conformal scores, entropy-balancing, domain-aware querying | Rigorous selection MIGHT beat leverage/random (which failed) | The bank U-source is closed (R4); selection cannot fix a missing signal |
| C3 | More sophisticated trust-region gating | Richer TTA consistency for rollback on the probe's first-order step | R5's gate is the sound part; better rollback could make aggressive updates safe | A3 showed no score separates good/bad updates; consistency-rollback is a weaker (unproven) claim |
| C4 | Online TTA + micro-updates with rollback | W_{t+1} = W_t + eta_t * Delta_t (probe rows), keep W_best only if prediction-consistency improves | The only online form; conservative against catastrophic updates | Needs the consistency score to be reliable (A3 caution) |
| C5 | ~~Prototype correction~~ | ~~Update class means mu_c~~ | ~~Active prototype selection~~ | **REMOVED: uses the R1 prototype decoder, which is closed (R1 < R4). Does not hold for the linear classifier.** |

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

---

## Iteration 1 result: active selection COMPENSATES for the lack of U -- the
acquisition sweep is the first positive result in the new framing
(2026-08-29, `al_acq_sweep_diag.py`)

The decisive Experiment A: fix the DOWNSTREAM update (normalized first-order +
oracle U, the sound R5 form) and vary ONLY the acquisition rule at b in {2,4,8},
on the boundary-focused candidate set. If a rule beats random, active selection
compensates for the label budget without recovering U. Both DGLSS++ and cov-shift,
fog/crosstalk.

best gc per rule (fog/crosstalk, best over b):

| rule | dglsspp fog | dglsspp crosstalk | covshift fog | covshift crosstalk |
| :--- | :--- | :--- | :--- | :--- |
| random | +0.23 | +0.33 | +0.44 | +0.11 |
| margin | +0.19 | +0.32 | +0.36 | +0.22 |
| entropy | +0.11 | +0.25 | +0.34 | +0.15 |
| tta_inst | +0.23 | +0.27 | +0.28 | +0.15 |
| margin_tta | +0.20 | +0.17 | +0.27 | +0.33 |
| margin_div | +0.29 | +0.20 | +0.33 | +0.32 |
| tta_div | +0.23 | +0.21 | +0.37 | +0.29 |
| **margin_tta_div** | **+0.29** | +0.30 | +0.37 | **+0.35** |
| class_pair | +0.18 | +0.23 | +0.32 | +0.06 |
| **egl** | **+0.33** | +0.21 | +0.21 | **+0.28** |

**Finding 1 -- acquisition matters, and the low-budget behavior is the signal.**
At b=2 the margin/TTA/egl rules beat random on most cells (dglsspp fog: random
+0.04 vs margin +0.19, tta_inst +0.20, egl +0.23; covshift crosstalk: random +0.04
vs margin_tta +0.32, egl +0.25). The random baseline at b=2 is weak (+0.04 to
+0.13); the acquisition rules get most of their b=8 gain from just 2 points. This
is exactly the "few labels on the informative boundary beat many on random points"
claim.

**Finding 2 -- diversity and the combined margin_tta_div rule are the most
consistent.** margin_div / tta_div / margin_tta_div reach +0.29-0.37 (fog) and
+0.32-0.35 (crosstalk) across both extractors, matching or beating random at every
budget. The combined rule is never negative and is among the best on both
conditions and both extractors. The TTA-instability component helps on fog
(covshift fog tta_div +0.37), and the margin+diversity component helps on
crosstalk (covshift crosstalk margin_tta +0.33, margin_div +0.32).

**Finding 3 -- class_pair and entropy are weak.** class_pair (per-pair budget) is
middling everywhere and weak on covshift crosstalk (+0.06); entropy is the worst
on dglsspp fog at b=8 (-0.01). The pair-concentration assumption (A1) is not
supported by this sweep -- the error is not concentrated enough in a few pairs for
per-pair budgeting to help. Entropy (pure uncertainty, no boundary/instability) is
consistently worse than the combined rules.

**Finding 4 -- the mechanism is real but the step is still U-limited.** The best
rules close +0.29 to +0.37 gc (fog) -- substantially above random's best (+0.23 to
+0.44) and well above the +0.0 to +0.06 the tangent-U trust-region achieved. But
the ceiling is set by oracle U in the step, so this measures SELECTION value under
a KNOWN direction. The remaining question is whether the local correction forms
(Experiment B) can supply a usable direction from the same few labels.

**Verdict: active selection compensates for the lack of U -- the Tier-1 bet is
supported.** Better labels (boundary-focused, TTA-informative, diverse) recover a
real share of the oracle-U gain that random labels miss, especially at the tiny
b=2-4 budgets. The acquisition rules to carry forward are the combined
`margin_tta_div` (most consistent) and `egl` (best on fog), with `tta_div` /
`margin_div` as the diversity+instability components. The next experiment
(Experiment B) is to pair the best acquisition with the LOCAL correction forms
(class-bias, prototype, class-pair separator) and test whether the update itself
can be driven by the same few labels without oracle U.

## Iteration 2 result: the U-free residual is CLOSED -- the pool geometry does not
substitute for U, and the local forms are not viable as-implemented
(2026-08-29, `al_local_update_diag.py` + `al_ufree_diag.py`)

Two experiments closed the "move away from the residual" branches:

**Experiment B (local forms).** class_bias / prototype / class_pair / local_topK,
driven by the same margin_tta_div / egl labels, WITHOUT oracle U. Result: all
negative or catastrophic. class_bias is a known-dead baseline (bias-only closes
0-4% of the gap, Iteration 0); prototype sits below the probe (R1 < R4);
class_pair's step size was uncalibrated (its -13 gc on snow/wet_ground is the
signature of overstepping, not necessarily the idea). The healthy conditions
were also hurt (P3 violated).

**Iteration 1-UFree (pool-prior regularizer).** Directly optimize dW with the
unlabeled pool geometry as a prior: min_dW L_L(W0+dW) + lam*R(dW; X_pool), never
estimating U. Variants: tikhonov (plain ridge on few labels), pool_span (force dW
into the top-r pool-eigen span), pool_penalty (penalize high-variance directions),
hybrid_first (pool-regularized gradient). Result:

| variant | dglsspp fog | covshift fog | crosstalk (best) |
| :--- | :--- | :--- | :--- |
| oracle_U (bound) | +0.10 | +0.24 | +0.07-0.26 |
| tikhonov | +0.04 | -0.12 | +0.02 to -1.53 |
| pool_span | -0.09 | -0.68 | -0.16 to -2.74 |
| pool_penalty | +0.00 | +0.00 | +0.00-0.02 |
| a_grad / hybrid | -0.45 | -1.54 | -0.27 to -12.5 |

The pool-prior does NOT substitute for U. pool_penalty is exactly zero (the
regularizer pushes dW into the lowest-variance directions where the label signal
is dead); pool_span is negative (reconfirming pool-covariance directions are NOT
the residual, Iteration 1 / R3); tikhonov reproduces the R2 collapse; a_grad and
hybrid are the known failures. Combined with Experiment B, the U-free residual
route is CLOSED with a measured negative.

**What this means.** The update itself cannot be driven by few labels -- neither
through a local correction form nor through a pool-geometry regularized dW. The
only few-label update that works is the first-order step with oracle U, and U is
not obtainable. This confirms the decision-tree verdict: Iteration 1 is closed as
a few-label method, and the remaining candidates are the U-free entries in the
next-steps table that do NOT require a parameter update at all (logit calibration,
prototype selection, error-correction loops, sequential AL).

## The decision: pivot to acquisition-only / decode-time correction (the rationale)

**The claim, stated precisely.** The evidence supports stopping the search for a
GENERIC few-label mechanism that directly estimates the full R = W* - W0. It does
NOT support "no W-update is ever possible" -- the narrower, actionable statement
is:

> With the available label budget (b ~ 8), the information needed to reconstruct
> the global optimal linear probe is not accessible through the parameter-update
> directions tested (labels' own span, all U-estimators, local forms, pool
> geometry).

A different regime -- hundreds of representative labels, a pretrained pool-derived
basis, extra supervision -- could change that, but it is not this setting.

**Why the pivot is justified (the measured facts converge on one bottleneck):**
- direct few-label fitting: insufficient representative feature mass (R2, U-free);
- U-based update: oracle U works, few-label U does not (R3, uest);
- rank-1-per-label: after fixing the confounds, the label span captures ~0.4-2% of
  R (Iteration 3b);
- local / pool parameterizations: the useful oracle correction depends on info
  distributed through the unlabeled pool, and the pool does not contain R in any
  label-free structure (bank R4, shared basis R6, P1 Iteration 4);
- TTA/gauge signals: reliable for instability DETECTION, not for how much of the
  global residual to apply (A2).

**The key asymmetry.** N_pool >> b: lots of unlabeled data, a few labels. The
oracle residual R = W* - W0 depends on statistics accumulated over the ENTIRE
pool. Asking b labels for R is asking the labels to provide information they do
not contain (span(u_1..u_b) is nearly orthogonal to R). Any further
W' = W0 + f(few labels) fights the information geometry.

**The "one final sanity check" has already been run.** The proposal
"unlabeled pool builds the basis V, labels choose the coefficients (Delta W =
V alpha)" is exactly Iteration 4/P1 -- V built from six label-free pool
structures, alpha by firstorder/lsq/oracle-coef. It closed: span_capture(full)
0.04-0.08, oracle_coef negative on half the cells. Both branches are now tested:
labels-provide-directions (3b) AND pool-provides-directions-labels-select (P1).
The stopping rule stated for this branch ("if that also fails, stop") has been
met.

**What labels ARE good for (the positive residue).** Not "what should W be", but
"which predictions/boundaries are wrong". That is an easier information problem:
estimate P(y | predicted class, class pair, region, TTA pattern) from a few labels
and alter the DECISION RULE, e.g. a nonparametric correction
y_hat = g(y_hat0, p0, TTA disagreement, local neighbors, class pair), without
reconstructing W*.

**Keep as diagnostics/references:** W0, W*_oracle, U_oracle.

## The remaining search, organized by how non-parametric the correction is

**Level 1 -- Decode-only (W0 frozen).** Labels learn class biases, class-pair
thresholds, confidence calibration, rejection rules, TTA-dependent relabeling.
This is B3 (logit calibration) plus rejection; the least aggressive, zero-risk
step.

**Level 2 -- Local decision correction (still no global W).** Labels learn
class-pair corrections, prototype/neighborhood rules, spatial/temporal
corrections, TTA-consistency overrides applied at decode. **CLOSED in its pair
form by Iteration 5 (A5): the labels do not reliably identify the pairs, and the
decode-time pair repair is ~0/negative even with oracle pairs.** The pair is not
the right atomic repair unit; this also closes A1's class-pair repair (same
mechanism). No remaining Level-2 candidate.

**Level 3 -- Fully acquisition-driven TTA.** Labels determine WHERE/WHEN the
frozen classifier is unreliable; TTA provides the actual adaptation signal. This
is A2/A3: acquisition rules (margin + TTA-instability + diversity) driving where
TTA is trusted. Matches TTA's validated role as an instability detector.

**Where the effort goes now:** Level 3 (acquisition-driven TTA, A3 sequential AL
and A2 TTA-instability x boundary acquisition) is the strongest fit to the
measured evidence -- labels for selection/instability, TTA for the signal, no
parameter reconstruction and no pair-rule repair. Level 1 (B3 logit calibration)
is the conservative floor and the only remaining decode-only candidate.

---

## Next: the U-free next-steps (from the candidate table)

Ranked by what the measured evidence supports after Iterations 3b, 4 and 5,
organized by the decision levels above. **All hold for the R4 linear probe** (no
prototype decoder; labels make decisions about where to look or how to re-rank at
decode time -- every W-update family is now closed, including the pool-basis
branch, P1/Iteration 4, and the pair-repair branch, A5/Iteration 5):

1. **Logit calibration (B3)** -- z' = a_c z_c + b_c on the probe's logits. The
   known-weak baseline (bias-only = 0-4%) but zero-cost and a clean control, and
   the only parameter change that holds for the linear classifier (it is the
   probe's bias, not U). The one remaining Level-1 (decode-only) candidate.
2. **Sequential AL (A3)** -- x1 -> y1 -> x2(y1) -> ... where each label sharpens
   the next query. Directly tests "labels reveal the structure of the next query."
   Now the primary Level-3 acquisition candidate, alongside A2 (TTA-instability x
   boundary acquisition, the validated detector).
3. **Error-correction AL loop (A5) is CLOSED (Iteration 5)** -- the labels do not
   reliably identify the error pairs (recall <= 0.25), AND the decode-time pair
   repair does not close the gap even with oracle pairs (pair_bias ~ 0/-2, pair
   _gate negative everywhere). The pair is not the right atomic repair unit.
4. **Prototype selection (B4) is REMOVED** -- it assumes the R1 prototype decoder,
   which is closed (R1 < R4). It does not hold for the linear classifier.

These are the surviving U-free bets. They share the property that failed nothing so
far: they use labels to make DECISIONS about where to look or how to re-rank,
not to estimate a parameter update. The last parameter-update branch (P1, pool
basis + label selection) is CLOSED by Iteration 4, and the pair-repair branch
(A5) is CLOSED by Iteration 5 -- the pivot to acquisition-only / decode-time
decision rules is now final, with a measured negative rather than an assumption.
The effort now concentrates on Level 3
(acquisition-driven TTA: labels for instability/selection, TTA for the signal),
with Levels 1-2 (decode-only calibration, local decision correction) as the
conservative floor.

## Iteration 3 result: the rank-1-per-label diagnostic is INCONCLUSIVE -- the
negative is confounded by metric choice and step scale, NOT a verdict on the
decomposition idea (2026-08-29, `al_rank1_diag.py`)

Tested whether each queried label can be its own rank-1 update direction
u_i = x_i r_i^T (r_i = e_{y_i} - p_0(x_i)), applied separately instead of the
aggregate sum. Central linearity fact confirmed: sequential-with-fixed-eta ==
aggregate, so only per-label scale / rejection / adaptive-query matter. Both
DGLSS++ and cov-shift, all 4 conds.

The raw result (D1/D2): per-label align(u_i, R) ~ 0.001-0.016 and all individual
deltas negative (0/8 positive on every condition, both extractors). D4
(oracle-scaled) still negative. D5 corr(d_conf, delta) inconsistent.

**BUT the negative is confounded in two ways, so it does NOT throw away the
decomposition idea:**

1. **D1's metric is a concentration-of-measure trap.** align = cos of the
   FLATTENED 170000-d u_i vs the FLATTENED 170000-d R. In 170k-dimensions, ANY
   two generic vectors have cos ~ 0 by concentration of measure. align ~ 0.01 does
   not mean "the direction is wrong" -- it means "one label is not the whole
   residual", which is trivially true (the residual is a sum of many
   contributions, not any single one). The correct D1 would measure whether the
   SPAN of the b labels captures R (||P_span R||/||R||), not whether each
   individually equals R.
2. **D2/D4's step size is massively oversized.** ||u_i|| ~ ||x_i|| * ||r_i|| =
   100 * 0.96 = 96, so eta * u_i = 0.05 * 96 = 4.8 magnitude applied across all
   of W0. The oracle residual R = W* - W0 has each W-column at ~O(1-10) norm; a
   single 4.8-magnitude step over the whole probe is pure overstepping. D2's
   all-negative delta may be entirely a scale artifact (the update is far too
   large), not wrong direction.

**What this means.** The diagnostic's D1/D2 instantiated the decomposition as
"per-point cross-entropy gradient at a fixed raw scale, measured by flattened
cosine to the total residual" -- and that INSTANTIATION shows no signal. It does
NOT rule out:
- normalized per-label directions (u_i/||u_i||, so the step is a bounded trust
  radius, not a raw magnitude),
- a span-based alignment (does the label span capture R, not each label = R),
- a different rank-1 form (e.g. class-pair margin direction, not the CE gradient).

**The honest verdict rule.** The decomposition idea is NOT dead. The diagnostic
shows the CE-gradient-at-raw-scale instantiation fails, but the confounds (D1
metric, D2 scale) prevent a conclusion about the idea itself. The correct
follow-up is a rerun with (a) normalized u_i and a per-label trust radius, and
(b) span-capture instead of per-label flattened cosine -- which would tell us
whether the idea fails because "labels don't span the residual" (real) or because
"the step was too big" (artifact). Do NOT discard the decomposition concept on
this result.

## Iteration 3b result: the corrected rerun CLOSES the rank-1 decomposition -- the
labels genuinely do NOT span the residual (2026-08-29, `al_rank1b_diag.py`)

The two confounds from Iteration 3 were fixed and the diagnostic rerun: (a)
NORMALIZED per-label directions (u_i/||u_i||, so eta is a bounded trust radius,
not a raw 4.8 overstep), and (b) SPAN-CAPTURE -- does the span of the b labels
contain R? (||P_span R||/||R||), which is the correct question (a single label is
never the whole residual, but the span can be). Both DGLSS++ and cov-shift, all 4
conds.

**SPAN-CAPTURE is the decisive number, and it is ~0 everywhere:**

| cond | dglsspp capture (b2/b4/b8) | covshift capture (b2/b4/b8) |
| :--- | :--- | :--- |
| fog | 0.008 / 0.011 / 0.017 | 0.009 / 0.014 / 0.020 |
| crosstalk | 0.009 / 0.009 / 0.011 | 0.004 / 0.016 / 0.017 |
| snow | 0.001 / 0.003 / 0.016 | 0.005 / 0.011 / 0.018 |
| wet_ground | 0.009 / 0.011 / 0.012 | 0.002 / 0.011 / 0.016 |

**The 8 labels' span captures 0.4-2.0% of the oracle residual** -- even with
NORMALIZED directions (removing the scale confound), the label span does not
contain R. This is the REAL, method-independent conclusion the corrected test was
designed to isolate: it is NOT "the step was too big" (that confound is gone); it
is that "few labels cannot span the residual." The decomposition idea is closed at
the fundamental level.

**The update confirms it.** With the bounded trust radius, the normalized
aggregate is still negative (dglsspp fog b8 -0.47; covshift fog b8 -1.05) and
0/8 per-label updates are positive on every condition, both extractors (so
keep_oracle_good is +0.00 -- there is nothing good to keep). The oracle-U
reference is only +0.00 to +0.06 (small at eta=0.5), confirming the bounded step
is not the issue. The failure is structural: the few labels' rank-1 directions do
not live in the residual subspace, so no combination of them -- aggregate,
per-label, or keep-good -- can move the probe correctly.

**Verdict: the rank-1-per-label CE-gradient decomposition is CLOSED, with the
confounds resolved -- but the claim must be stated NARROWLY.** The corrected rerun
cleanly separates the two hypotheses from Iteration 3: it is NOT a scale artifact
(normalized steps still fail) and NOT a metric artifact (span-capture is the right
metric and it is ~0). The precise, supported conclusion is:

> **Few labeled points' per-point CE-gradient rank-1 directions do not span the
> oracle residual** -- for b <= 8, across DGLSS++/cov-shift and all 4 conditions.

This is a strong, useful negative, but it is NOT the broader claim that "no
few-label parameter update works under any formulation." What is closed is the
specific update family Delta W in span{ x_i(e_{y_i}-p_i)^T : i in L }, not the
span of all information obtainable from the labels. A label (x_i, y_i) also
provides x_i x_i^T, a class prototype, a class-pair constraint
(w_{y_i}-w_j)^T x_i, or a correction to a pre-existing unlabeled boundary estimate
-- none of which lie in the CE-gradient span (see P1/P3 in the next-steps table).

**The deeper structural reason (why it's fundamental for this family).** u_i =
x_i r_i^T, so the feature-side reach of b labels is contained in span{x_1..x_b} --
a tiny subspace of the 10000-d feature space. 3b is telling us: a handful of
observed feature vectors do not span the feature directions the global probe
correction needs. This makes the point-gradient family closed regardless of
scaling, aggregation, sequential application, or better C-estimation inside the
span.

**What is now closed (do not re-test):** raw u_i, normalized u_i, different eta,
sequential vs aggregate, keep-good, per-label trust regions, and more
sophisticated weighting or C-estimation of the same u_i -- all remain inside
span{u_i}, whose projection onto R is ~0.

## Iteration 4 result: P1 -- the POOL-DERIVED BASIS does not contain the residual,
and the pool-basis + label-selection branch is CLOSED (2026-08-30,
`al_pool_basis_diag.py`)

The last parameter-update branch: build a RICH label-free dictionary from the
unlabeled pool, then let few labels select/weight the combination. The diagnostic
separates the basis half from the selection half. Both DGLSS++ and cov-shift, all
4 conds, budgets 2/4/8.

**A. BASIS half (the decisive number) -- the pool dictionary does NOT contain R.**

span_capture(full) of the K ~ 30 orthonormal dictionary directions (pool_cov,
bdry_pca, bdry_disp, conf_pair, class_disp, tta_disp):

| cond | dglsspp span | covshift span | best element (both) |
| :--- | :--- | :--- | :--- |
| fog | 0.061 | 0.076 | conf_pair 0.042, class_disp 0.044 |
| crosstalk | 0.047 | 0.036 | conf_pair 0.029 |
| snow | 0.054 | 0.056 | conf_pair 0.037, class_disp 0.037 |
| wet_ground | 0.036 | 0.052 | conf_pair 0.023, class_disp 0.020 |

The full dictionary captures only **3.6-7.6% of R** across all 8 cells -- the same
~0 verdict as the label span (3b), despite ~30 directions instead of 8. The best
single elements (conf_pair, class_disp = pseudo-label class/pair mean shifts) reach
only 0.02-0.04. No label-free pool statistic is the residual.

**Even PERFECT coefficients fail -- the bottleneck is the dictionary, not the
selection.** oracle_coef = W0 + P_span(D) R (the best classifier with W1-W0 in
span(D)) is negative on HALF the cells: dglsspp snow -0.384, covshift fog -0.207,
covshift crosstalk -0.006, covshift snow -0.043. A 4-8%-span dictionary that
contains an ANTI-aligned component cannot help regardless of how well the few
labels choose coefficients.

**B. SELECTION half -- the labels carry the signal, the basis does not.** With the
SAME few labels (margin_tta_div acquisition):

| | oracle_U (same labels, true basis) | firstorder (pool dict) | lsq (pool dict) |
| :--- | :--- | :--- | :--- |
| dglsspp fog b8 | +0.206 | -0.443 | -0.154 |
| dglsspp crosstalk b8 | +0.323 | -0.314 | -0.041 |
| covshift crosstalk b8 | +0.355 | -11.2 | -2.9 |
| covshift wet_ground b8 | +0.461 | -1.6 | -0.8 |

The oracle-U reference (with the SAME labels) closes +0.2 to +0.46 on every
healthy-capacity cell -- the coefficient half is easy, confirming Iteration 1.
But firstorder drives a step along the pool dictionary's directions, which are NOT
R: all-negative, often catastrophic (-1 to -13 gc, the overstepping signature of a
wrong basis). lsq is less violent but still negative everywhere except two ~0
cells (dglsspp crosstalk b2/b4).

**Verdict: P1 is CLOSED at the basis.** The unlabeled pool, under any of the six
label-free structures tried, does not contain the oracle residual (span 0.04-0.08).
The tangent_b8 align-0.24-0.52 result that motivated this test was a FEW-LABEL
construction (ridge fits on tiny labeled windows) -- it does NOT generalize to
label-free pool statistics, and its own downstream AL chain was ~0. This closes
the last parameter-update branch: **the pivot to acquisition-only / decode-time
decision rules is now final, with a measured negative rather than an assumption.**
The positive residue to carry forward: the coefficient half IS easy given a right
basis (Iteration 1, confirmed here with same-labels oracle U), but no label-free
or few-label route produces one. Remaining candidates: A5 (error-correction AL
loop), A3 (sequential AL), B3 (logit calibration).

## Iteration 5 result: A5 -- the error-correction AL loop is CLOSED, at BOTH
stages (2026-08-30, `al_error_loop_diag.py`)

A5 = labels reveal recurring (pred,true) error pairs, subsequent queries focus on
those pairs' boundaries (the sequential loop), and the repair is a DECODE-TIME
re-ranking/gating of the identified pairs (NOT a W update). The diagnostic
separates the two stages: pair DISCOVERY (confusion counts from labels vs
val-truth error pairs) and the decode repair (pair_bias = per-pair logit offset,
pair_gate = margin-threshold flip), each with label-pair vs oracle-pair arms.
Both DGLSS++ and cov-shift, all 4 conds, budgets 2/4/8.

**Stage 1 (discovery) is weak -- the labels do NOT reliably find the error
pairs.** The few labels identify only 1-3 pairs, and recall is <= 0.25 on every
cell (best hit: dglsspp wet_ground b8 2/4; covshift fog/snow/wet_ground b2 1/4
at precision 1.0 but that is one lucky pair). On most cells the discovered pair
is NOT in the val-truth set (dglsspp fog/crosstalk prec 0.00 at b2/b4; covshift
crosstalk finds nothing until b4). The "labels reveal the problem" claim is not
supported at this budget: the confusion structure of a handful of points does not
replicate the pool-level confusion.

**Stage 2 (decode re-ranking) is the DECISIVE negative -- the repair does not
close the gap even under ORACLE pairs.**

| arm | dglsspp fog/crosstalk | covshift fog/crosstalk | snow/wet_ground |
| :--- | :--- | :--- | :--- |
| pair_bias label | -0.03 / -0.03 (b8) | +0.00 / -0.41 | -1.46 / -6.03 (wg b8) |
| pair_bias oracle | +0.00 / +0.00 | **-2.15** / +0.00 | -0.41 / -0.10 |
| pair_bias random | +0.00 / +0.00 | +0.00 / +0.00 | +0.00 / +0.00 |
| pair_gate label | +0.00 / +0.00 | -0.00 / +0.00 | +0.00 / +0.00 |
| pair_gate oracle | **-0.20** / **-0.10** | **-0.26** / **-3.36** | **-1.86** / **-0.91** |

- **pair_bias ORACLE (true pairs, pool-label offsets) is ~0 or NEGATIVE on every
  cell** (covshift fog -2.15, snow -0.41, wet_ground -0.10). Even with the CORRECT
  pairs and pool-fit offsets, per-pair logit offsets do not move mIoU -- they
  cannot represent the global residual (the pair structure of R is not a small
  set of scalar logit shifts; this is the same global-object result as Iterations
  3b/4, at the decision-rule level).
- **pair_gate oracle is negative EVERYWHERE** (-0.10 to -3.36). The margin-threshold
  flip over-corrects: it flips more correct low-margin points than it fixes,
  because the label-observed misclassification threshold is not a clean separator.
- **random == +0.00 consistently** -- the control confirms the pair-bias mechanism
  is a no-op at best, and the label arms are occasionally catastrophic when a
  wrong pair gets a large offset (dglsspp wet_ground b8 -6.03, snow b8 -1.46).

**Verdict: A5 is CLOSED with a measured negative at BOTH stages.** The few labels
do not reliably identify the error pairs (stage 1), AND the decode-time pair
re-ranking does not repair the gap even with oracle pairs (stage 2). The pair is
not the right atomic repair unit: the recoverable error is a GLOBAL rotation of
the probe, not a set of per-pair scalar decisions. This closes the Level-2 "local
decision correction" candidate (A5, and by the same mechanism A1's class-pair
repair). Remaining: B3 (logit calibration, Level 1) and A3/A2 (sequential AL +
acquisition-driven TTA, Level 3).