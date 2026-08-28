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
