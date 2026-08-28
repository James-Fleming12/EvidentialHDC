# Improving DGLSS++ for the Linear-Probe HDC Model

Working doc for improving the plain DGLSS++ extractor (`supcon_vib_dglsspp`) as the
base for the linear-probe HDC decoder (the R4 probe, the `W_res` AL update, and
the closed-form TTA/AL framework in `docs/lin_probe_updates/`). The goal is to
raise the LABELED CEILING and the ZERO-SHOT on fog/crosstalk -- the conditions
where AL/TTA has real recoverable headroom -- without the cov-shift trade (which
fixes fog/crosstalk but compresses the healthy conditions).

This doc records what has been measured about DGLSS++ and what changes to the
training objective / head structure are worth trying next.

---

## Why DGLSS++ matters as the linear-probe base

The linear probe (fit on the 10000-d binarized HDC code, `docs/lin_probe_updates/tta_iterations.md`)
is the decoder-side machinery. Its gain over nearest-centroid is large, but the
recoverable ceiling is set by the FEATURES, not the decoder:

- The labeled ceiling (pool-refit oracle) is where AL headroom lives, and it is a
  property of the extractor's class structure under corruption.
- DGLSS++ is the CEILING extractor: at medium scale it has the highest labeled
  oracle on fog/crosstalk of the DGLSS family (`docs/robust_iterations.md`, the
  "DGLSS++ is the ceiling extractor" three-way vs Robust/InstanceNorm).
- cov-shift (`supcon_vib_dglsspp_inputin_in_chan`) fixes the fog/crosstalk
  collapse via channel-restricted input-IN {0,4} + internal InstanceNorm, but it
  compresses the healthy conditions (the cov-shift trade). DGLSS++ has the big
  AL-closeable gaps (KITTI-C 3-sev: fog zs 22.5 -> ceiling 35.2 = +12.7,
  crosstalk 11.9 -> 29.4 = +17.5) where AL/TTA has something to recover.

So: the extractor improvements that raise DGLSS++'s ceiling directly enlarge the
AL/TTA story; the ones that only push its zero-shot are the Pillar-1 (cov-shift)
story.

---

## The convergence baseline (2026-08-28, `run_overnight_convergence.sh`)

The current `logs/supcon_vib_dglsspp` checkpoint is a **24-epoch / 100% run** and
was never explicitly tested for convergence. An overnight 40-epoch / 100% run
(`conv40_supcon_vib_dglsspp`) was trained and evaluated on the SAME
`probe_linear_prop_diag.py` harness (200 frames, KITTI-C heavy, R4 linear probe).
Reference numbers: the 24-ep baseline is `probe_linear_prop.json` `dgl_kitti`;
the 40-ep is `probe_conv40_dglsspp.json`.

| metric | cond | 24ep | 40ep | delta |
| :--- | :--- | :--- | :--- | :--- |
| frozen (zs) | fog | 0.111 | **0.087** | **-0.023** |
| frozen (zs) | crosstalk | 0.133 | **0.129** | **-0.004** |
| **ceiling** | fog | 0.492 | **0.391** | **-0.102** |
| **ceiling** | crosstalk | 0.525 | **0.468** | **-0.056** |
| class_shift (clean->corr cos) | fog | 0.196 | **0.431** | +0.235 (LESS shift) |
| class_shift | crosstalk | 0.273 | **0.580** | +0.308 (LESS shift) |
| fisher_ratio | fog | 0.525 | **0.199** | -0.326 |
| fisher_ratio | crosstalk | 0.602 | **0.239** | -0.363 |
| within_class_var | fog | 0.466 | **0.519** | +0.054 |
| effrank | fog | 3.45 | 3.15 | -0.30 |
| effrank | crosstalk | 3.42 | 4.77 | +1.35 |

**Result: DGLSS++ at 24ep was already PAST its ceiling peak. Longer training
HURTS the linear-probe ceiling and zero-shot.**

The 40-ep ceiling drops substantially on both fog (-0.102) and crosstalk (-0.056)
-- the opposite of an under-converged model. The mechanism is now visible in the
Diagnostic-11 lever: class_shift IMPROVED (0.196 -> 0.431, less clean->corrupted
shift) but fisher_ratio COLLAPSED (0.525 -> 0.199) and within_class_var rose
(0.466 -> 0.519). Longer training makes the classes LESS shifted but LESS
separated -- the corruption-specific shift is being trained away while the
per-class variance grows, which is exactly what hurts a linear classifier that
needs clean class boundaries. This matches the Diagnostic-11 finding that
fisher (not shift) tracks the ceiling, and the gen_iterations "don't-erase"
(antianchor) line: objectives that un-learn the corruption shift reduce the
ceiling.

**Takeaway: do NOT run longer DGLSS++.** The 24ep checkpoint is converged-and-past-
peak for the R4 ceiling; more epochs cost the ceiling. If anything, the data
points the other way -- the DGLSS++ objective actively un-learns the shift
structure the ceiling needs. This is the same "erasure" failure mode the
antianchor/dircons line identified for the robust variants.

---

## What we've measured about DGLSS++ (background)

From `docs/robust_iterations.md`, `docs/gen_iterations.md`, `docs/cov_shift/`:

### 1. DGLSS++ is the sign-saturation-resistant member of the family
The plain DGLSS (mean-dominated consistency) saturates the HDC sign-projection
(dead coordinates, low effective rank); DGLSS++ (GMSIFC + LSCC on the bottleneck)
keeps the HDC code healthy. This is why it decodes better than plain DGLSS at
equal scale (robust_iterations Iteration 3-4).

### 2. DGLSS++ is the ceiling extractor at medium scale
In the three-way (DGLSS++ vs Robust-corsupcon vs InstanceNorm) at medium scale,
DGLSS++ has the highest labeled oracle on fog/crosstalk (robust_iterations
Iteration 19.9.1), even though its zero-shot and TTA are weaker than the robust
variant. The ceiling (what the linear probe / AL can reach) and the zero-shot
(what the frozen decode gives) are decoupled on DGLSS++.

### 3. The residual R = W* - W0 is a decision-rule object, not a shift object
The U-estimation diagnostics (`docs/lin_probe_updates/active_iterations_2.md`,
`al_uest_diag.py` + `al_uest_bdry_diag.py`) proved that the oracle residual
(R = W* - W0) is NOT recoverable from any unlabeled first/second-order statistic:
pool covariance, class-mean shift, CCA, near-boundary PCA, boundary outer product,
margin-weighted covariance, confused-pair PCA, and a weak-classifier ensemble all
land orthogonal (align 0.00-0.05) to the oracle U. The ONLY working U estimator
is the few-label tangent-space construction (`tangent_b8`: split 8 labels into 4
two-point provisional ridge fits, stack their dW, right-SVD) which recovers
0.3-0.5 alignment. Implication for training: an objective that makes the residual
MORE STRUCTURED (fewer effective directions, coherent per-class displacement)
would strengthen the tangent-space AL discovery.

### 4. Every training-side lever tried so far is closed or already incorporated
Checked against the docs (the full ledger is in the overnight-run discussion):
additive volumetric noise (closed at equal capacity -- worse on every condition),
GeoID inlier loss (no ceiling/zero-shot win, D13), dircons (crosstalk-ceiling-only,
costs healthy), dircons_w02 (down), corrsc / corrfree_corrsc (flat/negative),
antianchor (flat-to-negative), the C8 healthy-loss levers (closed at micro), and
InstanceNorm (positive but already folded into cov-shift's `inputin_in_chan`).
The current cov-shift method IS the accumulation of everything that worked.

---

## Directions worth trying for DGLSS++ (the linear-probe base)

Each entry states the hypothesis, the evidence it builds on, and the risk. The
priority ordering reflects the measured facts above.

### D1. Residual-structure training (train so R = W* - W0 is low-rank/coherent)
**Hypothesis.** The AL path works when U is discoverable (tangent-b8 found it),
but its C-fit can't exploit it. If the extractor is trained so the corruption
induces a MORE coherent per-class displacement (fewer effective residual
directions), the tangent-space U estimation gets stronger AND the downstream C-fit
works better.
**Builds on.** tangent_b8 is the only working U estimator (0.3-0.5 align);
dircons showed per-class displacement coherence is trainable (but was coupled on
all classes and cost healthy conditions). The modern version: displacement
coherence ONLY on the AL-relevant classes {7, 15, 14} on the plain DGLSS++ base,
with the healthy classes' residual left uncoupled.
**Risk.** The gen_iterations antianchor finding warns that objectives pulling
toward coherence can erase the shift the ceiling needs (see D2). Must micro-gate
on both ceiling AND zero-shot before any medium spend.

### D2. Don't-erase (antianchor) on DGLSS++ directly
**Hypothesis.** The convergence result (this doc) shows the DGLSS++ objective
actively un-learns the corruption shift (class_shift improved but fisher
collapsed). An explicit penalty on the corrupted->clean class cosine would retain
whatever shift develops naturally, protecting the ceiling.
**Builds on.** The antianchor variant exists (`supcon_vib_dglsspp_corsupcon_...`
family) but was micro-gated on the ROBUST base and was flat-to-negative there;
it was never tried on the plain DGLSS++ base where the erasure is the issue (per
this doc's convergence mechanism). The convergence numbers (fisher collapse at
40ep) are direct evidence the erasure is active.
**Risk.** The 19.9 antianchor isotropy was flat (fog HDC 0.057); the mechanism may
just not move the representation much. Cheap to micro-gate.

### D3. Healthy-condition headroom (the cov-shift trade, on the DGLSS++ side)
**Hypothesis.** cov-shift wins fog/crosstalk by compressing healthy conditions;
the gap is a continuous-features loss (C8). A training-side term that preserves
the healthy conditions' per-dimension anisotropy (scale-only internal InstanceNorm,
or the healthy-class residual decoupling) would let a fog/crosstalk-focused
objective be added without the trade.
**Builds on.** C8 proved the healthy-ceiling loss survives every decoding; the
`_scalein` / `_scope` / `_scalereg` levers were micro-gated but CLOSED at micro.
**Risk.** Closed at micro (C9). Low priority unless a new mechanism appears.

### D4. A U-friendly head: expose the residual subspace as an auxiliary output
**Hypothesis.** Instead of hoping the features make U discoverable, add a small
auxiliary head that PREDICTS the residual basis (or the per-class displacement
direction) from the corrupted view, supervised at train time by the clean/corrupted
pairing (which we uniquely have: KITTI-C is per-frame corruptions of seq-08).
This would give a learned, label-free U estimator at deployment -- the thing every
U-estimation diagnostic has failed to find from unlabeled statistics.
**Builds on.** The pairing exists; the tangent-space result proves the residual IS
predictable from a few labels; a trained head is the natural label-free version.
**Risk.** The residual is a decision-rule object in the 10000-d code space, and a
head on the 128-d bottleneck may not express it. Medium cost (new head + training).
Highest-upside new direction.

---

## Decision rules

- **Do NOT run longer DGLSS++.** The 40ep convergence test is decisive: the ceiling
  and zero-shot both drop. The current 24ep checkpoint is the operating point.
- **Micro-gate D1 (residual-structure) and D2 (don't-erase) on DGLSS++** before any
  medium spend; both must hold ceiling AND zero-shot on fog/crosstalk AND not
  regress the healthy conditions.
- **D4 (U-predictor head)** is the highest-upside new idea and does not conflict
  with the closed training-side levers; it targets the exact AL bottleneck (U
  estimation) that no unlabeled statistic has solved.

---

## Reproducibility

- Harness: `probe_linear_prop_diag.py` (frozen zs + labeled ceiling + Diagnostic-11
  levers on KITTI-C heavy, R4 probe), 200 frames.
- Convergence run: `run_overnight_convergence.sh` (40ep/100% DGLSS++ and cov-shift,
  isolated `conv40_*` logs). JSONs: `probe_conv40_dglsspp.json`,
  `probe_linear_prop.json` (24ep reference).
- The extractor-improvement target is the R4 labeled ceiling, which is what the
  AL/TTA framework (`docs/lin_probe_updates/`) acts on.
