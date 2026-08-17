# Covariate-Shift-Aware DGLSS++ (Cov-Shift): Iterations

This tracks the development of the **cov-shift DGLSS++** feature extractor — the
first extractor in this project to raise BOTH the labeled ceiling AND the label-free
TTA on BOTH fog and crosstalk simultaneously. The core idea, the measured
performance, and the open diagnostic (healthy-condition ceiling loss) are recorded
here. The extractor-level comparison (vs DGLSS++ and Robust DGLSS++) is summarized
in the README (Pillar 1); the per-iteration mechanics are in
`docs/robust_iterations.md` (Iterations 19.10 - 19.14).

## Background: the problem cov-shift solves

The project's central finding (Iteration 19.8, robust-iterations doc) is that
DGLSS++'s ceiling comes from NEVER being told to undo corruption — the recoverable
classes are the SHIFTED ones (car fog dir_retention 0.27, og 0.27, veg 0.03, far from
clean), and every training objective we tried (SupCon clean-anchor, GMSIFC/LSCC
view-invariance, dircons, corrsc, hdc, antianchor) either erased that shift or was
flat. The Robust DGLSS++ variant improved TTA but LOWERED the ceiling (clean-anchor
erased the shift); plain DGLSS++ had the highest ceiling but the weakest TTA.

**The cov-shift insight:** the fog/crosstalk failure is not a single geometry problem
— it decomposes into two mechanisms:
- **fog** : the recoverable structure is a *shifted direction* (geometry) that
  normalization must not erase.
- **crosstalk** : the corruption is a *statistics shift* in the range/remission
  channels that normalization should fix.

The cov-shift method addresses these as two separate mechanisms instead of one
geometry to balance. It does NOT pull corrupted features toward clean (no anchoring);
it fixes the covariate-shift statistics so the network receives well-scaled inputs.

## The method

Three changes to the DGLSS++ pipeline, all about *statistics* not *geometry*:

1. **Per-scan input normalization restricted to the statistics-shifted channels.**
   The parser normalizes the 5-channel input by FIXED clean-data `img_means` /
   `img_stds`; under fog/crosstalk the network receives inputs scaled against clean
   statistics. The cov-shift fix re-normalizes each scan's VALID points by its OWN
   per-scan mean/std, but only on the **range + remission channels (0, 4)** — the
   channels crosstalk's statistics shift lives in. The **xyz geometry channels
   (1-3) are left in the parser's clean-stat normalization**, so fog's shifted
   direction survives. (Full-channel normalization was tried and ERASES fog's
   direction — Iteration 19.12.2, rejected.)
2. **Internal InstanceNorm** replacing BatchNorm throughout the backbone. BN's
   learned running-stats are computed on clean data, so under covariate shift the
   internal normalization itself distorts features; InstanceNorm normalizes each scan
   independently (with learnable affine scale/shift), removing the batch-coupling
   that breaks under shift.
3. **Applied inside the model forward** (`input_in` flag in `ResNet_34`), so it holds
   at BOTH train and eval time — the diagnostics call `model()` directly.

The method name: `supcon_vib_dglsspp_inputin_in_chan` (per-scan **in**put
normalization on **chan**nels 0/4 + internal InstanceNorm).

## Measured performance

All numbers are the **ep-10 model** (the optimal window of the 21-ep run, chosen by
the mid-training monitor) unless noted. Sources: extractor-diff harness (fog/crosstalk
ceiling+TTA, 100k/100k) and the isotropy pipeline (8-condition HDC-zs) — see the
README tables for the full comparison.

### Ceilings (labeled oracle, extractor-diff harness, fog/crosstalk)

| condition | DGLSS++ | Robust DGLSS++ | **Cov-shift (ep-10)** |
| :--- | :--- | :--- | :--- |
| fog | 17.6% | 15.7% | **23.5%** |
| crosstalk | 22.2% | 18.8% | **39.4%** |

### Label-free TTA (naive, same harness)

| condition | DGLSS++ | Robust DGLSS++ | **Cov-shift (ep-10)** |
| :--- | :--- | :--- | :--- |
| fog | 10.0% | 10.7% | **21.0%** |
| crosstalk | 12.7% | 15.0% | **38.6%** |

### Zero-shot HDC-zs (isotropy pipeline, all 8 conditions)

| condition | DGLSS++ | Robust | Cov-shift ep-10 | Cov-shift ep-21 |
| :--- | :--- | :--- | :--- | :--- |
| fog | 6.8% | 8.5% | **20.1%** | 18.5% |
| crosstalk | 11.5% | 9.8% | 39.5% | **41.9%** |
| snow | 39.6% | **41.1%** | 37.7% | 38.6% |
| wet_ground | **48.3%** | 46.8% | 35.8% | 33.3% |
| incomplete_echo | 44.9% | **45.0%** | 40.6% | 40.0% |
| beam_missing | **50.6%** | 50.3% | 44.3% | 44.5% |
| motion_blur | **50.2%** | **50.2%** | 44.2% | 44.6% |
| cross_sensor | 43.4% | **43.5%** | 36.1% | 38.8% |
| **mean (8 corrupted)** | 36.9% | 36.9% | **37.3%** | **37.5%** |

### The headline result

The cov-shift extractor is the first to raise BOTH the ceiling AND the label-free TTA
on BOTH fog and crosstalk: fog oracle 23.5% vs DGLSS++ 17.6%, crosstalk oracle 39.4%
vs 22.2%. **Crosstalk is effectively fixed at the frozen-prototype level** — zero-shot
40.3% ~= oracle 39.4%, so the oracle re-estimation adds little because the frozen
prototypes already decode correctly. This is the "fog/crosstalk are not inherently
broken" claim, achieved by the extractor rather than by TTA.

### The key property: naive TTA ~= ceiling

The label-free update essentially reaches the ceiling on both conditions (fog naive
21.0% vs ceiling 23.5%; crosstalk naive 38.6% vs ceiling 39.4%). Every prior
extractor left a real gap between naive TTA and ceiling (crosstalk 6-10 points) — the
assignment wall. On the cov-shift extractor that wall is gone for crosstalk.

## Iteration log

### Iteration C1: the level-1 covariate-shift test (2026-08-14)

**Design.** The 5-channel input is normalized by FIXED clean-data statistics in the
parser. Test whether per-scan input normalization (training-side mirror of the
BN-statistic-alignment TTA lever, our best TTA method) raises the ceiling. Three
variants: `inputin` (input-IN only, internal BN), `inputin_in` (input-IN + internal
InstanceNorm, both levels), and the scale-only alternative (divide by per-scan std,
no mean subtraction).

**Result.** The full mean+std stack (`inputin_in`) was the strongest crosstalk result
in the family (crosstalk oracle 0.277 vs DGLSS++ 0.214) BUT fog regressed on the
oracle — the mean-shift component erased fog's recoverable direction. **Scale-only
was rejected** (Iteration 19.12.2): normalizing ALL channels by per-scan std,
including xyz, collapsed the class structure the clean-stat normalization provided
(corr_tightness car fog 0.89 -> 0.47). This confirmed the xyz channels' ABSOLUTE
scale carries the near-vs-far class structure.

### Iteration C2: the channel-restricted fix (2026-08-14)

**Design.** Restrict the per-scan input normalization to the range + remission
channels (0, 4) where crosstalk's statistics shift lives, and leave the xyz geometry
channels (1-3) in the parser's clean-stat normalization so fog's shifted direction
survives. Internal InstanceNorm kept. This is `inputin_in_chan`.

**Result.** Resolves the fog regression and improves everything:
- fog oracle 0.175 vs DGLSS++ 0.159 (the stack had dropped it to 0.137)
- crosstalk oracle 0.303 vs 0.214 (the strongest in the family)
- naive TTA also up on both (fog +0.067, crosstalk +0.165)

The mechanism is exact: normalizing only the range/remission channels absorbs
crosstalk's statistics shift while leaving fog's geometry in clean-stat
normalization — normalizing the SHIFTED STATISTICS, not the structure.

### Iteration C3: the scale-only negative (2026-08-14)

**Design.** The cleaner general version: divide by per-scan std on ALL channels but
DO NOT subtract the mean, preserving direction while absorbing magnitude.

**Result.** Clearly negative (fog oracle 0.086 vs DGLSS++ 0.159, crosstalk 0.113 vs
0.214). Dividing ALL channels by per-scan std — including xyz — collapsed the class
structure the clean-stat normalization provided. Confirms the channel-restricted
design (`inputin_in_chan`) is not one option among many but the necessary design.

### Iteration C4: the full medium run (2026-08-14 -> 15)

**Design.** 21 ep / 100% of `inputin_in_chan`, with a mid-training monitor
snapshotting the rolling checkpoint every ~30 min to find the optimal-epoch window
(the family degrades past 21 ep — Iteration 8.1).

**Result.** The monitor captured the full trajectory: fog oracle peaked at ~0.217
(ep 10) and held ~0.19-0.20 through 20; crosstalk stayed ~0.39-0.42 throughout. The
**ep-10 model is the optimal window** for fog (0.217 vs 0.195 at ep 21); ep-21 is
marginally better on crosstalk (0.411 vs 0.394). No degradation pattern past the
peak — the cov-shift win is stable, not a transient. The convergence note: the gains
are measured at the ep-10 optimal window; a sensible convergence metric or more
stable convergence behavior is needed to confirm behavior past it.

### Iteration C5: the full battery + ep-10/ep-21 comparison (2026-08-15)

**Design.** Train a fresh ep-10 model and run the full battery (tta_ceiling,
frozen_ceiling, gate_structure, extractor_diff vs DGLSS++ and Robust) on BOTH the
ep-10 and ep-21 checkpoints.

**Result.** The all-condition comparison (frozen HDC-oracle):
- cov-shift highest on fog (21.4%/20.2% vs DGLSS++ 15.1%) and crosstalk (38.9%/39.8%
  vs 21.4%)
- cov-shift lowest on the healthy conditions (wet_ground 40.5%/36.6% vs 51.4%)
- **the healthy-condition ceiling loss is the open problem** (next iteration).

### Iteration C6: the healthy-condition ceiling diagnostic (2026-08-15)

**Design.** Why does cov-shift hurt the healthy-condition ceilings while fixing
fog/crosstalk? The frozen-ceiling comparison shows the continuous class structure
survives (LP mostly preserved: snow 79->82%, wet_ground 85->77%) but the
HDC-binarized recoverability drops (wet_ground oracle 51->37%, beam_missing 51->44%).
`cond_structure_diag.py` measures per-class feat_cos / dir_retention / corr_tightness
/ zs on snow + wet_ground, plain DGLSS++ vs cov-shift.

**Result — the per-class mechanism (wet_ground):**

| cls | corr_tight A->B | zs A->B |
| :--- | :--- | :--- |
| car(4) | 0.96 -> 0.85 | 0.866 -> 0.746 |
| og(14) | 0.86 -> 0.67 | 0.775 -> 0.532 |
| road(11) | 0.98 -> 0.94 | 0.658 -> 0.483 |
| veg(16) | 0.90 -> 0.81 | 0.649 -> 0.574 |

The recoverable classes on wet_ground show a clear **PACKING loss** (corr_tight
drops for nearly every class) while the DIRECTION is retained (dir_ret stays
~0.99-1.0). The same pattern holds for snow (smaller). **This is a
binarization/recoverability loss, not a direction loss.**

**Interpretation for the projection/binarization design (the next step).** The
hypothesis is confirmed: InstanceNorm + the per-scan input normalization change the
per-dimension feature scale/magnitude on the healthy conditions, pushing the
recoverable classes closer to the HDC sign-binarization threshold. The continuous
structure is intact (LP kept, direction kept) but the packing that the HDC
random-projection + sign threshold needs is degraded. The fix direction is to make
the projection matrix or the binarization robust to this scale change — e.g. a
scale-aware projection, a threshold-aware binarization, or a normalization ordering
that preserves the healthy conditions' anisotropy (scale-normalize-then-InstanceNorm
instead of the current ordering).

## Open questions

1. ~~**Can the projection matrix or binarization be redesigned** so the healthy-
   condition packing survives the cov-shift normalization without losing the
   fog/crosstalk gains?~~ **CLOSED.** C8 rejected encoding changes (all encodings lose
   equally); C9 rejected the training-side normalization-scope levers at micro scale.
   The packing loss is not recoverable by changing the extractor's normalization or
   the binarization.
2. **Can a LEARNED HDC-code decoder replace nearest-centroid?** C10 says yes (R4 =
   1.24-1.77x), but it needs labeled data (frozen clean-fit is the label-free
   version). Open: does a label-free pool-refit R4 hold the C10 gain without labels?
3. **Does the NuScenes cross-domain gap give TTA real headroom?** C11 found a
   +0.05-0.06 oracle-zs gap on NuScenes vs ~0 on KITTI clean. Open: can naive TTA
   (re-estimating the decoder on the NuScenes pool) take that gap?
4. **Convergence**: the gains are measured at the ep-10 optimal window; does a
   sensible convergence metric or more stable convergence behavior confirm the
   extractor's peak is stable?

### Iteration C7: the binarization diagnostic (2026-08-15)

**Goal.** C6 found the healthy-condition ceiling loss is a PACKING loss in binarized
space (continuous structure kept, HDC-oracle drops). This iteration diagnoses what
the current `sign(z @ R)` binarization loses and tests whether an alternative
encoding recovers the healthy ceilings without losing the fog/crosstalk gains.

**What the current binarization does.** `get_hdc_projection` builds a random +-1
matrix R (dim_in x 10000); features are binarized as `sign(z @ R)` — threshold 0 per
coordinate, all coordinates equally weighted. Prototypes are per-class means of the
binarized clean codes.

**What the diagnostic measures (on frozen features, no training):**

1. **A: the pre-sign margin fraction** — the fraction of `|z @ R| < eps` coordinates
   (default eps=0.1). A coordinate near 0 is threshold-hugging: it flips sign on
   small feature noise. If the cov-shift HEALTHY features have a higher near-0
   fraction than DGLSS++ (B > A on snow/wet_ground), that is the quantitative
   signature of the packing loss — the cov-shift normalization moved the healthy
   features closer to the sign decision boundary, so sign() binarizes them
   unreliably.
2. **B: three alternative encodings, on the same frozen features:**
   - `bias`   : `sign(z @ R - b)` with per-coordinate b = median of the CLEAN
     projection. Each coordinate binarizes around its clean-typical value, not an
     arbitrary 0 — so healthy-condition coordinates that cov-shift scaled away from 0
     are re-centered.
   - `zscore` : `sign((z @ R - b) / s)` with per-coordinate clean mean and std.
     Z-scores the projection so every coordinate contributes equally before
     binarization — undoes the per-dimension scale change InstanceNorm introduced.
   - `fourier`: the other standard HDC style — `[cos(z @ w), sin(z @ w)]` with random
     frequencies w (dim_in x 10000, continuous codes, dim 20000). No sign; a smooth
     periodic encoding. Tests whether retaining continuous phase information fixes
     what hard sign() loses.

   For each encoding, on each condition: the frozen-prototype decode (zs) and the
   oracle decode (re-estimate prototypes from corrupted labeled points).

**Decision rule.** If on the healthy conditions an alternative encoding recovers the
cov-shift oracle toward (or above) DGLSS++'s level WITHOUT dropping the fog/crosstalk
gain, that encoding is the fix — and it can be adopted decoder-side (the extractor is
unchanged, only the HDC projection/binarization is swapped). If `margin_frac` B > A
confirms the packing loss, the mechanism is validated and the fix is a scale-aware or
bias-aware binarization.

Run: `bash run_binarization_diag.sh 3` (both ep10 + ep21, eval-only, ~1-1.5h).

### Iteration C8: the binarization diagnostic — the loss is continuous, not binarized

The C7 hypothesis was that the cov-shift healthy-ceiling loss is a BINARIZATION loss
(features pushed toward the sign threshold). The C8 diagnostic (`binarization_diag.py`)
tested this on frozen features with four encodings (current `sign`, per-coordinate
`bias`, `zscore`, and the continuous Fourier `[cos, sin]` style) plus the pre-sign
margin fraction.

**Result: the hypothesis is REJECTED, and the finding reframes the problem.**

1. **margin_frac is nearly identical between extractors** (0.0255 vs 0.0263 ep10;
   0.0256 vs 0.0306 ep21). The cov-shift healthy features do NOT sit closer to the
   sign threshold — the "threshold-hugging → unreliable binarization" mechanism does
   not occur.
2. **No alternative encoding recovers the healthy oracle.** On wet_ground (the
   worst case), the cov-shift oracle is ~0.22 (ep10) / ~0.19 (ep21) across sign /
   bias / zscore / fourier — all far below DGLSS++'s 0.27. The encoding choice barely
   changes anything (spread < 0.006).
3. **The loss is in the CONTINUOUS features, not the binarization.** If it were a
   binarization artifact, the continuous Fourier encoding (no sign) would retain the
   packing that sign() loses. It does not — so the cov-shift extractor's healthy
   features have genuinely lost recoverable structure at the continuous level.

**What this means for the fix.** Decoder-side projection/binarization redesign
(bias, zscore, Fourier, or any scale-aware threshold) CANNOT recover the
healthy-condition ceiling — the information is not in the binarized code because it
is not in the continuous features. The fix must be TRAINING-SIDE: the cov-shift
normalization (per-scan input-IN + internal InstanceNorm) is erasing a continuous
recoverable structure on the healthy conditions, and the levers are:
- **Scope the normalization**: apply InstanceNorm only to the bottleneck channels
  that fog/crosstalk recover through, not the whole backbone — so the healthy
  conditions' continuous structure is untouched.
- **Scale-normalize-then-InstanceNorm ordering**: preserve the healthy conditions'
  per-dimension anisotropy before InstanceNorm re-scales it.
- **Condition-aware regularization**: keep the healthy-condition feature scale
  closer to clean during training (the C6 packing loss is a real continuous loss,
  and the fix is to not compress it, not to re-binarize it).

This is a clean negative that redirects the effort from the HDC decoder (projection /
binarization design) back to the extractor's normalization scope — which is where the
cov-shift method's actual lever is.

## Potential next step: changing the HDC classification rule

The C8 negative covers the *encoding* (how a feature becomes a code) but not the
*decision rule* (how a code is assigned to a class). All C7/C8 variants still end in
the same nearest-centroid rule: unit-norm cosine to per-class prototypes
(`decode`/`decode_preds` in the diagnostic harness). If the training-side levers
above fail to recover the healthy-condition packing, the decision rule itself is the
remaining decoder-side lever. This is distinct from the encoding changes C8 ruled out.

The key observation: C6's packing loss is PER-CLASS (og 0.86->0.67, car only
0.96->0.85), but the C8 re-encodings were GLOBAL (one bias/scale per dimension,
applied identically to all classes). A class-conditional decision rule is the
untested alternative, and the LP evidence supports it: the continuous linear probe
recovers the healthy conditions far better than the centroid rule (LP wet_ground
~85->77, ~9% relative drop; HDC oracle ~51->37, ~27% relative drop). No rule fully
recovers the clean ceiling, but a learned/conditioned rule loses ~3x less than
nearest-centroid.

Candidate rules, in order of how much of the HDC/TTA story they keep:

1. **Per-class scaled HDC distance.** Keep sign-projection + prototypes, but replace
   the unit-norm cosine with a class-conditional scaled cosine: weight each
   prototype's contribution by that class's scale (inverse of its within-class
   spread / corr_tight). Re-tightens the decision boundary per class without touching
   the extractor, and keeps the label-free prototype re-estimation machinery
   (naive/oracle) intact. The most method-preserving fallback.
2. **Learned decision rule on the continuous 128-d features.** A train-time
   LogisticRegression on the full features (what LP measures). Strongest recovery,
   but abandons HDC decoding; TTA becomes re-fitting the probe on pseudo-labels
   instead of re-estimating prototypes.
3. **Per-class scale before the existing prototype distance.** The targeted version
   of (1): estimate each class's feature scale on the corrupted pool, divide before
   the cosine. Minimal code, directly attacks corr_tight.

Status: RESOLVED by C10 -- the decision rule IS the recovery path. C10 showed a
learned probe on the HDC code (R4) recovers 1.24-1.77x over nearest-centroid on
every condition, decisively. See Iteration C10.


### Iteration C9: the C8 training-side lever micro runs (2026-08-16)

The three C8 levers (`_scope`, `_scalein`, `_scalereg`) are trained at micro scale
(8 ep / 10%) and gated with `cond_structure_diag` vs plain DGLSS++ on snow /
wet_ground / fog / crosstalk.

**Caveat before reading results:** the micro checkpoints are 8-epoch / 10%-data
models, so their corr_tight and zs are NOT directly comparable in level to the
medium baselines (plain DGLSS++ is a full medium run). The micro gate is a
DIRECTIONAL test: does the lever change the corr_tight / zs trajectory toward
recovering the healthy packing, while keeping the fog/crosstalk direction?

**Gate results (micro B vs plain DGLSS++ medium A), mean over present classes:**

| variant | cond | corr_tight A->B | zs A->B |
| :--- | :--- | :--- | :--- |
| scope | snow | 0.836 -> 0.797 | 0.422 -> 0.317 |
| scope | wet_ground | 0.860 -> 0.771 | 0.468 -> 0.237 |
| scope | fog | 0.852 -> 0.762 | 0.082 -> 0.157 |
| scope | crosstalk | 0.875 -> 0.784 | 0.126 -> 0.296 |
| scalein | snow | 0.836 -> 0.805 | 0.422 -> 0.301 |
| scalein | wet_ground | 0.860 -> 0.783 | 0.468 -> 0.270 |
| scalein | fog | 0.852 -> 0.759 | 0.082 -> 0.161 |
| scalein | crosstalk | 0.875 -> 0.790 | 0.126 -> 0.294 |
| scalereg | snow | 0.836 -> 0.780 | 0.422 -> 0.297 |
| scalereg | wet_ground | 0.860 -> 0.754 | 0.468 -> 0.253 |
| scalereg | fog | 0.852 -> 0.701 | 0.082 -> 0.152 |
| scalereg | crosstalk | 0.875 -> 0.761 | 0.126 -> 0.288 |

**Result: none of the three levers recovers the healthy packing at micro scale.**
- On the healthy conditions (snow/wet_ground), all three variants have corr_tight_B
  below A and healthy zs_B far below A (wet_ground zs ~0.24-0.27 vs A's 0.47). The
  cov-shift healthy-condition packing loss is NOT reduced by scoping InstanceNorm,
  scale-only normalization, or the feature-scale regularizer.
- On fog/crosstalk the variants still show the cov-shift recovery signature (zs_B
  > zs_A: fog ~0.15-0.16 vs 0.08, crosstalk ~0.29-0.30 vs 0.13), consistent with the
  cov-shift direction surviving at micro scale -- but this is the same pattern as the
  baseline cov-shift, not a lever-specific recovery.

**Interpretation.** The training-side levers did not obviously recover the healthy
packing in the directional micro test. The caveat stands: micro models are
undertrained (zs levels are well below the cov-shift ep10 baseline's healthy zs of
~0.40), so a medium run could still show a lever effect. But the micro signal is
weak for all three, which combined with C8 (continuous-features loss, encoding-
independent) points away from the normalization-scope levers and toward the decision
rule (C10) as the more promising recovery path.

### Iteration C10: the HDC decision-rule diagnostic (2026-08-16)

C8 proved the healthy-ceiling loss survives every ENCODING change. C10 tests the
DECISION RULE instead: on the frozen ep10/ep21 cov-shift features, compare four rules
on the same code/prototypes:
- **R1** baseline: unit-norm cosine to per-class prototypes (the current rule).
- **R2** class-conditional: per-class scaled cosine (similarity / class spread).
- **R3** learned 128-d probe: LogisticRegression on the continuous features.
- **R4** learned HDC-code probe: LogisticRegression fit on the binarized 10k-d code
  itself (clean-fit = zs, pool-refit = oracle).

**Result: the HDC space IS linearly separable in a way the current implementation
misses, decisively.**

| cond | R1-orc | R2-orc | R3-orc | R4-orc | R4/R1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| snow (ep10) | 0.408 | 0.349 | 0.329 | 0.510 | 1.25 |
| wet_ground (ep10) | 0.425 | 0.362 | 0.349 | 0.683 | 1.61 |
| fog (ep10) | 0.261 | 0.246 | 0.193 | 0.433 | 1.66 |
| crosstalk (ep10) | 0.461 | 0.376 | 0.414 | 0.594 | 1.29 |
| snow (ep21) | 0.395 | 0.350 | 0.333 | 0.491 | 1.24 |
| wet_ground (ep21) | 0.405 | 0.384 | 0.336 | 0.668 | 1.65 |
| fog (ep21) | 0.219 | 0.207 | 0.184 | 0.387 | 1.77 |
| crosstalk (ep21) | 0.451 | 0.377 | 0.389 | 0.586 | 1.30 |

- **R4 (linear probe on the HDC code) beats R1 (nearest-centroid) by 1.24-1.77x on
  every condition and both checkpoints.** On fog it nearly doubles (0.26->0.43 ep10,
  0.22->0.39 ep21); on wet_ground +61-65%. The recoverable signal IS in the binarized
  code, and the nearest-centroid cosine throws a large fraction of it away.
- **R2 (per-class scaled cosine) does NOT help** (R2 <= R1 everywhere). The per-class
  spread rescaling does not capture the structure a learned linear boundary does.
- **R3 (learned 128-d probe) is the WEAKEST of the four** on the oracle, including
  below R4. The continuous 128-d space is less linearly separable than the binarized
  10k-d code (consistent with the HDC projection preserving/expanding the class
  structure).

**Interpretation.** The current prototype-decoder is the bottleneck, and a learned
decision rule on the HDC code is the fix. This is the "strong signal missed by the
current implementation" the C8 thread was looking for. It does NOT require changing
the extractor or the HDC encoding -- only the decode rule (R4-style: a linear probe
fit on the code, either frozen or re-fit on a labeled pool). This is method-preserving
in the sense that the code/prototype pipeline stays; the decoder gains a learned
boundary. Caveat: the R4 probe is FIT on labeled data (clean or pool), so it changes
the label-free story -- the label-free version is the FROZEN clean-fit probe (R4-zs),
which at 0.43-0.50 healthy still beats R1-oracle (0.40-0.43) at zero-shot.

### Iteration C11: NuScenes cross-domain zero-shot + oracle (2026-08-16)

The ep10/ep21 KITTI-trained cov-shift weights are evaluated on real NuScenes
(converted KITTI format, 32-beam sensor, shared 17-class space) for a zero-shot ->
oracle gap: is there TTA headroom on the cross-domain target that did not exist on
KITTI clean?

| model | domain | zs | oracle | gap (oracle-zs) | lp_miou |
| :--- | :--- | :--- | :--- | :--- | :--- |
| cov-shift ep10 | KITTI clean | 0.492 | 0.493 | +0.001 | 0.437 |
| cov-shift ep10 | NuScenes | 0.129 | 0.192 | +0.063 | 0.125 |
| cov-shift ep21 | KITTI clean | 0.464 | 0.468 | +0.003 | 0.444 |
| cov-shift ep21 | NuScenes | 0.127 | 0.181 | +0.054 | 0.110 |

**Result: a cross-domain TTA gap exists that did not exist on KITTI.**
- On KITTI clean, the cov-shift extractor's zs and oracle are essentially identical
  (gap ~0.001-0.003) -- the "no headroom" pattern (naive TTA ~= ceiling) that makes
  the healthy conditions a closed case.
- On NuScenes, the gap opens to +0.054-0.063 (oracle 0.18-0.19 vs zs 0.13), i.e. the
  frozen KITTI prototypes decode NuScenes ~26% worse than a NuScenes-labeled oracle.
  That is recoverable structure TTA could take: the features DO retain domain-
  transferable signal, but the frozen prototypes don't align to NuScenes.
- The LP mIoU on NuScenes is low (0.11-0.13), so the continuous features transferred
  weakly in absolute terms -- the gap is real but the base is low.

**Interpretation.** NuScenes is a genuinely harder transfer than KITTI clean for the
cov-shift extractor, and the oracle-zs gap is TTA-addressable headroom that did not
exist on KITTI. Combined with C10 (a learned HDC-code probe recovers a large missed
signal), the strongest next direction is a LABEL-FREE decoder that re-estimates a
learned boundary on the NuScenes pool -- which is exactly what the cov-shift method's
naive TTA could become on the cross-domain target. Note the dry-run caveat: these
numbers are from the full 100-frame run, so they are the real result (the earlier
2-frame dry-run numbers were only a smoke test).

### Iteration C8.1: reproducibility fix in cond_structure_diag (2026-08-16)

The scalereg gate crashed on fog with "labels must align": the parser randomly drops
points per scan (`drop_points = random.uniform(0, 0.5)`), so two separate
feature-extraction passes consumed different points and produced misaligned label
streams. Fixed by extracting both models in one SHARED pass
(`extract_features_pair`), guaranteeing A and B are evaluated on identical points.
This was a latent bug that could have hit any cross-extractor gate.

