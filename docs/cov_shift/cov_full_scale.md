# Cov-shift at Full Scale: Cross-Dataset Ceiling & Healthy-Condition Collapse

Tracking the full-scale comparison of the cov-shift DGLSS++ extractor against the
base DGLSS++ extractor on KITTI$\rightarrow$KITTI-C and NuScenes$\rightarrow$NuScenes-C, and
the search for what, *in the feature extractor itself*, causes
(1) the ceiling collapse on NuScenes-C and (2) the healthy-condition collapse on
KITTI-C -- so we can decide whether a new loss term is needed or an existing one
must change.

---

## Background: the corrected full-scale extractor comparison

### The shared-ARCH bug that invalidated the first comparison

The original full-dataset harness (`al_full_dataset_diag.py`) passed **one shared
`ARCH` dict** to every extractor. `GenTrainer.__init__` **mutates
`ARCH["train"]["twobranch"]` in place** (sets `norm`, `input_in`, `norm_channels`
for the method). Because the cov-shift method (`supcon_vib_dglsspp_inputin_in_chan`)
sets `input_in=True, norm_channels=(0,4)` and ran **before** DGLSS++/Robust in the
same process, the DGLSS++/Robust models were constructed with the cov-shift
input-normalization architecture:

- true DGLSS++ build = **6.796804 M** params (all standalone dglsspp logs)
- polluted build       = **6.786436 M** params (= the cov-shift arch)

and the true 6.796804M checkpoints loaded with a partial `strict=False` load. The
fix is a per-extractor `ARCH` deep-copy (now in `al_full_dataset_diag.py` and the
probe script). Verification: a single-extractor DGLSS++ run prints 6.796804 M and
reproduces the clean `al_nuscenes_c_dglsspp.json` numbers (fog 0.114/0.526).

### Corrected numbers (full dataset, R4 linear probe, ep-10)

KITTI$\rightarrow$KITTI-C (all frames of seq 08, ~300M pts/cond, 200k clean fit / 400k
pool, spectral-exact ridge):

| condition | cov-shift zs | cov-shift ceil | DGLSS++ zs | DGLSS++ ceil |
| :--- | :--- | :--- | :--- | :--- |
| fog | **32.2** | **36.9** | 9.7 | 25.3 |
| crosstalk | **47.7** | **49.1** | 11.8 | 29.1 |
| snow | 48.2 | 49.5 | **54.3** | **57.5** |
| wet_ground | 29.7 | 41.9 | **49.2** | **57.0** |
| incomplete_echo | 42.1 | 43.7 | **48.2** | **48.8** |
| beam_missing | 48.9 | 48.7 | **58.3** | **58.7** |
| motion_blur | 45.2 | 45.8 | **54.8** | **56.6** |
| cross_sensor | 42.0 | 44.6 | **46.9** | **49.4** |
| **mean** | 42.0 | 45.0 | 41.6 | **47.8** |

NuScenes$\rightarrow$NuScenes-C (NuScenes-trained extractors, 32-beam, heavy severity,
150 val scenes / 6019 keyframes):

| condition | cov-shift zs | cov-shift ceil | DGLSS++ zs | DGLSS++ ceil |
| :--- | :--- | :--- | :--- | :--- |
| fog | **14.4** | 40.4 | 11.4 | **52.6** |
| crosstalk | **19.1** | 50.1 | 11.9 | **53.3** |
| snow | **22.4** | 62.1 | 13.5 | **62.4** |
| wet_ground | **22.4** | 62.0 | 14.3 | **66.1** |
| incomplete_echo | **22.2** | **48.9** | 13.3 | 48.1 |
| beam_missing | **22.5** | 58.8 | 15.0 | **60.2** |
| motion_blur | **20.6** | 52.9 | 13.5 | **54.9** |
| cross_sensor | **18.6** | **49.4** | 13.1 | 47.2 |
| **mean** | **20.3** | 53.1 | 13.3 | **55.6** |

### What the corrected comparison shows

The cov-shift input normalization is a **trade, not a free win**:

- **It rescues the collapsed conditions.** cov-shift wins zero-shot on every
  NuScenes-C condition and wins fog/crosstalk on KITTI-C by 12-20 points
  (KITTI fog ceil 36.9 vs 25.3, crosstalk 49.1 vs 29.1).
- **But it compresses the healthy conditions.** DGLSS++ beats cov-shift at the
  ceiling on all 6 healthy KITTI-C conditions (snow 57.5 vs 49.5, wet_ground 57.0
  vs 41.9, beam 58.7 vs 48.7) and on 5/8 NuScenes-C conditions.
- **The mean cov-shift ceiling edge DISAPPEARS.** KITTI-C: 45.0 vs 47.8 (DGLSS++
  wins). NuScenes-C: 53.1 vs 55.6 (DGLSS++ wins). Zero-shot is even on KITTI-C
  (42.0 vs 41.6), cov-shift ahead on NuScenes-C (20.3 vs 13.3).

The earlier claim that "cov-shift beats DGLSS++ by 0.17-0.30 on the healthy
conditions" was an artifact of the ARCH leak (cov-shift's input normalization
applied to DGLSS++ features crushed them).

**The two symptoms to explain, both in the extractor itself:**

1. **NuScenes-C ceiling collapse.** cov-shift's recoverable ceiling is BELOW
   DGLSS++ on NuScenes-C (mean 53.1 vs 55.6), even though its zero-shot is much
   better. The labeled ceiling $W^*$ (fit on a 400k corrupted-pool reservoir)
   cannot recover as much as DGLSS++'s -- cov-shift's per-scan normalization on
   channels {0,4} appears to cap the recoverable structure under 32-beam
   corruptions.
2. **KITTI-C healthy-condition collapse.** cov-shift's ceiling on the healthy
   conditions is 8-15 points below DGLSS++ (snow 49.5 vs 57.5, wet_ground 41.9 vs
   57.0). The normalization trades healthy headroom for fog/crosstalk -- the same
   trade seen at the prototype level (Section 2 of the README).

Both point at the cov-shift input normalization (per-scan mean/std on range +
remission + internal InstanceNorm) as the mechanism: it de-biases the collapsed
conditions but suppresses the variance the labeled ceiling needs to re-draw
boundaries on the healthy / cross-domain conditions.

---

## Iteration 0: what in the feature extractor causes the collapse? (2026-08-23)

**Question.** Is the ceiling/healthy loss caused by (a) the input normalization
itself (a feed-forward preprocessing choice), (b) an existing loss term that
over-regularizes the healthy directions (VIB bottleneck, LSCC class-correlation
consistency, GMSIFC cross-view alignment), or (c) a missing term -- i.e. does the
extractor need a NEW loss that explicitly preserves the recoverable healthy /
cross-domain directions while keeping the collapsed-condition fix?

**Two reframings before running anything:**

- **The healthy collapse hits ZERO-SHOT too, not just the ceiling.** cov-shift
  loses zero-shot on all 6 healthy KITTI-C conditions (snow 48.2 vs 54.3,
  wet_ground 29.7 vs 49.2 -- a 20-point deficit, incomplete 42.1 vs 48.2, beam
  48.9 vs 58.3, motion 45.2 vs 54.8, cross 42.0 vs 46.9). The even means (42.0
  vs 41.6) hide this: cov-shift's zero-shot advantage is confined to
  fog/crosstalk. So the normalization degrades healthy-condition FEATURES, not
  merely their recoverable headroom.
- **Part of the deficit is clean-inherited.** Base DGLSS++ also wins on CLEAN
  (clean HDC mIoU 53.0 vs cov-shift 47.2). Any healthy-condition corruption
  deficit must be decomposed into (i) the clean-capacity gap and (ii) the extra
  gap the corruption induces. If (ii) ~ 0, the normalization is simply a
  capacity/regularization cost and the corruption story is only about
  fog/crosstalk rescue.

**Setup.** Full-scale harness (`al_full_dataset_diag.py`, deep-copied ARCH), the
four checkpoints we already have: cov-shift ep10/ep21 (KITTI-trained), base
DGLSS++ (KITTI-trained), cov-shift NuScenes (`nusc_covshift_21ep`), base DGLSS++
NuScenes (`nusc_dglsspp_21ep`). The `probe_nusc_c_dglsspp_vs_covshift_diag.py`
machinery (per-class frozen/ceiling/gap, pool support, code/feat nearest-mean
separability, residual norm, mean-shift gauge) already exists and is fixed.

**Diagnostics (decisive splits):**

1. **Clean-baseline decomposition (new, cheapest, do first).** Evaluate both
   extractors on the CLEAN val stream with the full harness (add `clean` as a
   pseudo-condition: frozen W0 / ceiling W* from a clean pool). Decompose each
   healthy-condition deficit into clean-gap + corruption-interaction:
   `deficit(corr) = [dgl_corr - cov_corr] - [dgl_clean - cov_clean]`. If the
   interaction term is small, the normalization is a flat capacity cost; if
   large, it specifically fails under shift-free inputs' corrupted variants.
   Note the harness already streams clean seq-08 for the W0 fit, so this is
   nearly free.

2. **Which classes lose, and where?** Per-class recoverable map (already in the
   flip probe). On NuScenes-C fog, cov-shift's biggest gaps lift classes from
   dead (unlabeled +0.716, bus/barrier/truck +0.50..+0.57 from frozen 0.00);
   majority classes lift modestly (car +0.316, driveable +0.301). Compare against
   DGLSS++'s per-class map on the SAME conditions to locate which classes'
   ceilings are structurally capped, and whether the capped set matches the
   classes that lose on KITTI-C healthy conditions.

3. **Input-statistics calibration (new).** Measure the per-scan mean/std of
   channels {0,4} (range, remission) that the input-IN divides by, for: clean
   KITTI, each KITTI-C condition, clean NuScenes, each NuScenes-C condition.
   Tests whether the normalization constant-behavior is calibrated to KITTI
   statistics and mis-engaged on NuScenes (32-beam density + different remission
   distribution). If the normalized outputs differ systematically in scale across
   domains, part of the NuScenes-C ceiling cap is domain miscalibration of the
   normalization, not feature information loss.

4. **Residual structure $\|W^* - W_0\|/\|W_0\|$ + conditioning (extended).**
   Keep the relative-residual comparison, but add: eigenvalue spectrum and
   condition number of $S = X^\top X$ per condition, and a ridge $\lambda$ sweep
   ($10^{-4}, 10^{-3}, 10^{-2}$) on the pool fits. If cov-shift's ceiling is
   $\lambda$-sensitive or its $S$ is ill-conditioned, part of the "ceiling cap"
   is a regularization/conditioning artifact of the binarized code rather than
   missing information. (Caveat: the residual is measured against each
   extractor's own $W_0$, so cross-extractor residual comparisons are suggestive,
   not decisive.)

5. **Code-vs-raw-feature separability + binarization health (extended).**
   `sep_code_*` vs `sep_feat_*` nearest-mean recall (exists). Add bit-balance
   (fraction of +1 per code coordinate) and the pre-sign margin distribution
   (|x . p_i| near zero = sign flips under small perturbations). If cov-shift's
   raw features separate the healthy classes but the code does not, or its
   features sit closer to the projection hyperplanes, the loss is in the HDC
   projection/binarization step, not the network.

6. **Variance / effective-rank of code and raw features.** Participation ratio
   and per-class variance for cov-shift vs DGLSS++ on the same condition. Direct
   test of the compression hypothesis. Pair with diagnostic 3: if variance is
   lower ONLY on channels the input-IN touches, the mechanism is confirmed at the
   input stage.

7. **Normalization-lever ablation (refined).** cov-shift changes TWO things vs
   base DGLSS++, configured independently in `GenTrainer`:
   (i) `input_in=True` (+ `norm_channels=(0,4)`): per-scan InstanceNorm on
   range+remission at the input, applied at train AND eval;
   (ii) `norm='in'`: internal BatchNorm replaced by InstanceNorm throughout the
   twobranch trunk.
   Disentangle them. Eval-only forward ablations have a caveat: the network was
   trained WITH these active, so disabling at eval creates a train/eval mismatch
   -- interpret "disable input-IN at eval" as testing inference-time gating, NOT
   as the counterfactual model. The decisive version of (i)/(ii) is training
   variants (diagnostic 8); the cheap eval-only ablation still tells us whether a
   data-dependent GATE (normalize when shift detected, skip when healthy) could
   work at deployment, and the existing AL-gauge signal (`mean_shift_cos`) is the
   candidate gate trigger.

8. **Loss-term attribution (training ablations).** Train NuScenes variants with
   one change at a time, measure the corrected ceiling metric on NuScenes-C +
   the healthy conditions: (a) input-IN only (no internal IN), (b) internal IN
   only (no input-IN), (c) cov-shift without VIB, (d) LSCC weight reduced,
   (e) GMSIFC weight reduced, (f) cov-shift + a variance-floor / healthy-direction
   preservation regularizer. This separates "an existing term over-regularizes"
   from "a new term is missing", and pins WHICH lever carries the fog/crosstalk
   rescue vs the healthy cost.

9. **W0-source control for the NuScenes comparison (new, cheap).** All current
   NuScenes numbers fit $W_0$ on KITTI clean seq-08 (64-beam) -- a cross-domain
   probe fit. Re-fit $W_0$ on nuScenes-clean val frames and re-measure: if
   cov-shift's NuScenes-C zero-shot lead shrinks/grows, part of the transfer
   story is probe-source interaction, not pure feature quality. (Ceiling $W^*$ is
   unaffected -- it is fit in-domain.)

10. **R1-vs-R4 headroom decomposition (existing data).** Compare proto_ceiling
    (R1, decoder-independent geometry) with linear_ceiling (R4) per condition per
    extractor: if cov-shift's R1 ceiling matches DGLSS++'s but its R4 ceiling is
    lower, the loss is in linear accessibility of the code; if both are lower,
    the geometry itself is poorer.

**Secondary (optional):** severity sweep (moderate/light on NuScenes-C) to see if
the ceiling cap scales with corruption strength; per-class $W$-column analysis
(norms, cosine to $W_0$ columns) for the capped classes.

**Note on pool composition:** verified extractor-invariant (identical reservoir
indices, +-1 point) since the pool samples the same corrupted scans with the same
seed regardless of extractor. It explains CONDITION-level differences (e.g. why
motorcycle cannot be recovered: ~180 points in the pool), never EXTRACTOR
differences. Use it as context for interpreting per-class ceilings only.

**Decision rule.** If diagnostic 1 shows most of the healthy deficit is
clean-inherited AND diagnostic 7's gated normalization recovers the
cross-domain/healthy ceiling without losing fog/crosstalk, the fix is in the
**forward normalization** (channel-restrict or gauge-gated), no new loss. If the
deficit is corruption-specific or the gate fails, move to **loss attribution**
(diagnostic 8): reduce the over-regularizing term or add a variance/healthy-
direction preservation term.

**Verdict / outcome:** (to fill in)

---

## Reproducibility

- Harness: `robust_diagnostic/al_full_dataset_diag.py` (deep-copied `ARCH`),
  runner `run_al_full_dataset.sh` (`EXTRACTORS=...` override).
- Flip probe: `robust_diagnostic/probe_nusc_c_dglsspp_vs_covshift_diag.py`,
  runner `run_probe_nusc_c_flip.sh`.
- Checkpoints: `logs/ep10_supcon_vib_dglsspp_inputin_in_chan/...` (KITTI cov),
  `logs/supcon_vib_dglsspp` (KITTI DGLSS++), `logs/nusc_covshift_21ep` (NuScenes
  cov), `logs/nusc_dglsspp_21ep` (NuScenes DGLSS++).
- Results: `al_full_dataset_ep10_custom.json` (corrected DGLSS++ KITTI-C),
  `al_nuscenes_c.json` / `al_nuscenes_c_dglsspp.json` (NuScenes-C),
  `probe_nusc_c_flip_ep10.json` (per-class + residual + separability).
