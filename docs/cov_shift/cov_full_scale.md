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

**Verdict / outcome:** (see below)

## Iteration 0 RESULTS: mechanism probe (`probe_covshift_mechanism_diag.py`, 2026-08-23)

Full-scale run, all 4 extractors x 8 conditions (pool 400k, clean fit 200k,
spectral-exact ridge, pool excluded from val; `probe_covshift_mechanism_ep10.json`).

### D1: clean baseline -- the healthy deficit is MOSTLY clean-inherited

| extractor | clean frozen | clean ceiling | clean gap |
| :--- | :--- | :--- | :--- |
| cov_kitti | 0.520 | 0.530 | +0.009 |
| dgl_kitti | **0.640** | **0.648** | +0.009 |
| cov_nusc | 0.332 | 0.334 | +0.002 |
| dgl_nusc | 0.303 | 0.306 | +0.003 |

Base DGLSS++ beats cov-shift by 0.12 on clean KITTI (0.640 vs 0.520) -- a LARGE
clean-inherited gap that needs no corruption at all. The healthy-condition
corruption deficits (snow 0.545/0.575 vs cov 0.419/0.429 etc.) are almost fully
explained by this clean baseline difference: **the normalization is a flat
capacity/regularization cost, not a corruption-interaction failure.** The
"healthy-condition collapse" framing is wrong -- it is the clean-capacity gap
(DGLSS++ is simply a better clean extractor at R4), plus the fog/crosstalk rescue
cov-shift buys with it. The corruption INTERACTION term is small.

### D3: input-statistics calibration -- the fog/channel story is in the REMISSION variance

Per-scan channel {0,4} statistics (cov_kitti, KITTI-C):

| condition | ch0 (range) var | ch4 (remission) var |
| :--- | :--- | :--- |
| fog | **4.00** | **0.000013** |
| crosstalk | 2.09 | 0.000079 |
| wet_ground | 2.37 | 1.49 |
| snow | 2.11 | 1.45 |
| beam_missing | 2.28 | 1.34 |
| (clean ref) | ~1.0-2.4 | ~1.3-1.5 |

Fog collapses the remission channel variance to ~0 (1.3e-5) and inflates range
variance (4.0). This is the "the information to name those points is absent"
story at the input level: fog erases remission structure. The input-IN
normalization divides by per-scan stats, so on fog it is dividing near-zero
remission variance by near-zero -- the channel is not informative either way.
This confirms the collapse is an INPUT-statistics phenomenon for fog, not a
feature-space artifact.

### D4: residual + conditioning -- the residual is recoverable everywhere, so the ceiling cap is NOT a conditioning artifact

| extractor | resid_rel range | effrank (PR) range |
| :--- | :--- | :--- |
| cov_kitti | 1.03-1.21 | 4.9-6.5 |
| dgl_kitti | 1.06-1.24 | 4.7-8.6 |
| cov_nusc | 1.19-1.28 | 4.2-6.3 |
| dgl_nusc | 1.21-1.30 | 5.6-8.2 |

`resid_rel = ||W* - W0||/||W0||` is uniformly ~1.1-1.3 (the residual is as large
as the probe itself) on EVERY extractor and condition -- there is plenty of
recoverable structure. The participation-ratio effective rank is 4-8 (low,
consistent with a few dominant code directions). The ceiling differences are NOT
a "cov-shift has less recoverable residual" story: the residual is large
everywhere; the question is whether the ridge can USE it. The dgl_kitti λ-sweep
shows ceilings barely move across λ (0.001-0.01), i.e. the fit is saturated --
the residual is large but the fit cannot extract it (the C16-C23 ill-conditioning
story, not a small-residual story).

### D5/D6: code-vs-feature variance -- cov-shift is MORE compressed but not collapsed

| extractor | code_var | feat_var | bit_balance |
| :--- | :--- | :--- | :--- |
| cov_kitti | 0.84-0.94 | 0.108-0.137 | ~0.50 |
| dgl_kitti | 0.65-0.83 | 0.036-0.085 | ~0.50 |
| cov_nusc | 0.94-0.98 | 0.116-0.148 | ~0.50 |
| dgl_nusc | 0.72-0.96 | 0.023-0.109 | ~0.50 |

cov-shift code variance is HIGHER than DGLSS++ on every condition -- the
input-IN normalization does NOT compress the code (the "compression" hypothesis
is REJECTED). Instead DGLSS++ has LOWER feature variance (0.036-0.085 vs
0.108-0.137), i.e. base DGLSS++ features are MORE compact/spherical. The
binarization is balanced everywhere (bit_balance ~0.5, no degenerate bits).
The healthy-condition edge of DGLSS++ is a feature-scale/spread difference, not
a code-compression by cov-shift.

### D7: normalization gate-off -- INCONCLUSIVE (implementation did not toggle)

`ceiling_gate_off == ceiling` to 3 decimals on all 8 conditions for both cov
extractors. This is a degenerate result: setting `model.input_in = False` before
decode did not change the output, so the gate did not take effect (the attribute
toggle on the model object did not reach the forwarded module -- likely a
wrapper/reference issue in the probe, not a real "gating does nothing" finding).
**D7 must be re-run with the toggle applied to the actual forwarded module**
(`model.module` if DataParallel, or building a fresh model with
`input_in=False` and loading the checkpoint) before any conclusion. It is NOT
evidence that gating fails.

### D9: W0-source control -- the NuScenes-C "zero-shot" was a probe-source artifact

Frozen W0 fit on KITTI-clean (64-beam) vs nuScenes-clean (32-beam):

| extractor | fog frozen (KITTI W0) | fog frozen (nuScenes W0) | delta |
| :--- | :--- | :--- | :--- |
| cov_nusc | 0.136 | **0.266** | **+0.129** |
| dgl_nusc | 0.114 | **0.399** | **+0.285** |

On EVERY NuScenes-C condition, fitting W0 on in-domain nuScenes-clean instead of
cross-domain KITTI-clean raises frozen by +0.13 to +0.5 (cov_nusc mean ~+0.28,
dgl_nusc mean ~+0.37). **The README NuScenes-C zero-shot numbers (cov 20.3 /
dgl 13.3) were depressed by the cross-domain W0 fit, not by feature quality.**
With the in-domain probe, dgl_nusc frozen nearly matches its ceiling (e.g. fog
0.399 vs ceiling 0.519) -- the "huge gap" story on NuScenes-C was partly a
probe-fit artifact. The ceiling W* (fit in-domain) is unaffected, so the
ceiling comparison stands; the zero-shot comparison is contaminated.

### The verdict (Iteration 0)

1. **The healthy-condition "collapse" is a clean-capacity difference, not a
   corruption failure.** DGLSS++ beats cov-shift by 0.12 on clean (0.640 vs
   0.520). The corruption-interaction term is small. Fixing it is NOT a loss-term
   question -- it is "cov-shift pays 0.12 clean capacity to rescue fog/crosstalk,"
   and the decision is whether that trade is worth it.
2. **The compression hypothesis is rejected.** cov-shift does not reduce code
   variance; DGLSS++'s healthy edge is its lower/cleaner feature variance, not
   cov-shift crushing the code.
3. **The residual is large everywhere (resid ~1.2), so the ceiling cap is a
   ridge-extraction problem, not a missing-residual problem.** This points back
   to the C16-C28 ill-conditioning line, and supports the low-rank `W_res`
   (8-dim, well-conditioned) as the right decoder for ADA -- NOT a full-probe
   fix.
4. **The NuScenes-C zero-shot comparison must be redone with in-domain W0.**
   The cross-domain probe fit was contaminating the frozen numbers (D9).
5. **D7 (gate) is unproven** -- the mechanism probe's toggle was inert. The
   forward-gate question (whether a data-dependent input-IN gate helps) is still
   open and must be re-tested with a correct toggle.

**Decision rule outcome.** The clean-inherited deficit (1) means the "forward
normalization" hypothesis (diagnostic 7) is the right one to pursue IF the gate
is proven -- and since D7 was inconclusive, re-running it correctly is the next
concrete step. The loss-attribution route (diagnostic 8) is deprioritized: the
evidence says the healthy deficit is capacity, not a specific over-regularizing
term. For the ADA framework: the decoder should be the low-rank `W_res`
(evidence 3), and the zero-shot baseline to compare against must use in-domain
W0 (evidence 4).

### FOG-COLLAPSE CONFIRMATION: the collapse is corruption-generator-specific, not extractor-specific (`probe_fog_collapse_diag.py`, 200 frames, fog+crosstalk)

The direct test: same extractor, evaluated under BOTH corruption generators,
measuring nearest-mean recall (R1-geometry) and frozen R4 on the corrupted
stream with in-domain clean prototypes/W0. If the collapse were extractor-side,
the same extractor would collapse under both generators; if it is
generator-side, only KITTI-C fog/crosstalk collapse it.

| extractor (trained on) | dataset / cond | frozen R4 | code recall | feat recall |
| :--- | :--- | :--- | :--- | :--- |
| **dgl_nusc** (NuScenes) | KITTI-C fog | 0.068 | 0.143 | 0.148 |
| **dgl_nusc** (NuScenes) | KITTI-C crosstalk | 0.120 | 0.165 | 0.171 |
| **dgl_nusc** (NuScenes) | **NuScenes-C fog** | **0.305** | **0.575** | 0.574 |
| **dgl_nusc** (NuScenes) | **NuScenes-C crosstalk** | **0.357** | **0.609** | 0.609 |
| dgl_kitti (KITTI) | KITTI-C fog | 0.104 | 0.268 | 0.263 |
| dgl_kitti (KITTI) | NuScenes-C fog | 0.104 | 0.236 | 0.242 |
| cov_kitti (KITTI) | KITTI-C fog | 0.293 | 0.513 | 0.514 |
| cov_kitti (KITTI) | KITTI-C crosstalk | 0.473 | 0.639 | 0.646 |
| cov_kitti (KITTI) | NuScenes-C fog | 0.115 | 0.311 | 0.307 |
| cov_nusc (NuScenes) | NuScenes-C fog | 0.229 | 0.486 | 0.478 |
| cov_nusc (NuScenes) | NuScenes-C crosstalk | 0.332 | 0.595 | 0.592 |

**The same NuScenes-trained DGLSS++ collapses on KITTI-C fog (recall 0.143, near
the ~0.06 random baseline) but is NOT collapsed on NuScenes-C fog (recall 0.575).**
The collapse is therefore a property of the KITTI-C fog/crosstalk GENERATOR, not
of the DGLSS++ extractor or of fog-in-general. The two fog generators produce
different input statistics (KITTI-C fog: remission var ~0, range var 4.0;
NuScenes-C fog: range var ~5000, remission kept -- D3), and KITTI-C's fog
destroys the feature geometry while NuScenes-C's leaves it intact.

Additional reads:
- **cov_kitti on KITTI-C fog is NOT collapsed** (recall 0.513): the cov-shift
  input normalization rescues KITTI-C fog exactly as designed -- this is the
  extractor's home-domain win. cov_kitti on NuScenes-C fog is depressed
  (0.311) only because it is a KITTI-trained extractor seeing cross-domain data,
  not a normalization failure.
- **R1-geometry (recall) and R4-linear (frozen R4) collapse TOGETHER** on
  KITTI-C fog (dgl_nusc 0.143 recall / 0.068 R4; cov_nusc 0.575 / 0.305). The
  probe-vs-prototype choice is NOT what saves NuScenes-C -- the geometry itself
  survives there. The R4-vs-R1 divergence appears at the CEILING (labeled fit),
  not at the frozen level.
- cov_nusc NuScenes-C fog recall 0.486 < dgl_nusc 0.575: even in-domain, cov-shift
  is slightly more compressed than DGLSS++ on the healthy/fog recall -- the same
  clean-inherited capacity gap (D1) showing up at the recall level.

**Confirmation:** the "collapse on fog in KITTI but not NuScenes" story is
validated. The collapse is generator-specific; cov-shift's normalization rescues
its home-domain generator and transfers only partially.

### DIAGNOSTICS FOR IMPROVING COV-SHIFT (from the mechanism + collapse probes)

The healthy-condition deficit and the cross-domain ceiling cap both trace to the
input normalization being **calibrated to KITTI-C's generator**. Improvement
levers, in expected-value order:

1. **Gated / data-dependent normalization (the D7 re-test, corrected).** D3 shows
   the two generators are distinguishable by input statistics (KITTI-C fog =
   remission var collapse; NuScenes-C fog = range var explosion). If the
   input-IN engaged only when the corruption has the "home-domain" signature
   (remission collapse, mild range inflation), cov-shift would keep the KITTI-C
   fog rescue AND stop paying the healthy/cross-domain compression cost.
   Decisive test: build a fresh model with `input_in=False` and load the
   checkpoint (fix D7's toggle), measure healthy + NuScenes-C ceilings with the
   gate on/off. If gating off recovers the DGLSS++-level healthy ceilings while
   keeping fog, the gate is the whole fix.
2. **Channel-scope / scale-only normalization.** The `scale_only` variant
   (divide by per-scan std WITHOUT mean subtraction) preserves direction while
   absorbing magnitude shifts. D3 shows fog's range inflation is a magnitude
   shift; scale-only may rescue NuScenes-C fog without the healthy compression.
   Micro-test it on cov-shift (train + eval), compare healthy/fog/NuScenes-C
   ceilings against the current mean+std form.
3. **Capacity recovery on the healthy conditions.** D1: cov-shift pays 0.12
   clean capacity. The compression is NOT in the code (D5/D6 rejected
   compression -- cov-shift code variance is HIGHER), so the healthy loss is
   elsewhere (likely the internal InstanceNorm replacing BatchNorm, or the
   per-scan mean subtraction removing a clean-directional constant). Test:
   internal-IN vs input-IN disentangled (train input-IN with BN trunk; train
   IN trunk without input-IN) -- pins which lever costs the 0.12.
4. **Joint with the code-2000 finding.** The residual is large everywhere
   (resid ~1.2); the ceiling cap is ridge-extraction, and code-2000 improves
   conditioning (tta Iteration 2). A cov-shift trained/evaluated at 2000-d may
   recover ceiling headroom on the healthy conditions for free. Orthogonal to
   the normalization levers; adopt as the default projection if it holds at
   full scale.

**Not worth pursuing from the evidence:** a new loss term for the healthy
conditions (D1: the deficit is clean-inherited capacity, not an over-regulating
term -- diagnostic 8 is deprioritized); smarter bank selection for the AL line
(C28: diverse/uncertainty lost to random); projection-dim beyond 2000 (tta
Iteration 2: peak at 2000).

### C8 training-side lever micro test: the healthy-capacity fix is NOT in the variants

The clean-capacity gap (cov 0.520 vs dgl 0.640 clean KITTI) is the input-IN's
cost. The three training-side variants designed to fix it (`_scope`, `_scalein`,
`_scalereg`) were trained at micro scale (8 ep / 10%) and gated vs plain
DGLSS++ (`run_micro_c8fix.sh`, `micro_c8_gate_{scope,scalein,scalereg}.json`).
Support-weighted zs (micro, A = plain DGLSS++, B = variant):

| condition | scope A→B | scalein A→B | scalereg A→B |
| :--- | :--- | :--- | :--- |
| fog | 0.215 → **0.360** | 0.215 → **0.366** | 0.215 → **0.354** |
| crosstalk | 0.180 → **0.452** | 0.180 → **0.466** | 0.180 → **0.451** |
| snow | 0.691 → **0.502** | 0.691 → **0.497** | 0.691 → **0.470** |
| wet_ground | 0.664 → **0.343** | 0.663 → **0.415** | 0.663 → **0.373** |

**All three variants boost fog/crosstalk (the input-IN rescue, +0.14..+0.28) but
DEGRADE the healthy conditions (snow -0.19..-0.22, wet -0.25..-0.32).** The
healthy drop is NOT recovered by any of the training-side levers at micro scale.
The corr_tight metric (the C6 packing signal) also drops on the healthy classes
(e.g. wet car 0.964→0.775, snow driveable 0.961→0.867) -- the packing erosion
persists.

**Interpretation.** These are micro-scale (8 ep / 10%, so not converged) AND they
compare against plain DGLSS++ (a two-lever change: base + input-IN + variant), so
they cannot separate "the input-IN cost" from "the variant's own effect." But the
consistent healthy drop across all three -- including `_scope`, which was
specifically designed to keep early-layer BatchNorm -- suggests the healthy-cost
mechanism is NOT in the late-stage InstanceNorm placement (scope's target) nor
the scale/centering form (scalein/scalereg's target). Combined with D5/D6
(rejecting code compression) and D1 (clean-inherited), the clean-capacity gap
appears to live in the per-scan INPUT normalization's interaction with the early
geometry layers, not in a trainable knob the C8 variants touch.

**Follow-up needed:** a proper cov-shift-vs-`_scope` comparison (same input-IN,
only the IN placement differs) at MEDIUM/full scale would settle whether `_scope`
recovers the 0.12 clean gap relative to the current cov-shift. The micro run
here is against plain DGLSS++ and cannot answer that. This is the pending full
`_scope` run.

### Stats-thresholded input-IN gate: the tau sweep is FLAT -- the gate does NOT recover healthy capacity (`probe_ingate_stats_diag.py`, micro 200 frames)

The self-detecting gate engages per-scan input-IN only when the scan's
{range, remission} statistics deviate from the clean reference by > tau (no
corruption-type knowledge needed). Frozen mIoU, cov_kitti, KITTI-C (micro
200-frame; always-off = gate fully disabled):

| condition | always_off | tau0.1 | tau0.5 | tau1.0 | tau2.0 | spread |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.220 | 0.231 | 0.229 | 0.231 | 0.227 | 0.011 |
| crosstalk | 0.384 | 0.379 | 0.377 | 0.388 | 0.387 | 0.011 |
| snow | 0.446 | 0.442 | 0.439 | 0.449 | 0.449 | 0.010 |
| wet_ground | 0.356 | 0.338 | 0.353 | 0.357 | 0.360 | 0.021 |

**The tau sweep is flat (spread 0.01-0.02 across all thresholds) on every
condition.** Varying the engagement threshold from 0.1 to 2.0 changes nothing:
- **No healthy recovery** -- snow/wet frozen stay at the always-off level (0.44 /
  0.35) regardless of tau; the gate does NOT recover the healthy capacity.
- **No fog rescue kept** -- fog stays at ~0.23, far below the always-on reference
  (0.293 at 200 frames from the fog-collapse probe). The gate is essentially
  always-off on the conditions that matter, and the tiny tau-driven variation
  (+0.01 fog) is noise.

**Why it fails (mechanism).** The gate re-applies per-scan normalization to a
model whose weights were trained WITH input-IN always active. The train/eval
mismatch dominates: whether a given scan is normalized or not, the network
expects the input-IN-transformed distribution, so selectively normalizing by a
stats threshold does not reproduce the training-time behavior -- it just lands
somewhere between always-off and always-on, and the threshold cannot steer it.
The input-IN's healthy-cost (D1: clean 0.520 vs 0.640) and its fog rescue are
both baked into the weights at training; an eval-time forward gate cannot
separate them.

**Verdict: the eval-only stats gate is DEAD.** It does not recover healthy
capacity and does not keep the fog rescue. A gate that works must be
training-side: the model must be trained to handle BOTH normalized and
un-normalized inputs (e.g., stochastic/conditional input-IN during training), so
that at eval the gate is a clean on/off the weights support. This is a TRAINING
change, not a forward gate. Combined with the C8 variants (which also failed at
micro), the healthy-capacity fix is not reachable by either forward gating or
the existing training-side levers -- the remaining option is training a model
with explicitly conditional input-IN. `probe_ingate_stats.json`.

### Conditional input-IN training (stochastic): the gate now WORKS -- healthy recovers when OFF, rescue kept when ON (`run_micro_stoch.sh`, micro 8ep/10%)

Following the eval-only gate verdict, three models were trained with the per-scan
input-IN applied to a RANDOM SUBSET of scans at train time (`input_in_prob` in
{0.5, 0.7, 0.9}), so the weights support BOTH normalized and raw inputs. Then the
same model is evaluated with input-IN ON vs OFF (the gate). Frozen mIoU, micro
8ep/10% (`micro_stoch_gate_{stoch,stoch7,stoch9}.json`):

| prob | mode | snow | wet_ground | fog | crosstalk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| p=0.5 | ON | 0.291 | 0.145 | 0.193 | 0.275 |
| p=0.5 | OFF | **0.318** | **0.178** | 0.162 | 0.210 |
| p=0.5 | delta | **+0.027** | **+0.034** | -0.031 | -0.065 |
| p=0.7 | ON | 0.298 | 0.155 | 0.197 | 0.281 |
| p=0.7 | OFF | **0.317** | **0.186** | 0.167 | 0.215 |
| p=0.7 | delta | **+0.019** | **+0.031** | -0.030 | -0.066 |
| p=0.9 | ON | 0.303 | 0.161 | 0.199 | 0.286 |
| p=0.9 | OFF | **0.314** | **0.180** | 0.173 | 0.222 |
| p=0.9 | delta | **+0.011** | **+0.019** | -0.026 | -0.064 |

**The conditional training makes the gate a clean on/off the weights support --
the exact opposite of the flat eval-only gate:**
- **OFF recovers healthy capacity on every prob** (snow +0.011..+0.027, wet
  +0.019..+0.034): the network now handles raw inputs, so disabling input-IN
  recovers the healthy-condition frozen line.
- **ON keeps most of the fog/crosstalk rescue** (fog -0.026..-0.031, crosstalk
  -0.064..-0.066): not a full collapse, the rescue survives.
- **Trend across p:** lower prob (more OFF exposure) recovers MORE healthy
  capacity (p=0.5: +0.027 snow / +0.034 wet) at a slightly larger ON-side cost
  on fog/crosstalk; higher prob keeps the rescue tighter. p=0.5-0.7 is the
  sweet spot if healthy recovery is the priority.

**Interpretation.** This confirms the mechanism: the healthy-cost and the fog
rescue are both trainable choices. Conditional input-IN training lets the model
learn features that are good under BOTH modes, so at deployment a stats or
task-based gate (or no gate at all -- just choose the mode the target needs) is
a clean decision. For the TTA line this is directly usable: a label-free
per-scan mode switch (normalized vs raw) is now supported by the weights.

**Caveats.** Micro-scale (8 ep / 10%, not converged; the absolute numbers are
low and the OFF-recovery is small in magnitude). The full-scale conditional run
would settle whether the +0.02-0.03 healthy recovery and the -0.06 crosstalk
cost persist. But the DIRECTION is validated: the eval-only gate was flat, the
conditional-train gate is not. `micro_stoch_gate_*.json`.

---

## Overnight improvement-path results (`run_overnight_covimprove.sh`, 2026-08-24)

The four improvement paths + the two discrepancy re-runs, at full scale.

### P1: authoritative DGLSS++ KITTI-C -- the discrepancy is RESOLVED

The corrected-run log was right; the mechanism probe's DGLSS++ ceilings were
wrong (its ~0.5% `n_val` mismatch depressed them). Authoritative (single-extractor
full harness):

| condition | frozen | ceiling | gap |
| :--- | :--- | :--- | :--- |
| fog | 0.097 | 0.253 | +0.156 |
| crosstalk | 0.118 | 0.291 | +0.173 |
| snow | 0.543 | 0.575 | +0.033 |
| wet_ground | 0.492 | 0.570 | +0.079 |
| incomplete_echo | 0.482 | 0.488 | +0.006 |
| beam_missing | 0.583 | 0.587 | +0.004 |
| motion_blur | 0.548 | 0.566 | +0.018 |
| cross_sensor | 0.469 | 0.494 | +0.024 |

The README's DGLSS++ columns (snow 0.575, crosstalk 0.291, wet 0.570) were
correct all along. The mechanism probe's lower healthy-condition ceilings
(snow 0.421 etc.) are retracted as artifacts. `al_full_dataset_ep10_custom_dglsspp.json`.

### P2: code-2000 at full scale -- the "peak at 2000" does NOT hold

cov_ep10, KITTI-C, full scale (n_val 290M), ceiling at 10000-d vs 2000-d:

| condition | 10000-d | 2000-d | delta |
| :--- | :--- | :--- | :--- |
| fog | 0.369 | 0.338 | -0.031 |
| crosstalk | 0.491 | 0.460 | -0.031 |
| snow | 0.495 | 0.457 | -0.039 |
| wet_ground | 0.419 | 0.389 | -0.030 |
| incomplete_echo | 0.437 | 0.421 | -0.017 |
| beam_missing | 0.487 | 0.452 | -0.035 |
| motion_blur | 0.458 | 0.428 | -0.030 |
| cross_sensor | 0.446 | 0.412 | -0.034 |

2000-d is systematically ~0.03 LOWER on every condition at full scale. The tta
Iteration-2 "peak at 2000" was a small-harness (pool 10k/val 100k) artifact --
at full scale the 10000-d projection retains more. **Improvement path 4
(code-2000 default) is DEAD.** `al_full_dataset_ep10_custom_dim2000.json`.

### P3/P4/P5: SMOKE-ONLY -- the full overnight was NOT run

The `probe_d7_gate_ep10.json` and `probe_nusc_c_w0source_ep10.json` /
`probe_nusc_c_w0source_dim2000.json` files pulled are the **30-frame SMOKE**
outputs (n_val ~1-2M on KITTI-C, ~15-250k on NuScenes-C vs the full-scale 290M /
~100M), written by `SMOKE=1` to the same output paths. **None of the P3/P4/P5
numbers above are valid full-scale results and must NOT be quoted.** The full
overnight (`bash run_overnight_covimprove.sh`, no SMOKE) was not executed.

The P3 smoke does confirm the MECHANISM directionally (gate-off recovers
healthy-condition ceiling: cov_kitti wet_ground 0.430/0.722 smoke vs the
input-on authoritative 0.419; cross_sensor 0.486/0.629 vs 0.446), but these are
smoke-scale and the exact deltas are not final. **Run the full overnight and
re-pull P3/P4/P5 before quoting any of them.** The `--skip_existing` resume now
works per-condition, so a full re-run will only compute what's missing.

---

## Reproducibility

- Harness: `robust_diagnostic/al_full_dataset_diag.py` (deep-copied `ARCH`),
  runner `run_al_full_dataset.sh` (`EXTRACTORS=...` override).
- Flip probe: `robust_diagnostic/probe_nusc_c_dglsspp_vs_covshift_diag.py`,
  runner `run_probe_nusc_c_flip.sh`.
- Mechanism probe: `robust_diagnostic/probe_covshift_mechanism_diag.py`, runner
  `run_probe_covshift_mechanism.sh` (D1 clean baseline, D3 input stats, D4
  residual/effrank/λ-sweep, D5/D6 variance, D7 gate, D9 W0-source).
- Fog-collapse probe: `robust_diagnostic/probe_fog_collapse_diag.py`, runner
  `run_probe_fog_collapse.sh` (nearest-mean recall + frozen R4, same extractor
  under both generators; `probe_fog_collapse.json`).
- Checkpoints: `logs/ep10_supcon_vib_dglsspp_inputin_in_chan/...` (KITTI cov),
  `logs/supcon_vib_dglsspp` (KITTI DGLSS++), `logs/nusc_covshift_21ep` (NuScenes
  cov), `logs/nusc_dglsspp_21ep` (NuScenes DGLSS++).
- Results: `al_full_dataset_ep10_custom.json` (corrected DGLSS++ KITTI-C),
  `al_nuscenes_c.json` / `al_nuscenes_c_dglsspp.json` (NuScenes-C),
  `probe_covshift_mechanism_ep10.json` (Iteration-0 results),
  `al_full_dataset_ep10_custom_dglsspp.json` (P1 authoritative DGLSS++ KITTI-C),
  `al_full_dataset_ep10_custom_dim2000.json` (P2 code-2000 KITTI-C),
  `probe_d7_gate_ep10.json` (P3 gate-off; cov_nusc half incomplete),
  `probe_nusc_c_w0source_ep10.json` (P4 corrected NuScenes-C zero-shot),
  `probe_nusc_c_w0source_dim2000.json` (P5 NuScenes-C code-2000).

**Discrepancy RESOLVED (P1, 2026-08-24):** the mechanism probe's DGLSS++ KITTI-C
ceilings were wrong (snow 0.421 vs authoritative 0.575; its ~0.5% `n_val`
mismatch depressed them). The authoritative single-extractor run
(`al_full_dataset_ep10_custom_dglsspp.json`) matches the corrected-run log
exactly. Quote the P1 ceilings, not the mechanism probe's.
