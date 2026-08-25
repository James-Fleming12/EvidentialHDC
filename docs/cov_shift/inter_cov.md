# Intermediate-Condition Robustness: a fog/crosstalk feature extractor that does not cost the healthy conditions

Tracking the design of a feature-extractor change (or gate) that recovers the
KITTI-C fog/crosstalk collapse **without** degrading DGLSS++ on the other
conditions -- because raising the fog/crosstalk zero-shot and ceiling is the
biggest lever for beating GeoID's adapted numbers.

---

## Background: why we are doing this

### The two extractors and their trade

The cov-shift extractor (`supcon_vib_dglsspp_inputin_in_chan`) adds per-scan
input normalization (InstanceNorm on range+remission channels {0,4}) on top of
base DGLSS++ (`supcon_vib_dglsspp`). Full-scale corrected numbers (from
`cov_full_scale.md`):

KITTI$\rightarrow$KITTI-C, frozen zero-shot / labeled ceiling (R4 linear probe):

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

The two conditions where cov-shift wins are **fog** (ceil 36.9 vs 25.3, +11.6)
and **crosstalk** (49.1 vs 29.1, +20.0) -- the two conditions DGLSS++ collapses
on. Everywhere else DGLSS++ wins, and the mean ceiling is DGLSS++ (47.8 vs 45.0).

### Why we want a fog/crosstalk extractor without the healthy cost

**1. The improvement is concentrated exactly where GeoID is weakest.** GeoID's
adapted nuScenes-C numbers (from the README comparison) are lowest on
**crosstalk (30.8)** and fog is its only adapted value above ours by a
double-digit margin in the *ceiling* direction (fog 57.4 vs our 54.7) -- but our
current **fog zero-shot is 9.7**, far below both. Lifting fog+crosstalk is the
single biggest swing for the GeoID comparison table:

| condition | our ceiling (R4) | GeoID adapted | GeoID source-only |
| :--- | :--- | :--- | :--- |
| fog | 54.7 | 57.4 | 47.2 |
| crosstalk | **55.3** | 30.8 | 19.7 |

Raising fog zero-shot 9.7 → 30+ (cov-shift's level) and crosstalk toward 47
would make the frozen/no-adaptation line itself competitive with GeoID's
adapted number on these conditions -- the exact conditions where the ceiling
comparison already wins.

**2. The recoverable gap is real (the decoder is not the cap).** The decoder
ceiling probe showed the R4 linear probe already reaches the code space's
information limit (kNN/raw/RFF/balanced all ≤ R4). So the fog/crosstalk gaps
(fog +0.156 frozen→ceiling, crosstalk +0.173) are genuine headroom, not a
decoder artifact -- a better feature extractor can actually unlock them.

**3. It separates the extractor question from the TTA/AL story.** The paper
novelty is the closed-form memory-bank AL / linear-probe TTA (no backprop) vs
GeoID's gradient test-time training. A fog/crosstalk feature extractor would
raise the *starting point* (zero-shot) AND the ceiling on the two conditions
that matter most for the comparison, so the TTA line can claim
"reaches GeoID's adapted result with a frozen + closed-form pipeline."

### What the collapse actually is (so far)

- **Trigger: input-statistics collapse (D3).** KITTI-C fog erases the remission
  channel: remission variance → 1.3e-5 (clean ~1.3-1.5), range variance inflates
  to 4.0. Crosstalk has the same signature (remission var 0.000079).
- **Symptom: total geometry collapse.** The same extractor's nearest-mean recall
  (R1, decoder-independent) drops to 0.143 on KITTI-C fog (near the 0.06 random
  baseline) and frozen R4 collapses in lockstep (0.068). The code/feature space
  stops separating classes -- not a decoder failure.
- **Generator-specific, not extractor-specific.** `dgl_nusc` collapses on
  KITTI-C fog (recall 0.143) but is healthy on NuScenes-C fog (recall 0.575) --
  same weights. NuScenes-C fog is a range-var *explosion* (~5000) that leaves
  remission intact; KITTI-C fog deletes the channel that names the points.
- **DGLSS++ has no rescue path.** It feeds raw per-scan volumes into the trunk;
  nothing re-anchors a collapsed input distribution. cov-shift's per-scan
  input-IN re-bases the channels, which is why it rescues fog/crosstalk.

### What we know about the healthy-cost mechanism (so far)

- **It is mostly clean-inherited capacity (D1).** DGLSS++ beats cov-shift on
  *clean* KITTI 0.640 vs 0.520 (0.12). The healthy-condition corruption deficits
  are almost fully explained by this clean gap; the corruption-interaction term
  is small. "Healthy-condition collapse" is really "cov-shift pays 0.12 clean
  capacity to rescue fog/crosstalk."
- **Compression hypothesis rejected (D5/D6).** cov-shift code variance is
  *higher* than DGLSS++ (0.84-0.94 vs 0.65-0.83); DGLSS++'s healthy edge is
  lower/cleaner feature variance (0.036-0.085 vs 0.108-0.137), not cov-shift
  crushing the code.
- **Eval-only gating is dead; conditional training works.** A post-hoc
  stats-thresholded input-IN gate is flat (no healthy recovery, no fog rescue)
  because of train/eval mismatch. Training with *stochastic* input-IN
  (`input_in_prob` ∈ {0.5,0.7,0.9}) makes the gate a clean on/off the weights
  support: OFF recovers healthy (+0.027 snow, +0.034 wet at p=0.5), ON keeps
  most of the fog/crosstalk rescue (fog −0.031, crosstalk −0.065). Micro-scale
  only (8ep/10%).

---

## Iteration 0: Diagnostics

**Goal.** Pin down *where within DGLSS++ the fog/crosstalk collapse originates*
-- the exact internal propagation path -- so we can design the least invasive
fix (input re-anchoring, late normalization, or a per-scan gate) that rescues
fog/crosstalk without touching DGLSS++'s healthy-condition performance.

### Diagnostic 1 (DONE): input-statistics calibration — fog/crosstalk collapse the remission channel (D3)

Full-scale mechanism probe (`probe_covshift_mechanism_diag.py`,
`probe_covshift_mechanism_ep10.json`), per-scan channel {0,4} stats
(cov_kitti, KITTI-C):

| condition | ch0 (range) var | ch4 (remission) var |
| :--- | :--- | :--- |
| fog | **4.00** | **0.000013** |
| crosstalk | 2.09 | 0.000079 |
| wet_ground | 2.37 | 1.49 |
| snow | 2.11 | 1.45 |
| beam_missing | 2.28 | 1.34 |
| (clean ref) | ~1.0-2.4 | ~1.3-1.5 |

**Finding.** Fog collapses remission variance to ~0 (the info to name points is
absent at the input) and inflates range variance. This is an *input-statistics*
phenomenon, not a feature-space artifact.

### Diagnostic 2 (DONE): geometry collapse is generator-specific (fog-collapse probe)

`probe_fog_collapse_diag.py`, `probe_fog_collapse.json`, nearest-mean recall
(R1) + frozen R4, same extractor under both generators, in-domain
clean prototypes/W0:

| extractor | dataset / cond | frozen R4 | code recall | feat recall |
| :--- | :--- | :--- | :--- | :--- |
| dgl_nusc | KITTI-C fog | 0.068 | 0.143 | 0.148 |
| dgl_nusc | KITTI-C crosstalk | 0.120 | 0.165 | 0.171 |
| dgl_nusc | **NuScenes-C fog** | **0.305** | **0.575** | 0.574 |
| dgl_nusc | **NuScenes-C crosstalk** | **0.357** | **0.609** | 0.609 |
| dgl_kitti | KITTI-C fog | 0.104 | 0.268 | 0.263 |
| dgl_kitti | NuScenes-C fog | 0.104 | 0.236 | 0.242 |
| cov_kitti | KITTI-C fog | 0.293 | 0.513 | 0.514 |
| cov_kitti | KITTI-C crosstalk | 0.473 | 0.639 | 0.646 |
| cov_kitti | NuScenes-C fog | 0.115 | 0.311 | 0.307 |
| cov_nusc | NuScenes-C fog | 0.229 | 0.486 | 0.478 |
| cov_nusc | NuScenes-C crosstalk | 0.332 | 0.595 | 0.592 |

**Finding.** Same NuScenes-trained DGLSS++ collapses on KITTI-C fog (recall
0.143, near random) but is NOT collapsed on NuScenes-C fog (0.575). The two fog
generators produce different input statistics (KITTI-C: remission var collapse;
NuScenes-C: range var explosion). The collapse is a property of the KITTI-C
fog/crosstalk *generator*, not the extractor or fog-in-general. cov_kitti on
KITTI-C fog is NOT collapsed (0.513) — the input-IN rescues its home-domain
generator.

### Diagnostic 3 (DONE): clean-capacity gap is the healthy cost (D1)

`probe_covshift_mechanism_diag.py` (clean pseudo-condition, full-scale):

| extractor | clean frozen | clean ceiling | clean gap |
| :--- | :--- | :--- | :--- |
| cov_kitti | 0.520 | 0.530 | +0.009 |
| dgl_kitti | **0.640** | **0.648** | +0.009 |
| cov_nusc | 0.332 | 0.334 | +0.002 |
| dgl_nusc | 0.303 | 0.306 | +0.003 |

**Finding.** DGLSS++ wins clean by 0.12 (KITTI). The healthy-condition deficits
are almost fully clean-inherited; the normalization is a flat
capacity/regularization cost, not a corruption-interaction failure.

### Diagnostic 4 (DONE): conditional (stochastic) input-IN makes a gate possible

`run_micro_stoch.sh`, `micro_stoch_gate_{stoch,stoch7,stoch9}.json`, micro
8ep/10%, frozen mIoU with input-IN ON vs OFF:

| prob | mode | snow | wet_ground | fog | crosstalk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| p=0.5 | ON | 0.291 | 0.145 | 0.193 | 0.275 |
| p=0.5 | OFF | **0.318** | **0.178** | 0.162 | 0.210 |
| p=0.7 | ON | 0.298 | 0.155 | 0.197 | 0.281 |
| p=0.7 | OFF | **0.317** | **0.186** | 0.167 | 0.215 |
| p=0.9 | ON | 0.303 | 0.161 | 0.199 | 0.286 |
| p=0.9 | OFF | **0.314** | **0.180** | 0.173 | 0.222 |

**Finding.** Training with stochastic input-IN makes the gate a clean on/off:
OFF recovers healthy (+0.027 snow, +0.034 wet at p=0.5), ON keeps most of the
fog/crosstalk rescue. Direction validated (micro-scale only). Full-scale run
pending.

### Diagnostic 5 (DONE): per-layer fog-collapse propagation probe

**Question.** *Where within DGLSS++ does the collapse originate?* The input
trigger (remission collapse) and the final geometry collapse are known, but not
the internal path: early-conv saturation from range inflation, BatchNorm
running-stat mismatch (frozen stats calibrated to clean remission), or gradual
degradation across blocks.

**Setup.** `probe_fog_collapse_layer_diag.py` (new): run `dgl_kitti` on clean
vs KITTI-C fog (200 frames, single extractor), record per-block:

- activation mean/var per channel,
- % units saturated (|x| large) or dead (near-zero variance),
- BN running-stat mismatch: `|x − μ_running| / σ_running` at each block.

**Runner.** `run_probe_fog_collapse_layer.sh` (DRY_RUN/SMOKE).
`probe_fog_collapse_layer.json` (200 frames, 6-7s/stream).

**Result (2026-08-24, dgl_kitti, 200 frames):**

*Per-stage activation stats (mean_act / mean_var):*

| stage | clean | fog | crosstalk |
| :--- | :--- | :--- | :--- |
| conv1 | 0.161 / 0.414 | 0.126 / 0.390 | 0.162 / 0.490 |
| conv2 | 0.023 / 0.128 | 0.065 / 0.153 | 0.063 / 0.164 |
| conv3 | 0.047 / 0.170 | 0.090 / 0.212 | 0.076 / 0.200 |
| layer1 | 0.176 / 0.467 | 0.204 / 0.589 | 0.169 / 0.463 |
| layer2 | 0.197 / 0.536 | 0.161 / 0.399 | 0.175 / 0.423 |
| layer3 | 0.231 / 0.753 | 0.234 / 0.615 | 0.261 / 0.634 |
| layer4 | 0.132 / 0.342 | 0.254 / 0.400 | 0.260 / 0.466 |
| conv_1 | 0.001 / 0.147 | 0.079 / 0.214 | 0.061 / 0.162 |
| conv_2 | 0.023 / 0.145 | −0.044 / 0.125 | −0.041 / 0.096 |

*BN running-stat mismatch |E[x]−μ_running|/σ_running (selected):*

| BN module | clean | fog | crosstalk |
| :--- | :--- | :--- | :--- |
| conv1.bn | 0.250 | 0.461 | 0.687 |
| conv2.bn | 0.360 | 0.401 | 0.690 |
| conv3.bn | 0.268 | 0.481 | 0.714 |
| **conv_1.bn** | **0.180** | **0.834** | **0.785** |
| **conv_2.bn** | **0.120** | **0.630** | **0.567** |
| layer1.2.bn1 | 0.110 | 0.428 | 0.477 |
| layer2.0.bn1 | 0.120 | 0.535 | 0.574 |
| layer4.2.bn1 | 0.143 | 0.630 | 0.626 |

*Saturation / dead fractions:* sat_frac ~0.003, dead_frac ~0.000 on ALL stages
across clean/fog/crosstalk.

**Findings (the decisive split):**

1. **NOT an early-conv saturation / dead-unit story.** No stage saturates or
   zeroes under fog/crosstalk (sat ~0.3%, dead 0%). conv1 activation stats are
   essentially unchanged (mean_act 0.161 vs 0.126; var 0.41 vs 0.39). The input
   collapse does NOT blow up or kill the first block.
2. **The collapse is a frozen-BatchNorm running-stat mismatch that builds up
   through the network and peaks at the bottleneck.** Under fog/crosstalk the BN
   mismatch is elevated at EVERY BN (2-5x clean), and the largest relative jumps
   are at the late fusion blocks: conv_1.bn 0.180→0.834 (4.6x) and conv_2.bn
   0.120→0.630 (5.3x). The early activations shift modestly, but by the
   640→256→128 bottleneck the frozen running stats (calibrated to clean KITTI
   range/remission) are badly out of calibration.
3. **The collapse is detectable per-scan with a single BN mismatch scalar
   (AUROC 1.000).** See Diagnostic 7.

**Interpretation for the design.** The fix is NOT at the raw input (conv1 is
healthy). It is at the **BN statistics** — the frozen running mean/var no longer
match the corrupted stream by the time the signal reaches the bottleneck. Two
levers fall out:
- a **late re-normalization** (re-estimate / re-base the late-stage BN or the
  bottleneck conv_1/conv_2 inputs) to realign the running stats on the fly — the
  "tiny fog/crosstalk bump, full healthy" target, since healthy scans already sit
  at their calibrated operating point;
- the existing **BN-statistic alignment TTA lever** (from the ResNet docstring:
  "the training-side mirror of the BN-statistic alignment TTA lever") is the
  directly-relevant mechanism, not a whole-extractor retrain.

This is consistent with why cov-shift's input-IN rescues fog/crosstalk but costs
healthy capacity: input-IN realigns the INPUT distribution so the frozen stats
stay valid, but it does so for ALL scans (paying the clean-capacity cost); a
late BN re-anchor would only act where the mismatch is large.

### Diagnostic 6 (PARKED): conditional input-IN at full scale

**Question.** Do the micro-scale stochastic-gate results
(+0.02-0.03 healthy recovery, −0.06 crosstalk cost) persist at full scale?

**Status: PARKED (2026-08-24).** The micro stoch signals are weak in BOTH
directions (OFF recovers +0.027 snow but loses −0.031 fog / −0.065 crosstalk),
which is the WRONG tradeoff for the goal. We want a *tiny* fog/crosstalk bump
with *full* healthy recovery — not a full-scale training run (~8h) to compromise
both sides. Diagnostic 5's finding (late BN-stat mismatch, not input saturation)
makes a targeted late re-normalization the preferred lever over retraining a
conditional extractor. Revisit only if the late-BN fix fails.

**Setup (if resumed).** `run_micro_stoch.sh` pattern, but the full dataset
harness (`al_full_dataset_diag.py`) with a `input_in_prob`-trained checkpoint
(cov-shift `supcon_vib_dglsspp_inputin_in_chan_stoch{,_7,_9}`), decode with
input-IN ON vs OFF on all conditions. Compare frozen AND ceiling.
`run_overnight_condin.sh` built but not run.

### Diagnostic 7 (DONE): per-scan detector signal for the gate

**Question.** What statistic reliably flags "this scan needs normalization" at
deployment, with no corruption-type knowledge?

**Setup.** From the Diagnostic-5 per-layer activations, extract per-scan scalar
signals (`{stage}_mean_act`, `{stage}_mean_var`, `bn_mismatch_*`); report AUROC
for "needs normalization" (fog/crosstalk) vs "healthy" (clean).

**Result (same run, `probe_fog_collapse_layer.json`):**

| condition | best detector | AUROC |
| :--- | :--- | :--- |
| fog | bn_mismatch_conv_1.bn | **1.000** |
| crosstalk | bn_mismatch_conv_1.bn | **1.000** |

The top candidates are ALL `bn_mismatch_*` signals (conv_1.bn, conv_2.bn,
layer1.2, layer2.0, layer2.1) at AUROC 1.000 on both conditions. The per-scan
BN-mismatch at the bottleneck (`conv_1.bn`, the 640→256 fusion) is a
**perfect, label-free, deployment-ready detector**: no corruption-type
knowledge, no labels, just the frozen running-stat deviation of one BN layer.

**Finding.** A per-scan gate signal exists and is trivial to compute. If a
late-BN re-normalization is engaged only when `bn_mismatch_conv_1` exceeds a
threshold, it would act exactly on the collapsed scans and leave healthy scans
at their calibrated operating point — the "tiny bump, full healthy" design.

### Diagnostic 8 (PENDING): BatchNorm re-anchor headroom probe

**Question.** How much of the frozen→ceiling gap on fog/crosstalk does a
label-free BatchNorm running-stat re-anchor close? This is the concrete
implementation of the Diagnostic-5 finding (the collapse is a frozen-BN
running-stat mismatch at the late bottleneck, not input saturation) and the
second online lever alongside the linear-classifier W update.

**Setup.** `probe_bn_reanchor_diag.py` + `run_probe_bn_reanchor.sh`: dgl_kitti,
fog+crosstalk. For each condition: W0 on clean (frozen BN) → frozen baseline;
re-estimate the late BN `running_mean/var` from the CORRUPTED stream (statistic
substitution, label-free, closed-form); decode with W0 → `bn_recal`; W* on the
corrupted pool → labeled ceiling. Report:

- `bn_recal` vs `frozen` (the label-free gain),
- `bn_recal_frac_of_gap` = (bn_recal − frozen)/(ceiling − frozen) — the fraction
  of the labeled headroom a BN re-anchor alone recovers,
- `bn_recal_Ws` (does the recal help the labeled ceiling too?),
- `--bn_scope` in {bottleneck, late, all} to locate which BN subset carries the
  recoverable signal.

**Decisive split.** If `bn_recal_frac_of_gap` ~ 1.0, a BN re-anchor *alone*
nearly reaches the labeled ceiling — the label-free lever is the whole fix, and
the TTA/AL pipeline becomes: per-scan BN re-anchor (gated on
`bn_mismatch_conv_1`, AUROC 1.000) + the online W update. If the fraction is
small, the recoverable gap needs the labeled pool after all (BN recal helps, but
the ceiling is pool-bound).

---

## Decision rule

Diagnostic 5 settled the decisive split: the collapse is NOT an input-saturation
or dead-unit phenomenon (conv1 is healthy; sat/dead ~0). It is a **frozen-BN
running-stat mismatch that builds gradually and peaks at the late bottleneck
(conv_1/conv_2, 4.6-5.3x clean mismatch)**, and Diagnostic 7 gives a **perfect
per-scan detector** (`bn_mismatch_conv_1`, AUROC 1.000) with no labels.

The fix to pursue is a **late, gated BatchNorm re-anchor** (re-estimate or
re-base the bottleneck BN statistics on the fly, engaged per-scan when
`bn_mismatch_conv_1` is high) — NOT a whole-extractor retrain and NOT the
parked conditional-input-IN path. Diagnostic 8 measures how much of the labeled
ceiling this lever alone recovers (`bn_recal_frac_of_gap`). If it is ~1.0, the
TTA/AL pipeline combines the BN re-anchor with the online W update for the
"tiny fog/crosstalk bump, full healthy recovery" target. If it underdelivers,
fall back to the AL/TTA pool re-anchoring of the corrupted stream instead.
