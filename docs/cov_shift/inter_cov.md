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

### Diagnostic 1 (DONE): input-statistics calibration: fog/crosstalk collapse the remission channel (D3)

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
KITTI-C fog is NOT collapsed (0.513): the input-IN rescues its home-domain
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
healthy). It is at the **BN statistics**: the frozen running mean/var no longer
match the corrupted stream by the time the signal reaches the bottleneck. Two
levers fall out:
- a **late re-normalization** (re-estimate / re-base the late-stage BN or the
  bottleneck conv_1/conv_2 inputs) to realign the running stats on the fly, the
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
with *full* healthy recovery, not a full-scale training run (~8h) to compromise
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

**Finding.** A per-scan gate signal exists and is trivial to compute, and it
perfectly flags collapsed scans (AUROC 1.000). **Caveat (Diagnostic 8):** the
mismatch is a perfect *detector* but NOT a *recoverable lever*: re-anchoring
the BN stats that produce it is negative (a symptom of the input collapse, not
the cause). As a gate it can still decide *when to engage an input-level
intervention* (e.g., input-IN) or the AL pool, but not to correct the features
itself.

### Diagnostic 8 (DONE): BatchNorm re-anchor headroom probe -- NEGATIVE

**Question.** How much of the frozen→ceiling gap on fog/crosstalk does a
label-free BatchNorm running-stat re-anchor close? The Diagnostic-5 finding (the
collapse is a frozen-BN running-stat mismatch at the late bottleneck) suggested
statistic substitution could be the whole fix.

**Setup.** `probe_bn_reanchor_diag.py` + `run_probe_bn_reanchor.sh`: dgl_kitti,
fog+crosstalk. W0 on clean (frozen BN) → frozen baseline; re-estimate the late
BN `running_mean/var` (scope=late: layer3/4 + conv_1/2) from the CORRUPTED
stream (statistic substitution, label-free, closed-form); decode with W0 →
`bn_recal`; W* on the corrupted pool → labeled ceiling.

**Result (dgl_kitti, 500 frames, `probe_bn_reanchor.json`):**

| condition | base (frozen) | bn_recal (TTA) | labeled ceiling | bn_recal Δ | frac of gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.076 | 0.046 | 0.296 | **−0.030** | **−14%** |
| crosstalk | 0.121 | 0.053 | 0.360 | **−0.068** | **−28%** |

`bn_recal_Ws` (W* decoded with re-anchored BN) is also *worse* than the frozen
ceiling on both (fog 0.028, crosstalk 0.052 vs ceiling 0.296/0.360).

**Finding: the BN re-anchor is NEGATIVE.** Substituting the corrupted-stream
BN running stats into the late layers *hurts* both conditions (−0.03 fog,
−0.07 crosstalk) and closes a negative fraction of the frozen→ceiling gap. The
re-anchored features are WORSE for the W0 probe than the frozen ones.

**Why (mechanism, from the combined evidence).** The Diagnostic-5 "BN mismatch"
was a *symptom of the input collapse*, not the cause. The probe showed conv1 is
healthy (no saturation); the geometry dies because the *remission channel
itself* is erased at the input (D3: var → 1.3e-5), so the information to name
points is absent before the network ever sees it. Re-calibrating the frozen BN
stats cannot create information that was destroyed at the input; it just moves
the features to a different operating point, which the clean-fitted W0 doesn't
recognize. This is consistent with the fog-collapse probe (the SAME extractor is
healthy on NuScenes-C fog where remission survives) and with D3.

**Implication.** The recoverable gap on fog/crosstalk is **input-bound, not
BN-bound**: no amount of late-network statistic realignment restores it. The
only thing that rescues KITTI-C fog/crosstalk is re-anchoring the INPUT
distribution (cov-shift's input-IN, which re-bases the collapsed remission),
and that costs healthy capacity (D1). The labeled ceiling remains reachable only
through the corrupted pool (W*), which is the AL/TTA pool mechanism.

### Diagnostic 8b (DONE): labeled BN-update probe -- NEGATIVE, and the separability data locates the collapse UPSTREAM of BN

**Question.** Diagnostic 8 only re-estimated the BN running *stats* (label-free
stat substitution). The stronger oracle: **with labels, fit the BN's affine
(γ, β) per channel** so each class's corrupted pre-BN activation maps to its
clean post-BN mean, the maximal expressivity a BN layer has. If even a
label-fitted affine cannot recover, BN is not the lever. The probe also measures
pre-BN class *separability* to locate where the collapse happens.

**Setup.** `probe_bn_labeled_diag.py` + `run_probe_bn_labeled.sh`: dgl_kitti,
fog+crosstalk, scope=late (22 BN modules). Per-channel least squares over
classes: `minimize Σ_c (γ·pre_corr[c,ch] + β − post_clean[c,ch])²`. Report W0
decoded with: label-fitted affine only, affine + re-estimated stats; and
pre-BN per-class separability (mean pairwise cosine distance of class means,
1 = separated, 0 = merged) on clean vs corrupted.

**Result (dgl_kitti, 500 frames, `probe_bn_labeled.json`):**

| condition | base (frozen) | labeled affine | +stats | labeled ceiling | affine Δ | frac of gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.076 | 0.013 | 0.015 | 0.299 | **−0.063** | **−28%** |
| crosstalk | 0.120 | 0.022 | 0.055 | 0.363 | **−0.098** | **−41%** |

**Finding 1: even the LABELED BN affine is NEGATIVE.** Fitting γ, β with
per-class labels makes the W0 decode *worse* on both conditions (−0.06 fog,
−0.10 crosstalk). The BN layer, even given labels and full freedom to scale +
shift each channel, cannot recover the collapse. This is a negative for *this
specific attempt at retraining the BN* (per-channel affine aligned to clean
per-class means), not a blanket "the BN idea is dead for all metrics": a later
iteration could switch the "correctness" metric (e.g. fit the affine to minimize
a different target, or to the corrupted pool's own class means) and get a
different answer. But the separable part is below.

**Finding 2 (the biggest point): the pre-BN class separability is COLLAPSED.**
The per-class mean vectors at the *input* of every scoped BN are nearly merged
under fog/crosstalk:

| BN module | clean sep | fog sep | crosstalk sep |
| :--- | :--- | :--- | :--- |
| conv_1.bn | 0.704 | **0.101** | **0.137** |
| conv_2.bn | 0.915 | **0.152** | **0.228** |
| layer4.2.bn1 | 0.788 | **0.121** | **0.227** |
| layer3.0.bn1 | 0.791 | **0.275** | **0.210** |

(fog ratio 0.14-0.45 across all 22 BNs; crosstalk ratio 0.19-0.56.) The classes
are already merged *before* the BN ever operates on them. A BN (scale + shift
per channel) cannot re-separate classes that arrive with near-identical means;
it has no rotational/class-specific freedom. The information to separate classes
is gone at the BN *input*.

**Mechanism tie-in.** This is the decisive complement to D3 and the fog-collapse
probe: the collapse is not *in* the BN, it is *upstream* of it. Diagnostic 5
(conv1 healthy, no saturation) + D3 (remission erased at input) + Diagnostic 8b
(classes merged before BN) all say the same thing: **the input distribution
collapse destroys class structure before any late network stage can be
re-calibrated to fix it.** The only intervention that re-creates the structure
is at the input (input-IN re-bases the collapsed remission); everything after
the input is downstream of a loss that already happened.

---

### Diagnostic 9 (DONE): gated input-IN TTA probe -- the gate works, recovery is small

**Question.** The collapse is input-bound (remission erased at input, classes
merged before BN), and the only rescue is per-scan input-IN on channels {0,4}.
Always-on costs 0.12 clean capacity (D1). Can a **label-free per-scan gate**
(apply input-IN only when the Diagnostic-7 detector fires) rescue fog/crosstalk
while leaving healthy scans at raw?

**Setup.** `probe_gated_inputin_diag.py` + `run_probe_gated_inputin.sh`:
plain DGLSS++ (`supcon_vib_dglsspp`), fog + crosstalk + snow, 200-frame subset
(100k clean fit / 200k pool). Gate threshold auto-calibrated on clean:
`bn_mismatch_conv_1` mean+3σ → tau=0.262 (clean mean 0.158, sd 0.035).
Per condition: raw (no input-IN), gated (input-IN when detector > tau),
always_on (input-IN every scan), and the labeled ceiling W*.

**Result (`probe_gated_inputin.json`, `probe_gated_inputin.log`):**

| condition | raw | gated | always_on | ceiling | gated Δ | always_on Δ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 0.0147 | 0.0171 | 0.0175 | 0.0290 | **+0.0023** | +0.0028 |
| crosstalk | 0.0311 | 0.0369 | 0.0374 | 0.0663 | **+0.0058** | +0.0063 |
| snow | 0.0749 | 0.0755 | 0.0482 | 0.0768 | **+0.0006** | **−0.0267** |

**Findings.**

1. **The gate works as designed: it helps the collapsed conditions without the
   always-on healthy penalty.** On snow (healthy), gated stays at raw (+0.0006,
   i.e. no false-positive re-anchoring), while always_on HURTS snow (−0.0267) —
   the 0.12 clean-capacity cost showing up at this subset scale. The gate
   isolates the input-IN rescue to exactly the scans that need it.
2. **But the recovery is SMALL (~16% of the ceiling gap):** gated closes
   0.0023/0.0142 (fog) and 0.0058/0.0352 (crosstalk) of the raw→ceiling gap,
   barely less than always_on (which closes ~20%). The gate correctly avoids
   harming healthy, but the input-IN rescue on a plain DGLSS++ extractor only
   recovers a fraction of the labeled headroom.

**Interpretation.** The gated-input-IN TTA validates the *mechanism* (a
label-free detector can route the input re-anchor to exactly the collapsed
scans, with zero healthy cost) but shows the *recovery magnitude is small* at
this scale. Two limits: (a) it is a 200-frame subset with a 100k clean fit, so
absolute numbers are depressed; (b) the input-IN on a plain DGLSS++ model is a
*train/eval mismatch* (the model never saw normalized inputs), so the rescue is
weaker than it would be for a conditional-trained model. The full-scale run and
a conditional (input_in_prob) variant would settle the magnitude. For the paper
pivot (linear-classifier TTA/AL), this is the label-free detection signal that
can gate *where* to spend AL/TTA effort, even if the input-IN itself is not the
final lever.

### Diagnostic 10 (DONE): representation-robustness probe -- invariance does NOT predict the ceiling (`probe_rep_robustness_diag.py`, `run_probe_rep_robustness.sh`)

**Question.** Why does the plain-supervised HyperLiDAR extractor (`baseline`)
beat DGLSS++/cov-shift on KITTI-C (R4 ceiling hyper 53.9 vs dgl 50.7 vs cov
46.7) but LOSE on NuScenes-C (55.1 vs 63.5 vs 61.0)? Hypothesis: hyper's features
are more *invariant* to KITTI-C corruptions but less to NuScenes-C's. This probe
measures, per (extractor, condition) on its home dataset: feature-space shift
(clean→corrupted per-class mean cosine), per-class separability retention,
BN running-stat mismatch at conv_1.bn, code variance, and effective rank.

**Setup.** 6 extractors (hyper/dgl/cov × kitti/nusc), 200 frames, fog+crosstalk+snow,
in-domain clean reference. (Note: the initial full run OOM'd at ~600 GB in
`hdc_codes`; the probe was made memory-bounded by subsampling to 100k pts before
the 10000-d projection.)

**Result (`probe_rep_robustness.json`):**

| dataset | extractor | cond | shift | sep_ret | bn_clean→corr |
| :--- | :--- | :--- | :--- | :--- | :--- |
| KITTI-C | hyper | fog | 0.215 | **0.12** | 0.17→0.82 |
| KITTI-C | hyper | crosstalk | 0.261 | **0.16** | 0.17→0.84 |
| KITTI-C | hyper | snow | 0.951 | 0.96 | 0.17→0.18 |
| KITTI-C | dgl | fog | 0.169 | 0.38 | 0.16→0.82 |
| KITTI-C | dgl | crosstalk | 0.211 | 0.47 | 0.16→0.83 |
| KITTI-C | dgl | snow | 0.949 | 1.00 | 0.16→0.16 |
| KITTI-C | cov | fog | 0.914 | 1.05 | 0→0* |
| KITTI-C | cov | crosstalk | 0.977 | 0.97 | 0→0* |
| KITTI-C | cov | snow | 0.983 | 1.00 | 0→0* |
| NuScenes-C | hyper | fog | 0.763 | **0.94** | 0.20→0.38 |
| NuScenes-C | hyper | crosstalk | 0.799 | 1.04 | 0.20→0.26 |
| NuScenes-C | hyper | snow | 0.802 | 1.07 | 0.20→0.28 |
| NuScenes-C | dgl | fog | 0.937 | 0.99 | 0.18→0.30 |
| NuScenes-C | dgl | crosstalk | 0.917 | 0.95 | 0.18→0.23 |
| NuScenes-C | dgl | snow | 0.924 | 0.97 | 0.18→0.20 |
| NuScenes-C | cov | fog | 0.905 | 1.03 | 0→0* |
| NuScenes-C | cov | crosstalk | 0.937 | 1.02 | 0→0* |
| NuScenes-C | cov | snow | 0.941 | 1.01 | 0→0* |

\* cov-shift uses internal InstanceNorm (no BatchNorm), so its `bn_mismatch` is
an artifact (the hook skips non-BN); the 0→0 is not meaningful for cov.

**Findings (the invariance hypothesis is REVERSED):**

1. **hyper is NOT more invariant on KITTI-C -- it is the LEAST invariant, yet it
   has the HIGHEST KITTI-C ceiling.** On KITTI-C fog/crosstalk hyper's separability
   retention is the worst (0.12/0.16 vs dgl 0.38/0.47, cov ~1.0) and its BN
   mismatch is the largest (0.82/0.84). Its raw feature geometry COLLAPSES most.
   Yet its R4 ceiling (29.1/31.3) beats dgl (25.3/29.1). So the invariance
   metrics (sep_retention, shift, bn_mismatch) do NOT predict the recoverable
   ceiling in the direction hypothesized.
2. **On NuScenes-C hyper is MORE invariant (sep_ret 0.94-1.07, low bn drift) yet
   has the LOWEST ceiling (47.9-49.5 vs dgl 58.5-61.3).** The exact opposite
   pattern: hyper's NuScenes-C features stay geometrically intact but are less
   recoverable by the labeled probe.
3. **cov-shift (input-IN) is the invariant outlier: it preserves separability on
   BOTH datasets (~1.0 retention) and has the KITTI-C fog/crosstalk ceiling edge
   (36.9/49.1) -- but NOT the NuScenes-C edge, where dgl wins.** So cov-shift's
   input-IN is invariant everywhere, yet still loses NuScenes-C to dgl.

**Interpretation.** The feature-space invariance (raw geometry retention, BN
drift, class shift) does NOT determine the labeled ceiling. The recoverable
ceiling is set by something else: how much *linearly separable* structure the
features retain under the probe, which is a different property than geometric
invariance. Hyper's KITTI-C advantage and NuScenes-C disadvantage are therefore
NOT explained by "more/less invariant representations" -- they are a property of
how the linear probe interacts with each extractor's feature geometry, likely
tied to the input statistics (D3: KITTI-C fog collapses remission; NuScenes-C
fog inflates range) and how each extractor's normalization reacts. This
reframes the earlier "hyper is more robust" claim: hyper's features are NOT more
invariant; they are differently *distributed* such that the R4 linear probe
rewards them on KITTI-C but not NuScenes-C.

### Diagnostic 11 (DONE): linear-property probe -- class_shift is the zero-shot predictor, and it IS reachable at training time (`probe_linear_prop_diag.py`, `run_probe_linear_prop.sh`)

**Question.** Diagnostic 10 measured the CEILING interaction, but the comparison
of interest (hyper best on KITTI-C, worst on NuScenes-C) is the ZERO-SHOT line.
Zero-shot uses the CLEAN-fit W0 decoded on corrupted features -- there is no
"fit on corrupted data" -- so the question is: does the clean-fit boundary
survive the clean→corrupted feature shift, and which feature property predicts
that?

**Setup.** Same 6 extractors, fog+crosstalk, 200 frames. Per (extractor, cond),
in the binarized 10000-d code space: `class_shift` (clean→corr per-class
cosine), `fisher_ratio` (between/within scatter), `pre_sign_margin_lt05_frac`
(fraction of codes near the sign-flip boundary), `within_class_var`, `effrank`,
`margin_sweep` (frozen W0 accuracy after zeroing fragile pre-sign activations),
plus `frozen` (the zero-shot) and `ceiling`.

**Result (`probe_linear_prop.json`):**

| extractor | cond | class_shift | fisher | frozen (zs) | margin_sweep tau0.5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| hyper_kitti | fog | 0.199 | 0.21 | 0.110 | 0.116 |
| dgl_kitti | fog | 0.196 | 0.52 | 0.111 | 0.107 |
| cov_kitti | fog | **0.933** | 0.75 | **0.344** | 0.328 |
| hyper_kitti | crosstalk | 0.247 | 0.20 | 0.124 | 0.126 |
| dgl_kitti | crosstalk | 0.273 | 0.60 | 0.132 | 0.126 |
| cov_kitti | crosstalk | **0.980** | 0.96 | **0.579** | 0.540 |
| hyper_nusc | fog | 0.760 | 0.28 | 0.211 | 0.211 |
| dgl_nusc | fog | **0.931** | **1.59** | **0.266** | 0.257 |
| cov_nusc | fog | 0.899 | 0.99 | 0.222 | 0.208 |
| hyper_nusc | crosstalk | 0.799 | 0.22 | 0.255 | 0.252 |
| dgl_nusc | crosstalk | 0.897 | 1.21 | 0.285 | 0.280 |
| cov_nusc | crosstalk | **0.928** | 1.13 | **0.299** | 0.282 |

**Pearson correlation with frozen (zero-shot), n=12:**
- `class_shift` **r=+0.776** (strongest)
- `fisher_ratio` r=+0.480
- `within_class_var` r=+0.338, `effrank` r=+0.291, `pre_sign_margin` r=+0.202

**Findings.**

1. **`class_shift` (clean→corr feature stability) is the dominant zero-shot
   predictor** (r=+0.776), and it tracks the exact zero-shot ordering: cov has
   the lowest shift on KITTI fog/crosstalk (0.93/0.98) and the highest frozen;
   dgl the lowest shift on NuScenes fog (0.931) and highest frozen. Where the
   per-class means stay put, the clean-fit W0 transfers.
2. **`fisher_ratio` is secondary** (r=+0.48) -- the corrupted code's linear
   separability. `pre_sign_margin` is NOT predictive (r=+0.20), i.e. the zero-shot
   loss is not primarily "codes sitting at sign-flip boundaries."
3. **The `margin_sweep` confirms the mechanism is shift, not boundary
   fragility**: cov_kitti crosstalk frozen 0.579 drops only to 0.540 at tau0.5
   (robust coordinates), while its tau2.0 craters (0.045) -- the stable cores
   survive; hyper/dgl drop more at tau0.5 because their features shifted off the
   clean W0's support entirely.

**The extractor objective this motivates.** The property to optimize for
zero-shot robustness is `class_shift` = minimizing clean→corrupted per-class
feature displacement, i.e. a TRAINING-TIME consistency/invariance objective
between clean and corrupted views. This is exactly the desideratum-3 "Invariance
to First-Order Sensor Energy Shifts" from cov_shift_iterations.md -- which is
what cov-shift's input-IN partially achieves at the input level.

**Relation to the prior attempt (Phase 24 `supcon_vib_additive`).** We DID try
this as the "noise invariance" / volumetric-noise augmentation (`supcon_vib_additive`),
and threw it out -- but the Diagnostic-11 result says it was NOT a bad idea, it
was badly implemented. The Phase 24.6 medium verdict (gen_iterations.md:924-956):
at equal capacity the additive regimen was WORSE than plain supcon_vib on every
condition (fog 8.4 vs 10.1 mIoU), fog LP 30.6% BELOW plain (36.3%), and the
micro-scale healing (fog LP 57%, ArtSurv 3.8x) did NOT scale. Critically, the
autopsy showed the additive space DID reduce feature shift at micro scale (Cos
shift 0.821→0.738, norm gap healed) but the 128D→10kD binarization transfer
failed (BinCos 0.076) AND the volumetric injection traded away the 6 healthy
conditions (-1.6 to -4.2 pts). So the mechanism (reduce class_shift) was
targeted correctly but the implementation (a) didn't survive convergence, (b)
didn't transfer through binarization, and (c) cost the healthy conditions. The
Diagnostic-11 result says: **a clean→corrupted class_shift penalty is the right
loss family, but it must (i) be measured/trained on the actual corrupted views
that matter, (ii) preserve the healthy-condition geometry (not trade it), and
(iii) be validated at the code/binarized level, not just the 128D LP.**

### Diagnostic 12 (DONE): why the Phase-24 noise-invariance attempt failed -- the mechanism WORKED, the implementation was too weak (`run_overnight_noiseinv.sh`, `probe_noiseinv.json`)

**Question.** Diagnostic 11 identified `class_shift` (clean→corrupted per-class
feature cosine) as the dominant zero-shot predictor (r=+0.776), and the Phase-24
`supcon_vib_additive` (volumetric noise injection) was the prior attempt at
exactly this "sensor-noise invariance". It was thrown out at medium scale (lost
on every condition). This diagnostic re-tests it CAPACITY-MATCHED (plain vs
additive, both 8 ep / 10%) and measures whether the additive training actually
reduced the class_shift Diagnostic 11 says drives zero-shot.

**Setup.** Train `supcon_vib` (plain) and `supcon_vib_additive` at identical micro
scale (8 ep / 10%), then evaluate both with the linear-property probe
(class_shift, fisher, frozen, ceiling, margin_sweep) on fog+crosstalk.

**Result (`probe_noiseinv.json`):**

| extractor | cond | class_shift | fisher | frozen (zs) | ceiling | ms0.5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| plain | fog | 0.815 | 0.23 | 0.072 | 0.268 | 0.054 |
| additive | fog | 0.841 | 0.27 | 0.086 | 0.271 | 0.076 |
| plain | crosstalk | 0.878 | 0.38 | 0.120 | 0.320 | 0.076 |
| additive | crosstalk | 0.875 | 0.43 | 0.135 | 0.334 | 0.103 |

**Findings (the mechanism did NOT reduce class_shift -- the augmentation was too
weak to matter):**

1. **Additive barely moved `class_shift`** (fog 0.815→0.841, crosstalk 0.878→0.875
   -- essentially unchanged). The volumetric-noise injection did NOT reduce the
   clean→corrupted feature shift, which Diagnostic 11 says is the zero-shot
   driver. So the mechanism was not "wrong" -- it was **not actuated**: the
   augmentation was too weak (density 0.05 into empty space) to change the
   per-class feature geometry.
2. **The small frozen gains are noise-level** (fog +0.014, crosstalk +0.015) and
   track the tiny fisher gain (+0.04/+0.05), not a class_shift change. The
   additive training slightly improved code separability (fisher) but did not
   address the shift that actually limits zero-shot.
3. **No healthy trade observed** (the Phase-24 concern) -- because the additive
   change was so weak it affected nothing negatively either. Ceiling is flat
   (0.268→0.271 fog, 0.320→0.334 crosstalk).

**Interpretation.** The Phase-24 additive failure was NOT "noise invariance is a
bad idea" -- it was "the noise invariance was too weak to change the feature
shift." Diagnostic 11 says class_shift is the lever, and Diagnostic 12 shows the
previous attempt never actually pulled it (shift unchanged). A re-do must use a
**much stronger displacement** than volumetric empty-space injection at density
0.05: e.g. GeoID-style on-manifold displacement of valid points (the
`supcon_vib_geoid` port), or an explicit clean→corrupted class-shift penalty in
the loss, trained to actually reduce the measured shift. This directly
motivates the `run_train_geoid.sh` port (GeoID's inlier-discrimination loss)
currently running.

---

## Decision rule

Diagnostic 5 showed the collapse is NOT input saturation (conv1 healthy); the
BN-mismatch signal (Diagnostic 7, AUROC 1.000) is a perfect *detector*; and
Diagnostics 8 + 8b proved that NEITHER label-free stat re-estimation NOR a
label-fitted BN affine recovers the collapse; both are negative.

**The separable finding (8b) is the biggest point:** the pre-BN per-class
separability is collapsed (conv_1.bn clean 0.704 → fog 0.101, crosstalk 0.137;
ratio 0.14-0.45 across all late BNs). The classes are merged at the BN *input*,
so no BN, which is only a per-channel scale + shift, can re-separate them. The
BN mismatch measured in Diagnostic 5 is a *symptom*, not the cause: the
information to name classes is destroyed at the input (D3: remission var →
1.3e-5), before any late stage. Caveat: the 8b negative is scoped to this affine
fit (clean-per-class-means target); a later iteration could pick a different
"correctness" metric for the BN update and get a different answer, but the
separability evidence locates the collapse upstream of BN regardless of metric.

So the direction is **NOT a BN re-anchor**. The two things that DO work:

1. **Input re-anchoring** (cov-shift's per-scan input-IN) rescues KITTI-C
   fog/crosstalk (+12-20 ceiling): it re-bases the collapsed remission channel
   at the input, where the information loss is (the only place the separability
   data says it can be recovered). Its cost is the 0.12 clean capacity gap (D1),
   which is why the conditional/stochastic-gate idea existed. The gate signal
   (Diagnostic 7) is a perfect detector, but Diagnostic 4/6 showed the
   stochastic model only weakly recovers healthy in OFF mode.
2. **The labeled corrupted pool** (W* / AL) reaches the ceiling; this is the
   existing AL/TTA mechanism and is unaffected by this result.

For the GeoID comparison, the pool-based AL is the cleaner claim: it does not
touch the healthy conditions at all, and it is what the existing TTA/AL line
already does. If DGLSS++ stays the default extractor (per the earlier decision:
a feature-extractor compromise as bad as the current cov-shift is not worth it),
then the fog/crosstalk story rides on the AL/TTA pool, not on the extractor or
the BN.
