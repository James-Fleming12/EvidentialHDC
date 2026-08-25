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

### Diagnostic 5 (PENDING): per-layer fog-collapse propagation probe

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

**Decisive split.** If the first block already saturates/zeros, the fix is at
the input (re-anchor range/remission before the trunk). If the collapse builds
gradually, a late per-scan re-normalization (or a data-dependent gate) can
rescue it without touching the input branch. This decides the design of the
extractor change.

**Runner.** `run_probe_fog_collapse_layer.sh` (DRY_RUN/SMOKE).

### Diagnostic 6 (PENDING): conditional input-IN at full scale

**Question.** Do the micro-scale stochastic-gate results
(+0.02-0.03 healthy recovery, −0.06 crosstalk cost) persist at full scale?

**Setup.** `run_micro_stoch.sh` pattern, but the full dataset harness
(`al_full_dataset_diag.py`) with a `input_in_prob`-trained checkpoint
(cov-shift `supcon_vib_dglsspp_inputin_in_chan_stoch{,_7,_9}`), decode with
input-IN ON vs OFF on all conditions. Compare frozen AND ceiling.

**Decisive split.** If the healthy recovery persists and the fog/crosstalk
rescue holds, the conditional extractor is the base for the TTA/AL line. If the
healthy cost reappears at full scale, revert to plain DGLSS++ + a *labeled*
pool-dependent fix (e.g., re-anchor the corrupted pool in the AL line).

### Diagnostic 7 (PENDING): per-scan detector signal for the gate

**Question.** What statistic reliably flags "this scan needs normalization" at
deployment, with no corruption-type knowledge? Candidates:

- input-channel stats (range/remission var — the failed eval-only gate used
  these, but with the wrong (non-conditional) model),
- activation statistics at a mid-network layer (mean/var/saturation),
- network-flow / entropy of the code or prototypes.

**Setup.** From the Diagnostic-5 per-layer activations, extract a per-scan
scalar signal (e.g., first-block mean activation magnitude, or BN-mismatch
norm). Calibrate a threshold on clean+corrupted scans; report AUROC for
"needs normalization" vs "healthy."

**Decisive split.** If a clean scalar separates fog/crosstalk from healthy
with high AUROC, the gate is label-free and deployable. If not, the mode switch
falls back to per-condition or task-aware selection.

---

## Decision rule

If Diagnostic 5 shows the collapse originates at the input (early saturation /
BN mismatch in the first block), the fix is **input re-anchoring** (per-scan
re-basing of range/remission, or a data-dependent input-IN gate) trained into a
single conditional model (Diagnostic 6/7). If the collapse is gradual or the
healthy cost persists, move to a **late normalization** or accept the plain
DGLSS++ healthy line and push fog/crosstalk through the AL/TTA pool instead.
