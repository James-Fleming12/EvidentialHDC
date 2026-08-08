# Evidential Head and Per-Class Generalization Iterations

The encoder/head-side attempts to close the oracle gap (full-label oracle: fog 16.6%, crosstalk 26.2%, vs the ~10-12% label-free ceiling) after the decode-side TTA line closed (`docs/tta_iterations.md`). Two mechanisms were proposed on the plain `supcon_vib` base: **Addition 1** (per-class generalization of the fragile classes) and **Addition 2** (an intrinsic uncertainty head). The goal for both is a **fog AND crosstalk** mechanism, since the method must not favor one condition.

## Prerequisites (the evidence each design was built on)

- **The collapse is class-conditional** (Phase 24.2): under fog/crosstalk the casualties (2, 7, 13, 14, 15) collapse while Terrain/Truck survive; the collapse is regimen-invariant (Phase 24.6).
- **The fragile classes are not linearly separable under corruption** (Iteration 4B): per-class LP corrupt accuracy is 0-2.5% for every class except Terrain (0.91); the "LP fog 36%" headline was almost entirely Terrain.
- **The fragile classes get absorbed by neighbors** (Iteration 4A): Building's clean nearest-other distance is 0.042; the collapsers sit close to a majority neighbor.
- **A recoverability signal exists and is measurable** (Iteration 4C extension): the combined label-free signals reach AUROC 0.68-0.80 for separating recoverable from stuck points.
- **Confidence is miscalibrated under fog** (Phase 22.2): 99.96% of fog misclassifications are confident artifacts.

## Addition 1: per-class generalization (fragile-class SupCon): DELAYED (revisit after the evidential head)

**Design intuition**: up-weight the SupCon anchors of the casualty classes (2, 7, 13, 14, 15) across the clean-corrupted pair, so the contrastive signal concentrates on making those classes separable under corruption.

**Setup**: `supcon_vib_fragile` (per-anchor weight 3x on the casualties), micro probe vs plain (3-epoch config):

| Metric | plain | fragile w=3 | Delta |
| :--- | :--- | :--- | :--- |
| clean proto mIoU | 36.1% | **38.9%** | +2.8 |
| fog proto mIoU | 2.5% | **3.4%** | +34% |
| **crosstalk proto mIoU** | 5.6% | **3.7%** | **-34%** |

**Results**: clean improves (+2.8, no regression), fog improves (+34%), but **crosstalk gets worse (-34%)**; the lever is also weight-sensitive (w=6 went negative on fog).

**Verdict: delayed, not closed.** The blanket fragile-class weighting is fog/clean-positive but crosstalk-negative: the same casualties it targets under fog are hurt under crosstalk, so it does not qualify for the both-conditions goal *yet*. The fog/clean benefit is real and reproducible (+34% fog, +2.8 clean, no regression), and the likely fix is the same crosstalk-style augmentation being added to the loss-prediction head (sparse wrong-beam returns, which the current fog-ish views lack). It is deferred behind the evidential head, which is the higher-priority both-conditions mechanism, and should be revisited right after that line resolves.

## Addition 2: the evidential uncertainty head: the iterated design

The goal: the model's own head outputs calibrated uncertainty (the gating signal), replacing the hand-built recoverability combiner (AUROC 0.68-0.80). Iterations in order:

| Iteration | Config | fog AUROC | crosstalk AUROC | fog u vs clean | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 25.2 | EDL, KL cap 1.0 | 0.403 | 0.363 | uniform | collapsed (KL 255x) |
| 25.3 | EDL, cap 0.005 | 0.332 | 0.464 | uniform | fixed structure, not calibrated |
| 25.4 | EDL, edl_w 1.0 | **0.740** | 0.351 | separated | fog validated, crosstalk not |
| 25.5 | EDL, selective KL | 0.208 | 0.309 | none | regressed, reverted |

### 25.2: EDL with KL cap 1.0: collapsed (KL 255x the CE)

**Setup**: `supcon_vib_evidential`: a 1x1-conv Dirichlet head on the 128D bottleneck; evidential CE on clean+augmented views plus a KL-to-uniform regularizer on the augmented view, `lam_kl = min(1, epoch/10)`. 26-epoch medium run.

**Results**:

| Metric | value |
| :--- | :--- |
| in-training `kl_ratio` | 255 (KL ~63 dominates; total loss sat at ~64, backbone healthy) |
| fog LP / proto mIoU | 9.3% / 1.5% (plain 17.1% / 5.2%) |
| clean LP | 91.7% (plain 91.4%) |
| mean uncertainty (clean / fog / crosstalk) | 0.586 / 0.589 / 0.588 (uniform) |
| correct-vs-wrong AUROC (fog / crosstalk) | 0.403 / 0.363 (anti-calibrated) |

**Fix**: the KL cap was the bug (255x). Made configurable (`--edl_kl_cap`, default 0.005, annealed `min(cap, epoch/100)`).

### 25.3: EDL with cap 0.005: structurally fixed, still not calibrated

**Setup**: same, cap 0.005 (26-epoch medium run).

**Results**:

| Metric | value |
| :--- | :--- |
| in-training `kl_ratio` | 1.28 (balanced); `edl` 2.24 -> 1.19 (head learned to classify) |
| fog LP / proto mIoU | 8.0% / 3.7% (plain 17.1% / 5.2%) |
| clean LP | 91.9% |
| mean uncertainty (clean / fog / crosstalk) | 0.583 / 0.587 / 0.586 (uniform) |
| correct-vs-wrong AUROC (fog / crosstalk) | 0.332 / 0.464 (anti-calibrated on targets) |

**Root cause**: the 0.1 evidential-CE weight starved the head, so it never built evidence on clean either (clean uncertainty at the softplus floor ~0.59); and the blanket KL-on-augmented forces uncertainty indiscriminately, teaching no selectivity.

**Decision**: the Dirichlet-KL design is closed as-is; the fix is a stronger classification weight and/or a direct target (see 25.4/25.6).

### 25.4: edl_w=1.0: validated for fog (AUROC 0.74), not crosstalk

**Setup**: same EDL, but `edl_w=1.0` (stronger classification weight). ~1h micro probe (8 epochs).

**Results**:

| Metric | value |
| :--- | :--- |
| probe decode (clean / fog / cross proto mIoU) | 42.8% / 5.1% / 6.0% |
| mean uncertainty (clean / fog / crosstalk) | 0.566 / **0.594** / 0.580 (clean-fog separation) |
| correct-vs-wrong AUROC (fog / crosstalk) | **0.740** / 0.351 |
| per-class fog uncertainty | casualties (0.586-0.599) > Terrain (survivor) |

**Verdict**: fix (a) works for fog; crosstalk's calibration gap is a coverage problem (the augmented views are fog-ish).

### 25.5: selective KL (fix b): regressed, revert to blanket

**Setup**: apply the KL only to augmented points the head currently predicts wrong (the "be uncertain where wrong" idea). ~1h micro probe.

**Results**:

| Metric | value |
| :--- | :--- |
| mean uncertainty (clean / fog / crosstalk) | 0.570 / 0.569 / 0.572 (no clean-fog separation) |
| correct-vs-wrong AUROC (fog / crosstalk) | 0.208 / 0.309 (regressed vs blanket) |

**Why**: the selective KL self-cancels: once the head classifies an augmented point correctly, the KL stops applying to it, so it never learns uncertainty behavior (matches the Bengs et al. result that second-order predictors cannot be faithfully optimized without a ground-truth uncertainty target).

**Decision**: revert to blanket edl_w=1.0 (fog AUROC 0.74 is the keeper); the selective-KL mechanism is abandoned.

## Current direction: direct loss prediction (supcon_vib_losspred)

**Design intuition** (from the Bengs et al. analysis and the measured recoverability signal): instead of an indirect Dirichlet-KL, a head that **directly regresses the main classifier's per-point CE** on clean + augmented views. The supervision is the semantic head's own error (no OOD labels), the output is a predicted per-point loss (the gating signal), and it is condition-agnostic. **Crosstalk-style augmentation added** (sparse wrong-beam returns, density 0.005) so the head sees crosstalk-hard points in training.

**Setup**: `supcon_vib_losspred` (128 -> 1 conv, smooth-L1 vs the per-point CE, weight `--edl_w`; `--edl_kl_selective` unused here). ~1h micro probe: `phase25_probe.py --methods supcon_vib_losspred --epochs 8 --edl_w 1.0`, then `evidential_eval.py --method supcon_vib_losspred`.

**Decision rule**: fog AND crosstalk correct-vs-wrong AUROC both > 0.5 -> the both-conditions head is validated and the 10h run is justified. If fog good but crosstalk ~0.5, tune the crosstalk augmentation (density/injection pattern). If both ~0.5, reconsider.

## Current state

- **Addition 1 is delayed, not closed**: a real fog/clean win (+34% fog, +2.8 clean, no regression) that fails crosstalk as-is; the likely fix is the crosstalk-style augmentation, and it is queued to revisit right after the evidential head resolves.
- **EDL (Dirichlet-KL) is closed**: the only validated config (blanket edl_w=1.0) gives a fog-calibrated head (0.74) with no crosstalk gain; both cap- and selectivity-based attempts fail on crosstalk.
- **The open bet is the direct loss-prediction head** (25.6), pending the ~1h probe; its validation criteria are the fog + crosstalk correct-vs-wrong AUROCs.
