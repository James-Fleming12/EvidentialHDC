# Test-Time Adaptation (TTA) Iterations

## Goal: the target each TTA method must chase

The prototype decoder is not exhausted. Re-estimating the 10kD prototypes from the corrupted stream **with true labels** (the full-label oracle) recovers the collapsed conditions on the **full scene, no points removed**, and the result is pool-size-stable (200k / 500k / 1M pools agree; Phase 24.10). The project goal is a **label-free TTA method whose prototype re-estimation replicates the oracle's per-point weighting**, reaching each condition's oracle mIoU without labels.

### Per-condition targets

| Condition | Zero-shot mIoU | Full-label oracle (target) | Gain to close |
| :--- | :--- | :--- | :--- |
| **Fog** | 10.1% | **16.6%** | +6.5 |
| **Crosstalk** | 12.0% | **26.2%** | +14.2 |
| Wet Ground | 49.0% | 51.2% | +2.2 |
| Cross Sensor | 41.5% | 43.6% | +2.1 |
| Snow | 39.4% | 40.7% | +1.3 |
| Incomplete Echo | 41.2% | 41.2% | 0.0 |
| Beam Missing | 53.7% | 53.6% | −0.1 |
| Motion Blur | 44.3% | 44.8% | +0.5 |

**Reading the table:**
- **Fog and crosstalk are the primary targets** (the only conditions with large label-gated headroom; full-scene, no point removal).
- **The geometric conditions are already at or near the oracle**: a TTA method must not regress them; their target is "hold zero-shot."
- **Key constraints learned so far:**
  - Artifact-free selection ≈ full-label (Phase 24.9): the problem is estimating the oracle's per-point *weights*, not dropping artifacts.
  - Every self-supervised variant tested so far fails to reach the oracle: naive EMA (fog 9.3), soft-dual-weight (9.4), BN-stat alignment (10.7), artifact gating (retained-subset only), prior correction (closed), prototype rebalancing (flat). The oracle is the only thing that moves fog full-scene.

## Baseline: where every tested TTA variant currently lands

Full-scene mIoU, plain medium encoder, Phase 24.9 harness (200k pool / 100k val):

| Condition | Zero-shot | naive EMA | SDW | BN-align | Full-label oracle |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 10.1% | 9.3% | 9.4% | 10.7% | **16.6%** |
| crosstalk | 12.0% | 10.7% | 10.6% | 15.1% | **26.2%** |
| snow | 39.4% | 38.1% | 37.9% | 39.9% | 40.7% |
| wet_ground | 49.0% | 47.7% | 47.4% | 40.5% | 51.2% |
| incomplete_echo | 41.2% | 39.4% | 39.2% | 40.6% | 41.2% |
| beam_missing | 53.7% | 51.6% | 51.4% | 53.6% | 53.6% |
| motion_blur | 44.3% | 43.0% | 42.8% | 44.8% | 44.8% |
| cross_sensor | 41.5% | 39.7% | 39.4% | 42.6% | 43.6% |

**The gap to close**: the difference between the "label-free TTA" columns and the "full-label oracle" column, which is ~6-14 mIoU on fog/crosstalk and ~0-2 on the geometric conditions.

## Iteration Log

### Iteration 0: what the labels give (complete)

**Setup**: `--iter0_label_info` on the plain medium encoder (500k pool / 100k val). naive EMA and the full-label oracle use the **same weighted-mean prototype operator**; the only difference is the per-point class assignment. So the labels' information is *assignment*, and this run quantifies it.

**Overall: pseudo-label accuracy on the collapsed conditions is near-random.**

| Condition | Zero-shot mIoU | naive EMA | Full-label oracle | Pool pseudo-label acc |
| :--- | :--- | :--- | :--- | :--- |
| fog | 10.1% | 9.3% | **16.4%** | **26.4%** |
| crosstalk | 12.0% | 10.7% | **26.2%** | **33.7%** |
| snow | 39.4% | 38.1% | 40.7% | 66.5% |
| wet_ground | 49.0% | 47.7% | 51.5% | 68.8% |
| incomplete_echo | 41.2% | 39.5% | 41.2% | 78.9% |
| beam_missing | 53.7% | 51.6% | 53.7% | 77.2% |
| motion_blur | 44.3% | 43.0% | 44.8% | 73.3% |
| cross_sensor | 41.5% | 39.7% | 43.5% | 68.8% |

**Per-class: the labels rescue exactly the class-conditional casualties** (per-class val IoU: zero-shot → naive → oracle):

| Class | fog | crosstalk |
| :--- | :--- | :--- |
| Traffic-sign (13) | 0.000 → 0.000 → **0.243** | 0.001 → 0.001 → **0.309** |
| Other-ground (14) | 0.005 → 0.007 → **0.160** | 0.004 → 0.011 → **0.177** |
| Vegetation (16) | 0.093 → 0.062 → **0.177** | 0.050 → 0.031 → **0.434** |
| Building (15) | 0.024 → 0.045 → 0.047 | 0.114 → 0.073 → **0.235** |
| Car (4) | 0.157 → 0.114 → **0.208** | 0.346 → 0.247 → **0.423** |
| Terrain (11) | 0.526 → 0.514 → 0.462 | 0.424 → 0.474 → 0.475 |
| Road (7) | 0.003 → 0.003 → 0.014 | 0.022 → 0.016 → 0.035 |

**Findings:**
1. **The labels' information is correct class assignment, and it is currently near-missing entirely**: pool pseudo-label accuracy is 26.4% (fog) and 33.7% (crosstalk) versus 66-79% on the geometric conditions. Every label-free prototype re-estimate is built from majority-contaminated means (e.g., fog Road's pseudo-prototype has precision 0.003, with 160k unlabeled points assigned into it).
2. **The oracle's full-scene gain is concentrated in the class-conditional casualties**: Traffic-sign, Other-ground, Vegetation, Building, and Car all jump dramatically under correct assignment (Traffic-sign 0.000 → 0.24-0.31, Vegetation up to 0.434). Road stays dead even with labels (features too collapsed); Bicycle is absent from the pool.
3. **naive EMA does not merely fail, it actively degrades**: classes the zero-shot decode handles OK (Car, Vegetation) get *worse* under pseudo-label re-estimation (fog Car 0.157 → 0.114, Vegetation 0.093 → 0.062), because the majority-wrong assignments contaminate their prototypes.
4. **The geometric conditions are assignment-healthy** (66-79% pseudo accuracy): their prototypes are only mildly contaminated, which is why they sit at or near the oracle already.

**Implication for Iteration 1**: the loss-estimator head does not need to solve assignment globally; it needs to recover the *rare-class* assignments (13, 14, 15, 16, and to a lesser extent 4) where pseudo-label precision is currently 0.0-0.24, and it must not degrade the already-correct majority assignments (Terrain, Car).

### Iteration 0.1: how the prototypes should be updated: gating is NOT the path (complete)

**Setup**: `--iter0_update_diag` on the plain medium encoder (500k pool / 100k val). Distinguishes a gating problem (correctly-assigned pseudo points are informative but drowned out) from an overrun problem (minority classes swamped by majority artifacts regardless of weighting). Key number: the **correct-subset gate bound** (re-estimate prototypes from pseudo-assigned points restricted to the *correct* subset, i.e., perfect gating of pseudo-labels).

**The gate-bound ladder (full-scene mIoU): zero-shot → naive pseudo → gate bound → full-label oracle**

| Condition | Zero-shot | naive pseudo | **Gate bound** | Full-label oracle |
| :--- | :--- | :--- | :--- | :--- |
| fog | 10.1% | 9.3% | **10.3%** | 16.4% |
| crosstalk | 12.0% | 10.7% | **13.3%** | 26.2% |
| snow | 39.4% | 38.1% | 39.7% | 40.7% |
| wet_ground | 49.0% | 47.7% | 49.3% | 51.5% |
| incomplete_echo | 41.2% | 39.5% | 40.8% | 41.2% |
| beam_missing | 53.7% | 51.6% | 53.0% | 53.7% |
| motion_blur | 44.3% | 43.0% | 43.8% | 44.8% |
| cross_sensor | 41.5% | 39.7% | 42.0% | 43.5% |

**Findings:**
1. **Perfect gating of pseudo-labels does NOT reach the oracle**: the gate bound is 10.3% on fog and 13.3% on crosstalk, essentially at zero-shot, while the oracle is 16.4% and 26.2%. The gating hypothesis is falsified; even keeping only the correct pseudo-assigned points cannot re-estimate the prototypes well enough.
2. **The reason is recall starvation, not weighting.** The dying classes' points are overwhelmingly mis-assigned elsewhere, so the pseudo-decoder sees almost none of them and their correct-subset is tiny: fog Traffic-sign (13) has 73,193 true pool points but only 32 correct pseudo-assignments; crosstalk Traffic-sign has 101,158 true but 74 correct; Other-ground (14) 38-82k true but 198-327 correct; Vegetation (16) 26-76k true but 3.4k-3.9k correct. A 10kD prototype cannot be re-estimated from 32-198 points, so gating the existing assignments has nothing to work with.
3. **The oracle's power is assignment recall on the rare classes**: it assigns the bulk of each class's points to the right prototype (Traffic-sign 0.000 → 0.24-0.31 IoU, Vegetation up to 0.434). No weight or gate on the current pseudo-labels can recover this; the assignments themselves must be fixed.
4. **The healthy classes are gating-fixable but don't need it**: Road/Terrain/Car/Building have large correct subsets and their naive prototypes already sit at cosine 0.91-1.0 to the oracle; their problem is not prototype estimation.
5. **The 128D linear probe is a strictly better assignment source than the 10kD zero-shot** (fog LP 36.3% vs zs 26.4%, and 57% under the additive regimen), so an improved assignment source is available label-free; whether it is enough to approach the oracle is the next test.

**Revised implication for Iteration 1**: a *weighting* head on the existing pseudo-labels is closed. Iteration 1 must **improve rare-class assignment** (recall on classes 13, 14, 16, 15, 4), i.e., the head's role is to produce better pseudo-labels (a better label-free decoder), not to weight them. The natural first test is the 128D-probe-based re-estimate (the LP's assignments are already ~10 points more accurate than the 10kD zero-shot).

### Iteration 1: the learned loss-estimator head (proposed, not yet run)

The leading candidate: a small head trained (on clean/self-supervised signal) to predict each point's cos-to-true at test time, used as the per-point weight (or pseudo-label confidence) in a prototype re-estimate (Phase 24.4 #3). Rationale: the oracle's advantage over every label-free variant is exactly the per-point cos-to-true it has and they lack; Iteration 0 shows the labels' information is *assignment*, concentrated in the rare classes (13, 14, 15, 16, 4), where pseudo-label precision is currently 0.0-0.24.

Success criterion: fog full-scene mIoU moving from ~10% toward 16.6%, and crosstalk from ~12% toward 26.2%, with the gain concentrated in the rare classes and without regressing the geometric conditions (or Terrain/Car) beyond their zero-shot level, using a label-free (or clean-only-supervised) estimator.
