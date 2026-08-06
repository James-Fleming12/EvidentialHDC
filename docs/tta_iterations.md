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

### Iteration 0: the labels' information is assignment

The full-label oracle and naive EMA use the **same weighted-mean prototype operator**; the only difference is the per-point class assignment. The oracle's advantage is therefore *assignment*, and this section quantifies that information.

**Pseudo-label accuracy on the collapsed conditions is near-random:**

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

**The labels rescue exactly the class-conditional casualties** (per-class val IoU: zero-shot → naive → oracle):

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
1. **The labels' information is correct class assignment, currently near-missing entirely**: pool pseudo-label accuracy is 26.4% (fog) and 33.7% (crosstalk) versus 66-79% on the geometric conditions. Every label-free prototype re-estimate is built from majority-contaminated means (fog Road's pseudo-prototype has precision 0.003, with 160k unlabeled points assigned into it).
2. **The oracle's full-scene gain is concentrated in the class-conditional casualties**: Traffic-sign, Other-ground, Vegetation, Building, and Car all jump dramatically under correct assignment (Traffic-sign 0.000 → 0.24-0.31, Vegetation up to 0.434). Road stays dead even with labels (features too collapsed); Bicycle is absent from the pool.
3. **naive EMA does not merely fail, it actively degrades**: classes the zero-shot decode handles OK (Car, Vegetation) get *worse* under pseudo-label re-estimation (fog Car 0.157 → 0.114, Vegetation 0.093 → 0.062), because the majority-wrong assignments contaminate their prototypes.
4. **The geometric conditions are assignment-healthy** (66-79% pseudo accuracy): their prototypes are only mildly contaminated, which is why they sit at or near the oracle already.

**Consequence**: recovering the oracle needs the *rare-class* assignments (13, 14, 15, 16, and to a lesser extent 4), where pseudo-label precision is currently 0.0-0.24, without degrading the already-correct majority assignments (Terrain, Car).

### Iteration 0.1: gating the pseudo-labels cannot reach the oracle

The **correct-subset gate bound** (re-estimating prototypes from pseudo-assigned points restricted to the *correct* subset, i.e., perfect gating of the pseudo-labels) lands at zero-shot, not the oracle:

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
1. **Perfect gating of pseudo-labels does NOT reach the oracle** (fog 10.3%, crosstalk 13.3% vs 16.4%/26.2%): even keeping only the correct pseudo-assigned points cannot re-estimate the prototypes well enough.
2. **The reason is recall starvation, not weighting.** The dying classes' points are overwhelmingly mis-assigned elsewhere, so the pseudo-decoder sees almost none of them and their correct-subset is tiny: fog Traffic-sign (13) has 73,193 true pool points but only 32 correct pseudo-assignments; crosstalk Traffic-sign has 101,158 true but 74 correct; Other-ground (14) 38-82k true but 198-327 correct; Vegetation (16) 26-76k true but 3.4k-3.9k correct. A 10kD prototype cannot be re-estimated from 32-198 points.
3. **The oracle's power is assignment recall on the rare classes**: it assigns the bulk of each class's points to the right prototype. No weight or gate on the current pseudo-labels can recover this; the assignments themselves must be fixed.
4. **The healthy classes are gating-fixable but don't need it**: Road/Terrain/Car/Building have large correct subsets and their naive prototypes already sit at cosine 0.91-1.0 to the oracle; their problem is not prototype estimation.

**Consequence**: a *weighting* head on the existing pseudo-labels is closed; the fix must improve the assignments themselves (a better label-free decoder).

### Iteration 1: no better label-free assignment source helps

The two candidate assignment sources for the re-estimate (the 128D linear probe, which is ~10 points more accurate on fog, and Multi-View Augmented Consensus with the canonical D3CTTA-style views) are no better than the 10kD zero-shot pseudo-labels:

| Condition | Zero-shot | zs-pseudo re-est | LP-pseudo | MVAC-LP | MVAC-proto | Full-label oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 10.1% | 9.2% | 8.5% | 8.4% | 9.2% | **16.5%** |
| crosstalk | 10.6% | 9.6% | 8.7% | 8.5% | 9.6% | **22.8%** |

**Findings:**
1. **MVAC is a no-op on this encoder.** MVAC-LP reproduces LP-pseudo and MVAC-proto reproduces zs-pseudo (identical within 0.001-0.003). The canonical geometric views (scale 0.9-1.1, yaw ±5-8°, dropout) do not change the 128D or 10kD predictions, so cross-view consensus adds nothing.
2. **A more accurate assignment source does NOT produce a better prototype re-estimate.** The LP's pseudo-labels are 36.7% accurate on fog and 35.1% on crosstalk (10 and 1.5 points above the 10kD zero-shot's 26.0%/33.6%), yet the LP-pseudo re-estimate (8.5%/8.7%) is *slightly worse* than the zs-pseudo one (9.2%/9.6%). Assignment accuracy alone does not determine re-estimate quality: the LP's error structure (which classes it gets wrong) matters more than its overall accuracy, and both sources fail on the rare-class recall the oracle's gain depends on.

**Consequence**: the "better assignment source" direction is closed. The oracle's advantage (rare-class assignment) is not reachable by any label-free classifier tested.

### Iteration 2: forcing rare-class support hurts the re-estimate

Sinkhorn-Knopp balanced the re-estimate pool's class marginals toward the source prior (SHOT's diversity guardrail without entropy-min or backprop) to counter the rare-class recall starvation. The prior was enforced (support-match 0.5; all small τ converge to the same prior-support fixed point), and the result is worse:

| | Zero-shot | zs-pseudo re-est | balanced-hard | balanced-soft | Full-label oracle |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | 10.1% | 9.4% | **3.9%** | **8.5%** | 16.5% |
| crosstalk | 12.0% | 10.7% | **5.6%** | **7.2%** | 26.2% |

**Finding: the diversity guardrail at the assignment level hurts.** Forcing rare-class support assigns *wrong* points into the rare classes (the features cannot separate them), contaminating their prototypes and dropping the re-estimate below the already-weak zs-pseudo baseline. The oracle's gain requires *correct* assignment; balance-forcing cannot substitute for it.

### Iteration 3: magnitude clipping cannot touch the binarized decode

ReAct-style clipping of the 128D feature norms (thresholds 3/4/5/6/8/inf) before the HDC projection + Sign() binarization had **zero effect at every threshold, on every condition** (fog mIoU 0.1011, bin_cos 0.1586 identical at clip=3 with 98.4% clipped and at no-clip).

**Finding: Sign() is invariant to positive per-point scaling.** `sign(s·(x @ proj)) == sign(x @ proj)`, so the binarized vectors are bit-for-bit identical and the decode cannot change. The autopsy's magnitude inflation is a *correlate* of the fog artifacts (useful as a gating signal), not the *mechanism* of the binarized-decode failure: the direction (angle) of the corrupted features is what is wrong. The norm-based retention gate works because it *removes* points, not because it alters their vectors.

### Iteration 4 (planned): what the labels carry that the features cannot derive

The remaining question is not "can we reach the oracle" but "precisely what information do the labels provide that the feature space and the 10kD prototypes cannot, and where does it live?" Implemented as `--deep_label_analysis` (keeps the clean features; use `--corruptions fog,crosstalk` first, a full 8-condition run is fine at a few hours). Three analysis tracks, all per condition:

**A. Feature geometry: why do classes collapse under corruption but not under supervised clean training?**
- Per class (clean vs corrupt): centroid cosine shift, norm inflation, intra-class tightness (mean cosine to own centroid), and **inter-class absorption** (distance to the nearest *other* clean centroid, clean vs corrupt).
- The hypothesis this tests: the collapsing classes (Road, Building, Other-ground, Traffic-sign, Bicycle) sit close to a majority neighbor in clean space, so under corruption their shifted features get absorbed; survivors (Terrain, Truck) sit far from competitors. Also reports the LP's per-class clean vs corrupt accuracy to confirm the collapse is corruption-specific, not a clean-decodability difference.

**B. Pseudo-label error analysis: how wrong, why wrong, and the impact of being wrong.**
- Per true class: the top-3 destinations of its points under the 10kD zs assignment (the confusion structure).
- Per class and per source (zs, LP): prototype contamination (assignment precision + cosine of the pseudo re-estimated prototype to the oracle prototype).
- Per-class IoU of the zs re-estimate, the LP re-estimate, and the oracle re-estimate: which classes' assignment errors cost the most.

**C. Recoverability / confidence: why does the oracle retrieve good results when no confidence signal can?**
- Split the val points the oracle rescues (zs-wrong, oracle-right) from the points wrong even under the oracle.
- AUROC of every label-free signal (norm, margin, LP confidence, oracle perceptron loss) for separating the two groups. **If all AUROCs ≈ 0.5, the recovered points are feature-space-indistinguishable from the stuck ones: the label information is genuinely inaccessible from the features, which is the precise sense in which the oracle is label-gated.**

### Candidate mechanisms: all closed or judged not worth running

- **ReAct (norm clipping before projection): CLOSED (Iteration 3).** Sign() is scale-invariant, so magnitude clipping cannot change the binarized decode (verified empirically: identical at every threshold).
- **LAME (latent-space marginalization): not worth running.** Its precondition (Iteration 2 being at least flat) failed; the balanced assignment it would build on assigns wrong points to rare classes, and affinity smoothing would only spread that further.
- **Black-box label shift: not worth running.** It is a data-driven variant of the decision-level prior correction (Phase 24.8, flat-to-negative everywhere) whose test distribution is ≈ the source distribution under fog; the distribution-forcing mechanism was additionally tested in Iteration 2 and hurt.
