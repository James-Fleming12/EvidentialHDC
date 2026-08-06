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

### Consolidated iteration table

All full-scene mIoU (no points removed) on the plain medium encoder. The pool/val split varies slightly between harnesses, so the oracle baseline reads 16.4-16.6% (fog) and 22.8-26.2% (crosstalk) depending on the run; **within-row comparisons are the valid ones**, and the canonical oracle target is the pool-size-stable 16.6% / 26.2% (Phase 24.10).

| Iteration | Method | fog mIoU | crosstalk mIoU | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| baseline | Zero-shot (frozen clean prototypes) | 10.1% | 12.0% | reference |
| target | Full-label oracle (true assignment) | **16.6%** | **26.2%** | the goal; label-gated |
| 0 | Labels = correct assignment (diagnostic) | n/a | n/a | rare classes starved: pseudo acc 26.4% / 33.7%; oracle gain is rare-class recall |
| 0.1 | Correct-subset gate bound (perfect gating of pseudo-labels) | 10.3% | 13.3% | gating closed: correct pseudo points too few to re-estimate |
| 1 | zs-pseudo prototype re-estimate | 9.2% | 9.6% | flat, below zero-shot |
| 1 | LP-pseudo re-estimate (36.7% / 35.1% acc) | 8.5% | 8.7% | assignment accuracy ≠ re-estimate quality: closed |
| 1 | MVAC-LP / MVAC-proto consensus | 8.4% / 9.2% | 8.5% / 9.6% | no-op: canonical views don't change predictions |
| 2 | Balanced (Sinkhorn) hard / soft | 3.9% / 8.5% | 5.6% / 7.2% | hurts: forcing rare-class support assigns wrong points |
| 3 | ReAct norm clipping (all thresholds) | 10.1% | 12.0% | non-starter: Sign() is scale-invariant |
| 4 | Deep label analysis (diagnostic) | n/a | n/a | labels = tight shifted clusters; fog norm AUC 0.75, crosstalk label-only |
| 5 | Combined-recoverability gate (weighted re-estimate / retention) | 9.4% | 10.9% | not validated: AUROC does not transfer to full-scene decode gains |

**Bottom line**: every label-free route to the oracle is tested and closed. The full-label oracle (fog 16.6%, crosstalk 26.2%) remains the only thing that recovers the collapsed conditions; its information is the rare-class *assignment* the features cannot provide label-free under fog/crosstalk.

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

### Iteration 4: what the labels carry that the features cannot derive

`--deep_label_analysis` on the plain medium encoder (fog + crosstalk). Three tracks: feature geometry (survivor vs collapser), pseudo-label error/contamination, and recoverability (which label-free signals separate oracle-rescued from oracle-stuck points).

**A. Feature geometry: the collapse is not dispersion, not drift magnitude, and not LP-separability in 128D.**

- **Every class decodes fine under supervised clean training** (LP clean acc 0.72-0.98 for classes 4, 11, 13, 14, 15, 16), so the collapse is corruption-specific.
- **The surviving class is Terrain (11), and it is distinguished by isolation, not stability**: its clean nearest-other-class distance is 0.989 (nearly orthogonal to every competitor) and it stays far after drifting. The collapsing classes sit closer to neighbors in clean space and several move *into* a neighbor under corruption (Traffic-sign 13: 0.677 → 0.598/0.479; Other-ground 14: 0.517 → 0.507).
- **Drift magnitude is NOT the killer**: Terrain has the *largest* centroid shift (cos_shift 0.513 fog / 0.715 crosstalk) and survives; Traffic-sign/Vegetation/Other-ground drift almost not at all (cos_shift 0.007-0.061) and collapse. The collapsers' clusters stay tight under corruption (tight_corrupt 0.90-0.97): they do not disperse, they shift as coherent clusters.
- **The LP's "36% fog accuracy" is almost entirely Terrain**: per-class LP corrupt accuracy is near-zero for every other class (Car 0.005, Road 0.000, Building 0.006, Traffic-sign 0.000, Vegetation 0.025). The continuous 128D space is not actually linearly separable for the collapsing classes under corruption; the LP figure overstates the representational headroom.

**B. Pseudo-label errors: the re-estimate prototypes are majority-contaminated.**

- zs-assignment precision on the collapsing classes is 0.10-0.25 (Traffic-sign 0.102, Other-ground 0.148, Building 0.100, Vegetation 0.254); the pseudo re-estimated prototypes are mostly wrong-class points, and their cosine to the oracle prototype (0.71-0.91) is not close enough.
- Per-class impact confirms it: the oracle recovers Traffic-sign 0.002 → 0.234, Other-ground 0.005 → 0.159, Vegetation 0.070 → 0.185, Car 0.120 → 0.193, while the LP re-estimate is *worse* than the zs re-estimate on Car (0.027 vs 0.120) and Vegetation (0.056 vs 0.070), consistent with Iteration 1.

**C. Recoverability: crosstalk is label-only; fog has a real, exploitable signal.**

AUROC of each label-free signal for separating oracle-rescued (zs-wrong, oracle-right) from oracle-stuck points:

| Signal | fog AUC | fog mean rec/stuck | crosstalk AUC |
| :--- | :--- | :--- | :--- |
| norm | **0.747** | 8.2 / 6.5 | 0.544 |
| margin | 0.588 | 0.110 / 0.116 | 0.423 |
| LP confidence | 0.577 | 0.907 / 0.887 | 0.597 |
| oracle loss | 0.436 | 0.161 / 0.207 | 0.397 |

- **Crosstalk: genuinely label-only** (all AUROCs 0.40-0.60): no label-free signal separates the points the oracle rescues from the points it cannot. This is the precise sense in which the crosstalk oracle is label-gated.
- **Fog: the recovered points are the HIGH-norm zs-wrong points** (norm AUC 0.747, means 8.2 vs 6.5). This is a new, actionable lead: among zs-wrong fog points, higher norm predicts oracle-recoverability. It is the first label-free signal shown to carry information about which fog errors are fixable, and it inverts the naive "high norm = dead artifact" reading for the recoverable set.

**Implication**: the labels provide the per-point cluster assignment that no decoder follows because the collapsing classes shift as tight clusters toward neighbors while the clean prototypes stay stale (and the LP never saw the corrupted clusters). The fog norm-AUROC result suggests a norm-conditioned re-estimate or weighting could recover part of the fog gap label-free; the crosstalk case has no such lead.

**Extension for the both-conditions goal**: Part C was extended with a signal battery + a joint classifier to look for a mechanism that separates oracle-recovered from oracle-stuck points on BOTH conditions, not just fog. New signals: cos128 (nearest clean-prototype cosine), per-class z-scored norm (norm_z), LP softmax entropy and top-2 margin, kNN local agreement (fraction of 128D neighbors sharing the assignment), plus a logistic-regression combination of all label-free signals (trained/evaluated on halves of the recovered/stuck sets).

**Extended recoverability AUROCs (recovered vs stuck):**

| Signal | fog AUC | crosstalk AUC |
| :--- | :--- | :--- |
| norm | **0.747** | 0.544 |
| norm_z | 0.619 | 0.545 |
| LP confidence | 0.576 | 0.595 |
| LP margin | 0.577 | **0.593** |
| margin | 0.588 | 0.423 |
| cos128 | 0.479 | 0.393 |
| kNN agreement | 0.505 | 0.496 |
| LP entropy | 0.425 | 0.399 |
| **combined (all signals)** | **0.799** | **0.680** |

**Finding: a joint label-free signal separates recovered from stuck points on BOTH conditions.** Single signals are weak on crosstalk (best 0.60), but the combined classifier reaches 0.799 on fog and 0.680 on crosstalk. The recoverability information is present in the *joint* label-free feature space on both conditions, not just fog. The two conditions lean on different features (fog on the magnitude signals norm/norm_z, crosstalk on the probe-confidence signals LP-conf/LP-margin), so a shared mechanism must combine both families, exactly the feature set a learned per-point recoverability/loss-estimator head would take as input. This is the first evidence that a both-conditions label-free mechanism is reachable in principle; the open question is whether the combiner can be learned without oracle labels (clean/self-supervised training).

### Iteration 5: the combined-recoverability path is not validated

`--combined_gate_sweep` on all 8 conditions: does the joint recoverability signal (Part C combined AUROC 0.68-0.80) translate into decode gains? Five fixed z-scored configs (norm, lp, norm+lp, full, full_no_norm; no oracle) were applied two ways: (a) re-estimate-side weighting (weight the prototype re-estimate toward high-recoverability points, full-scene mIoU) and (b) decode-side retention (top-25/50/75% retained mIoU).

**Results (full-scene mIoU):**

| Condition | Zero-shot | zs-pseudo re-est | **Best weighted** | Oracle |
| :--- | :--- | :--- | :--- | :--- |
| fog | 10.1% | 9.3% | **9.4%** | 16.4% |
| crosstalk | 12.0% | 10.7% | **10.9%** | 26.2% |
| snow | 39.4% | 38.1% | 38.2% | 40.7% |
| wet_ground | 49.0% | 47.7% | 47.7% | 51.5% |
| beam_missing | 53.7% | 51.6% | 51.7% | 53.7% |
| motion_blur | 44.3% | 43.0% | 43.1% | 44.8% |
| cross_sensor | 41.5% | 39.7% | 39.8% | 43.5% |

Decode-side retention was flat-to-worse: the best recoverability config reached fog 9.6% and crosstalk 12.2% at top-50%, *below* the Phase 23 margin+cosine gate (crosstalk 23.1% @ 51.6%).

**Findings:**
1. **Weighting the re-estimate by the recoverability score changes nothing** (fog 9.3 → 9.4, crosstalk 10.7 → 10.9). The reason is structural and reconfirms Iteration 1: weighting cannot change the *assignment*, and the recoverable points are assigned to the wrong prototypes regardless of their weight.
2. **The recoverability configs are worse retention gates than the old margin+cos gate** on crosstalk (12.2% vs 23.1% @ ~50%). The joint signal separates *recovered from stuck* (a classification of the val) but is a weaker *artifact-rejection* signal than the Phase 23 margin/cosine geometry.
3. **The one positive: no collapse in the geometric conditions** (all hold at the plain re-estimate level), so the weighting mechanism is safe even if useless.
4. **Conclusion: the combined-recoverability path is not validated.** The Part C AUROC did not transfer to full-scene decode gains through either weighting or retention. This is consistent with the arc across all iterations: the oracle's information is the rare-class *assignment*, and no label-free weighting, gating, balancing, assignment-source, or magnitude intervention of the existing assignment reaches it. The recoverability signal identifies *which* points are recoverable but not *what class* they should carry, and the decode cannot use the former without the latter.

### Candidate mechanisms: all closed or judged not worth running

- **ReAct (norm clipping before projection): CLOSED (Iteration 3).** Sign() is scale-invariant, so magnitude clipping cannot change the binarized decode (verified empirically: identical at every threshold).
- **LAME (latent-space marginalization): not worth running.** Its precondition (Iteration 2 being at least flat) failed; the balanced assignment it would build on assigns wrong points to rare classes, and affinity smoothing would only spread that further.
- **Black-box label shift: not worth running.** It is a data-driven variant of the decision-level prior correction (Phase 24.8, flat-to-negative everywhere) whose test distribution is ≈ the source distribution under fog; the distribution-forcing mechanism was additionally tested in Iteration 2 and hurt.
