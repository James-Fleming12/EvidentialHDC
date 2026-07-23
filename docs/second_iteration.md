# Phase 2: Advanced Class-Balancing and Multi-View Architectures
**Date:** July 21, 2026

## Objective
Following the establishment of our core 4-tier adaptation pipeline (Network, Hypervector, and Temporal uncertainties with basic frequency protection) in the preliminary phase, the objective of Phase 2 is to rigorously refine the class-balancing mechanisms and introduce multi-view architectures. We aim to identify and exploit the highest leverage areas for improvement, specifically targeting rare-class preservation (inter-class balance), sub-cluster geometric integrity (intra-class balance), and multi-perspective spatial consistency.

## Background & Context

### 1. The Class-Balancing Challenge
During preliminary testing, we discovered that standard threshold-based Test-Time Adaptation suffers from massive "Voronoi Shattering," where majority classes (like Road or Vegetation) confidently dominate updates and geometrically crush rare class prototypes. 
While our initial heuristic—a deterministic frequency soft-dampener ($\gamma=0.1$)—successfully mitigated the worst of this "Majority Class Paralysis" without completely freezing the network, it remains a blunt instrument. 

In this phase, we will explore:
* **Advanced Inter-Class Balancing:** Moving beyond simple inverse frequency to true dynamic semantic weighting that adapts to the shifting class distributions of varying environments.
* **Intra-Class Balancing:** Rare sub-clusters (e.g., a specific pose of a Pedestrian) within a single class are often vetoed as Out-of-Distribution (OOD) noise. We will investigate methods to protect rare geometries from being smoothed out by the dense "core" of the class manifold, revisiting concepts like the Subcluster Ledger with more refined, non-restrictive math.

### 2. Multi-View Architectures (Microscopic Temporal Consistency)
Single-frame adaptation is inherently vulnerable to transient noise and occlusions. By leveraging multi-view architectures, we can enforce spatial and semantic consistency across multiple perspectives or consecutive temporal frames before committing to a permanent weight update. 

Crucially, if we move to disable static Macroscopic Temporal Drift tracking (e.g., the static `anchor_spring`) to allow the prototypes greater flexibility to adapt, Microscopic Temporal Consistency becomes the primary defense mechanism against catastrophic drift, dynamically filtering out transient geometric noise before it ever enters the continuous epistemic gates.

In this phase, we will investigate:
* **Multi-View Consistency Gating:** Requiring consensus across multiple spatial projections (or consecutive LiDAR sweeps) before allowing high-magnitude updates.
* **Feature Fusion:** Aggregating features from multiple views into the 10,000D hyperdimensional space to generate a more robust, physically grounded Dirichlet uncertainty prior.

## Part 1: Architecture Calibration & Normalization (Test A1 & C2)

Before implementing complex Multi-View or Intra-Class mechanisms, we conducted a structural audit to ensure the foundational pipeline was sound. We discovered three critical mathematical flaws in the adaptation update rules that were severely penalizing rare classes:

1. **Global Calibration (Test A1):** The Dirichlet gating (Hypervector Uncertainty) was calibrated globally. Rare classes naturally have lower geometric coherence (higher standard deviation) than dense classes like Road. By applying a global Z-score, the gate was inherently filtering out rare classes as "uncertain." We shifted to **per-class Z-score calibration** (`source_mu_cos[c]`, `source_density_std[c]`), allowing each class to be gated on its own respective sub-manifold scale.
2. **Update Normalization (Test C2):** The TTA update formula `c_update = (valid_enc * weights).mean(dim=0)` inherently coupled the magnitude of the rotation step to both the class point count and intra-class coherence. Diffuse tail classes yielded small-norm means, rotating slower than tight head classes. We applied `F.normalize(c_update, p=2, dim=0)` *before* multiplying by the learning rate, turning the learning rate into a pure angular rotation budget.
3. **Veto-Purity Bug:** Corrected a shape misalignment in the metric tracking where the model was falsely reporting how accurately it was vetoing errors.

### Results (Post-Calibration)

The impact of normalizing the adaptation steps and dynamically scaling the uncertainty gates per-class was staggering:

| Corruption | Base (Frozen) | Adapted (New) | $\Delta$ mIoU | Tail (Base) | Tail (New) | $\Delta$ Tail |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Snow | 0.4078 | **0.4840** | +7.62% | 0.1223 | **0.2846** | +16.23% |
| Incomplete Echo | 0.3760 | **0.4263** | +5.03% | 0.0504 | **0.1276** | +7.72% |
| Cross Sensor | 0.2587 | **0.3132** | +5.45% | 0.0564 | 0.0496 | -0.68% |
| Beam Missing | 0.3779 | **0.4253** | +4.74% | 0.1155 | 0.1211 | +0.56% |
| Motion Blur | 0.4061 | **0.4489** | +4.28% | 0.1354 | **0.1710** | +3.56% |
| Fog | 0.0583 | **0.0986** | +4.03% | 0.0006 | 0.0000 | -0.06% |
| Crosstalk | 0.0700 | **0.1139** | +4.39% | 0.0033 | 0.0000 | -0.33% |
| Wet Ground | **0.4245** | 0.3917 | -3.28% | 0.1805 | 0.1619 | -1.86% |

### Key Takeaways
* **Structural Recovery:** Under the previous legacy pipeline, `beam_missing` was being actively degraded by the network (dropping to 0.1936). With the fixed normalized tracking and independent gates, `beam_missing` leaps by +4.74% mIoU.
* **Massive Rare-Class Rescue:** By normalizing the `c_update` step, tail classes finally had enough rotation budget to adapt to environmental snow noise. The Tail mIoU for `snow` skyrocketed by **+16.23%**. `incomplete_echo` saw similar tail recoveries (+7.72%). 
* **Wet Ground Degradation:** `wet_ground` remains uniquely adversarial. Despite structural improvements elsewhere, adaptation actively harmed it by -3.28%. This points towards a fundamental domain shift where the reflection geometry entirely shatters the semantic manifold in a way that continuous tracking cannot heal.

## Phase 2 Test Plan (v2): Post-Calibration Next Steps
**Updated July 21 (Post A1+C2)**

The empirical results above fundamentally re-sequence the Phase 2 agenda. The A1+C2 fixes successfully rescued the tail on noise corruptions (Snow, Incomplete Echo, Motion Blur), but failed on structural corruptions (Beam Missing) and actively regressed reflection corruptions (Wet Ground). 

The likely mechanism behind the Wet Ground regression: reflections produce confident-but-wrong road labels. Road has a high intrinsic $\sigma$, so per-class calibration (A1) widened the road acceptance band, admitting reflections. C2 then gave those noisy updates full rotation magnitude instead of shrinking them, causing the head prototypes to drift on phantom geometry. 

Phase 2 will now proceed as follows:

### Part 0: Instrumentation Debt (Immediate)
Before testing new methods, we must unblock interpretation by tracking mechanism diagnostics per-class:
* **Per-class prototype rotation** ($\angle(w_c^t, w_c^0)$) to see if tail classes are even moving on structural corruptions.
* **Per-class firing rate** to see what classes are triggering updates.
* **Per-class veto purity** to verify if the road gate is now leaking reflections on `wet_ground`.

### Part 1: Fix the Wet Ground Regression (Update: July 22)

**Overnight Sweep Results (Test 1b and 1c)**
We ran a grid sweep over `test_1b` (Step Magnitude Throttles) and `test_1c` (Partial Calibration Shrinkage) to attempt to decouple the C2 *direction* from the C2 *magnitude*. The results definitively isolated the bug, though the proposed mitigations failed:
1. **The Diagnostic Truth (Test 1a):** The per-class diagnostics revealed that on `wet_ground`, classes 11 (bicyclist) and 13 (motorcyclist) were rotating a staggering **50 to 60 degrees**, despite only firing on **0.1%** of frames. The C2 normalization fix (`F.normalize(c_update)`) gave tail classes the budget to adapt to noise, but it unintentionally granted a single noisy phantom reflection point a full `update_lr=0.01` unit step. Over 4000 frames, these isolated false-positives completely shattered the prototypes.
2. **The Failure of Evidence Throttling (Test 1b):** 
   - `mean_evidence` caused complete model collapse (mIoU dropped to ~0.27). Because Dirichlet evidence (`softplus(5*Z)`) is unbounded, highly coherent clusters generated scalars of 5.0+, massively amplifying the learning rate and exploding the gradients.
   - `coherence` (L2 norm of the batch) successfully constrained the gradients but did so uniformly (scaling all steps by ~0.65). This slowed the `wet_ground` destruction slightly but failed to selectively punish noisy single-point updates.
3. **The Failure of Statistical Shrinkage (Test 1c):** Shrinking the calibration bounds (`0.75` and `0.5`) had almost zero effect on `wet_ground` performance. The reflections geometrically lie inside the core road manifold; statistical gating cannot distinguish them.

**The New Hypothesis & Solution (Morning Update: July 22):**
The initial assumption that the -3.28% `wet_ground` regression was entirely due to tail-class shattering (bicyclist rotating 60 degrees on isolated noise) was incomplete. While the tail *did* shatter due to the C2 fix granting unit steps to 0.12%-confidence noise, the bulk of the mIoU drop must have stemmed from high-mass head classes like Road drifting on large specular puddles. 
We tested three new mechanisms to address the accumulation of this drift:
## Phase 2: Audit of Phase 1 Takeaways and Balancing Test Suite
**Date:** July 22, 2026 (v3)

## Objective
Establish mathematically rigorous diagnostics to audit the early Phase 1 hypotheses, and explicitly shift from heuristic fixes to explicit magnitude/rotation schedules and inter/intra-class balancing. 

---

## Part A: The True Phase 1 Takeaways

### A0. The PyTorch `.data` bug was a Per-Class, Time-Decaying Learning Rate
`model.classify.weight[c].data = F.normalize(...)` failed to overwrite the tensor in-place, meaning the un-normalized weight accumulated steps infinitely: `w_c += step`.
Because the classification layer always normalized on the forward pass (`proto_norm = F.normalize(model.classify.weight, dim=1)`), the unconstrained magnitude *only* affected the angular step size:
`rotation per step ≈ step_mag / ||w_c||`
As `||w_c||` grew linearly via accumulation, the effective angular learning rate decayed like **$1/t$**. This produced a textbook Robbins–Monro annealing schedule, not "early stopping". 

### A1. The 50–60° Rotation was a Prototype-Norm Artifact
The claim that a single phantom point dragged prototypes 60° was false. Step magnitudes were gated by `step_mag = update_lr * update_weights`. The massive rotations occurred because rare classes happened to have small initial prototype norms, making their `1/||w_c||` multiplier massive. 

### A2. The Epistemic Anchor ($k=0$) does nothing
The regression from $k=0.0001$ to $k=0.0$ was only 0.0004 mIoU—well below the noise floor. The spring is neither the cause of the `0.4840` shortfall nor a necessary anchor. 

### A3. The `0.4840` vs `0.4129` Comparison is Confounded
The `0.4840` metric was reported as a *cumulative online* mIoU during the adaptation pass. The `0.4129` metric was reported as a *final frozen* mIoU post-adaptation. This, alongside the addition of the hard 40° cap, the anchor spring, and the normalization fix, means the two metrics are fundamentally uncomparable.

### A4. The `count_throttle` and Inverse-Frequency Dampeners
`count_throttle` was aimed at a mechanism (dense false positives causing runaway rotation) that was never operating. The true mechanism was the norm artifact. Inverse-Frequency dampening ($\gamma$) was also operating on the magnitude, suppressing both direction and step-size simultaneously.

---

## Part B: Verification Tests (Diagnostics)

| ID | Test | Goal |
| :--- | :--- | :--- |
| **V1** | **Protocol equivalence** | Report both cumulative-online mIoU and final-frozen mIoU for every run to bridge the 0.4840 metric gap. |
| **V2** | **Log `\|\|w_c\|\|` per class** | Confirms the norm-artifact explanation. Reveals whether per-class effective LR varied by orders of magnitude. |
| **V3** | **Class Index Fix** | Fix the Head/Tail indices. Classes 11, 13, 14, 15, 16 are head/background classes (Sidewalk, Building, Fence, Vegetation, Trunk). 2, 3, 6, 7, 10 are tail classes (Bicycle, Motorcycle, Person, Bicyclist, Parking). |
| **V4** | **Firing Rate Logging** | Log `n_points, n_fired, mean_w, sum_w` to separate true firing frequency from mean weight. |
| **V5** | **Argparse Plumbing** | Plumb `--test_1b` and `--reproduce_bug` to cleanly run ablations. |
| **V6** | **Bug-reproduction** | Disable only the post-step normalization (`--reproduce_bug`). Does snow return to ~0.4840 online? This is the decisive test for the $1/t$ annealing hypothesis. |
| **V7** | **Noise Floor** | Establish a noise floor using semantically-neutral perturbations. |

---

## Part C: The Step-Size Schedule

The evidence says the mechanism producing the best-ever number was a decaying effective LR ($1/t$), and that everything since has been a global shrink rather than a schedule.

1. **S1 (LR Schedule Sweep):** Explicitly test constant vs $1/t$ vs cosine decay.
2. **S2 (Per-Class Norm Equalization):** `step = lr * direction / max(||w_c||, epsilon)`.
3. **S3 (Explicit Early Stopping):** Adapt for first N frames, then freeze. 
4. **S4 (Soft Rotation Barrier):** Replace the 40° hard cap with a soft exponential barrier.
5. **S5 (Spring, Properly Evaluated):** Apply $k$ to all classes every frame, evaluated against the V7 noise floor.

---

## Part D & E: Inter/Intra-Class Balancing
* **IC1 (Per-class rotation budget):** Equalize angular displacement, not weights.
* **IC2 (Split $\gamma$):** Apply inverse-frequency to direction weighting only, vs magnitude only.
* **IC4 (Confusion-aware weighting):** Weight class $c$ by how many points it is actively losing in the confusion matrix.
* **XC1 (Per-subcluster Dirichlet calibration):** Compute calibration per $K$-means subcluster.
* **XC2 (Equal-weight-per-subcluster aggregation):** The non-restrictive replacement for the Subcluster Ledger.
