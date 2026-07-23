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

### 2. Multi-View Architectures
Single-frame adaptation is inherently vulnerable to transient noise and occlusions. By leveraging multi-view architectures, we can enforce spatial and semantic consistency across multiple perspectives or temporal frames before committing to a permanent weight update. 

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
1. **`count_throttle` (The NO-OP):** Throttling by mass (`min(1.0, count / 10.0)`) failed to stop `wet_ground` degradation (mIoU 0.3916). Puddle reflections on `wet_ground` generate hundreds of points per frame, easily clearing the throttle threshold. Mass throttling is blind to dense, coherent false-positives.
2. **`rotation_cap` (The Over-Correction):** A hard 10-degree rotation cap perfectly halted the 60-degree runaways on `wet_ground`, but actively destroyed our previous `snow` tail rescue (mIoU dropped from 0.4840 $\rightarrow$ 0.3862). To successfully adapt to severe `snow`, tail prototypes organically need to rotate up to 40 degrees. 
3. **`anchor_spring` (The Silent PyTorch Bug):** A mechanism to gently pull drifting prototypes back to their initialization (`w_t = (1-k)*w_t + k*w_0`) yielded results mathematically identical to the baseline. This exposed a critical PyTorch assignment bug: `model.classify.weight[c].data = ...` silently replaces a temporary view's pointer rather than modifying the tensor in-place. 
   - **Crucially, this meant our post-step `F.normalize()` had also been silently failing for the entire project.** Prototypes were never re-normalized, causing their magnitudes to grow unbounded over thousands of frames and systematically shrinking the effective angular step size of the $0.01$ learning rate.

**Immediate Action (Completed):** 
We patched the script to use explicit in-place memory copies (`.data.copy_()`) for all tensor updates, widened the `rotation_cap` to 40 degrees, and ran a sweep to observe the true impact of the spring mechanism.

### Part 1.5: The Epistemic Anchor Hypothesis (Update: Late July 22)

**The `anchor_spring` Sweep Results:**
With the magnitude scaling correctly fixed, we ran a sweep over the spring constant $k$ ($0.001, 0.0005, 0.0001$). We observed that the mathematically perfect pipeline peaked at **0.4129** on Snow ($k=0.0001$, allowing $\sim 18^\circ$ of rotation), which surprisingly underperformed the bugged Phase 1 code (which reached **0.4840**).

**The Hypothesis:**
In Phase 1, the PyTorch magnitude bug caused the effective learning rate to decay to zero after taking massive, fast steps (rotating up to 40-50 degrees) in the first evaluation chunk. The model adapted deeply, and then permanently froze.

In the current fixed pipeline, the `anchor_spring` (even at a tiny $k=0.0001$) creates a continuous physical equilibrium that arrests rotation prematurely. For severe structural shifts like Snow, the prototypes organically *need* to rotate 40+ degrees to envelop the sparsified geometry. The physical spring is preventing them from reaching the true adapted state.

Because our continuous magnitude gates (Tier 1 Euclidean Density and Tier 2 Dirichlet Evidence) are now fully functional, they naturally act as an **Epistemic Anchor**. If a 10,000D prototype drifts too far into OOD noise, the 128D Euclidean gate (`tier1_decay`) will exponentially crush the step magnitude to zero because the points no longer match the frozen 128D clean manifold.

**Action:** We will run a final test with $k=0.0$ (disabling the macroscopic drift spring entirely) to see if the epistemic gates alone (backed by a hard 40-degree ceiling) can achieve the 0.4840+ Snow performance without shattering on Wet Ground.

### Part 2: Structural Tail Rescue
Why did `beam_missing` head classes adapt while the tail didn't? Single-frame adaptation cannot recover geometry that is completely missing (e.g. absent scan lines). 
* **Test 2a:** Check tail rotation on `beam_missing`. If zero, the tail is starved of points.
* **Test 2b:** Compare surviving point counts between `incomplete_echo` (dropout) and `beam_missing` (absent lines). 

### Part 3 & 4: Inter/Intra-Class Refinement
* **Test 3a:** The inverse-frequency dampener ($\gamma$) was permanently fixed to $0.1$ as it successfully suppressed majority class confirmation bias without causing paralysis.
* **Test 4a (Per-Subcluster Calibration):** Push the A1 Dirichlet calibration one level down to the $K$-means subclusters so rare poses don't look OOD against their own core class manifold.

### Part 5: Multi-View Architectures (Microscopic Temporal Consistency)
With the Macroscopic Temporal Drift (`anchor_spring`) potentially being disabled, **Microscopic Temporal Consistency** becomes critical. Without a physical spring pulling the network back, we need absolute certainty in our gradient steps to defend against transient noise (like snow flakes or puddle reflections).
Multi-view now has two concrete, evidence-grounded jobs:
1. Enforce frame-to-frame geometric agreement before allowing high-magnitude updates (acting as a dynamic replacement for the static anchor spring).
2. Supply missing geometry for structural corruptions (pose-warped temporal sweeps).
* We will fix the naive 5x5 temporal gate, build an AUROC screening harness to rank view augmentations, and fuse the top signals into the Dirichlet evidence term.
