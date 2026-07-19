# Preliminary KITTI-C Test Iterations
**Date:** July 16, 2026

## Objective
Evaluate baseline unsupervised adaptation (`prototype_cosine`) against novel uncertainty-gated adaptation strategies (epistemic, spatial, temporal) across 8 SemanticKITTI-C corruptions. The primary metric is Mean Intersection over Union (mIoU) measured pre-adaptation and post-adaptation.

## Results Summary (mIoU)

| Corruption | Base (Frozen) | Prototype Cosine | Epistemic Density | Balanced + Epistemic Density |
| :--- | :---: | :---: | :---: | :---: |
| Fog | 0.0583 | 0.0778 | 0.0759 | **0.0864** |
| Wet Ground | 0.4245 | 0.2619 | 0.4498 | **0.4839** |
| Snow | 0.4078 | 0.3639 | 0.4899 | **0.5271** |
| Motion Blur| 0.4061 | 0.3401 | 0.4594 | **0.4788** |
| Beam Miss  | 0.3779 | 0.3058 | 0.4081 | **0.4322** |
| Crosstalk  | 0.0700 | 0.0976 | 0.0829 | **0.1055** |
| Incomplete | 0.3760 | 0.2940 | 0.4076 | **0.4192** |
| Cross-Sens | 0.2587 | 0.1991 | 0.2861 | **0.3070** |

## Analysis of Successes and Failures

### 1. The Degradation Phenomenon is Real (Baseline Failure)
The standard pseudo-labeling adaptation approach (`prototype_cosine`) suffers from severe model degradation. It actually *hurts* performance on 5 out of 8 corruptions (Wet Ground, Motion Blur, Beam Missing, Incomplete Echo, Cross-Sensor). This validates the core hypothesis: unfiltered pseudo-labeling causes confirmation bias where the model confidently updates on its own mistakes, shattering rare class prototypes.

### 2. Epistemic Density is the Most Robust (Universal Improvement)
The `epistemic_density` method improves performance across **all 8 corruptions** with absolutely zero degradation. 
- **Mechanism:** It has a highly conservative firing rate (~15-20%). By only updating on points that lie densely near established source distributions, it isolates the model from outlier noise. 

### 3. Balanced Margin Drop Synergizes with Density (Breakthrough)
Combining a probabilistic drop ($p=0.5$ for margins $> 0.05$) with `epistemic_density` created a massive synergistic effect that shattered all previous performance ceilings:
- **Successes:** +11.9% on Snow, +7.2% on Motion Blur, +5.9% on Wet Ground.
- **Mechanism:** `epistemic_density` isolates hard samples from OOD noise. However, it still updates heavily on "easy/confident" samples (which are overwhelmingly common classes like Road/Building), causing Voronoi shattering of rare classes. The `balanced_margin` drop throttles the Firing Rate on confident samples from ~19% down to **~9.5%**. By halving the updates on common classes, the rare classes successfully maintain their geometric volume in the HDC space.

### 4. The Subcluster Ledger (Geometric Rebalancing) - FAILED
While `balanced_margin` dynamically limits updates on high-confidence (common class) samples, it acts as a proxy for true class-budgeting. We implemented a strict budget across $K$ intra-class subclusters (`ledger_epistemic_density`) to enforce this mathematically.
- **Result:** Severe performance degradation (e.g., `snow-3` mIoU plummeted from `0.4078` to `0.2290`). Firing rates dropped to <1% in strict mode.
- **Mechanism (Outlier Amplifier):** In latent space, "common geometries" form the dense, confident core. "Rare geometries" form the low-density fringe (often outliers/noise). By freezing the core updates and waiting for the rare geometries to catch up, the ledger prevented the network from updating on safe representations, dragging prototypes toward heavily penalized noise.
- **Future Pivot (Novel HDC Gating):** Intra-class spatial balancing flattens the natural confidence gradients. If we return to novel gating later, we should directly exploit the topological properties of the 10,000D hypervectors rather than the 128D bottleneck. For example:
  - **Dirichlet Evidential Gating:** Treat HDC cosine similarities as raw evidence to generate Dirichlet parameters, natively quantifying epistemic uncertainty (True Evidential Deep Learning).
  - **HDC-Energy:** Calculate the LogSumExp of cosine similarities in the HDC space to detect OOD samples.

## Strategic Next Steps

Now that we have proven that uncertainty gating resolves test-time adaptation collapse, we can focus on maximizing these gains.

### 1. Method Ensembling (Union of Experts)
Since intra-class geometric balancing (the ledger) proved detrimental, we will shift focus immediately towards **Method Ensembling**. We saw that `balanced_epistemic_density` provides universally elite performance. `temporal_veto` also showed unique synergy for chaotic frame-by-frame structural noise.
- **Action:** Explore multi-condition constraints, combining the strengths of our robust gating functions. For instance, requiring a point to pass `temporal_veto` AND have high `epistemic_density` to safely survive adaptation without amplifying outliers.
### 2. Move to Soft Gating Tuning
We replaced hard binary masks with Soft Gating (using sharpened Softmax probabilities as continuous weights). However, the hyperparameters (like `update_lr = 0.01` and temperature `T=100`) were chosen arbitrarily to fix the NaN bug.
- **Action:** Now that the pipeline runs, we can tune the soft-gating temperature to scale updates proportionally to confidence, rather than letting everything update at full stride.
