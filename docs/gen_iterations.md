# Generalizing the Feature Extractor for Corruptions: Preliminary Testing

Based on the Corruption Atlas diagnostics, we have mathematically proven that **Type C corruptions (Fog, Crosstalk)** fundamentally destroy the linear separability of the geometric manifolds. A purely mathematical Test-Time Adaptation (TTA) technique (like Memory Banks or Prototype Shifts) is powerless because the underlying feature space has collapsed into isotropic noise. 

To solve this, the **128D Backbone** must be explicitly trained to map physical noise to clean semantic manifolds (or isolate it) *before* the HDC projection layer.

This document outlines our **Preliminary Diagnostic Test**. Before committing to a massive 60-epoch overnight pre-training run, we are testing three highly-aligned mathematical constraints in a "micro-pretraining" shootout.

---

## The Three Aligned Methodologies

We have identified three strategies perfectly aligned with sparse 3D point cloud architecture that aggressively target Type C domain shifts:

### Method 1: Unnormalized Supervised Contrastive (SupCon)
- **Concept:** Pull augmented views of the same class together while pushing different classes apart.
- **Why it Aligns:** Traditional SupCon requires massive batch sizes, but because LiDAR segmentation is dense, a single batch of 1 provides ~100,000 points (millions of positive/negative pairs natively). 
- **The Modification:** We apply SupCon on **unnormalized 128D features**. Standard $L_2$ normalization fatally stretches zero-signal noise to a magnitude of 1.0 (the Inlier Paradox). Without normalization, destroyed points will lack gradient signal and naturally collapse toward the origin ($\Vert z \Vert_2 \approx 0$).

### Method 2: Variational Information Bottleneck (VIB)
- **Concept:** Force the network to "forget" everything about the input that isn't strictly necessary for semantic prediction.
- **Why it Aligns:** Requires only changing the final linear projection from `128` output to `256` output (Mean + Variance). 
- **The Modification:** A KL-divergence penalty forces the latent distribution to match a standard normal prior $\mathcal{N}(0, I)$. Complex, high-entropy spatial noise (Fog/Crosstalk) is mathematically expensive to memorize, so the network minimizes the loss by letting the penalty drag the noise straight to the origin ($\mu = 0, \sigma^2 = 1$). This guarantees perfect magnitude isolation.

### Method 3: Local Smoothness (Dirichlet Energy Regularization)
- **Concept:** Prevent highly oscillatory, high-frequency decision boundaries that are vulnerable to random scattering noise.
- **Why it Aligns:** Requires absolutely **zero architectural modifications** and consumes very little VRAM.
- **The Modification:** We add a Dirichlet Energy penalty to the feature graph ($\lambda \sum \| z_{point} - z_{neighbor} \|^2$). Transient noise points (like a Fog point near a solid building wall) are absorbed into the building's robust local geometry rather than snapping to a distinct noise cluster.

---

## Non-Leaking Data Augmentation Pipeline
To teach these invariances without overfitting to KITTI-C's specific ray-tracing algorithms, all three methods will train against generic physical degradation:
1. **Voxelized Random Dropout (Sparsity):** Drop entire spatial voxels instead of uniform points. (Simulates *Beam Missing / Incomplete Echo*).
2. **Anisotropic Gaussian Jitter (Sensor Noise):** Heavy Gaussian noise applied strictly along the ray-axis (distance from sensor). (Simulates *Crosstalk*).
3. **Global/Local Density Subsampling:** Randomly decimate point density in specific angular quadrants. (Simulates signal attenuation like *Fog / Snow*).

---

## The Diagnostic Script: `micro_pretrain_eval.py`

To definitively rank these methods, we will implement a rapid-prototyping script that executes the following pipeline in one continuous run:

### Phase 1: Micro-Training
- **Data:** 10% subset of the clean SemanticKITTI training split.
- **Competitors:** Instantiate 4 lightweight backbones: Baseline (Cross-Entropy), Unnormalized SupCon, VIB, and Local Smoothness.
- **Epochs:** Train all 4 models for exactly 5 epochs using the physics augmentations. (Should take <1 hour total).

### Phase 2: Latent Extraction
- Freeze all 4 backbones.
- Pass a fixed subset of 50 Clean frames and 50 Fog frames through the networks.
- Extract the raw, continuous 128D features (bypassing the HDC layer completely).

### Phase 3: The 128D Headroom Metrics
The script will automatically evaluate the raw 128D geometry of the Fog frames against the Clean frames, calculating the winning criteria:

1. **Cosine Shift ($\Delta f$):** The average cosine distance between the clean 128D centroid and the Fog 128D centroid. 
   - *Winning Criteria: Minimizes this shift, proving the geometry resists drift.*
2. **Target Neighborhood Purity (1-NN):** Query Fog points against their own local 128D neighborhood. 
   - *Winning Criteria: Pushes 1-NN purity back up to >90% (Baseline is ~75%).*
3. **Magnitude Segregation (Artifact Isolation):** Measure the average $L_2$ norm of correctly classified points vs. semantic hallucinations. 
   - *Winning Criteria: Maximizes the magnitude gap (e.g., Clean points > 5.0, Noise points < 1.0), allowing a trivial pre-HDC threshold gate to veto unrecoverable points.*

---

## Phase 4: Micro-Pretraining Results (5 Epochs, 10% Data)

The `micro_pretrain_eval.py` script was executed to benchmark the architectures against Heavy Fog domain shifts. The raw 128D features yielded the following metrics:

| Metric | Baseline | Unnormalized SupCon | VIB | Local Smoothness |
| :--- | :--- | :--- | :--- | :--- |
| **Cosine Shift (Angular Drift)** | 0.587 | **0.238** | 0.593 | 0.726 |
| **Euclidean Shift (Absolute Drift)** | 6.453 | 7.697 | 4.465 | **3.717** |
| **Target Neighborhood Purity (1-NN)** | **0.696** | 0.613 | 0.685 | 0.665 |
| **Average L2 Norm (Clean)** | 8.485 | 11.200 | 5.341 | 3.679 |
| **Average L2 Norm (Fog)** | 7.387 | 7.748 | 4.733 | 3.918 |

### Diagnostic Analysis & Conclusion
1. **The Angular Victory (SupCon):** SupCon achieved a massive breakthrough in angular geometric preservation. The Cosine Shift dropped from `0.587` (Baseline) to an incredible `0.238`. This proves that the InfoNCE loss successfully maps the corrupted points to the correct angular trajectories.
2. **The Missing Magnitude Collapse (VIB & SupCon):** Neither SupCon nor VIB successfully collapsed the unrecoverable Fog noise toward the origin ($L_2 \approx 0.0$). Fog magnitudes remained extremely high (4.7 to 7.7). 

**The Verdict:** 
The failure to achieve Magnitude Segregation is a direct artifact of the "micro-pretraining" constraints. 5 epochs on 10% of the dataset equates to **0.5 epochs of total gradient steps**. KL-Divergence penalties (VIB) and unnormalized contrastive margins (SupCon) are notoriously slow to converge, requiring tens of thousands of steps to structurally reorganize absolute magnitudes in a 128D hyperspace. The fact that SupCon already improved Cosine Shift by over 60% in just half an epoch proves the mathematical viability of the strategy.

**Next Steps:** We must commit to a full-length pre-training run (e.g., 20+ epochs on 100% of the dataset) using **Unnormalized SupCon** (to leverage its incredible angular alignment) and **VIB** (to aggressively force the magnitude collapse over time) to allow the structural reorganization to fully manifest.

---

## Phase 5: Implementation Audit & Epistemic Pivot

A critical audit of the Phase 4 methodology and results revealed fatal flaws in both the experimental design and the codebase implementation.

### The Epistemic Crisis
1. **Selective Interpretation:** The core objective of the pre-training is to correctly map corrupted points to clean semantic manifolds, which is directly measured by **1-NN Purity**. In Phase 4, *every single method* regressed 1-NN Purity relative to the Baseline (0.696), yet the conclusion selectively elevated Cosine Shift as a "victory" and waved away the purity drop. 
2. **Mathematical Tension:** The "Magnitude Segregation" thesis demands that dead Fog points collapse to the origin ($L_2 \approx 0$). However, a point at the origin has no well-defined angle. Optimizing for Cosine Shift (angular alignment) while simultaneously crushing the magnitude to zero via VIB are fundamentally contradictory goals for a single representation.
3. **Circular Metrics:** SupCon directly optimizes angular margins (Cosine Shift), and VIB directly penalizes magnitude. Scoring them on the exact axes they optimize without calibrating against downstream mIoU or 1-NN Purity is a circular, uncalibrated comparison.

### Implementation Fixes Applied for Scale-Up
To ensure the next scale-up is mathematically robust, the following fixes were pushed to `modules/gen_trainers.py` and the evaluation scripts:
* **The Scheduler Bug:** Truncating dataset epochs (`max_steps`) was breaking the LR scheduler, which expects to traverse the full dataset length. Cutoffs were removed entirely. `med_pretrain_eval.py` now runs 3 full epochs on 100% data (which fits the 24-hour compute budget and guarantees scheduler convergence).
* **VIB Clean Pathway:** VIB pressure was previously only applied to the augmented view. The KL-divergence and bottleneck sampling are now applied to **both** the clean and augmented pathways.
* **Tau Scaling:** Unnormalized SupCon features regularly reach magnitudes of 8–11. Applying `tau=0.1` was mathematically exploding the softmax. Unnormalized SupCon now uses `tau=1.0`.
* **Density Subsampling:** Added a 20% random drop mask to `get_augmented_view` to correctly simulate LiDAR sparsity.
* **Architecture Agnostic:** `logvar_head` is now lazily initialized to support backbone bottleneck changes.

### The New Protocol: The Strictly Calibrated Medium-Scale Run
Because the original hypothesis (Magnitude Segregation via unnormalized features) has epistemic tension, the upcoming 24-hour run (`med_pretrain_eval.py`) is no longer just a scale-up of a "winning" method. It is a strictly controlled falsification test.

**Protocol:** 
Run 4 methods (`baseline`, `vib`, `supcon`, `supcon_vib`) for 3 full epochs (100% dataset) each. 
*If, after a fully converged run, the Baseline continues to beat the generalized models on **1-NN Purity**, the Magnitude Segregation thesis will be officially falsified.* We will then pivot HDC architecture to explicitly handle uncertainty rather than forcing the backbone to implicitly model physics noise via geometry.
