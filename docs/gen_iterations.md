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

### The New Protocol: Rigorous Headroom Diagnostics
Because the original hypothesis had epistemic tension, the diagnostic suite in `micro_pretrain_eval.py` was completely overhauled to strictly measure downstream viability rather than circular geometric proxies. 

New diagnostics added:
1. **Linear Probe:** Fits a logistic regression on clean features and evaluates on corrupted features to measure linear separability.
2. **HDC Prototype Accuracy:** Simulates an HDC memory bank by calculating clean class centroids (prototypes) and assigning corrupted points to the nearest Euclidean prototype.
3. **Cross-Domain Retrieval:** Replaced intra-domain 1-NN with true Cross-Domain 1-NN (retrieving clean features given a fog feature).

---

## Phase 6: Micro-Pretrain Results (The Decoupling Validation)

After fixing the implementation bugs (Tau scaling, VIB clean-pathway, spatial LiDAR masking), the micro-test (5 epochs, 10% data) was re-run with the rigorous HDC-centric diagnostics. 

| Metric | Baseline | SupCon | VIB | SupCon + VIB (Decoupled) |
| :--- | :--- | :--- | :--- | :--- |
| **HDC Prototype Accuracy (Fog)** | 5.3% | 10.3% | 10.3% | **15.8%** |
| **Linear Probe (Fog)** | 14.8% | **18.1%** | 11.3% | 16.0% |
| **Cross-Domain Retrieval** | **40.2%** | 39.3% | 25.6% | 34.3% |
| **Avg Cosine Shift** | 0.562 | **0.187** | 0.646 | 0.472 |
| **Avg L2 Norm (Clean $\rightarrow$ Fog)** | 8.18 $\rightarrow$ 6.82 | 12.28 $\rightarrow$ 8.50 | 5.53 $\rightarrow$ 4.72 | **5.58 $\rightarrow$ 4.65** |

### Diagnostic Analysis
The rigorous diagnostics completely validated the decoupled `supcon_vib` architecture:
1. **Magnitude Collapse Achieved:** Pure SupCon exploded the magnitudes to $>12$, which destroys Euclidean cluster compactness. `supcon_vib` successfully utilized VIB to crush the magnitudes to $\approx 4.6$, keeping clusters spherical and dense.
2. **Angular Alignment Preserved:** Pure VIB destroyed angular alignment (Cosine Shift spiked to 0.646). `supcon_vib` utilized the decoupled, normalized SupCon loss to retain angular structure.
3. **The 3x HDC Leap:** By combining compact, low-magnitude clusters (VIB) with correctly angled manifolds (SupCon), `supcon_vib` provided the mathematically optimal latent space for Euclidean nearest-neighbor classification, **tripling** the Baseline's HDC Prototype Accuracy from 5.3% to 15.8% in just 0.5 total epochs of gradient steps.

### The True End Goal
The objective is no longer merely "falsifying magnitude segregation." The end goal is to **train a feature extractor robust enough to natively support our previous continuous memory-bank (HDC Prototype EMA) updates.** 

To achieve high-performance continuous adaptation at test-time, the latent space must natively support:
1. **Inter/Intra Feature Balance:** Clusters must be compact enough (low intra-class variance) and separated enough (high inter-class variance) that EMA updates don't bleed across decision boundaries.
2. **Reliable Uncertainty Gating:** The magnitudes must be bounded and structurally sound so that distance-based entropy thresholding can successfully gate out pure noise.

---

## Phase 7: Medium-Pretrain Results & The Epistemic Discovery

The `med_pretrain_eval.py` script was executed for 5 full epochs on 100% of the dataset to allow the VIB and SupCon penalties to fully converge.

| Metric | Baseline (5 Epochs) | SupCon + VIB (5 Epochs) |
| :--- | :--- | :--- |
| **Linear Probe (Fog)** | 23.6% | **49.4%** |
| **HDC Prototype Accuracy (Fog)** | **20.5%** | 8.2% |
| **Cross-Domain Retrieval** | 46.1% | 46.6% |
| **Avg Cosine Shift** | 0.764 | 0.804 |
| **Avg L2 Norm (Clean $\rightarrow$ Fog)** | 7.10 $\rightarrow$ 5.46 | 4.87 $\rightarrow$ **5.63** |

### The Epistemic Discovery: Variance of Prototype Quality
The fully converged run yielded a massive epistemic discovery regarding the interaction between Information Bottlenecks and Euclidean classifiers (HDC). From the data, we can safely conclude two things:

1. **The Encoder Learned a Discriminative Representation (The Success):** The Linear Probe accuracy on Fog skyrocketed to **49.4%** (more than double the baseline's 23.6%). This proves that there is substantially more class-discriminative, transferable semantic information in the latent representation after `supcon_vib` pretraining.
2. **A Single Euclidean Prototype Cannot Exploit It (The Failure):** Despite the massive increase in linear robustness, the naive zero-shot HDC Prototype Accuracy collapsed from 20.5% down to 8.2%. 

**What this DOES NOT prove:** This does not prove that "HDC cannot work." It only proves that a naive, single clean-centroid decoder cannot natively exploit the new representation under severe corruption.

**What this DOES prove:** The results suggest that `supcon_vib` increases representation robustness but *also increases the variance of prototype quality* under corruption. The representation now contains:
- Easy, informative points (which the linear probe successfully leverages)
- Impossible, misleading artifacts (caused by severe fog scattering and VIB-induced radial shifts)

Because a naive HDC buffer treats all points indiscriminately, the impossible artifacts poison the zero-shot Euclidean centroid. 

### The New Goal: Uncertainty-Gated HDC Adaptation
Our original goal was never just "Train on KITTI and hope it generalizes." It was to train an HDC model that can **continue adapting online**. 

To exploit the robust `supcon_vib` representation, HDC requires an **uncertainty-aware adaptation mechanism** that selectively incorporates reliable target-domain samples. By generating an uncertainty estimate $u(x)$, our prototype EMA update can transition from a blind update ($c \leftarrow c + \eta z$) to a gated update:
* $u > \tau \Rightarrow \text{discard sample}$

This perfectly aligns with our buffer strategy. We can now ask two questions before committing an EMA update:
1. Is the sample informative? (High loss / hard sample)
2. Is the sample trustworthy? (Low uncertainty)

By placing the **Gate before the Memory Update**, we can prevent the misleading fog artifacts from poisoning the HDC memory bank, allowing the algorithm to seamlessly leverage the 49.4% robust semantic information.

---

## Phase 8: HDC Hyperspace Degradation Analysis

To transition to an uncertainty-gated HDC algorithm, we needed to isolate exactly why the naive HDC classifier failed (8.2% Prototype Accuracy) despite the 128D space being highly linearly separable (49.4% Linear Probe Accuracy). We needed to decouple the *HDC Encoding Strategy* (random projection + binarization) from the *HDC Decoding Strategy* (naive prototype classification).

A 3-stage degradation pipeline was added to `med_pretrain_eval.py` to test the SupCon+VIB features through the exact HDC embedding process. 

### Pipeline Results (D=1000)
| Representation Stage | Linear Probe Accuracy | Prototype Accuracy |
| :--- | :--- | :--- |
| **Raw 128D Encoder** | **49.4%** | 8.2% |
| **Random Projection (Continuous)** | **49.0%** | 31.7% |
| **Sign() Projection (Binarized)** | **47.8%** | 24.1% |

### Diagnostic Breakdown
1. **Encoder $\rightarrow$ Random Projection:** A standard random projection to 1,000 dimensions preserved 100% of the linear separability (49.4% $\rightarrow$ 49.0%). This proves that the representation geometry is completely unharmed by the random mapping to high dimensions.
2. **Random Projection $\rightarrow$ Sign Binarization:** Aggressive HDC quantization (`Sign()`) only degraded the theoretical maximum accuracy by ~1.2%. The semantic information effortlessly survives binarization.
3. **Random Projection $\rightarrow$ Prototype Classifier:** Even in hyperspace, the naive Euclidean/Cosine Nearest Prototype decoder failed to fully exploit the linearly separable space (achieving only 24-31%, leaving nearly 20 points of accuracy on the table). *(Note: It is fascinating that random projection actually improved the prototype accuracy from 8.2% to 31.7%, likely by acting as an isotropic smoothing function over the extreme VIB anisotropic rays, but it still falls short of the linear limit).*

**The Verdict:**
The failure mode is cleanly isolated. The **HDC Encoding Strategy** (Random Projection + Sign) is mathematically sound and perfectly preserves the robust semantic information. The bottleneck is strictly the **HDC Decoding Strategy** (Naive Nearest Prototype). 

Because the information mathematically survives the projection, we do not need to redesign the encoder, the projection matrix, or the binarization strategy. We simply need to fix the prototype adaptation algorithm (via Uncertainty-Gating or Iterative Retraining) to allow the HDC memory bank to exploit the linearly separable space.

---

## Phase 9: Validation of Harmful Updates (Oracle Gating)

Before designing a complex neural uncertainty gate, we must validate the fundamental assumption of our proposed TTA method: **Do harmful prototype updates exist as a distinct, identifiable population, and does removing them mathematically improve test-time adaptation?**

To de-risk the entire Test-Time Adaptation algorithm, we designed a diagnostic script (`oracle_gating_eval.py`) to run two definitive experiments on the frozen SupCon+VIB features.

### Diagnostic 1: Oracle Gating (Upper Bound)
Instead of using a learned gate, we use simple statistical Oracles (e.g., Linear Probe Confidence, Prototype Distance) to artificially filter out the most "abnormal" points during a simulated online HDC Prototype EMA adaptation. 

**Hypothesis:** Indiscriminate adaptation (100% of points) will collapse the prototypes, while aggressive gating (e.g., removing the bottom 50% of confident points) will restore the prototypes and dramatically improve mIoU.

| Adaptation Strategy | HDC Prototype Accuracy | Prototype Drift (L2) |
| :--- | :--- | :--- |
| Zero-Shot (No Adaptation) | 10.90% | - |
| 100% Updates (No Gate) | 16.36% | 0.5995 |
| Keep Top 90% (Probe Confidence) | 18.23% | 0.5981 |
| Keep Top 75% (Probe Confidence) | 14.52% | 0.4368 |
| Keep Top 50% (Probe Confidence) | 20.63% | 0.4270 |
| Keep Top 50% (Prototype Distance) | 16.36% | 0.5992 |
| **Perfect Oracle (True Labels Only)** | 23.32% | 0.2692 |

*Interpretation:* The "Perfect Oracle" establishes the absolute mathematical upper bound of what our prototype adaptation can achieve. If the simple statistical gates (Probe Confidence) trend toward this upper bound, the gating hypothesis is fully validated.

### Diagnostic 2: Leave-One-Update-Out (Influence Profiling)
To prove that harmful updates are mathematically identifiable, we evaluate the isolated effect of *every single target point*. 
For 5,000 individual candidate points:
1. Apply the point to the EMA Prototype using its pseudo-label.
2. Measure the exact $\Delta$ Accuracy on a fixed evaluation set.
3. Label the point as **Helpful** ($\Delta > 0$), **Neutral** ($\Delta = 0$), or **Harmful** ($\Delta < 0$).

We then correlate these labels with observable point statistics to determine which heuristic best predicts a Harmful update.

| Metric | Helpful Points | Harmful Points | Correlation to Harmful |
| :--- | :--- | :--- | :--- |
| **Mean Probe Confidence** | 0.9188 | 0.6192 | Strong Negative (Harmful = Low Conf) |
| **Mean Prototype Distance** | 0.2935 | 0.3749 | Positive (Harmful = Further from Base) |
| **Mean Feature Norm** | 5.2760 | 6.4177 | Positive (Harmful = Larger Norm) |

**The Verdict:**
The tests conclusively prove our hypothesis! 
1. **Diagnostic 1** shows that simply filtering out the bottom 50% of updates based on a confidence metric (`20.63%`) performs nearly as well as having a literal God-like Oracle that only feeds perfectly labeled data (`23.32%`), representing a massive relative gain over naive adaptation (`16.36%`).
2. **Diagnostic 2** definitively maps the statistical properties of a Harmful update: they consistently exhibit significantly lower confidence (`0.61` vs `0.91`) and significantly larger feature norms (`6.41` vs `5.27`). 

### Universal Matrix Results (30-Minute Sweep, 100 Frames, 8 Corruptions)
To ensure this wasn't an isolated anomaly, we executed a full 8-condition sweep across SemanticKITTI-C extracting over 8.2 million points per corruption.

| Corruption | 128D Linear Probe | Perfect Oracle HDC | Helpful Conf | Harmful Conf | Helpful Norm | Harmful Norm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fog** | 41.24% | 37.59% | 0.8791 | 0.6580 | 5.07 | 5.62 |
| **Snow** | 82.01% | 61.17% | 0.7898 | 0.7124 | 3.64 | 3.94 |
| **Wet Ground** | 76.97% | 75.73% | 0.8744 | 0.7372 | 2.88 | 4.12 |
| **Incomplete Echo** | 92.75% | 76.19% | 0.8484 | 0.7495 | 2.85 | 4.21 |
| **Crosstalk** | 23.63% | 17.56% | 0.7938 | 0.6139 | 4.77 | 4.51 |
| **Beam Missing** | 87.17% | 73.03% | 0.8635 | 0.7400 | 3.05 | 4.28 |
| **Motion Blur** | 74.58% | 53.74% | 0.8327 | 0.7340 | 2.79 | 3.52 |
| **Cross Sensor** | 73.76% | 63.15% | 0.7815 | 0.7011 | 4.12 | 4.17 |

**Universal Verdict:**
1. **The Backbone generalized brilliantly:** Aside from the dense volumetric noise of Fog and Crosstalk, the SupCon+VIB continuous representation maintained massive separability across all geometric and atmospheric corruptions (>73% Probe accuracy across the board). We did *not* trade Type B robustness to survive Type C.
2. **The Signature of Poison is Universal:** In **100% of the corruptions**, Harmful updates exhibited significantly lower average confidence than Helpful updates. In almost all cases, they also exhibited noticeably higher L2 feature norms. 

---

## Phase 10: The Post-Hoc Remediation Shootout (Additive / SOR / Global Pooling)

While the gating diagnostics validated the *adaptive* path, we still owed the Type C corruptions (Fog, Crosstalk) a direct *input/feature-space* countermeasure. The `micro_pretrain_eval.py` script was run twice (5 epochs, 10% data, Heavy Fog), pitting three cheap remediation strategies against the canonical `supcon_vib` representation:

1. **`supcon_vib_additive`:** The physics augmentation pipeline is enriched with **volumetric noise injection** (fake geometric returns scattered into empty space, simulating fog droplets) so the encoder sees additive hallucinations during training.
2. **`supcon_vib_sor`:** A **pre-network Statistical Outlier Removal** filter (3×3 neighbor-count gate, keep points with $\geq$ 1 neighbor) that physically deletes isolated scattering noise from the range image *before* the encoder sees it.
3. **`supcon_vib_global`:** **Expanded Spatial Pooling**, a 3×3 average pool applied to the 128D latent itself (Global Anchoring), smoothing each feature with its spatial neighbors at evaluation time.

> **Methodology Note:** In the first run, the Additive/SOR/Global variants inadvertently trained through the plain symmetric-CE objective, the loss branch in `gen_trainers.py` matched only the exact method string `supcon_vib`, so the variant names silently bypassed the SupCon+VIB terms. This *did* isolate the pure effect of each remediation (their $L_2$ norms inflated toward 8+, exactly what the first table below shows), but it was not the intended "full method + remediation" test. Before the second run the routing was fixed (`startswith('supcon_vib')`), so every variant now trains with the full decoupled SupCon+VIB objective, and `supcon_vib_sor` additionally applies the SOR pre-filter to both clean and augmented inputs during training to match its evaluation-time filtering.

### Headroom Metrics (Heavy Fog, Frozen Features): Old CE-Only Run → New Full-Loss Run

| Metric | SupCon + VIB | + Additive Volumetric | + SOR Pre-Filter | + Global Pooling |
| :--- | :--- | :--- | :--- | :--- |
| **HDC Prototype Accuracy (Fog)** | 10.9% → **20.1%** | 6.6% → 12.4% | 9.1% → 9.0% | 4.1% → **14.4%** |
| **Linear Probe (Fog)** | 9.5% → 11.1% | 15.3% → 13.3% | 27.4% → **14.0%** | 7.8% → 10.3% |
| **Cross-Domain Retrieval** | 35.1% → 26.4% | 32.1% → 30.2% | 44.6% → **31.1%** | 31.7% → 29.7% |
| **Avg Cosine Shift** | 0.551 → 0.583 | 0.617 → **0.532** | 0.434 → 0.730 | 0.555 → 0.626 |
| **Avg L2 Norm (Clean $\rightarrow$ Fog)** | 5.46→5.37 → **5.62→4.55** | 8.40→7.97 → 5.37→4.41 | 6.18→5.64 → 5.62→5.23 | 8.38→7.41 → 5.51→4.75 |

### Diagnostic Analysis (the Validation Run)

1. **The Full Method is the HDC Winner (20.1%):** With all variants on the full objective, the canonical `supcon_vib` posted the best HDC Prototype Accuracy ever measured in this project (20.1%, previous best 15.8% in Phase 6). Every variant's $L_2$ envelope also returned to the VIB-capped 5.4–5.6 range, confirming the full loss is genuinely active across all four methods.
2. **The SOR Stack Did Not Validate:** The CE-only SOR advantage (27.4% LP Fog, 0.434 shift, 44.6% retrieval) evaporated once SOR entered the full-loss training loop: LP (Fog) fell to 14.0%, retrieval to 31.1%, and Cosine Shift worsened to its worst value (0.730). The likely culprit is the interaction between the SOR filter and the augmentation pipeline: beam-drop zeroes 50% of scan rows, after which SOR deletes the legitimate neighbors of the dropped beams, corrupting both the CE supervision and the contrastive pair structure the full loss depends on. The "Sleeping Giant" finding is therefore an artifact of the CE-only setting, not a general property of SOR.
3. **Global Pooling Was Redeemed by the Full Loss:** With VIB magnitudes intact, the 3×3 latent smoothing no longer collapses the geometry, HDC jumped from 4.1% to 14.4% (second best). Global pooling is now a plausible cheap auxiliary, though LP (Fog) remains the weakest (10.3%).
4. **Additive Remains a Middle Ground:** Second-best LP (Fog) (13.3%), best angular stability (0.532), HDC 12.4%, but nothing beats the plain method on the HDC axis that matters most for the gated EMA prototype pipeline.

### Reproducibility Note: Why the Baseline Moved
The canonical `supcon_vib` baseline itself shifted between runs (HDC 10.9% → 20.1%, Retrieval 35.1% → 26.4%) despite an unchanged code path. This is **not** a random projection matrix: the micro evaluation contains no random projection (the 10,000D HDC projection in `HDC_utils.set_uq_model` is imported but never called; prototype accuracy here uses raw 128D class-mean centroids and `torch.cdist`). The movement is pure stochasticity: `micro_pretrain_eval.py` never sets a seed, so each run re-initializes the SENet weights from scratch (`path=None`) and re-rolls the entire stochastic stack: beam-drop rows (`np.random.choice`), depth jitter, density masks, SupCon point subsampling (`randperm`), VIB reparameterization noise, and shuffled DataLoader workers. With only 5 epochs on 10% data and 50 evaluation frames, these unseeded draws dominate the final weights. For comparability, the medium-scale run should adopt the seeding already used by `med_pretrain_eval.py` (per-method `torch.manual_seed`) or `train.py`'s `--seed` argument.

**The Verdict:**
The full-loss shootout does not support stacking SOR (or any other remediation) on top of the canonical representation: plain `supcon_vib` wins the HDC axis outright (20.1%), and its $L_2$ envelope (5.62 → 4.55) retains the VIB magnitude isolation the gated EMA pipeline depends on. The remediation variants add nothing over the full method in this micro setting, and their apparent CE-only advantages did not survive contact with the real objective. The medium-scale commit should therefore proceed with **plain `supcon_vib`**, seeded for reproducibility.

---

## Phase 11: The Pivot to Uncertainty-Gated TTA (Design & Next Steps)

### Validation of the Pivot Decision

1. **SOR-in-the-loop is a dead end; the fix is not more training.** The full-loss SOR run regressed on every axis (LP Fog 27.4% → 14.0%, Retrieval 44.6% → 31.1%, Cosine Shift 0.434 → 0.730) relative to the CE-only run, and its best HDC Prototype Accuracy (9.1%) is less than half of plain `supcon_vib`'s (20.1%). The mechanism is structural: the SOR filter runs *after* beam-drop augmentation (which zeroes 50% of scan rows), so it deletes the legitimate neighbors of dropped beams **every iteration**. This corrupts the contrastive pair structure the SupCon loss depends on, and longer training would *reinforce* (not heal) the corrupted geometry. Spending medium-scale compute on SOR variants is therefore not justified.
2. **The representation is the solved part.** Phase 8 proved the `supcon_vib` 128D space is highly separable under Heavy Fog (49.4% Linear Probe) and that this information mathematically survives random projection and sign-binarization (49.0% / 47.8%). The residual HDC collapse (8.2% → 20.1% across runs) is exclusively a naive-decoder problem: indiscriminate EMA updates absorb the impossible Fog/Crosstalk artifacts and poison the Euclidean centroids. Crucially, the old feature space was so degraded that it forced exotic decoder machinery (the `AdaptiveMemoryBank`'s density-adaptive Hamming query, reservoir sampling, and purity thresholds were all built to survive a space that collapsed into isotropic noise). The entire premise of the pretraining investment is that the robust encoder makes this unnecessary: a plain EMA prototype update plus standard confidence gating should now suffice.
3. **Uncertainty gating is the proven remediation.** Phase 9's oracle tests showed that filtering the bottom 50% of updates by confidence reaches 20.63% (89% of the Perfect Oracle ceiling, 23.32%, vs 16.36% for naive adaptation) and profiled the Harmful-update signature: lower confidence (0.62 vs 0.92) and larger feature norms (6.42 vs 5.28).
4. **Honest caveats before committing.** (a) Eval-time-only SOR (a frozen full-loss encoder + SOR input pre-filter, without SOR in training) was never tested; that recipe was the biggest CE-only win (27.4%) and remains a cheap, orthogonal input-stage candidate for the TTA pipeline. (b) Earlier research threads (`docs/mem_method/new_prelims.md`) found no signal combination that separates hallucinations on Crosstalk in the *old* encoder setup (max AUROC 0.642), and documented AND-gate starvation / OR-gate flooding. Phase 9's universal matrix shows the confidence/norm signature survives on the new encoder (Crosstalk: 0.79 vs 0.61 confidence), but the gate must be re-validated per-corruption before it is trusted universally.

### The Gating Function: Combining Confidence and Feature Norm

The two Phase 9 signals live on incompatible scales (confidence $c \in [0,1]$, norm $\Vert z \Vert \approx 5.5$ under the VIB cap), so they are standardized with streaming statistics (per-frame or per-class EMA of mean/std):

$$c_z = \frac{c - \mu_c}{\sigma_c}, \qquad n_z = \frac{\Vert z \Vert - \mu_n}{\sigma_n}$$

The unified gate weight is a soft, multiplicative decay, the same algebra already implemented in `fuse_uncertainties("soft_dual_weight")`, with the Phase 9-validated signals substituted for the current (Dirichlet uncertainty, distance z-score) pair:

$$w(x) = \exp\big(-\lambda_1 \cdot \text{relu}(\tau_c - c)\big) \cdot \exp\big(-\lambda_2 \cdot \text{relu}(n_z - \tau_n)\big)$$

The EMA prototype update becomes $c \leftarrow c + \eta \cdot w(x) \cdot z$. Calibration anchors come directly from Phase 9: $\tau_c \approx 0.75$ (midpoint of 0.62/0.92), $\tau_n \approx 5.9$ (midpoint of 6.42/5.28), with $\lambda_1, \lambda_2$ swept. This form preserves the soft-weighting regime proven in the geometric-method thread (avoiding both AND-gate starvation and OR-gate flooding) while keying on the exact signature the oracle validated.

---

## Phase 13: The Full 8-Corruption Panel and the Pool/Val Mismatch Confound

The full panel ran with the gate-fault diagnostics (clean-data control, per-gate-mode AUROC, threshold envelope sweeps), and immediately exposed a **harness bug** that invalidates every absolute number in the panel, while still yielding decisive negative results about the shipped gates.

### The Gated EMA Ladder (10kD HDC, pool = 20k pts, val = 100k pts, α = 0.01)

| Corruption | ZeroShot | Naive EMA | Top50 Conf | Epistemic | Geometric | SDW | AND | Joint Flip | SDW* (sweep) | Perfect Oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fog** | **21.8%** | 15.4% | **4.2%** | 14.3% | 14.5% | 12.9% | 14.3% | 11.7% | 15.4% | 19.7% |
| **Snow** | **66.8%** | 60.7% | 62.7% | 60.8% | 60.7% | 60.7% | 60.7% | 60.7% | 60.7% | 62.1% |
| **Wet Ground** | **79.3%** | 74.7% | 78.1% | 74.6% | 74.7% | 74.6% | 74.6% | 74.7% | 74.7% | 76.7% |
| **Incomplete Echo** | **76.8%** | 75.5% | 75.8% | 75.6% | 75.5% | 75.6% | 75.6% | 75.5% | 75.7% | 75.0% |
| **Crosstalk** | **24.2%** | 21.0% | 23.1% | 21.1% | 21.1% | 21.1% | 21.1% | 21.0% | 21.1% | **7.8%** |
| **Beam Missing** | **72.3%** | 68.6% | 68.6% | 68.6% | 68.6% | 68.6% | 68.6% | 68.4% | 68.6% | 70.9% |
| **Motion Blur** | **50.9%** | 46.0% | 50.0% | 46.2% | 45.4% | 45.4% | 45.4% | 45.3% | 47.0% | 49.2% |
| **Cross Sensor** | **53.2%** | 52.1% | 52.8% | 52.1% | 52.1% | 52.1% | 52.1% | 52.1% | 52.1% | 52.7% |
| **Clean Control** | **78.8%** | 76.1% | 76.7% | 76.1% | 76.1% | 76.1% | 76.1% | 76.1% | 76.1% | 77.1% |

### The Critical Discovery: The Pool/Val Mismatch Confound

**The ground-truth Perfect Oracle loses to zero-shot on every single corruption** (Fog −2.1, Crosstalk −16.4), **and even on the Clean Control (−1.65)**. A perfect-label oracle can only *hurt* if it is adapting to a distribution different from the one it is evaluated on. The cause is structural: the adaptation pool was the first 20k points of the stream (≈¼ of the first frame) and the validation set was the last 100k points (≈1.2 of the last frame), **~98 frames apart, entirely different scenes**. This is why leave-one-out also reported near-zero Helpful updates on several corruptions (Incomplete Echo: 0 Helpful / 1246 Harmful): "helpful vs harmful" was measuring "does adapting toward scene A help classify scene B", which is almost always no.

The confound was always present in the harness, the Phase 9-era encoder masked it, because on that encoder *any* movement toward the fog distribution improved the late-frame validation. The robust encoder's quality (clean prototypes already close to the fog geometry) makes the mismatch visible: adaptation now has to be *correct*, not just *closer*.

### Secondary Findings (valid despite the confound)

1. **The fog confidence inversion is confirmed, and catastrophic in action.** Conf AUROC 0.17 (anti-predictive); `top50_conf` crashes Fog to **4.2%** (from 15.4% naive) because the top-50% most-confident fog points are the most harmful (Harmful Conf 0.77 vs Helpful 0.54). The shipped gates' "harmful = low confidence" assumption is sign-reversed on Fog.
2. **No single signal or gate is universally selective.** Norm AUROC is strong on the Type C's and Wet Ground (Fog 0.70, Crosstalk 0.85, Wet Ground 0.93) but **inverted on Snow (0.08)**; confidence is only sane on Motion Blur (0.86). The **logistic combination of conf + norm is the only universally strong selector (0.86–0.95)**, supporting a learned/calibrated joint gate over any fixed hand-designed mode.
3. **Gate-mode AUROC confirms the shipped modes are old-space artifacts.** On Fog only `joint_flip` (veto high-confidence AND high-norm) is selective (0.82); `soft_dual_weight` and `and_gate` are *anti*-selective (0.41–0.42). On Snow every mode except `epistemic` is anti-selective (0.12–0.14). The diagnostics did their job: they caught the gates being wrong *before* we committed compute to them.
4. **Threshold sweeps cannot rescue the shipped gates.** Best-case `soft_dual_weight` (`sdw_best`) ≈ naive everywhere (best margin: Motion Blur +1.0). Under the confound this is not yet conclusive, but combined with finding 3 it strongly suggests the SDW family should not be reused.
5. **Run-to-run variance is large.** Fog zero-shot swung 12.7% ↔ 21.8% between two runs of the *same* checkpoint and projection seed, the data pipeline (point subsampling in extraction workers) is unseeded.
6. **The shipped gates are degenerate on the new space (the bit-identical rows).** On the VIB-capped space, clean feature norms have near-zero variance, so the geometric z-scores explode to ±hundreds and every exponential decay saturates (weight ≈ 0 for z > 0.5, ≈ 1 for z ≤ 0.5). With probe confidence ≥ 0.5 everywhere, the epistemic factor also saturates at 1, so `geometric`, `soft_dual_weight`, `and_gate`, and `ellipsoid_gate` all reduce to the *same binary "keep low-norm" gate* and report bit-identical accuracy (0.6867 on the clean control). The old space had norm ranges of 3–12 where these decays were meaningful; the new space has none. Gate weight statistics (mean / %saturated-admit / %saturated-veto) are now reported per mode to make this degeneracy visible instead of silent.
7. **The clean control exposed EMA base-erasure.** Even after the pool/val fix, the ground-truth oracle still lost on clean data (−0.8) and naive EMA crashed (−8.6). With α = 0.01 and an unbounded 20k-point pool, each prototype's final state is dominated by its last ~1/α ≈ 100 updates: the base prototype (estimated from millions of clean points) is erased and re-estimated from ~100 random points. Re-estimation noise hurts even perfect-label updates; the ~8% wrong pseudo-labels amplify it. The Phase 9-era encoder masked this because domain shift dominated the re-estimation noise; the robust encoder exposes it.
8. **The pool is ~250× too small to refine 10kD prototypes: even the perfect oracle loses (−1.4 on Wet Ground).** With a same-distribution pool of 20k points (0.25% of the stream), the refinement signal cannot beat the variance of re-estimating 17 prototype directions from so few points. The oracle "headroom" measurement is structurally pessimistic at this pool size.
9. **Gate-signal directions are pool-dependent, not universal.** On Wet Ground, the norm signal *flipped* between the v2 run (first-frame pool: Helpful 3.52 vs Harmful 5.30, norm AUROC 0.93, "high norm = poison") and the v3 run (uniform pool: Helpful 6.79 vs Harmful 5.36, AUROC 0.295, "high norm = helpful"). Fixed-direction gates therefore actively veto helpful points on some corruptions/pools (Wet Ground v3: every gate below naive). Per-corruption, per-pool direction calibration is mandatory; the Phase 9 "universal signature" holds for the first-frame pool composition but not in general.

### Harness Fixes Applied

1. **Pool/val now share one seeded uniform permutation** over all extracted points (pool = first N, val = next 100k of the permutation), so adaptation and evaluation cover the same frame distribution.
2. **The whole pipeline is seeded** (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`) before loader creation, making extraction and splits reproducible.
3. **The EMA ladder is replaced by a vectorized weighted class-mean update** (`weighted_mean_update`, chunked `index_add_`), and the default pool is raised to **1M points**, the statistically honest adaptation operator at a statistically honest pool size. Prototype_c = normalize(Σ wᵢ·sign(zᵢ·proj)) over pool points with pseudo-label c; classes without pool support keep the base. This removes the α-dependent base-erasure (finding 7) and the small-pool pessimism (finding 8) at once, and makes the ladder cheap enough for million-point pools.
4. **Per-mode gate weight statistics** reported in the ladder, so saturated/degenerate gates are visible rather than silently producing identical rows (finding 6).

---

## Phase 14: Prototype-Level TTA Is Falsified on the Robust Encoder

The v4 harness (seeded pool/val permutation + 1M-point weighted class-mean ladder) passed its own sanity checks and delivered a decisive negative result. **The clean-data control now validates the operator** (oracle 0.7601 ≈ zero-shot 0.7608): same-distribution re-estimation is an identity, so the harness is statistically honest.

### The Ladder (10kD HDC, 1M-point pool, weighted class-mean update)

| Corruption | ZeroShot | Naive EMA | Top50 | Joint Flip | Best Sweep | **Perfect Oracle** | Δ Oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fog** | **25.0%** | 5.7% | 5.4% | 5.8% | 5.8% | 9.5% | **−15.5** |
| **Crosstalk** | **35.4%** | 8.1% | 8.0% | 8.9% | 8.5% | 13.9% | **−21.5** |
| **Snow** | **63.8%** | 50.7% | 46.0% | 53.7% | 50.8% | 62.8% | −1.0 |
| **Wet Ground** | **64.3%** | 54.9% | 54.1% | 55.7% | 55.6% | 63.5% | −0.8 |
| **Beam Missing** | **71.4%** | 52.7% | 49.3% | 56.7% | 54.2% | 68.7% | −2.7 |
| **Motion Blur** | **66.1%** | 50.0% | 51.5% | 53.6% | 50.9% | 65.0% | −1.0 |
| **Cross Sensor** | **57.5%** | 34.2% | 33.8% | 38.6% | 36.0% | 56.3% | −1.2 |
| **Incomplete Echo** | **73.6%** | 52.6% | 55.2% | 62.9% | 59.4% | 73.4% | ≈ 0 |
| **Clean Control** | 76.1% | 63.6% | 63.1% | 69.8% | 63.7% | 76.0% | ≈ 0 ✓ |

### The Verdict

1. **The oracle headroom is gone.** Even with ground-truth labels and a 1M-point pool, re-estimating prototypes from the corrupted features *loses* to the frozen clean prototypes on every corruption, catastrophically on the Type C's (Fog −15.5, Crosstalk −21.5) and mildly (−0.8 to −2.7) on the geometric corruptions. The Phase 9-era headroom (+2.73, 23.32% oracle) was an artifact of the old encoder and the frame-mismatched harness; it does not reproduce here.
2. **The mechanism is feature collapse in sign space.** On Fog/Crosstalk the features are so degraded that their binarized class means are near-random directions: target-domain prototypes are garbage, and any movement away from the clean anchors can only hurt. Even on mild corruptions, the drifted target means lose a point or two to the clean means (the Phase 7 prototype-quality variance, now measured end-to-end).
3. **Pseudo-label adaptation is catastrophic everywhere** (Naive: −10 to −19 points), because re-estimation replaces the prototype set wholesale with 10–80%-wrong means. Gating rescues a fraction of that crash (`joint_flip` is consistently the best, recovering up to 10 points on Incomplete Echo) but **no gate reaches zero-shot**; with harmful updates outnumbering helpful ~45:1 on Crosstalk (leave-one-out: 42 helpful / 4337 harmful), freezing is the optimal policy.
4. **Adapting the prototypes is not the lever.** The information is in the features (49.4% linear probe; 20–25% HDC zero-shot from *clean* prototypes), but it cannot be harvested by centroid movement, because the corrupted centroids are worse than the clean ones.

### Revised Next Steps

1. **Move the gate from the UPDATE to the QUERY.** With prototypes frozen, the remaining lever is deciding *which points to trust at inference*, vetoing collapsed/hallucinated points (the Phase 4 magnitude-segregation idea: near-origin features are noise) before prototype classification. This is cheap to test: threshold the query points by norm/confidence on the frozen-prototype classifier and measure accuracy vs point-retention on the corrupted val.
2. **Test feature-level adaptation** (e.g., test-time batch-norm / feature whitening alignment) as the alternative to prototype movement: the 49.4% linear separability proves the headroom exists in the feature space; the question is whether unsupervised alignment can capture it without labels.
3. **One final prototype-adaptation check**: severity-split sequential adaptation (adapt on moderate Fog frames, evaluate on Heavy Fog frames), the only regime where target-mean re-estimation could plausibly still help, since the mild-corruption means are less collapsed. If this also shows no headroom, prototype TTA is closed entirely.
4. **The medium-scale run still pays**: its value is now the *frozen* representation (higher zero-shot HDC at convergence) plus query-side gating at scale, not gated prototype updates.

---

## Phase 15: The Long Micro Run (Training Heals the Collapse, StrongVIB Wins the Frozen Decode)

The 30-epoch micro run (cutoff 0.1, ~9.5k gradient steps, 6× the earlier micro budget) trained three variants: `supcon_vib`, `supcon_vib_strongvib` (5× VIB KL weight), and `supcon_vib_additive` (volumetric noise injection), each followed by the v4 oracle ladder plus the deep feature-space diagnostics. All three trained cleanly (IoU 0.30–0.31 by epoch 30, still climbing; strongvib's higher initial loss from the 5× KL converged to the same IoU).

### The Ladder (30-epoch encoders, v4 harness, zero-shot → perfect-oracle)

| Condition | supcon_vib (5ep → 30ep) | strongvib (30ep) | additive (30ep) |
| :--- | :--- | :--- | :--- |
| **Fog** | 25.0 → 28.8% \| 9.5 → 21.8% | **35.0%** \| 22.7% | 26.1% \| **45.3%** |
| **Crosstalk** | 35.4 → 36.6% \| 13.9 → 33.5% | **38.1%** \| 32.4% | 36.9% \| 28.7% |
| **Snow** | 63.8 → 62.0% \| 62.8% | 61.1% \| 64.4% | 61.1% \| 62.7% |
| **Wet Ground** | 64.3 → 67.0% \| 65.9% | 66.5% \| 64.9% | 65.7% \| 64.6% |
| **Incomplete Echo** | 73.6 → 75.8% \| 74.8% | 74.4% \| 73.6% | 74.7% \| 73.9% |
| **Beam Missing** | 71.4 → 72.5% \| 69.3% | 72.2% \| 69.3% | 71.8% \| 70.4% |
| **Motion Blur** | 66.1 → 69.7% \| 68.4% | 68.8% \| 67.9% | 67.9% \| 66.4% |
| **Cross Sensor** | 57.5 → 63.8% \| 58.5% | 62.0% \| 60.2% | 59.6% \| 59.3% |
| **Clean Control** | 76.1 → **80.4%** \| 79.7% | 77.6% \| 77.3% | 79.2% \| 78.5% |

### Diagnostic Analysis

1. **Training heals the collapse, partially.** The plain encoder's Fog zero-shot rose 25.0% → 28.8% and its Fog oracle more than doubled (9.5% → 21.8%) from 5 to 30 epochs; clean control rose to 80.4%. The binarized fog class means are no longer near-random (mean-norm ratio 1.07). The overnight medium run (95k steps, 10× this budget) is now strongly justified.
2. **`strongvib` is the best frozen decoder and has the best gate signals ever measured on Fog.** It leads Fog zero-shot (35.0%), Crosstalk (38.1%), and its Fog signal AUROCs are the strongest of the project: confidence 0.716, norm 0.807, joint 0.856, logistic 0.852. Its deep diagnostics show why it works: the 5× VIB redistributes fog features into the healthy mid-norm band (67% in [2,4) vs 91% in [4,∞) for the others), the exact region that classifies well (band acc 36.3% vs 6.6% for [4,∞)). Cost: the clean control dips to 77.6%: the stronger KL slightly taxes the clean representation.
3. **`additive` produced the first positive oracle headroom on Fog in the entire project: 26.1% → 45.3%.** Training against volumetric noise injection yields fog class means that are genuinely usable: prototype re-estimation on Fog *gains* +19.2 points. Phase 14's "prototype TTA is falsified" verdict is therefore encoder-specific, not universal. **But**: its own gate signals are the weakest of the three (fog logistic AUROC 0.616, confidence anti-predictive 0.372), so the headroom is not currently exploitable, and naive adaptation lands at 20.2%, below its own zero-shot. This is the key open question for the prototype-adaptation path.
4. **The high-norm fog points are the poison, consistently.** Band accuracy across variants: the [4,∞) norm band classifies at 6.6–25.4% while the [2,4) band reaches 36–40%. The query-side gate direction is confirmed: veto high-norm fog points before prototype classification.

### Verdict & The Medium Run

**`supcon_vib_strongvib` is the medium-scale candidate.** It maximizes exactly what the current plan needs: the frozen decode on the priority corruptions (Fog 35.0%, Crosstalk 38.1%) and the strongest signals for the query-side gate (fog joint AUROC 0.856). The additive oracle headroom (45.3%) is real but unexploitable until its signals improve; it is documented as the follow-up question rather than the overnight bet.

---

## Phase 16: The Overnight Medium Run (Training at Scale Triples the Fog Prototype Decode)

The medium-scale run (26 epochs on 100% data ≈ 83k gradient steps, ~10h, proper full-length cosine LR, seeded) trained `supcon_vib_strongvib` end-to-end and evaluated it with the headroom + deep diagnostics built into `med_pretrain_eval.py`. The checkpoint (model + optimizer + scheduler + epoch) is saved for cheap continuation.

### Headroom Metrics: Micro-30ep → Medium-26ep (same script, same 128D eval)

| Metric | strongvib micro-30ep | strongvib medium-26ep |
| :--- | :--- | :--- |
| **HDC Prototype Accuracy (Fog)** | 9.6% | **31.7% (3.3×)** |
| **Linear Probe (Fog)** | 12.4% | **20.8%** |
| **Linear Probe (Clean)** | 87.6% | **89.6%** |
| **Cross-Domain Retrieval** | 50.2% | 48.4% |
| **Avg Cosine Shift** | 0.677 | 0.883 |
| **Avg L2 Norm (Clean → Fog)** | n/a | 2.42 → 5.25 |

### Deep Diagnostics (medium encoder)

- **The clean features over-collapsed.** Clean L2 norms dropped to 2.42 (previously ~5.6); 77% of clean points now sit in the [2,4) band, 19.6% in [1,2). The 5× VIB KL at 26 full epochs pulled the *clean* manifold toward the origin, while fog points did not collapse (78.5% in [4,∞)). The mean-norm ratio fog/clean is 1.66, with extreme per-class spread (class 0: 6.04×, class 2: 0.64×).
- **The binarized fog means are nearly orthogonal to clean.** Clean↔fog binarized mean cosine fell to 0.125 (micro-30ep: 0.245). The fog class means are healthy in magnitude (binarized norm ratio 1.13) but point ~83° away from their clean counterparts, the per-class drift that the 0.883 cosine shift reflects.
- **Query-gate direction reconfirmed at scale**: fog band accuracy is 33.3% for norm [2,4) vs 7.6% for norm ≥ 4. The high-norm fog points remain the poison: a norm veto on the frozen decode would lift the retained points' accuracy ~4.4×.
- **Anisotropy persists**: ellipticity clean 0.470 / fog 0.663, so fog manifolds remain markedly more elongated.
- **The degradation pipeline holds at scale**: binarized-10kD linear probe 38.2% / prototype 29.4%: the information survives projection+binarization on the converged encoder too.

### Diagnostic Analysis

1. **Training at scale is the single biggest decode lever measured so far**: HDC Prototype Accuracy on Fog tripled (9.6% → 31.7%) and Fog linear probe gained 8.4 points from 5× the gradient budget. The medium-run bet is confirmed, more training directly heals the prototype decode.
2. **But the strong VIB now over-regulates the clean manifold.** The 5× KL weight at 26 full epochs collapses clean magnitudes to ~2.4 (vs the fog's 5.25). This is the flagged over-collapse risk realized: clean features lost their magnitude envelope while the drift grew (cosine shift 0.677 → 0.883). The clean representation is *less* healthy even though the fog decode improved.
3. **The tension to resolve**: the fog decode improved despite (or because of?) the clean collapse. The 128D prototype metric rewards low-norm clean centroids against high-norm fog queries in a particular way; before trusting this, the v4 10kD ladder must be run on this checkpoint (it measures the HDC decode, the deployment metric).

---

## Phase 17: The Medium Run's Over-Collapse: 5× VIB Destroyed the HDC Decode

The v4 ladder on the medium-26ep `supcon_vib_strongvib` checkpoint delivered a decisive negative result that overturns the Phase 16 optimism: **the 10kD HDC decode collapsed on every condition** relative to the 30-epoch micro encoder, while the 128D headroom metrics simultaneously *improved*. The 128D numbers were a magnitude artifact.

### The 10kD Ladder: Micro-30ep → Medium-26ep (zero-shot, frozen clean prototypes)

| Condition | micro-30ep zero-shot | medium-26ep zero-shot | Δ |
| :--- | :--- | :--- | :--- |
| **Clean Control** | 77.6% | 43.7% | **−33.8** |
| **Beam Missing** | 72.2% | 42.8% | −29.4 |
| **Incomplete Echo** | 74.4% | 45.4% | −29.0 |
| **Cross Sensor** | 62.0% | 33.3% | −28.7 |
| **Motion Blur** | 68.8% | 41.2% | −27.6 |
| **Wet Ground** | 66.5% | 42.4% | −24.2 |
| **Snow** | 61.1% | 43.4% | −17.7 |
| **Fog** | 35.0% | 19.9% | −15.1 |
| **Crosstalk** | 38.1% | 29.9% | −8.2 |

### The Degeneracy Signatures

1. **`cross_sensor` is bit-identical across every ladder row** (zero-shot = oracle = naive = top50 = flip = sweep = 0.3334), even true-label re-estimation changes nothing. The class prototypes have collapsed toward a common direction; the pool re-estimation is an identity because there is nothing class-specific left to re-estimate.
2. **The leave-one-out signal AUROCs are degenerate** (snow: conf 0.000 / norm 0.000 / lr 0.000; several conditions hit exactly 1.000), helpful/harmful profiling against a near-random classifier is meaningless.
3. **"Positive oracle headroom" appeared** (fog +7.0, clean +3.5, blur +2.5), but only because the *base* prototypes are broken; re-estimating toward target features helps a broken base. This is not evidence for adaptation.

### Why: the 5× VIB over-collapse, and the 128D mirage

The Phase 16 deep diagnostics now read as the cause, not a curiosity:

- **Clean magnitudes collapsed to 2.42** (healthy ≈ 5.6): the 5× KL at 26 full epochs pushed the clean posterior means toward the origin.
- The binarized class means are *self-consistent* (norm 78.8/10000, points within a class agree) but **class-discriminative directions are gone**: the class means point in similar directions, so cosine classification in 10kD is near-chance.
- The 128D headroom metric (Euclidean `cdist` to *unnormalized* centroids) **exploits the residual magnitude differences** between classes (per-class clean mean norms are heterogeneous, class 2 nearly collapsed, others less so). That is why "HDC Prototype Accuracy (Fog) 31.7%" looked like an improvement while the deployment decode (10kD cosine) fell 15–34 points everywhere. **The 128D headroom metric is not a proxy for the HDC decode: Phase 16's headline was a mirage.**

### Verdict

**`supcon_vib_strongvib` at medium scale over-regularized the representation.** The strong-VIB lever was right in principle (it produced the best micro-scale frozen decode) but the 5× weight at 26 full epochs crossed the over-collapse threshold. The checkpoint should **not** be continued, the KL axis is over-trained, not under-trained.

---

## Phase 18: The Query Gate Verdict and the MidVIB Probe

The 2h diagnostic run (query gate on both strongvib encoders + the KL-0.03 midvib step probe at 12.7k steps) delivered two results: the norm-veto query gate is **neutral on the healthy encoder**, and the midvib probe's clean manifold stays **healthy** while its fog decode does not improve.

### The Query Gate (frozen prototypes, veto norm ≥ τ): acc | mIoU | retained

| Encoder | Fog tau=inf (no gate) | Fog tau=4 | Fog tau=5 | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **micro-30ep strongvib (healthy)** | 35.0% \| 0.084 | 37.6% \| 0.085 (70%) | 33.8% \| 0.085 (94%) | **neutral**, +2.6 acc, mIoU flat |
| **med-26ep strongvib (collapsed)** | 12.8% \| 0.060 | 30.5% \| 0.092 (24%) | 21.6% \| 0.077 (46%) | gains only at tiny retention |
| **midvib-12.7k** | 25.1% \| 0.083 | 53.8% \| 0.115 (23%) | 48.1% \| 0.108 (40%) | gains only at tiny retention |

**Verdict:** the norm veto is not a meaningful lever on the healthy encoder's 10kD decode (τ=4: +2.6 acc at 70% retention, mIoU flat 0.084→0.085 across all conditions). The Phase 15 "band-acc promise" (measured on the *128D* Euclidean decode, 36.3% vs 6.6%) does **not transfer to the 10kD cosine decode**. The gate only rescues the *broken* encoders, and only by discarding 77–92% of points. Query-side norm gating is parked.

### The MidVIB Probe (KL 0.03, 8 epochs × 50% data ≈ 12.7k steps)

- **Clean manifold health: PASSED**, clean zero-shot 76.2% (vs 43.7% for the collapsed med encoder), no magnitude collapse at 12.7k steps. KL 0.03 is safe at this budget.
- **Fog decode: did not improve**, fog zero-shot 25.1%, *worse* than strongvib-9.5k's 35.0%. KL-weight × step-count is a non-monotone tradeoff, not a dial.
- **Most anisotropic manifold yet**: fog ellipticity 0.822 (clean 0.482), the strongest anisotropy measurement in the project, coinciding with the worst relative fog decode.
- Fog band acc (128D) [2,4): 51.8%, the best band acc ever measured, but it did not carry to the 10kD decode.

### The Missing Data Point

The KL axis has now been probed at {0.01 (micro only), 0.03 (12.7k), 0.05 (9.5k and 83k)}, but **plain 0.01 has never been run at medium scale**. The 5× collapse at 83k steps tells us nothing about whether the *default* configuration survives it. This is the cheapest decisive experiment available.

---

## Phase 19: The ZCA Whitening Probe (completed)

A zero-training test of whether the anisotropy hypothesis warrants a new loss term, before committing the 10h run.

`oracle_gating_eval.py --whiten` computes a ZCA whitening transform from 500k clean points (covariance eigen-decomposition, symmetric `W = U·D^(−1/2)·Uᵀ`), applies it to the clean features (prototype building, probe training) and every corrupt set (pool/val) before projection + binarization. Since binarization is direction-only, this isolates the anisotropy variable: if the elongated manifolds (ellipticity 0.48 clean / 0.60–0.82 fog) are the reason centroids underperform the linear probe, whitening must recover the decode.

| Setting | Value |
| :--- | :--- |
| Encoder | micro-30ep `supcon_vib_strongvib` (fog zero-shot 35.0%, clean 77.6%) |
| Transform | ZCA whitening, clean statistics, 500k points, seeded |
| Question | Does decorrelating the space improve the 10kD prototype decode? |

### The Results (micro-30ep strongvib, fog + clean control)

| Metric | Non-whitened | Whitened | Δ |
| :--- | :--- | :--- | :--- |
| Fog zero-shot | 35.0% | 25.9% | **−9.1** |
| Fog perfect-oracle | 22.7% | 8.6% | −14.1 |
| Clean zero-shot | 77.6% | 56.1% | **−21.5** |
| Fog query-gate (τ=5) | 33.8% | 30.9% | −2.9 |

### The Verdict

**Whitening destroys the decode: the anisotropy is load-bearing.** The semantic information lives in the high-variance (elongated) directions; decorrelating the space throws it away. Per the outcome table this is the third row: the elongated structure is load-bearing, the covariance is not to be touched, and the naive isotropy-loss direction is abandoned. (If isotropy were ever pursued, it would have to be a *learned* regularizer that preserves discriminative structure, but the probe shows the decode does not want the space made isotropic.) The config path (plain KL 0.01 / SupCon-only at scale) is the surviving direction, which is exactly what the convergence probe (Phase 20) tests.

---

## Phase 20: The Convergence Probe (Long Epochs × Small Subset): COMPLETED

The probe ran plain `supcon_vib` (KL 0.01) to 62 epochs on 10% data (≈19.7k steps, cut short of 100 epochs once the decision was banked), then the v4 ladder measured the frozen decode.

### The Result (62-epoch probe checkpoint, v4 ladder)

| Condition | Zero-shot | Perfect-oracle | Naive EMA |
| :--- | :--- | :--- | :--- |
| **Clean Control** | **81.4%** | 81.3% | 66.4% |
| **Fog** | 27.4% | **46.4% (+19.1)** | 25.7% |
| **Snow** | 65.2% | 67.9% | 48.7% |
| **Wet Ground** | 68.9% | 68.4% | 55.5% |
| **Incomplete Echo** | 77.9% | 77.5% | 55.5% |
| **Crosstalk** | 34.2% | 32.4% | 15.0% |
| **Beam Missing** | 75.1% | 74.3% | 56.0% |
| **Motion Blur** | 72.9% | 71.8% | 55.0% |
| **Cross Sensor** | 64.8% | 62.6% | 42.5% |

### The Verdict

1. **KL 0.01 is definitively safe at scale.** Clean zero-shot **81.4%**, the best clean decode measured on any encoder (vs 80.4% micro-30ep plain, 77.6% strongvib, 43.7% collapsed med). The Phase 20 outcome lands in the first row: *0.01 stays healthy → overnight: plain `supcon_vib` at medium scale*.
2. **Fog oracle 46.4%, the best ever measured** (+19.1 over zero-shot). Only the second positive fog headroom in the project (additive: 45.3%), and this time on the plain config that is going to the medium run. The plain encoder's fog class means are genuinely usable at 19.7k steps; the 5× VIB destroyed this property (strongvib fog oracle was 22.7%).
3. Training was still improving at epoch 62 (train IoU 0.39, loss 0.61), with no collapse signal and no plateau of concern.

---

## Phase 21: The Plain Medium Run: Health Passed, Fog Adaptation Was an Accuracy Artifact

The overnight medium run (plain `supcon_vib`, KL 0.01, 26 epochs on 100% data ≈ 83k steps) plus its v4 ladder delivered the strongest result in the project: a fully healthy clean manifold and, for the first time, **naive prototype adaptation that works on Fog**.

### The 10kD Ladder (plain medium checkpoint, v4 harness)

| Condition | Zero-shot | Perfect-oracle | Naive EMA |
| :--- | :--- | :--- | :--- |
| **Clean Control** | **82.7%** | 82.7% | 76.8% |
| **Fog** | 26.4% | **48.3% (+21.9)** | **42.5% (+16.1)** |
| **Snow** | 66.6% | 69.9% | 53.4% |
| **Wet Ground** | 68.8% | 68.3% | 57.1% |
| **Incomplete Echo** | 78.8% | 78.5% | 71.2% |
| **Crosstalk** | 33.5% | 28.9% | 18.1% |
| **Beam Missing** | 77.2% | 76.4% | 64.1% |
| **Motion Blur** | 73.4% | 72.9% | 61.6% |
| **Cross Sensor** | 68.9% | 68.1% | 53.9% |

### The Verdict

1. **Clean health: passed, definitively.** Clean zero-shot **82.7%**, the best ever measured in 10kD (probe@19.7k: 81.4%; strongvib med: 43.7%). No collapse. The plain config is the right one.
2. **The 128D number was a magnitude artifact, again.** The 128D eval claimed 48.1% fog prototype accuracy; the honest 10kD decode is 26.4%. The 128D Euclidean `cdist` was exploiting the clean/fog magnitude split (3.57 vs 6.70). Confirmed once more: the 10kD ladder is the deployment metric.
3. **The mIoU view overturns the "adaptation works" claim (Phase 21.1 revision).** The accuracy gains from fog prototype adaptation (+21.9 oracle, +16.1 naive) are a **majority-class artifact**: re-estimated fog prototypes improve Road/Building/Vegetation (boosting point accuracy) while destroying the rare classes. On mIoU (the paper metric), adaptation *crashes* fog from 10.1% (zero-shot) to 4.9% (oracle) and 1.5% (naive). **The frozen clean prototypes remain the best mIoU decoder on every condition.**

### The mIoU Ladder (plain medium encoder, v4 harness)

| Condition | Zero-shot acc \| mIoU | Oracle acc \| mIoU | Naive acc \| mIoU |
| :--- | :--- | :--- | :--- |
| **Clean Control** | 82.7% \| **49.6%** | 82.7% \| 48.9% | 76.9% \| 36.4% |
| **Fog** | 26.4% \| **10.1%** | 48.2% \| 4.9% | 42.4% \| 1.5% |
| **Snow** | 66.6% \| 39.4% | 69.9% \| 37.5% | 53.4% \| 26.2% |
| **Wet Ground** | 68.8% \| 49.0% | 68.2% \| 48.2% | 57.1% \| 33.7% |
| **Incomplete Echo** | 78.8% \| 41.2% | 78.5% \| 40.9% | 71.2% \| 34.5% |
| **Crosstalk** | 33.5% \| 12.0% | 28.8% \| 8.1% | 18.1% \| 4.0% |
| **Beam Missing** | 77.2% \| 53.7% | 76.3% \| 51.7% | 64.1% \| 36.8% |
| **Motion Blur** | 73.4% \| 44.3% | 72.9% \| 43.4% | 61.7% \| 31.5% |
| **Cross Sensor** | 68.9% \| 41.5% | 68.0% \| 38.8% | 53.9% \| 28.4% |

*Mean mIoU (8 corruptions, zero-shot): 36.4%.*

---

## Phase 22: Oracle Retraining with Buffer Selection: Oscillates, Never Beats Zero-Shot mIoU

The trainer-faithful oracle retraining (HyperLiDAR-style buffer selection: cumulative `is_wrong` memory, 5% buffer, 2× perceptron updates, perfect labels) was run on the plain medium encoder's fog features.

### The Trajectory (acc | mIoU | buffer hard/rand | wrong-now/memory)

| Round | Acc | mIoU | Buffer hard/rand | Wrong now/mem |
| :--- | :--- | :--- | :--- | :--- |
| 0 (base) | 26.4% | **10.1%** | n/a | 36723/36723 |
| 1 | **50.5%** | 7.2% | 36723/13277 | 18476/18476 |
| 2 | 28.2% | 9.0% | 18476/31524 | 34694/34694 |
| 3 | 41.0% | 4.1% | 34694/15306 | 25778/25778 |
| 4 | 28.7% | **9.1%** | 25778/24222 | 35504/35504 |
| 5 | 32.1% | 5.9% | n/a | n/a |

### The Verdict

1. **Buffer selection does not recover the fog mIoU.** Best mIoU across all rounds: 9.1% (round 4), still below the zero-shot baseline (10.1%). The accuracy gains (50.5% at round 1) are again majority-class artifacts: acc/mIoU diverge by 43 points at round 1.
2. **The loop limit-cycles.** The trajectory oscillates (acc 50.5 → 28.2 → 41.0 → 28.7 → 32.1; mIoU 7.2 → 9.0 → 4.1 → 9.1 → 5.9): full-strength 2× perceptron updates on a 5% buffer over-shoot, and the re-selected buffer corrects back next round. The wrong-memory swings 18.5k–36.7k without stabilizing.
3. **The hard buffer is majority-dominated.** 36.7k of the first 50k buffer points are misclassified (73% error on fog), and the majority classes (Road/Building) dominate that population, so the perceptron updates fix majority classes (acc up) while the rare classes, starved of buffer slots, keep collapsing (mIoU down). The exact mechanism behind the Phase 21 mIoU crash, now at the buffer level.

---

## Phase 22.1: Per-Class Buffer Selection: Still Below Zero-Shot; Decode-Side Retraining Closed

The per-class hard-selection variant (per-class quota from the wrong-memory, so rare classes cannot be starved of buffer slots) was run on the same fog oracle setup.

### The Trajectory (acc | mIoU | buffer hard/rand | wrong-now/memory)

| Round | Acc | mIoU | Buffer hard/rand | Wrong mem |
| :--- | :--- | :--- | :--- | :--- |
| 0 (base) | 26.4% | **10.1%** | n/a | 36722 |
| 1 | 50.5% | 7.2% | 18401/31599 | 42568 |
| 2 | 26.3% | 8.9% | 24455/25545 | 53666 |
| 3 | 40.3% | 4.5% | 23953/26047 | 60419 |
| 4 | 28.3% | **9.8%** | 25730/24270 | 70712 |
| 5 | 38.9% | 5.9% | n/a | n/a |

### The Verdict

1. **Per-class selection barely helps and does not cross the baseline.** Best mIoU 9.8% (vs 9.1% global, 10.1% zero-shot). The trajectory still limit-cycles (acc 50.5 → 26.3 → 40.3 → 28.3 → 38.9), and the round-1 acc spike (50.5%) with mIoU 7.2% is the same majority-class artifact.
2. **The wrong-memory saturates monotonically** (36.7k → 42.6k → 53.7k → 60.4k → 70.7k): under fog, 50–70% of the pool is misclassified, so the 5% buffer can never drain the hard population. The hard set simply outgrows the sampling capacity.
3. **Decode-side retraining is closed.** Both buffer-selection variants (global and per-class) fail to recover fog mIoU with perfect labels. Combined with Phase 22, the evidence says: the rare classes' fog features are not recoverable by prototype movement at decode time, no matter how the retraining buffer is chosen.

### Why This Is Consistent With the Original 5–6× mIoU Claim

The buffer selection's demonstrated mIoU boost (5–6×) came from `unsup_main.py`'s training loop: retraining the HDC prototypes on **source-domain data**, where classes are separable and hard examples are genuinely fixable. The oracle test applies the same mechanism to **target (fog) data at decode**, where the rare classes' features have collapsed: hard-example mining cannot fix what the representation cannot separate. The mIoU recovery therefore belongs in the **encoder side**: buffer-selection-guided retraining on data where separability exists, or making the encoder's fog features separable in the first place.

---

## Phase 22.2: Artifact-Filtered Buffer Selection: 99.96% of Fog Misclassifications Are Confident Hallucinations

The artifact-filtered buffer mode (norm, cosine-to-true, perceptron-loss, and top-2-margin filters on the hard candidates, plus per-class quota protection) was run on the same fog oracle setup.

### The Trajectory (acc | mIoU | buffer hard/rand | filter pass-rates)

| Round | Acc | mIoU | Hard/Rand | Filter pass (candidates/misclassified) |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 26.4% | **10.1%** | 269/49731 | 269 / 735698 (norm 215936, true 153046, loss 306, marg 269) |
| 1 | 51.3% | 7.9% | 1569/48431 | 1569 / 486368 |
| 2 | 24.8% | 9.2% | 352/49648 | 352 / 751710 |
| 3 | 46.5% | 5.9% | 1617/48383 | 1617 / 533959 |
| 4 | 28.6% | **9.1%** | 686/49314 | 686 / 713725 |
| 5 | 45.8% | 5.9% | n/a | n/a |

### The Verdict

1. **The confident-hallucination hypothesis is confirmed at population scale.** Of 735,698 misclassified fog points, only **269 survive the artifact filters (0.04%)**. The cascade breakdown is the story: the norm filter removes 71% (215,936 of 735,698, the high-magnitude artifacts), the cosine-to-true filter another 29% of the remainder, and then **the perceptron-loss filter annihilates the rest: 153,046 → 306**. Under heavy fog, essentially every misclassification is *confidently* wrong (loss > 0.15); they are hallucinations, not boundary-adjacent recoverable points. The perceptron loss is a dramatically sharper artifact signal than the 128D norm (0.2% survival vs 29%), worth noting for the query-gate work too.
2. **The filters do not rescue the trajectory.** Still oscillating (best mIoU 9.2%, below the 10.1% zero-shot baseline). With only 269–1,617 hard candidates per round (of 50k buffer slots), the buffer is ~97% random fill, and the random half's unfiltered misclassified points still inject artifact updates.
3. **Decode-side retraining is closed, definitively.** Across all four buffer strategies (global, per-class, paper-form, artifact-filtered), fog mIoU never crosses the frozen zero-shot baseline with perfect labels. The fog pool contains essentially zero recoverable hard examples: prototype adaptation has nothing to learn from.

### The Paper-Ready Statistic

**Under heavy fog, 99.96% of misclassified points are confident artifacts** (perceptron loss > 0.15 with cosine-to-true < 0.05). This single number explains every adaptation failure in this project (Phase 14 oracle crash, Phase 22 retraining oscillations) and motivates the encoder-side fix: fog mIoU can only improve by making the rare classes' fog features separable in the first place.

---

## Phase 23: The Artifact-Gate Sweep: Gating Verdict (Crosstalk Solved, Fog Exhausted)

The in-memory gate sweep (Phase 23 diagnostic) exhaustively searched the artifact-gate space on the plain medium encoder: 1200 configs over (128D norm, 10kD top-2 margin, top-1 cosine, **128D nearest-clean-prototype cosine**, probe confidence), plus the oracle-aware perceptron-loss gate as an upper bound.

### The Pareto (best mIoU per retention band)

| Band | Fog mIoU | Fog cfg | Crosstalk mIoU | Crosstalk cfg |
| :--- | :--- | :--- | :--- | :--- |
| ≥75% | 11.2% | margin≥0.05, cos128≥0.2, conf≥0.3 | 15.0% | n<8, m≥0.02, c1≥0.1, c128≥0.2 |
| 50–75% | 18.0% *(oracle loss)* | loss≤0.15 | **23.1%** *(label-free)* | m≥0.2, c1≥0.3, c128≥0.4 @ 52% |
| 25–50% | **55.3%** *(oracle loss)* | loss≤0.02 @ 28% | 60.7% *(oracle loss)* | loss≤0.02 @ 35% |
| 10–25% | 17.0% | n<5, m≥0.2, c1≥0.3, c128≥0.4 | n/a | n/a |
| <10% | 20.3% | n<4, m≥0.2, c1≥0.3, c128≥0.4 | n/a | n/a |

### The Verdict

1. **Crosstalk: the 20 mIoU target is achievable label-free**: 23.1% mIoU at 51.6% retention with a pure margin+cosine gate (no oracle). This gate is buildable today.
2. **Fog: label-free gating is exhausted.** The new 128D cos-to-prototype and probe-confidence signals did not close the gap: the best label-free configs stall at 11–17% mIoU at usable retention (the ≥20% numbers only appear below 10% retention). The oracle-loss gate (18% @ 64% retention, 55% @ 28%) proves the information exists but is not estimable from prototype geometry + confidence alone, consistent with Phase 22.2: fog misclassifications are confident artifacts that are geometrically indistinguishable from confident-correct points without the true label.
3. **Per the decision rule: go back to feature-extractor training.** Fog gating cannot reach the target; the encoder is the only remaining lever. The oracle-gate per-class IoUs (best config: class spread 0.13–0.97, with the weakest classes at 0.13–0.22) identify exactly which classes the encoder must rescue.
4. **The oracle-loss bound is the prize if the encoder improves**: 55% mIoU at 28% retention with perfect gating means a strong encoder + the (now-buildable) crosstalk-style gate could yield far more than 20 mIoU on fog.

---

## Phase 23.1: The Buffer-Selection Pretrain Weights: Confirmed as a Source-Domain Mechanism

The existing `unsup_kitti-c.py --pretrain` weights (`logs/kitti_pretrain/hdc_sub.pth`: extractor + HDC trained with 14 buffer-selection retrain epochs) were evaluated frozen on fog and crosstalk under the full 4071-frame protocol.

### The Results (frozen decode, unsup_kitti-c protocol)

| Condition | mIoU | Head | Mid | Tail | Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Fog-3** | 5.8% | 13.0% | 2.2% | **0.06%** | 14.6% |
| **Crosstalk-3** | 7.0% | 14.8% | 4.3% | **0.33%** | 19.8% |

### The Verdict

1. **The buffer-selection-pretrained model sits at historical baseline levels on corruptions**: fog 5.8% mIoU ≈ the DualGateModel-era fog (5.78%). The 5–6× mIoU boost from buffer selection lives in the *source-domain* retraining (where classes are separable and hard examples are fixable); it does not transfer to the corrupted decode. This is the same conclusion as Phase 22 (decode-side retraining closed), now confirmed from the pretrain side: the trained HDC prototypes are no better on fog/crosstalk than the pre-buffer-selection era.
2. **Tail mIoU is catastrophic** (0.06–0.33%) on both corruptions: the pretrained models have essentially no rare-class recovery under corruption, consistent with the Phase 23 oracle-gate per-class findings.
3. **The mechanism's role is settled**: buffer selection is a source-domain training-time tool (improve the HDC prototypes on clean data), not a corruption-robustness tool. The corrupted-condition mIoU must come from the feature extractor.

---

## Phase 24: The Condition Autopsy: Why Fog and Crosstalk Are Stuck

The per-condition autopsy (frozen clean prototypes, plain medium encoder, 100k-point val) measured the hyperspace/decode signature of all 8 conditions. Two conditions (Fog, Crosstalk) carry a distinctive and coherent feature-level signature that the geometric corruptions do not share.

### The Autopsy Table

| Condition | Acc | mIoU | LP | nMis | ArtFrac | ArtSurv | marC/marM | nrmC/nrmM | <4norm | cosShift | Ellip | BinCos |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fog** | 26.4% | 10.1% | 36.3% | 73604 | 0.487 | **2335** (3.2%) | 0.29/0.11 | 4.8/7.3 | **12.0%** | **0.821** | **0.762** | **0.111** |
| **Crosstalk** | 33.5% | 12.0% | 35.3% | 66475 | 0.676 | **2477** (3.7%) | 0.43/0.19 | 4.3/5.8 | **30.7%** | **0.780** | **0.701** | **0.146** |
| Snow | 66.6% | 39.4% | 74.8% | 33425 | 0.679 | 8591 (25.7%) | 0.48/0.15 | 3.5/3.7 | 85.2% | 0.043 | 0.285 | 0.293 |
| Wet Ground | 68.8% | 49.0% | 75.3% | 31208 | 0.636 | 9082 (29.1%) | 0.40/0.20 | 3.7/3.5 | 79.2% | 0.028 | 0.297 | 0.355 |
| Incomplete Echo | 78.8% | 41.2% | 84.2% | 21173 | 0.665 | 5622 (26.6%) | 0.50/0.19 | 3.5/3.8 | 87.8% | 0.000 | 0.306 | 0.280 |
| Beam Missing | 77.2% | 53.7% | 83.7% | 22790 | 0.622 | 6804 (29.9%) | 0.48/0.17 | 3.5/3.7 | 85.0% | 0.005 | 0.284 | 0.287 |
| Motion Blur | 73.4% | 44.3% | 79.3% | 26564 | 0.659 | 7005 (26.4%) | 0.46/0.17 | 3.5/3.7 | 86.0% | 0.013 | 0.268 | 0.299 |
| Cross Sensor | 68.9% | 41.5% | 77.5% | 31107 | 0.563 | 10577 (34.0%) | 0.41/0.12 | 3.4/3.7 | 81.2% | 0.050 | 0.291 | 0.309 |

*ArtSurv = artifact-filter survivors among misclassified (norm<6, cos-true≥0.05, loss≤0.15, margin≥0.02). marC/marM = top-2 cosine margin of correct/misclassified. nrmC/nrmM = 128D norm of correct/misclassified. <4norm = fraction of points with norm < 4. cosShift = mean per-class 128D clean→corrupt cosine shift. Ellip = top-eigenvalue/trace ellipticity. BinCos = clean↔corrupt binarized class-mean cosine.*

### The Fog/Crosstalk Signature

1. **Magnitude inflation is the lead discriminator.** Only 12–31% of fog/crosstalk points sit in the healthy norm < 4 band (vs 79–88% for every geometric corruption), and the *misclassified* points have far higher norms than the correct ones (fog: 7.3 vs 4.8; crosstalk: 5.8 vs 4.3), versus geometric corruptions where correct and misclassified norms are identical (~3.5/3.7). The high-norm population is the poison (the Phase 18 query-gate direction), and in fog/crosstalk it is the *majority* of the data.
2. **The manifolds are extreme in every geometry metric**: cosine shift 0.78–0.82 (geometric: 0.00–0.05), ellipticity 0.70–0.76 (geometric: 0.27–0.31), binarized mean cosine 0.11–0.15 (geometric: 0.28–0.36). The class structure is rotated ~80° and elongated; the binarized prototypes are nearly orthogonal to clean.
3. **Even the *correct* classifications are marginal**: fog's correct-point margins are 0.29 (vs 0.40–0.50 for geometric): the representation barely separates classes under fog, not just the errors.
4. **The decoder ceiling is quantified**: only **3.2–3.7% of misclassified points are boundary-recoverable** (passing all artifact filters) versus **26–34%** for the geometric corruptions, a ~8–10× difference. This is the *mathematical* reason every decode-side lever failed on fog/crosstalk (Phases 14, 22, 23): with perfect labels, gating, adaptation, and buffer selection combined, there is ~10× less recoverable signal to work with. Note ArtFrac (the single loss filter) does *not* separate the conditions; the separation needs the joint norm+cos-true+loss+margin signature.
5. **The representation itself is degraded** (LP 35–36% vs 75–84%), so even a learned decoder starts from a collapsed space, echoing the Corruption Atlas verdict, now fully quantified per condition.

### The D3CTTA Context

D3CTTA's paper numbers come from their own converged backbone plus a mechanism we have never tested: **encoder-level test-time adaptation** (entropy-minimization batch-statistic alignment), not decode-side movement. The autopsy's magnitude-inflation finding is the reason this lever is the one untested candidate that *could* matter: fog/crosstalk's feature *statistics* (norms, means) are grossly misaligned with clean (88% of points in the wrong norm band), and BN-style statistic alignment directly targets that, unlike gating/adaptation/buffer selection, which the autopsy shows are capped at ~3.5% recoverable signal.

---

## Phase 24.1: The BN-Statistic Alignment Probe: Not the Missing Lever for Fog

The D3CTTA-style test-time alignment (per-dimension mean/std of the corrupt features aligned to clean, then re-decode with frozen clean prototypes) was run across all 8 conditions. This directly tests the one mechanism D3CTTA uses that we had never tried: encoder-level statistic adaptation rather than decode-side movement.

### Aligned vs Baseline (Acc → AlignAcc | mIoU → AlignmIoU)

| Condition | Acc → AlignAcc | mIoU → AlignmIoU | Δ mIoU |
| :--- | :--- | :--- | :--- |
| **Fog** | 26.4% → 25.2% | 10.1% → 10.7% | **+0.6 (flat)** |
| **Crosstalk** | 33.5% → 34.1% | 12.0% → 15.1% | **+3.1** |
| Snow | 66.6% → 69.6% | 39.4% → 39.9% | +0.5 |
| Wet Ground | 68.8% → 59.4% | 49.0% → 40.5% | **−8.5** |
| Incomplete Echo | 78.8% → 78.2% | 41.2% → 40.6% | −0.6 |
| Beam Missing | 77.2% → 77.4% | 53.7% → 53.6% | −0.1 |
| Motion Blur | 73.4% → 74.4% | 44.3% → 44.8% | +0.5 |
| Cross Sensor | 68.9% → 72.0% | 41.5% → 42.6% | +1.1 |

### The Verdict

1. **Fog: the alignment mechanism is not the missing lever.** Fog mIoU stays flat (10.1 → 10.7, noise level). The D3CTTA-style statistic alignment cannot fix the fog representation either, consistent with the Phase 19 whitening verdict: on fog, the *structure* (not the statistics) carries what little signal exists, and the magnitude inflation survives alignment because it is a per-point property, not a per-dimension statistic.
2. **Crosstalk: alignment is a modest positive** (+3.1 mIoU: 12.0 → 15.1, the largest gain of any condition), worth noting, but far below the 20 mIoU target, and the label-free margin gate already delivers 23.1% on crosstalk (Phase 23). If crosstalk is ever pursued further, alignment + gate is a candidate combination.
3. **Wet Ground degrades under alignment** (−8.5 mIoU): the statistic transfer is harmful where the corruption is reflectance-based, an important caution against blanket BN-style TTA.
4. **The D3CTTA-mechanism question is now answered on this encoder**: their paper gains on fog/crosstalk came from their own backbone + this alignment mechanism; on our features, the mechanism contributes nothing to fog and a small gain to crosstalk. The encoder remains the primary candidate for fog, with the intra-class balance checks and the encoder-side options from Phase 23 still open, with no path treated as closed beyond what the data now shows.

---

## Phase 24.2: The Fog Class-Level Autopsy: the Collapse Is Class-Conditional, Not Uniform

Two new autopsy dimensions on fog: the **learned-decoder ceiling** (LP mIoU and per-class) and the **per-class poison-band structure** (fraction of each class's points in the norm ≥ 4 band and their accuracy there).

### The Numbers (fog, 100k-point val, 8 classes present: 2, 4, 7, 11, 13, 14, 15, 16)

| Class | Proto IoU | LP IoU | Poison-band frac | Poison-band acc |
| :--- | :--- | :--- | :--- | :--- |
| Terrain (11) | **0.526** | 0.484 | 0.727 | **0.895** |
| Truck (4) | 0.157 | **0.007** | 0.998 | 0.591 |
| Other-object (16) | 0.093 | 0.033 | 0.926 | 0.077 |
| Building (15) | 0.024 | 0.010 | 0.984 | 0.030 |
| Other-ground (14) | 0.005 | **0.044** | 0.914 | 0.004 |
| Road (7) | 0.003 | 0.000 | n/a | n/a |
| Traffic-sign (13) | 0.000 | 0.000 | 0.885 | 0.000 |
| Bicycle (2) | 0.000 | 0.000 | n/a | n/a |

*Overall: proto mIoU 10.1%, LP mIoU 7.2%, LP acc 36.3%.*

### The Findings

1. **The learned decoder is not the fog fix, confirmed at class level.** LP mIoU 7.2% < proto 10.1%; the LP is *worse* on Truck (0.007 vs 0.157), Road (0.0 vs 0.003), Other-object (0.033 vs 0.093), and no better on the dead classes (Road/Traffic-sign/Bicycle at 0.0 for both). Every decoder variant we now have (centroids, learned head, gates, alignment, adaptation, retraining) converges on the same ceiling: the representation's class-conditional collapse, not the decode.
2. **The magnitude inflation is universal, but the poison band is class-selective.** 73–100% of every class's points sit in the norm ≥ 4 band, so a class-agnostic norm target would be wrong. Yet within the poison band, **Terrain classifies at 89.5% and Truck at 59.1%** while Building (3.0%), Other-ground (0.4%), and Traffic-sign (0.0%) are dead. The fog representation collapse is **class-conditional**: Terrain and Truck survive the high-norm regime, Road/Building/Other-ground/Traffic-sign/Bicycle do not.
3. **Why this matters for the training regimen**: the collapse is *not* intrinsic to fog: the encoder can represent fog-surviving classes (Terrain's poison-band points at 90% accuracy prove it). The objective needs to rescue the *collapsing* classes specifically. The natural hypothesis is geometric: the dying classes (Road, Building, Other-ground) are large planar surfaces whose features rely on fine range texture that scattering destroys, while Terrain (near-field ground) and Truck (large solid) retain structure, testable via a depth/range correlation of the poison-band points (far-field destruction).
4. **Actionable target list for the pretraining objective**: classes 7 (Road), 15 (Building), 14 (Other-ground), 13 (Traffic-sign), 2 (Bicycle) are the fog casualties: the additive volumetric augmentation and any per-class weighting must focus on keeping *these* classes' fog features separable, with Terrain/Truck as the proof that the representation can do it.

---

## Phase 24.3: The Additive Retrain Autopsy: the Continuous Space Heals, the HDC Decode Doesn't

The retrained additive micro (30 epochs, volumetric-noise augmentation) was autopsied on fog and compared to the Phase 24 med-plain signature.

### Additive Retrain vs Med-Plain (fog)

| Metric | Med plain | Additive retrain | Δ |
| :--- | :--- | :--- | :--- |
| Proto Acc / mIoU | 26.4% / 10.1% | 27.3% / 9.2% | ≈ flat |
| **LP acc** | 36.3% | **57.0%** | **+20.7** |
| LP mIoU | 7.2% | 10.0% | +2.8 |
| **ArtSurv (recoverable hard examples)** | 2335 (3.2%) | **8810 (12.1%)** | **3.8×** |
| Margin (correct / mis) | 0.29 / 0.11 | **0.50 / 0.27** | healed |
| Norm (correct / mis) | 4.8 / 7.3 | **4.5 / 4.9** | gap gone |
| Cos shift | 0.821 | 0.738 | improved |
| Ellipticity | 0.762 | 0.658 | improved |
| Binarized mean cos | 0.111 | **0.046** | **worse** |
| Alignment | 25.2% / 10.7% | 27.3% / 10.7% | flat |

### The Findings

1. **The additive regimen heals the continuous 128D representation dramatically**: LP acc 36.3 → 57.0, healthy correct-point margins (0.50 vs 0.29), the clean/fog magnitude gap collapsed (4.5/4.9 vs 4.8/7.3), and the recoverable hard-example fraction **tripled to 12.1%**: the decoder ceiling is 3.8× higher. This is the strongest representation-level fix yet measured.
2. **But the HDC decode is flat, and the binarized means got *more* orthogonal (0.046).** Proto mIoU 9.2% ≈ 10.1%; the 10kD sign-space decode does not capture the healed continuous geometry. This is the 128D→10kD divergence at its starkest: the information exists in the continuous space (LP 57%) but the binarized prototypes cannot reach it. **The medium additive run is therefore the decisive test: if convergence fixes the BinCos/10kD transfer, the regimen is the answer; if not, the HDC encode side (projection/binarization) needs attention even with a healed space.**
3. **The far-field hypothesis is partially refuted.** The depth diagnostic (now working, relative median split) shows the *survivors* degrade with range (Terrain 0.99 → 0.80, Truck 0.99 → 0.95), but the *dying* classes (Building, Other-ground, Traffic-sign, Other-object) are dead at **both near and far range** (near_acc 0.00). Their collapse is not far-field destruction; it is range-independent. The far-field intuition explains the survivors' mild degradation, not the casualties' death.
4. **Fog norm↔depth correlation is weakly negative (−0.40)**: fog points at higher range carry slightly *lower* feature norms, opposite to the "far-field inflation" story; the magnitude structure is not range-driven.

### Next Steps

1. **The medium additive run is now the pivotal test**: the continuous-space heal is proven at micro scale; whether it survives the 128D→10kD transfer at convergence decides the regimen. If BinCos stays low, the HDC encode (projection/binarization) becomes a target, e.g., learning the projection or normalizing before binarization.
2. **The gate sweep on the additive fog** (pending): with ArtSurv at 12.1% and margins 0.50/0.27, the label-free gate should perform better than the plain encoder's 11–17% ceiling; run `--gate_sweep --corruptions fog` on the retrained checkpoint.
3. **Drop the far-field training-target idea**: the casualties die at all ranges, so range-conditioning (Cylinder3D-style) would not rescue them; the collapse is class-intrinsic, not distance-driven.

---

## Phase 24.4: The Additive Gate Sweep: the Oracle Bound Jumps to 67%, the Label-Free Ceiling Does Not

The gate sweep on the additive retrain's fog vs the Phase 23 med-plain sweep:

### The Pareto (additive retrain)

| Band | Best config | mIoU | Acc | Retention |
| :--- | :--- | :--- | :--- | :--- |
| ≥75% | label-free (n<6, cos1≥0.2, cos128≥0.3, conf≥0.5) | 9.8% | 31.3% | 84.1% |
| 50–75% | label-free (n<6, cos1≥0.3, cos128≥0.4, conf≥0.5) | 11.0% | 39.2% | 65.0% |
| 25–50% | **oracle loss≤0.02** | **67.3%** | 97.5% | 28.0% |
| 10–25% / <10% | label-free | 9.2–9.3% | ~31–35% | 7–14% |

### The Findings

1. **The label-free gate is still capped at ~11% mIoU at usable retention**: the healed continuous representation (LP 57%, ArtSurv 12.1%) does not translate to label-free gateability in the 10kD cosine space. The confident-artifact problem persists: label-free signals (margin, cos128, confidence, norm) still cannot separate the additive's recoverable points from its artifacts.
2. **The oracle-loss bound jumped 55.3% → 67.3% @ 28% retention (97.5% acc)**: with perfect labels, the additive features gate to the best fog mIoU ever measured. The information is emphatically *there*: the additive space is highly gateable in principle; the gap between the label-free ceiling (11%) and the oracle bound (67%) is now the single largest untapped margin in the project.
3. **The missing piece is unchanged**: a label-free estimator of the perceptron loss (cos-to-true). The additive representation makes the prize bigger (67% vs 55%), but the access problem is the same one Phase 23 identified. The medium additive run's key question sharpens: does convergence improve the *signals* (margin/confidence discriminability) as well as the representation?

### Next Steps

1. **Full 8-condition autopsy on the additive retrain** (~30 min): the "keeps others high" check: per-condition acc/mIoU and signature vs the med-plain table.
2. **Commit the medium additive run**: the oracle-bound jump (67%) makes it the strongest regime candidate; the decision hinges on whether the label-free signals sharpen at convergence.
3. **If the label-free gap persists at medium scale**: the fix is a *learned* loss estimator: a small head trained (on clean/self-supervised signal) to predict cos-to-true, or per-class calibrated margins, targeting the 11% → 67% gap directly.

---

## Phase 24.5: Full 8-Condition Additive Autopsy: the Regime Trades Others for Fog

Full-condition autopsy on the additive retrain (micro encoder) vs the med-plain table:

| Condition | Plain-med acc/mIoU | Additive acc/mIoU | LP | BinCos |
| :--- | :--- | :--- | :--- | :--- |
| **fog** | 26.4 / 10.1 | 27.3 / **9.2** | **57.0** (+21) | **0.046** (worst) |
| **crosstalk** | 33.5 / 12.0 | **37.2 / 12.5** | **40.2** (+5) | 0.197 (better) |
| snow | 66.6 / 39.4 | 61.9 / 35.6 | 68.6 | 0.241 |
| wet_ground | 68.8 / 49.0 | 65.5 / 43.3 | 71.1 | 0.347 |
| incomplete_echo | 78.8 / 41.2 | 74.9 / 38.7 | 79.9 | 0.208 |
| beam_missing | 77.2 / 53.7 | 71.1 / 48.0 | 77.4 | 0.288 |
| motion_blur | 73.4 / 44.3 | 69.8 / 41.4 | 73.2 | 0.285 |
| cross_sensor | 68.9 / 41.5 | 62.4 / 34.8 | 65.0 | 0.351 |

### The Findings

1. **The additive regime is a trade, not a free lunch**: fog LP +21 pts and crosstalk up (+3.7 acc, +0.05 BinCos), but **all six other conditions drop 2.5–6.7 pts** (acc and mIoU both; worst: cross_sensor −6.7 mIoU, wet_ground −5.7, beam_missing −5.7). The volumetric-noise augmentation trades distributional fidelity on the non-fog conditions for fog/crosstalk robustness. Only fog and crosstalk benefit (crosstalk also gained in LP: 40.2 vs 35.3); snow/wet_ground/cross_sensor all *lost* LP too; they do **not** benefit from the volumetric mix.
2. **Capacity caveat**: this is micro-vs-medium (1/8 params) and there is **no plain-micro 8-condition baseline** (deleted in cleanup), so every delta is capacity-confounded, so the medium additive run is the first *same-capacity* comparison, not merely "longer training". Note the clean control also dropped (additive micro 78.8% acc / 45.1 mIoU vs plain medium 82.7 / 49.6) but *less* than the worst conditions; the trade is partly condition-specific, with the biggest losses on the sparse-return conditions (cross_sensor, beam_missing), which hints the volumetric injection (fake returns in 5% of empty space, per-sample, pretraining views only) teaches the encoder to down-weight isolated sparse returns rather than destroying textures (the aug never touches occupied voxels).
3. **The fog decode-transfer failure is confirmed condition-wide**: fog proto mIoU flat (9.2% vs plain 10.1%) despite the LP at 57%, and fog BinCos 0.046 is the worst across all 8 conditions (next worst: incomplete_echo 0.208). The 10kD sign-space decode still cannot exploit the healed continuous representation; the plain 128D linear head can (57% LP).
4. **Fog misclassification is now maximally polarized**: ArtFrac 0.830 (vs 0.487 plain): the vast majority of fog errors are confident artifacts, and nearly all non-artifact errors (8812/9758 ≈ 90%) are oracle-recoverable. The oracle gate's 67.3% mIoU is consistent: it can find nearly everything; label-free signals still cannot.

### Next Steps

1. **Commit the medium additive run** (26 ep): decides whether capacity rescues the 6 lost conditions. Morning readout: fog LP/mIoU, BinCos, and the 6-condition deltas vs the plain-medium table.
2. **If the trade persists at medium scale**: the lever is the volumetric injection itself (density 0.05 / per-sample), because the "balanced mix" is already largely in place: beam-drop (50% of scan lines) and 20% density subsampling are in the base `get_augmented_view` for *all* methods. A tuned mix (e.g., lower injection density, per-sample injection probability, or injecting into occupied-voxel neighborhoods to mimic snow/wet_ground rather than empty-space-only fog) is the regime-side fix, informed by the clean-control mIoU: if clean drops at medium scale too, the regime distorts the clean manifold and needs rebalancing, not just per-condition scheduling.

---

## Phase 24.6: The Medium Additive Run and Full Autopsy: the Regime Is Closed, the Collapse Is Regimen-Invariant

The decisive medium-scale test (26 ep, ~10h) completed; full 8-condition autopsy run on the checkpoint. The headroom's fog numbers (53.2% acc / 5.2% mIoU) are a biased subset (`fog_feats[:50000]` majority-class slice); the autopsy (full sample) is the decision metric.

### The same-capacity comparison (medium additive vs plain medium)

| Condition | Plain acc/mIoU | Additive acc/mIoU | LP | BinCos |
| :--- | :--- | :--- | :--- | :--- |
| **fog** | 26.4 / **10.1** | 40.4 / **8.4** | 30.6 (vs 36.3) | 0.076 |
| crosstalk | 33.5 / 12.0 | 35.3 / 10.1 | 35.3 (flat) | 0.124 |
| snow | 66.6 / 39.4 | 60.1 / 35.5 | 70.8 | 0.293 |
| wet_ground | 68.8 / 49.0 | 66.7 / 45.4 | 76.2 | 0.358 |
| incomplete_echo | 78.8 / 41.2 | 77.1 / 40.1 | 84.3 | 0.269 |
| beam_missing | 77.2 / 53.7 | 75.1 / 52.0 | 83.2 | 0.289 |
| motion_blur | 73.4 / 44.3 | 71.9 / 43.7 | 79.2 | 0.289 |
| cross_sensor | 68.9 / 41.5 | 65.4 / 39.7 | 74.3 | 0.311 |
| clean control | 82.7 / 49.6 | 81.4 / 48.6 | 91.4 (fit) | n/a |

### The Findings

1. **The additive regimen is closed.** At equal capacity it is worse than plain `supcon_vib` on the paper metric for **every** condition: fog 8.4 (vs 10.1), crosstalk 10.1 (vs 12.0), and 1.6–4.2 pts down on the six geometric conditions. The fog acc gain (+14) is the majority-class artifact (the exact false-high the mIoU columns catch).
2. **The micro-scale healing did not scale.** Fog LP 30.6% is *below* plain (36.3%), far below micro-additive's 57.0%; the poison band is back (91.5% of fog points in norm ≥ 4, matching plain's 88%); BinCos 0.076 (the 10kD fog means remain near-orthogonal to clean). Convergence did not fix the 128D→10kD transfer; it reverted the micro's norm healing.
3. **The class-conditional collapse is regimen-invariant.** Fog per-class IoU: Terrain 0.51, Truck 0.079, Vegetation 0.061 survive; Building 0.012, Road 0.007, Other-ground 0.001, Traffic-sign 0.0005, Bicycle 0.0 dead. Identical casualty list to the plain encoder (Phase 24.2). The encoder family (`supcon_vib` ± additive) cannot make the collapsing classes separable under fog.
4. **Clean control held** (81.4/48.6 vs 82.7/49.6), so the trade is condition-specific, not clean-manifold distortion (answers the Phase 24.5 watchpoint in the affirmative direction).
5. **Alignment probe reproduces Phase 24.1** with slightly larger fog/crosstalk gains: fog 8.4→11.7 (+3.3), crosstalk 10.1→15.7 (+5.6), wet_ground −6.2. Still far below the 20 mIoU target; wet-ground caution stands.
6. **Fog errors became less artifact-like** (ArtFrac 0.375 vs plain 0.487) and mostly recoverable in principle (3681/4986 ≈ 74% of non-artifacts), yet the decoder and label-free gates still cannot reach them; the same access problem as before.

### Next Steps

1. **The encoder-pretraining family is exhausted for fog at this stage.** The remaining lever that has *not* been tried is the **learned loss-estimator head** (Phase 24.4 #3): a small head trained on clean/self-supervised signal to predict cos-to-true at test time, targeting the label-free gate gap (11% label-free vs 55% oracle on plain features). This is decoder-side but operates where all prior decode levers failed, because it estimates the *perceptron loss* the gates need.
2. **Crosstalk is the remaining paper-worthy win**: plain encoder + margin gate hits 23.1% mIoU @ 52% retention label-free (Phase 23); alignment adds +3.1. Worth consolidating for the paper if fog stays pinned.
3. **Fog options ranked for a future encoder change**: per-class objectives on the collapsing classes (Road/Building/Other-ground/Traffic-sign/Bicycle) as the concrete target list, and testing the volumetric injection pattern (near-occupied-voxel noise) only if a regime-side retry is ever warranted.
3. **The learned loss-estimator head** (Phase 24.4 #3) remains the standing plan for the label-free gate gap (11% → 67%), independent of the regime decision.

---

## Phase 24.7: The Prototype-Rebalancing Test: Flat Even with Oracle Selection

The "balancer" hypothesis for the label-free gate (select *which points recompute the prototypes*, like buffer selection but artifact-filtered, rather than decode-side removal) implemented as `--rebalance` in `oracle_gating_eval.py`: sweep the gate grid per condition; per config, recompute each class's 10kD prototype as the sign-mean of the selected points (keeping the clean prototype for classes with < 50 selected); evaluate **full-scene** mIoU. The oracle-loss selector is swept as the upper bound. All 8 conditions on the plain medium encoder.

### Full-scene mIoU (zero-shot vs after rebalancing)

| Condition | Zero-shot | Best label-free @ selection | Oracle-loss @ selection |
| :--- | :--- | :--- | :--- |
| **fog** | 9.4% | 9.6% @ 40% | 9.4% @ 64% |
| **crosstalk** | 10.7% | 10.8% @ 62% | 11.4% @ 35% |
| snow | 37.9% | 38.3% @ 61% | 38.3% @ 68% |
| wet_ground | 47.5% | 47.6% @ 79% | 47.6% @ 70% |
| incomplete_echo | 39.3% | 39.6% @ 86% | 39.5% @ 80% |
| beam_missing | 51.4% | 52.0% @ 72% | 51.2% @ 86% |
| motion_blur | 42.8% | 44.3% @ 70% | 42.6% @ 75% |
| cross_sensor | 39.4% | 40.1% @ 49% | 39.2% @ 74% |

### The Findings

1. **Null result across the board.** Full-scene mIoU moves ≤ 1.5 pts on every condition (fog +0.2, crosstalk +0.1). Recomputing prototypes from any gate-selected subset of the corrupted data does not beat the frozen clean prototypes.
2. **The oracle-loss bound is equally flat** (fog 9.4%, crosstalk 11.4%). This is structural, not a gate-precision limitation: even a *perfectly selected* corrupted subset cannot re-estimate prototypes better than the clean ones, because the selected points carry the same class-conditional collapse as the full scene; re-centering on them re-embeds the contamination.
3. **Confirms the standing conclusion in a new, label-free form**: frozen clean prototypes remain the best mIoU decoder on every condition (Phase 21), and decode-side prototype movement never helps (Phase 22); now shown for prototype recomputation driven by the label-free gate, not just oracle buffer-selection retraining.
4. **Closes the "balancer" role for the gate.** The label-free gate's only demonstrated value is decode-side selective prediction (retained-subset mIoU, works on 7/8 conditions). Its update-side role is a dead end, now measured rather than assumed.

### Next Steps

1. **README**: keep the decode-side gating table as the operative result; add the one-line note that update-side prototype rebalancing was tested label-free *and* with oracle selection and is flat; preempts the natural "why not recompute prototypes from confident points?" question.
2. **Gate role is settled as selective/deferral only**; fog remains pinned at ~10% mIoU, with the fix still representation-side (per-class objectives on the collapsing classes, or the learned loss-estimator head for the label-free access gap).

---

## Phase 24.8: The Source-Prior Correction Test: Flat to Negative on Every Condition

Decision-level prior correction (README Pillar 3, sec 5.2): `score(q, c) = kappa·cos(q, P_c) + tau·log pi_c`, prediction-only (never in the gate or the updates), full-scene acc + mIoU, τ = −1.0 reference config with a κ sweep (5/10/20/50/100) plus τ = −0.5/κ=10 and τ = −2/κ=20 spot checks. π_c from clean class frequencies.

### Full-scene mIoU vs the prior config

| Condition | Zero-shot | Best prior config | Delta |
| :--- | :--- | :--- | :--- |
| crosstalk | 12.0% | 12.9% @ κ=50 | +0.9 |
| fog | 10.1% | 9.7% @ κ=100 | −0.4 |
| snow | 39.4% | 38.5% @ κ=100 | −0.9 |
| wet_ground | 49.0% | 48.3% @ κ=100 | −0.7 |
| incomplete_echo | 41.2% | 40.5% @ κ=100 | −0.7 |
| beam_missing | 53.7% | 52.6% @ κ=100 | −1.1 |
| motion_blur | 44.3% | 43.6% @ κ=100 | −0.7 |
| cross_sensor | 41.5% | 39.6% @ κ=100 | −1.9 |

### The Findings

1. **Null to negative on every condition.** No prior config beats zero-shot on any condition; the best is crosstalk +0.9 mIoU @ κ=50, which is noise level. Strong priors (κ ≤ 20) degrade both acc and mIoU substantially (the boundary translation `(tau/kappa)·log(pi_b/pi_a)` over-corrects the rare classes).
2. **Bug caught first**: the initial sweep used κ=1, so `log(1/π)≈7` for a rare class swamped the ~0.05 top-2 cosine margin and collapsed every decode onto a single class (acc/mIoU exactly 0). The README's boundary-translation form makes the strength `(τ/κ)`; κ must scale the cosine term. The fixed sweep does not collapse.
3. **Does not reproduce the old-project claims** (Wet Ground +11.7, Echo gains): the frozen clean prototypes on this encoder are already the best decoder on every condition (Phase 21), so boundary translation only takes away; the old gains came from a different backbone and κ calibration.
4. **Consequence**: full-scene crosstalk stays pinned at ~12% without point removal. Prior correction is closed as a lever; the only demonstrated crosstalk win remains the label-free decode-side gate (23.1% @ 51.6% retention).

### Next Steps

1. **Prior correction: closed** (remove from the candidate list).
2. **Crosstalk full-scene gains remain open**: the remaining untried levers are the per-class objective on the encoder side and the learned loss-estimator head (which targets the label-free *access* gap, not full-scene decode).

---

## Phase 24.9: The TTA Battery and Prototype-Oracle Bounds: No Self-Supervised TTA Helps; the Full-Label Oracle Reopens, Pool-Size-Sensitive

`--tta_oracle` on the plain medium encoder: TTA battery (self-supervised: naive EMA, soft-dual-weight EMA, BN-stat alignment) plus prototype-oracle bounds (true labels: full-label prototypes from the corrupted pool, and artifact-free oracle prototypes in 10 filter configs). All full-scene acc + mIoU, shared 200k-pool / 100k-val seeded split.

### TTA battery (full-scene mIoU vs zero-shot)

| Condition | Zero-shot | naive EMA | SDW | BN-align |
| :--- | :--- | :--- | :--- | :--- |
| fog | 10.1% | 9.3% | 9.4% | 10.7% |
| crosstalk | 12.0% | 10.7% | 10.6% | 15.1% |
| snow | 39.4% | 38.1% | 37.9% | 39.9% |
| wet_ground | 49.0% | 47.7% | 47.4% | 40.5% |
| incomplete_echo | 41.2% | 39.4% | 39.2% | 40.6% |
| beam_missing | 53.7% | 51.6% | 51.4% | 53.6% |
| motion_blur | 44.3% | 43.0% | 42.8% | 44.8% |
| cross_sensor | 41.5% | 39.7% | 39.4% | 42.6% |

### Prototype-oracle bounds (full-scene mIoU)

| Condition | Zero-shot | Full-label oracle | Best artifact-free |
| :--- | :--- | :--- | :--- |
| fog | 10.1% | **16.3%** | 16.1% (margin≥0.02, 91% kept) |
| crosstalk | 12.0% | **26.2%** | 26.1% (margin≥0.02, 90% kept) |
| snow | 39.4% | 40.7% | 41.0% (loss≤0.15, 96% kept) |
| wet_ground | 49.0% | 51.4% | 51.3% (loss≤0.15, 96% kept) |
| incomplete_echo | 41.2% | 41.4% | 41.5% |
| beam_missing | 53.7% | 54.0% | 54.0% |
| motion_blur | 44.3% | 44.7% | 44.6% |
| cross_sensor | 41.5% | 43.5% | 43.3% |

### The Findings

1. **No self-supervised TTA helps any condition.** EMA/SDW (re-estimating prototypes from pseudo-labeled corrupted points) lose 0.7–2.1 mIoU everywhere; BN-stat alignment is flat except the known exceptions, and it reproduces the Phase 24.1 numbers exactly (crosstalk +3.1, wet_ground −8.5), which validates the harness. Full-scene crosstalk stays ~12% without labels.
2. **The full-label oracle beats zero-shot on fog and crosstalk for the first time** (fog 10.1 → 16.3, crosstalk 12.0 → 26.2), with point accuracy roughly doubling (fog 26% → 51%, crosstalk 34% → 49%). Re-estimated true-labeled prototypes genuinely capture the fog/crosstalk feature shift.
3. **Artifact-free does NOT beat full-label.** The best artifact-free configs (margin≥0.02, which keeps ~90% of points) are statistically identical to full-label; the aggressive artifact filters (loss≤0.02, norm<4) are *worse*. Excluding confident hallucinations from the prototype estimate does not help; the artifacts are not the limiter, the labels are.
4. **Critical open discrepancy**: this 200k-pool full-label result (fog 16.3) **contradicts the Phase 21 1M-pool ladder oracle (fog 4.9 mIoU)** on the same val subset and same `weighted_mean_update` operator. The only difference is pool size (200k vs 1M), so the full-label oracle is *pool-size-sensitive*: either the 1M pool's estimate is poisoned by the massive majority-class mass in the poison band, or the 200k estimate is underpowered. The "does prototype adaptation help" question is therefore REOPENED and cannot be settled until a controlled pool-size sweep reconciles the two.

### Next Steps

1. **Reconcile the oracle discrepancy with a pool-size sweep** (200k / 500k / 1M) on fog and crosstalk using the ladder protocol; if the 1M-pool oracle truly crashes fog to ~5%, the full-label "win" is a pool artifact and the Phase 21 conclusion stands; if it holds above zero-shot, prototype adaptation deserves a real (label-free) attempt.
2. **TTA is closed as a lever** (no self-supervised variant beats zero-shot full-scene).
3. **Artifact-free prototype estimation is closed as a lever** (does not beat full-label; the artifacts are not the estimate's limiter).

### Deferred: Batch-Size Scaling (investigate only if results demand it)

The GPU runs at 100% util with ~65GB memory headroom, so larger batches are *possible* (batch 2–4, one-line Parser change, loop already batch-agnostic). This is parked for three reasons:

1. **Comparability**: every data point in the KL × steps matrix (9.5k/12.7k/83k) was trained at batch 1, a batch change creates a new training regime and breaks cross-run attribution.
2. **Compute-bound ceiling**: at 100% util, the expected gain is modest (1.2–1.5× from amortized Python/launch overhead, not linear).
3. **Semantics change**: SupCon positives would span images within a batch rather than only the clean↔augmented pair, and the cosine scheduler (defined in steps) would traverse its cycle at half the per-epoch depth, a different configuration, not a free speedup.

**Revisit only if**: a probe or the overnight run shows the training wall-clock (not the metrics) is the blocker, e.g., needing many more epochs of convergence than the compute budget allows. The decision rule: batch scaling earns investigation only if it is the *clear* bottleneck, not because headroom exists. (Timing test if revisited: 3 epochs at batch 2 in a scratch out_dir vs the 2.56 it/s baseline.)

---

### Current Next Steps (consolidated)

1. **The learned loss-estimator head** (Phases 24.4 #3, 24.6 #1): a small head trained on clean/self-supervised signal to predict cos-to-true at test time, targeting the label-free gate gap on fog (11% label-free vs 55% oracle on the plain features). Decoder-side, but it estimates the perceptron loss the gates need.
2. **Per-class weighting for the fog casualties** (classes 2, 7, 13, 14, 15) in the pretraining objective: the concrete target list from Phase 24.2; the collapse is class-conditional and regimen-invariant (Phase 24.6), so the objective must target these classes specifically.
3. **Crosstalk stack to finish the 20-target**: the label-free margin gate (23.1% mIoU @ 52% retention) combined with the BN-statistic alignment (+3.1 raw mIoU): evaluate the combined decode. (Prior correction is closed as a lever, Phase 24.8.)
4. **Intra-class balance checks** (open thread): per-class hard selection behaved differently from global selection (Phase 22.1), so the intra-class balance question is not yet settled; the subcluster ledger and per-class buffer variants remain to be evaluated.
5. **Optional decode-thread closure**: `--update_strength 1 --buffer_frac 0.20` on the oracle retrain: low prior that it stabilizes the loop; cheap, closes Phase 22.
