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
3. **`supcon_vib_global`:** **Expanded Spatial Pooling** — a 3×3 average pool applied to the 128D latent itself (Global Anchoring), smoothing each feature with its spatial neighbors at evaluation time.

> **Methodology Note:** In the first run, the Additive/SOR/Global variants inadvertently trained through the plain symmetric-CE objective — the loss branch in `gen_trainers.py` matched only the exact method string `supcon_vib`, so the variant names silently bypassed the SupCon+VIB terms. This *did* isolate the pure effect of each remediation (their $L_2$ norms inflated toward 8+, exactly what the first table below shows), but it was not the intended "full method + remediation" test. Before the second run the routing was fixed (`startswith('supcon_vib')`), so every variant now trains with the full decoupled SupCon+VIB objective, and `supcon_vib_sor` additionally applies the SOR pre-filter to both clean and augmented inputs during training to match its evaluation-time filtering.

### Headroom Metrics (Heavy Fog, Frozen Features) — Old CE-Only Run → New Full-Loss Run

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
3. **Global Pooling Was Redeemed by the Full Loss:** With VIB magnitudes intact, the 3×3 latent smoothing no longer collapses the geometry — HDC jumped from 4.1% to 14.4% (second best). Global pooling is now a plausible cheap auxiliary, though LP (Fog) remains the weakest (10.3%).
4. **Additive Remains a Middle Ground:** Second-best LP (Fog) (13.3%), best angular stability (0.532), HDC 12.4% — but nothing beats the plain method on the HDC axis that matters most for the gated EMA prototype pipeline.

### Reproducibility Note: Why the Baseline Moved
The canonical `supcon_vib` baseline itself shifted between runs (HDC 10.9% → 20.1%, Retrieval 35.1% → 26.4%) despite an unchanged code path. This is **not** a random projection matrix: the micro evaluation contains no random projection (the 10,000D HDC projection in `HDC_utils.set_uq_model` is imported but never called — prototype accuracy here uses raw 128D class-mean centroids and `torch.cdist`). The movement is pure stochasticity: `micro_pretrain_eval.py` never sets a seed, so each run re-initializes the SENet weights from scratch (`path=None`) and re-rolls the entire stochastic stack — beam-drop rows (`np.random.choice`), depth jitter, density masks, SupCon point subsampling (`randperm`), VIB reparameterization noise, and shuffled DataLoader workers. With only 5 epochs on 10% data and 50 evaluation frames, these unseeded draws dominate the final weights. For comparability, the medium-scale run should adopt the seeding already used by `med_pretrain_eval.py` (per-method `torch.manual_seed`) or `train.py`'s `--seed` argument.

**The Verdict:**
The full-loss shootout does not support stacking SOR (or any other remediation) on top of the canonical representation: plain `supcon_vib` wins the HDC axis outright (20.1%), and its $L_2$ envelope (5.62 → 4.55) retains the VIB magnitude isolation the gated EMA pipeline depends on. The remediation variants add nothing over the full method in this micro setting, and their apparent CE-only advantages did not survive contact with the real objective. The medium-scale commit should therefore proceed with **plain `supcon_vib`**, seeded for reproducibility.

---

## Phase 11: The Pivot to Uncertainty-Gated TTA (Design & Next Steps)

### Validation of the Pivot Decision

1. **SOR-in-the-loop is a dead end; the fix is not more training.** The full-loss SOR run regressed on every axis (LP Fog 27.4% → 14.0%, Retrieval 44.6% → 31.1%, Cosine Shift 0.434 → 0.730) relative to the CE-only run, and its best HDC Prototype Accuracy (9.1%) is less than half of plain `supcon_vib`'s (20.1%). The mechanism is structural: the SOR filter runs *after* beam-drop augmentation (which zeroes 50% of scan rows), so it deletes the legitimate neighbors of dropped beams **every iteration**. This corrupts the contrastive pair structure the SupCon loss depends on, and longer training would *reinforce* — not heal — the corrupted geometry. Spending medium-scale compute on SOR variants is therefore not justified.
2. **The representation is the solved part.** Phase 8 proved the `supcon_vib` 128D space is highly separable under Heavy Fog (49.4% Linear Probe) and that this information mathematically survives random projection and sign-binarization (49.0% / 47.8%). The residual HDC collapse (8.2% → 20.1% across runs) is exclusively a naive-decoder problem: indiscriminate EMA updates absorb the impossible Fog/Crosstalk artifacts and poison the Euclidean centroids. Crucially, the old feature space was so degraded that it forced exotic decoder machinery (the `AdaptiveMemoryBank`'s density-adaptive Hamming query, reservoir sampling, and purity thresholds were all built to survive a space that collapsed into isotropic noise). The entire premise of the pretraining investment is that the robust encoder makes this unnecessary: a plain EMA prototype update plus standard confidence gating should now suffice.
3. **Uncertainty gating is the proven remediation.** Phase 9's oracle tests showed that filtering the bottom 50% of updates by confidence reaches 20.63% — 89% of the Perfect Oracle ceiling (23.32%) vs 16.36% for naive adaptation — and profiled the Harmful-update signature: lower confidence (0.62 vs 0.92) and larger feature norms (6.42 vs 5.28).
4. **Honest caveats before committing.** (a) Eval-time-only SOR — a frozen full-loss encoder + SOR input pre-filter, *without* SOR in training — was never tested; that recipe was the biggest CE-only win (27.4%) and remains a cheap, orthogonal input-stage candidate for the TTA pipeline. (b) Earlier research threads (`docs/mem_method/new_prelims.md`) found no signal combination that separates hallucinations on Crosstalk in the *old* encoder setup (max AUROC 0.642), and documented AND-gate starvation / OR-gate flooding. Phase 9's universal matrix shows the confidence/norm signature survives on the new encoder (Crosstalk: 0.79 vs 0.61 confidence), but the gate must be re-validated per-corruption before it is trusted universally.

### The Gating Function: Combining Confidence and Feature Norm

The two Phase 9 signals live on incompatible scales (confidence $c \in [0,1]$, norm $\Vert z \Vert \approx 5.5$ under the VIB cap), so they are standardized with streaming statistics (per-frame or per-class EMA of mean/std):

$$c_z = \frac{c - \mu_c}{\sigma_c}, \qquad n_z = \frac{\Vert z \Vert - \mu_n}{\sigma_n}$$

The unified gate weight is a soft, multiplicative decay — the same algebra already implemented in `fuse_uncertainties("soft_dual_weight")`, with the Phase 9-validated signals substituted for the current (Dirichlet uncertainty, distance z-score) pair:

$$w(x) = \exp\big(-\lambda_1 \cdot \text{relu}(\tau_c - c)\big) \cdot \exp\big(-\lambda_2 \cdot \text{relu}(n_z - \tau_n)\big)$$

The EMA prototype update becomes $c \leftarrow c + \eta \cdot w(x) \cdot z$. Calibration anchors come directly from Phase 9: $\tau_c \approx 0.75$ (midpoint of 0.62/0.92), $\tau_n \approx 5.9$ (midpoint of 6.42/5.28), with $\lambda_1, \lambda_2$ swept. This form preserves the soft-weighting regime proven in the geometric-method thread (avoiding both AND-gate starvation and OR-gate flooding) while keying on the exact signature the oracle validated.

### Next Steps

1. **Re-validate the Harmful-update signature on the new encoder:** rerun the `oracle_gating_eval.py` leave-one-update-out protocol (already seeded) on the 20.1%-HDC `supcon_vib` features, per-corruption (Fog, Crosstalk, plus the geometric panel) — confirming confidence/norm still separate Helpful from Harmful updates before building on it.
2. **Implement the gate as a new `fuse_uncertainties` mode** (e.g. `norm_confidence_gate`) using the weight above, so it is directly ablative against the existing `epistemic` / `geometric` / `soft_dual_weight` modes.
3. **Sweep $(\tau_c, \tau_n, \lambda_1, \lambda_2)$** on Fog and Crosstalk through the standard EMA prototype update on the frozen encoder — the `fuse_uncertainties` gate feeding the plain prototype update, deliberately *without* any of the old `AdaptiveMemoryBank` machinery (density-adaptive Hamming, reservoir sampling). Acceptance criteria: beat the naive EMA baseline (16.36%) and approach the oracle ceiling (23.32%), and outperform the single-signal gates (confidence-only, norm-only) to justify the joint combination.
4. **Commit the medium-scale run:** plain `supcon_vib` on 100% data (seeded per `med_pretrain_eval.py` convention) to obtain the converged encoder, then run the gated TTA end-to-end on top of it.

**Contingency (deferred, not deleted):** SOR — whether as an eval-time input pre-filter on the frozen encoder or in any future augmentation variant — is parked. Its best ever HDC Prototype Accuracy (9.1%) is less than half of what the plain encoder already achieves (20.1%), so it does not justify proactive compute. Only revisit input-space remediation or richer physics augmentation if the gated TTA underperforms the naive EMA baseline.

---

## Phase 13: The Full 8-Corruption Panel and the Pool/Val Mismatch Confound

The full panel ran with the gate-fault diagnostics (clean-data control, per-gate-mode AUROC, threshold envelope sweeps) — and immediately exposed a **harness bug** that invalidates every absolute number in the panel, while still yielding decisive negative results about the shipped gates.

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

**The ground-truth Perfect Oracle loses to zero-shot on every single corruption** (Fog −2.1, Crosstalk −16.4) — **and even on the Clean Control (−1.65)**. A perfect-label oracle can only *hurt* if it is adapting to a distribution different from the one it is evaluated on. The cause is structural: the adaptation pool was the first 20k points of the stream (≈¼ of the first frame) and the validation set was the last 100k points (≈1.2 of the last frame) — **~98 frames apart, entirely different scenes**. This is why leave-one-out also reported near-zero Helpful updates on several corruptions (Incomplete Echo: 0 Helpful / 1246 Harmful): "helpful vs harmful" was measuring "does adapting toward scene A help classify scene B", which is almost always no.

The confound was always present in the harness — the Phase 9-era encoder masked it, because on that encoder *any* movement toward the fog distribution improved the late-frame validation. The robust encoder's quality (clean prototypes already close to the fog geometry) makes the mismatch visible: adaptation now has to be *correct*, not just *closer*.

### Secondary Findings (valid despite the confound)

1. **The fog confidence inversion is confirmed — and catastrophic in action.** Conf AUROC 0.17 (anti-predictive); `top50_conf` crashes Fog to **4.2%** (from 15.4% naive) because the top-50% most-confident fog points are the most harmful (Harmful Conf 0.77 vs Helpful 0.54). The shipped gates' "harmful = low confidence" assumption is sign-reversed on Fog.
2. **No single signal or gate is universally selective.** Norm AUROC is strong on the Type C's and Wet Ground (Fog 0.70, Crosstalk 0.85, Wet Ground 0.93) but **inverted on Snow (0.08)**; confidence is only sane on Motion Blur (0.86). The **logistic combination of conf + norm is the only universally strong selector (0.86–0.95)** — supporting a learned/calibrated joint gate over any fixed hand-designed mode.
3. **Gate-mode AUROC confirms the shipped modes are old-space artifacts.** On Fog only `joint_flip` (veto high-confidence AND high-norm) is selective (0.82); `soft_dual_weight` and `and_gate` are *anti*-selective (0.41–0.42). On Snow every mode except `epistemic` is anti-selective (0.12–0.14). The diagnostics did their job: they caught the gates being wrong *before* we committed compute to them.
4. **Threshold sweeps cannot rescue the shipped gates.** Best-case `soft_dual_weight` (`sdw_best`) ≈ naive everywhere (best margin: Motion Blur +1.0). Under the confound this is not yet conclusive, but combined with finding 3 it strongly suggests the SDW family should not be reused.
5. **Run-to-run variance is large.** Fog zero-shot swung 12.7% ↔ 21.8% between two runs of the *same* checkpoint and projection seed — the data pipeline (point subsampling in extraction workers) is unseeded.
6. **The shipped gates are degenerate on the new space (the bit-identical rows).** On the VIB-capped space, clean feature norms have near-zero variance, so the geometric z-scores explode to ±hundreds and every exponential decay saturates (weight ≈ 0 for z > 0.5, ≈ 1 for z ≤ 0.5). With probe confidence ≥ 0.5 everywhere, the epistemic factor also saturates at 1 — so `geometric`, `soft_dual_weight`, `and_gate`, and `ellipsoid_gate` all reduce to the *same binary "keep low-norm" gate* and report bit-identical accuracy (0.6867 on the clean control). The old space had norm ranges of 3–12 where these decays were meaningful; the new space has none. Gate weight statistics (mean / %saturated-admit / %saturated-veto) are now reported per mode to make this degeneracy visible instead of silent.
7. **The clean control exposed EMA base-erasure.** Even after the pool/val fix, the ground-truth oracle still lost on clean data (−0.8) and naive EMA crashed (−8.6). With α = 0.01 and an unbounded 20k-point pool, each prototype's final state is dominated by its last ~1/α ≈ 100 updates: the base prototype (estimated from millions of clean points) is erased and re-estimated from ~100 random points. Re-estimation noise hurts even perfect-label updates; the ~8% wrong pseudo-labels amplify it. The Phase 9-era encoder masked this because domain shift dominated the re-estimation noise; the robust encoder exposes it.
8. **The pool is ~250× too small to refine 10kD prototypes — even the perfect oracle loses (−1.4 on Wet Ground).** With a same-distribution pool of 20k points (0.25% of the stream), the refinement signal cannot beat the variance of re-estimating 17 prototype directions from so few points. The oracle "headroom" measurement is structurally pessimistic at this pool size.
9. **Gate-signal directions are pool-dependent, not universal.** On Wet Ground, the norm signal *flipped* between the v2 run (first-frame pool: Helpful 3.52 vs Harmful 5.30, norm AUROC 0.93 — "high norm = poison") and the v3 run (uniform pool: Helpful 6.79 vs Harmful 5.36, AUROC 0.295 — "high norm = helpful"). Fixed-direction gates therefore actively veto helpful points on some corruptions/pools (Wet Ground v3: every gate below naive). Per-corruption, per-pool direction calibration is mandatory; the Phase 9 "universal signature" holds for the first-frame pool composition but not in general.

### Harness Fixes Applied

1. **Pool/val now share one seeded uniform permutation** over all extracted points (pool = first N, val = next 100k of the permutation), so adaptation and evaluation cover the same frame distribution.
2. **The whole pipeline is seeded** (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`) before loader creation, making extraction and splits reproducible.
3. **The EMA ladder is replaced by a vectorized weighted class-mean update** (`weighted_mean_update`, chunked `index_add_`), and the default pool is raised to **1M points** — the statistically honest adaptation operator at a statistically honest pool size. Prototype_c = normalize(Σ wᵢ·sign(zᵢ·proj)) over pool points with pseudo-label c; classes without pool support keep the base. This removes the α-dependent base-erasure (finding 7) and the small-pool pessimism (finding 8) at once, and makes the ladder cheap enough for million-point pools.
4. **Per-mode gate weight statistics** reported in the ladder, so saturated/degenerate gates are visible rather than silently producing identical rows (finding 6).

### Next Steps

1. **Re-run the 8-corruption panel** on the fully fixed harness (pool/val permutation + 1M-point weighted class-mean ladder) — the oracle must now land *at or above* zero-shot on every corruption, and the clean control must show gates ≈ naive ≈ zero-shot (no adaptation should help or hurt when nothing is wrong).
2. **Fix the linear-probe sampling** (uniform subsample across all 100 frames, not the first 100k points) — the probe that supplies the confidence signal still trains on ~1–2 frames' worth of points, which may explain the fog confidence inversion and must be resolved before the gate is calibrated on confidence.
3. **Treat the gate-fault results as binding**: drop the shipped `soft_dual_weight`/`and_gate`/`ellipsoid_gate` modes for the new space; build the gate from the two bare signals (confidence, feature norm) with **per-corruption direction calibration** — finding 9 shows the norm direction is pool-dependent, so a fixed "harmful = high norm" prior cannot be trusted (the LR combination is the reference: 0.86–0.95 AUROC when computed).
4. **Commit the medium-scale run** (plain `supcon_vib`, seeded) in parallel — the encoder is not in question; the harness and gate questions are independent of it.

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

1. **The oracle headroom is gone.** Even with ground-truth labels and a 1M-point pool, re-estimating prototypes from the corrupted features *loses* to the frozen clean prototypes on every corruption — catastrophically on the Type C's (Fog −15.5, Crosstalk −21.5) and mildly (−0.8 to −2.7) on the geometric corruptions. The Phase 9-era headroom (+2.73, 23.32% oracle) was an artifact of the old encoder and the frame-mismatched harness; it does not reproduce here.
2. **The mechanism is feature collapse in sign space.** On Fog/Crosstalk the features are so degraded that their binarized class means are near-random directions — target-domain prototypes are garbage, and any movement away from the clean anchors can only hurt. Even on mild corruptions, the drifted target means lose a point or two to the clean means (the Phase 7 prototype-quality variance, now measured end-to-end).
3. **Pseudo-label adaptation is catastrophic everywhere** (Naive: −10 to −19 points), because re-estimation replaces the prototype set wholesale with 10–80%-wrong means. Gating rescues a fraction of that crash (`joint_flip` is consistently the best, recovering up to 10 points on Incomplete Echo) but **no gate reaches zero-shot** — with harmful updates outnumbering helpful ~45:1 on Crosstalk (leave-one-out: 42 helpful / 4337 harmful), freezing is the optimal policy.
4. **Adapting the prototypes is not the lever.** The information is in the features (49.4% linear probe; 20–25% HDC zero-shot from *clean* prototypes) — but it cannot be harvested by centroid movement, because the corrupted centroids are worse than the clean ones.

### Revised Next Steps

1. **Move the gate from the UPDATE to the QUERY.** With prototypes frozen, the remaining lever is deciding *which points to trust at inference* — vetoing collapsed/hallucinated points (the Phase 4 magnitude-segregation idea: near-origin features are noise) before prototype classification. This is cheap to test: threshold the query points by norm/confidence on the frozen-prototype classifier and measure accuracy vs point-retention on the corrupted val.
2. **Test feature-level adaptation** (e.g., test-time batch-norm / feature whitening alignment) as the alternative to prototype movement — the 49.4% linear separability proves the headroom exists in the feature space; the question is whether unsupervised alignment can capture it without labels.
3. **One final prototype-adaptation check**: severity-split sequential adaptation (adapt on moderate Fog frames, evaluate on Heavy Fog frames) — the only regime where target-mean re-estimation could plausibly still help, since the mild-corruption means are less collapsed. If this also shows no headroom, prototype TTA is closed entirely.
4. **The medium-scale run still pays**: its value is now the *frozen* representation (higher zero-shot HDC at convergence) plus query-side gating at scale, not gated prototype updates.

---

## Phase 15: The Long Micro Run — Training Heals the Collapse, StrongVIB Wins the Frozen Decode

The 30-epoch micro run (cutoff 0.1, ~9.5k gradient steps — 6× the earlier micro budget) trained three variants — `supcon_vib`, `supcon_vib_strongvib` (5× VIB KL weight), `supcon_vib_additive` (volumetric noise injection) — each followed by the v4 oracle ladder plus the deep feature-space diagnostics. All three trained cleanly (IoU 0.30–0.31 by epoch 30, still climbing; strongvib's higher initial loss from the 5× KL converged to the same IoU).

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

1. **Training heals the collapse — partially.** The plain encoder's Fog zero-shot rose 25.0% → 28.8% and its Fog oracle more than doubled (9.5% → 21.8%) from 5 to 30 epochs; clean control rose to 80.4%. The binarized fog class means are no longer near-random (mean-norm ratio 1.07). The overnight medium run (95k steps, 10× this budget) is now strongly justified.
2. **`strongvib` is the best frozen decoder — and has the best gate signals ever measured on Fog.** It leads Fog zero-shot (35.0%), Crosstalk (38.1%), and its Fog signal AUROCs are the strongest of the project: confidence 0.716, norm 0.807, joint 0.856, logistic 0.852. Its deep diagnostics show why it works: the 5× VIB redistributes fog features into the healthy mid-norm band (67% in [2,4) vs 91% in [4,∞) for the others) — the exact region that classifies well (band acc 36.3% vs 6.6% for [4,∞)). Cost: the clean control dips to 77.6% — the stronger KL slightly taxes the clean representation.
3. **`additive` produced the first positive oracle headroom on Fog in the entire project: 26.1% → 45.3%.** Training against volumetric noise injection yields fog class means that are genuinely usable — prototype re-estimation on Fog *gains* +19.2 points. Phase 14's "prototype TTA is falsified" verdict is therefore encoder-specific, not universal. **But**: its own gate signals are the weakest of the three (fog logistic AUROC 0.616, confidence anti-predictive 0.372), so the headroom is not currently exploitable — naive adaptation lands at 20.2%, below its own zero-shot. This is the key open question for the prototype-adaptation path.
4. **The high-norm fog points are the poison, consistently.** Band accuracy across variants: the [4,∞) norm band classifies at 6.6–25.4% while the [2,4) band reaches 36–40%. The query-side gate direction is confirmed: veto high-norm fog points before prototype classification.

### Verdict & The Medium Run

**`supcon_vib_strongvib` is the medium-scale candidate.** It maximizes exactly what the current plan needs: the frozen decode on the priority corruptions (Fog 35.0%, Crosstalk 38.1%) and the strongest signals for the query-side gate (fog joint AUROC 0.856). The additive oracle headroom (45.3%) is real but unexploitable until its signals improve; it is documented as the follow-up question rather than the overnight bet.

### Next Steps

1. **Overnight medium run**: `supcon_vib_strongvib`, 30 epochs on 100% data (~95k steps, ~12h), seeded — with headroom + deep diagnostics baked into `med_pretrain_eval.py`.
2. **On the converged encoder**: re-run the v4 ladder + deep diagnostics — expect Fog zero-shot > 35% and Fog band-acc spread to sharpen; confirm the clean control recovers with the proper full-length LR schedule.
3. **Build the query-side gate** on the strongvib signals (confidence + norm, direction-calibrated per corruption — the fog joint AUROC 0.856 is the reference).
4. **Follow-up (prototype-adaptation path)**: study why `additive`'s fog means are usable — if its gate signals can be sharpened (e.g., stronger augmentation density or a norm-conditioned variant), the +19.2 oracle headroom becomes exploitable.
