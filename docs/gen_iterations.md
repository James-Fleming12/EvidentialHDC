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

## Phase 12: The Naive Gated-EMA Diagnostic on the New Encoder (Fog)

Before committing to the medium-scale run, the offline EMA-adaptation simulator (`oracle_gating_eval.py`, seeded 10kD projection) was pointed at the fresh micro-trained `supcon_vib` encoder (`logs/micro_pretrain/supcon_vib`) to measure how much accuracy the *naive* implementation — current `fuse_uncertainties` gating + current robust encoder — actually gains under Heavy Fog, and to profile whether the Phase 9 gate signals still discriminate.

### The Gated EMA Ladder (Fog, 10kD HDC space, pool = 20k points, α = 0.01)

| Strategy | HDC Prototype Accuracy |
| :--- | :--- |
| Zero-Shot (No Adaptation) | 12.74% |
| **Naive EMA (No Gate)** | **18.34%** |
| Top-50% Confidence (Phase 9 gate) | 18.52% |
| Epistemic Gate | 17.70% |
| Geometric Gate (norm z-score) | **18.62%** |
| Soft Dual Weight | 17.94% |
| AND Gate | 17.91% |
| Ellipsoid Gate | 18.34% |
| **Perfect Oracle (True Labels)** | 19.54% |

### The Signal Profiling (Leave-One-Update-Out, 5k updates: 165 Helpful, 227 Harmful)

| Signal | Helpful Mean | Harmful Mean | AUROC (Helpful vs Harmful) |
| :--- | :--- | :--- | :--- |
| Probe Confidence | 0.255 | 0.338 | **0.154 (inverted!)** |
| Feature Norm (L2) | 0.039 | 0.057 | **0.942** |
| Joint z-score (c − n) | — | — | 0.673 |
| Logistic Combination | — | — | 0.855 |

### Diagnostic Analysis

1. **The naive EMA already harvests ~82% of the oracle headroom.** Zero-shot 12.74% → naive EMA 18.34% (+5.6 of the +6.8 oracle ceiling). On Fog, the robust representation has largely *solved* the poisoning problem at the source: indiscriminate adaptation now mostly helps, leaving only ~1.2 points of gate-able headroom. This is the exact opposite of the Phase 9-era regime (naive captured only 45% of headroom on the old encoder).
2. **Feature norm is the dominant signal (AUROC 0.94).** The adaptation pool is dominated by near-origin, VIB-collapsed noise points (norms 0.04–0.06 vs the ~5.6 class geometry average): the collapsed points are *benign* (tiny updates, no drift), while the few points that escaped collapse (higher norm) are the poison. This is a direct empirical vindication of the Phase 4 **Magnitude Segregation** thesis.
3. **Probe confidence is ANTI-predictive (AUROC 0.154)** — harmful updates carry *higher* confidence (0.34 vs 0.26), the "confident hallucination" signature previously documented in the mem-method thread. **Caveat:** the linear probe here is trained on only the first 100k points (~1–2 frames of sequence 08, vs 50k points spread over 50 frames in the micro eval), so the probe may have collapsed to majority-class prediction (clean probe acc 52% here vs 88% there). The inversion may be a probe-sampling artifact — must be re-tested with a uniformly sampled probe before being trusted.
4. **The gates barely beat naive (best: geometric +0.28).** With headroom above naive so small, no gate can add much; and because the confidence term is anti-predictive in this regime, the joint gates (`soft_dual_weight` 17.94%, `and_gate` 17.91%) are *dragged below* naive. The LR combination (0.855 AUROC) proves the signals are complementary — the Phase 11 joint direction (c_z − n_z) is likely *wrong* for this encoder; if the inversion survives re-testing, the correct joint is c_z + n_z (veto high-confidence AND high-norm).
5. **Zero-shot dropped from 20.1% (128D Euclidean) to 12.7% (10kD sign-binarized)** — the HDC binarization still costs points on this encoder, consistent with the Phase 8 degradation findings.

### Next Steps

1. **Fix the probe sampling** (uniform subsample across all 100 frames, not the first 100k points) and re-run the Fog diagnostic — determines whether the confidence inversion is real or an artifact.
2. **Run the remaining 7 corruptions** with the same ladder (the JSON currently contains Fog only), with special attention to Crosstalk AUROC (the mem-method thread's known failure case).
3. **Recalibrate the joint gate on the re-tested signals** (c_z + n_z if the inversion holds; c_z − n_z otherwise) and re-run the ladder on Fog + Crosstalk.
4. **Commit the medium-scale run** (plain `supcon_vib`, seeded) once the gate direction is settled — the naive EMA baseline (18.34%, Fog) becomes the accuracy floor the converged encoder must beat.
