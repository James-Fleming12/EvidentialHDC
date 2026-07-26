# Phase 3 Final Document Outline: Evidential HDC for Continual Online Adaptation in 3D LiDAR Perception
**Status:** Active Draft (Updated July 25, 2026)  
**Scope:** Specification and outline for the final Phase 3 publication/technical report, incorporating established empirical findings and marking upcoming experimental validations with `[TODO]`.

---

## Abstract & Executive Summary
* **The Challenge:** Real-world 3D LiDAR perception systems face continuous, sequential physical degradation (e.g., adverse weather, sensor dropout, surface reflectivity, and ego-motion distortion). Standard unsupervised test-time adaptation (TTA) methods suffer from pseudo-label confirmation bias and catastrophic forgetting under continuous domain shifts.
* **Core Contributions:**
  1. **Multi-View Epistemic Disagreement Veto (MV-2):** A lightweight Dirichlet uncertainty gating mechanism that actively rejects high-error pseudo-labels during online adaptation.
  2. **Synergistic Prior Calibration ($\tau=-1.0$):** Demonstrating that prior-calibrated evidential density stabilizes mid- and tail-class prototypes during TTA.
  3. **Dual-Uncertainty Complementarity:** Re-evaluating the union of Geometric (HDC Latent Free Energy) and Network (Dirichlet) uncertainty without representation shrinkage.
  4. **Physics-Based Plasticity (The Global Spring):** A self-regulating momentum vector that automatically thaws frozen class prototypes upon detecting domain shifts, enabling continual learning without catastrophic forgetting.

---

## 1. Theoretical Foundations & Prior Calibration
### 1.1 Unsupervised Epistemic Uncertainty via Dirichlet Density
* Formulation of hyperdimensional evidential density and Dirichlet concentration parameters for 3D point cloud segmentation.
* Derivation of epistemic uncertainty as the reciprocal of total evidential strength: $u = \frac{K}{\sum_{k} \alpha_k}$.

### 1.2 The Majority Amplifier Prior Calibration ($\tau=-1.0$ vs $\tau=0.0$)
* Formal analysis of uncalibrated ($\tau=0.0$) vs prior-calibrated ($\tau=-1.0$) initial prototype norms.
* **Empirical Validation (Snow Benchmark):**
  * Uncalibrated prior ($\tau=0.0$) adaptation achieves $0.4112$ mIoU (from $0.4078$).
  * Prior calibration ($\tau=-1.0$) elevates initial baseline mIoU to **$0.5524$**, providing a superior geometric starting point for unsupervised adaptation.

### 1.3 Mathematical Derivation of Multi-View Epistemic Consensus
* Deriving the joint evidential probability across overlapping spatial/temporal views without increasing inference latency.

---

## 2. Multi-View Active Disagreement Veto (MV-2 Architecture)
### 2.1 The Ensemble Paradox & Architecture Pruning
* Theoretical explanation of representation shrinkage observed in multi-view feature bundling and $K$-means subclustering.
* **Methodological Decision:** Permanent deprecation and removal of complex subclustering in favor of streamlined single-view feature extraction gated by multi-view epistemic disagreement vetoes.

### 2.2 Prediction-Path Consensus vs Veto-Disagree Mechanisms
* Formulation of `veto_disagree` (vetoing adaptation when view predictions diverge) and `conf_pred` (prediction-path consensus).

### 2.3 Point-Level Precision Tracking & Pseudo-Label Purity
* **Empirical Validation (Snow Benchmark Findings):**
  * **Agreeing Points:** Maintain high pseudo-label purity with **~88.0% precision** (525M points evaluated).
  * **Disagreeing Points:** Exhibit severe degradation, dropping to **~39.0% to 43.0% precision** (8M points).
  * **Tail-Class Vulnerability:** In vulnerable classes such as Class 10 (Sidewalk) and Class 3, disagreeing points collapse to an error-prone **11.9% to 21.6% precision**.
* **Conclusion:** `veto_disagree` actively filters out these 8M high-error disagreeing points, stabilizing Mid ($0.4539 \rightarrow 0.4673$) and Tail ($0.4045 \rightarrow 0.4098$) mIoU under $\tau=-1.0$ calibration.

---

## 3. Re-Evaluating Dual-Uncertainty Gating (Geometric HDC + Network Epistemic Density)
* **The Core Scientific Question:** While Phase 2 pruned complex subclustering due to representation shrinkage, this temporarily sidelined the purely geometric **HDC Latent Density (Free Energy / 128D Gaussian)** in favor of Network Epistemic Uncertainty (Dirichlet Density). Is combining Geometric HDC and Network Epistemic uncertainty fundamentally beneficial when implemented cleanly without subclustering?

### 3.1 Decoupled Dual-Uncertainty Gating (The AND vs. OR Paradox)
* **Theoretical Hypothesis:** Network Epistemic certainty (classifier confidence) and Geometric HDC certainty (latent feature clustering free energy) capture distinct error distributions.
* **`[TODO]` Experimental Validation:** Evaluate logical `OR` gating (admitting points if *either* Network certainty *or* Geometric certainty is high) vs logical `AND` gating. Determine whether Geometric HDC density rescues true-positive hard examples (such as structurally deformed road boundaries) that the Dirichlet gate incorrectly vetoes during adaptation.

### 3.2 The Cross-View Orthogonality Hypothesis
* **Theoretical Hypothesis:** In single-view TTA, Network Epistemic and Geometric HDC uncertainties are heavily correlated (e.g., both degrade similarly under static snow occlusion).
* **`[TODO]` Experimental Validation:** Compute Geometric HDC consistency *across* multi-view spatial/temporal sweeps rather than within a single static view. Validate whether cross-view geometric consensus provides an orthogonal gating signal that improves pseudo-label precision over single-view Dirichlet gating alone.

### 3.3 Class-Conditioned Dynamic Geometric Thresholding
* **Theoretical Hypothesis:** Frozen source variance thresholds create rigid hard boundaries that fail when global domain shifts alter feature scale.
* **`[TODO]` Experimental Validation:** Implement batch-adaptive geometric normalization (running batch variance) to establish soft decision boundaries, allowing the model to smoothly admit structurally deformed in-distribution points while strictly rejecting structureless out-of-distribution scatter (e.g., volumetric fog).

---

## 4. Physics-Based Plasticity & Continual Learning Dynamics
### 4.1 The Starvation-Thawing Hypothesis
* Analysis of the S2 Norm-Driven Dynamic Learning Rate schedule and the vulnerability of permanently frozen momentum vectors ($M_c$) when encountering sudden domain transitions.

### 4.2 Uncoupling the Spring from the Firing Loop
* Mathematical formulation of the unconditional global anchor spring:
  $$ M_c \leftarrow (1-k)M_c + k \cdot 1.0 $$
* **Mechanism:** Confident classes fire frequently, maintaining inflated $M_c$ (frozen stability). Upon encountering a domain shift, error-generating classes are vetoed by Dirichlet gates; as they starve, the global spring erodes $M_c$ back to $1.0$, automatically **"thawing"** the class for rapid adaptation.

### 4.3 `[TODO]` Continual Pipeline Implementation & Empirical Validation
* **`[TODO]`** Modify the core adaptation loop in `unsup_kitti-c.py` with the `--continual` flag (disabling weight resets between sequences).
* **`[TODO]`** Evaluate continuous trajectory execution across sequential domain shifts (e.g., Clean $\rightarrow$ Snow $\rightarrow$ Rain $\rightarrow$ Night).

---

## 5. Comprehensive Multi-Corruption Evaluation & Generalization
### 5.1 Benchmark Protocol Across the Diagnostic Panel
* Definition of the 5-corruption diagnostic panel covering the primary axes of LiDAR sensor degradation:
  1. *Sensor Sparsity / Beam Dropout* (`beam_missing`)
  2. *Specular Reflections / Multipath Ground Returns* (`wet_ground`)
  3. *Volumetric Scattering / Atmospheric Attenuation* (`fog`)
  4. *Temporal Dynamics / Ego-Motion Artifacts* (`motion_blur`)
  5. *Adverse Weather / Precipitation* (`snow` — completed)

### 5.2 `[TODO]` Comparative Analysis Across LiDAR Degradation Axes
* **`[TODO]`** Execute the overnight benchmark suite (`run_phase3_overnight.sh`) across `beam_missing`, `wet_ground`, `fog`, and `motion_blur` for all 6 method-prior combinations.
* **`[TODO]`** Populate the final benchmark tables comparing `baseline`, `veto_disagree`, and `conf_pred` at $\tau=0.0$ and $\tau=-1.0$.

### 5.3 `[TODO]` Forward Transfer vs Catastrophic Forgetting Metrics
* **`[TODO]`** Quantify forward transfer plasticity (adaptation speed in new domains) and catastrophic forgetting (performance degradation upon revisiting clean/earlier domains) in continual learning mode.
