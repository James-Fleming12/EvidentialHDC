# Adaptive Mechanism Preliminaries

This document compiles the foundational empirical results and active test structures that drive the controller-based adaptation mechanisms (M-A through M-D).

## 1. Background Information & Ceilings

The development of the M-series components is anchored by the absolute performance ceilings established in earlier diagnostic runs. By evaluating the baseline, the standard geometric update, and oracle-bound strategies, we map the available headroom across all corruptions.

### 1.1 The Established Ceilings (Severity 3, Mean mIoU)
- **Frozen (No Adaptation):** 33.68
- **Uniform Geometric Adaptation (`full_method`):** 33.28 (Net -0.40)
- **Gate Oracle (Perfect Admission):** 35.43 (+1.75)
- **Prior Oracle (Perfect Decision Boundary):** 35.46 (+1.78)
- **Per-Domain Decision Oracle:** **36.30 (+2.62)**

The Per-Domain Decision Oracle is the ultimate ceiling for any per-domain controller. It combines the optimal categorical action for each domain:
- **Freeze:** fog, snow, motion_blur
- **Adapt:** beam_missing, crosstalk
- **Prior-Fix:** wet_ground, incomplete_echo, cross_sensor

### 1.2 Key Diagnostic Findings
- **G3 (Crosstalk False-Veto Breakdown):** The epistemic gate aggressively over-vetoes correct pseudo-labels in high-uncertainty domains like crosstalk, rejecting ~20 times more correct predictions than true errors. This justifies **M-C (Uncertainty-loosened gate)**.
- **G4 (Prototype Drift Audit on Wet Ground):** The standard method loses -7.25 mIoU on wet ground. Head classes (e.g., Road, Building) rotate by a massive average of ~46 degrees, leading to a catastrophic -15.0 mIoU drop. This necessitates a hard safety boundary: **M-A (Per-class rotation cap)**.
- **Prior Tracking:** An earlier test substituting the source prior with the target distribution yielded up to +11.7 mIoU on wet ground purely at inference, justifying **M-D (Always-on prior estimation)**.

---

## 2. Active Component Ladder Tests

We are currently executing an additive 3-seed component ladder against the +2.62 ceiling from the G1 decision-oracle. The tests incrementally enable the four continuous mechanisms:

1. **M-A:** Per-class rotation cap (20 degrees)
2. **M-AB:** M-A + Continuous gain control (domain-gap LR scaling)
3. **M-ABC:** M-AB + Uncertainty-loosened gate
4. **M-ABCD:** M-ABC + Always-on prior estimation
5. **M-D (Prior Only):** Isolated inference-only prior estimation (no adaptation)

### 2.1 STAGE 1: The Additive Component Ladder
*Tests the additive effect of M-A through M-D on top of the frozen baseline.*

| Method | Mean | fog | wet_ground | snow | motion_blur | beam_missing | crosstalk | incomplete_echo | cross_sensor |
|---|---|---|---|---|---|---|---|---|---|
| `frozen` | 33.68 | 6.04 | 42.02 | 50.22 | 50.52 | 39.44 | 10.71 | 43.90 | 26.56 |
| `m_a_cap` | 33.36 | 5.56 | 39.50 | 49.29 | 50.57 | 39.94 | 12.33 | 42.77 | 26.93 |
| `m_ab_gain` | 33.36 | 5.56 | 39.50 | 49.29 | 50.57 | 39.94 | 12.33 | 42.77 | 26.93 |
| `m_abc_loosen` | 33.36 | 5.56 | 39.50 | 49.29 | 50.57 | 39.94 | 12.33 | 42.77 | 26.93 |
| `m_abcd_prior` | 32.56 | 2.74 | 46.06 | 48.24 | 49.88 | 38.84 | 4.82 | 42.70 | 27.21 |

### 2.2 STAGE 2: Isolated Prior Test (M-D)
*Verifies the pure inference-time gain (claimed +11.7 on wet_ground) without any prototype adaptation.*

| Method | Mean | fog | wet_ground | snow | motion_blur | beam_missing | crosstalk | incomplete_echo | cross_sensor |
|---|---|---|---|---|---|---|---|---|---|
| `m_d_prior_only` | 35.46 | 5.27 | 53.73 | 49.91 | 49.25 | 39.28 | 11.80 | 46.70 | 27.76 |

---

## 3. Diagnostic Tests (D1-D4)

Before building new mechanisms, we must run four specific diagnostic tests to validate our assumptions and gather tuning parameters.

*   **D1, The Prior-Switch Signal:** The prior is a precision tool that suppresses rare classes (killing false positives on wet ground) but hurts when the model is already precise (deleting correct predictions on fog/snow/blur). We need to log the label-free *rare-class prediction rate* vs. the *source rare-class rate*. If this signal correlates with the prior's per-domain $\Delta$ (+helps vs -hurts), we have a zero-label controller for switching the prior on/off.
*   **D2, Label Shift vs. Covariate Shift (The BBSE Trap):** BBSE assumes pure label shift ($p(x|y)$ is fixed, $p(y)$ changes). However, LiDAR corruptions like fog physically alter $p(x|y)$ (covariate shift). We must test if BBSE is viable by estimating the target confusion structure on a held-out labeled slice of corruption and comparing it to the source confusion matrix $C$. If they diverge sharply, BBSE will invert the wrong matrix and inject error.
*   **D3, The Drift-Knee Sweep:** The 20° cap still lost -2.5 mIoU on wet ground. We must sweep $\Phi \in \{2, 5, 10, 20\}^\circ$ on wet ground alone to find the exact angle where the loss crosses zero. This identifies the "damage knee" and sets the target $\Phi_{\text{max}}$ for any future adaptive budget.
*   **D4, Confirm M-B/M-C Saturation in Code:** `m_ab_gain` and `m_abc_loosen` were byte-identical to the cap-only run. We must log the actual gain multiplier and effective threshold per frame to guarantee they are constant. If they vary but simply don't matter, un-saturating the Dirichlet mapping won't help.

---

## 4. Reprioritized Improvement Steps

The honest framing of our current architecture is now **"frozen model + selectively-applied inference-time prior correction,"** with bounded adaptation treated purely as an optional per-domain extra. 

1.  **Prior Switch (Highest Priority - Target: 35.78):** Gate the inference prior using D1's rare-class-rate signal. An oracle that applies the prior only where it helps (wet ground, echo) and turns it off where it hurts scores 35.78 (+2.10 over frozen). This is a simple logic switch, requires zero new machinery, and is the single biggest available gain.
2.  **Adaptive Budget (Targeted Scope):** Implement the confidence-keyed adaptive budget $\Phi_c = \Phi_{\text{max}} \cdot (1 - \text{conf}_c)$. Crucially, this is *not* a global mechanism to raise the mean. It is scoped strictly as bounded adaptation on crosstalk only, layered on top of the frozen+prior baseline to claw back performance where the prior alone fails.
3.  **BBSE (Gated on D2):** Pursue Black Box Shift Estimation *only* if diagnostic D2 shows that label shift dominates covariate shift. If the assumption holds, it is genuinely promising for pushing wet ground past +11.7. If it fails, BBSE is a theoretical trap.
4.  **Un-saturate Dirichlet (Lowest Priority):** Scaling the Dirichlet prior to un-saturate the domain-gap scalar is only worth doing if the adaptive budget (Step 2) specifically needs the domain-gap scalar as a trigger. Otherwise, fixing inert mechanisms (M-B/M-C) to chase a +1.75 gate ceiling is counterproductive when the prior switch already yields +2.10.

---

## 5. Preliminary Redo Results (Prior Switch & T-DRIFT)

### 5.1 Stage 1: Frozen Prior Methods
*Note: The initial run of Stage 1 suffered from a cache key collision in the `ablation_kitti-c.py` test runner. Upon fixing the cache key and re-running, a second major bug was discovered: `target_prior` was missing from `RESET_ATTRS`, meaning the estimated prior was leaking across corruptions.*

**Clean Re-evaluation Findings:**
When properly isolated and initialized from the clean `source` distribution, the `ratio >= 1.15` heuristic in `m_d_prior_switch` **fails to trigger on wet ground**. Because the frozen model is highly conservative, its initial predictions lack sufficient tail mass to pull the estimated `target_prior` up to the 1.15x threshold. Consequently, the switch remains OFF for the entire sequence, and performance identically matches the `frozen` baseline (42.02).

### 5.2 Stage 3: Prior Switch (D5 Test)
*In this run, `fog` evaluated before `wet_ground`. Due to the `target_prior` leak, `wet_ground` inherited the corrupted prior from `fog`.*

| Method | Mean | fog | wet_ground | snow | motion_blur | beam_missing | crosstalk | incomplete_echo | cross_sensor |
|---|---|---|---|---|---|---|---|---|---|
| `frozen` | 33.68 | 6.04 | 42.02 | 50.22 | 50.52 | 39.44 | 10.71 | 43.90 | 26.56 |
| `m_d_prior_switch` (Original) | 32.24 | 1.29 | 47.84 | 49.80 | 49.42 | 39.15 | 5.57 | 38.00 | 26.84 |

### 5.3 T0a & T0b Clean Reproductions
*Because the original Stage 3 results were contaminated by a state leak, we ran a fully guarded repro (T0a) and a deliberate kick-start (T0b) on a fast subset of 4 corruptions.*

**T0a (Fully Cleaned, No Leaks):**
| Method | fog | crosstalk | wet_ground | incomplete_echo |
|---|---|---|---|---|
| `frozen` | 6.04 | 10.71 | 42.02 | 43.90 |
| `m_d_prior_only` | 1.29 | 5.68 | 44.47 | 42.44 |
| `m_d_prior_switch` | 1.29 | 5.68 | 42.02 | 42.93 |
| `m_d_prior_ramp` | 1.29 | 5.68 | 42.30 | 42.92 |
| `m_d_prior_inverse` | 6.04 | 10.71 | 44.27 | 43.39 |

**T0b (Deliberate Tail-Prior Kick-Start):**
| Method | wet_ground |
|---|---|
| `m_d_prior_boosted` | 42.29 |

**Observations (The State Leak Phenomenon & T0a/T0b Reproduction):**
- **The "Kick" is a Fluke (T0b Failure):** We hypothesized that the switch needed a "kick" to bootstrap out of the frozen model's bias. However, a deliberate, clean kick-start of the tail prior on `wet_ground` (T0b) yielded only **42.29** mIoU (reverting almost instantly to the frozen baseline). The massive 47.84 score in Stage 3 was a pure fluke: it inherited a highly skewed, hallucinated prior from the end of the `fog` sequence, which just happened to spuriously align with `wet_ground`!
- **False Triggers on Noise:** In the clean, fully-guarded T0a run, `m_d_prior_switch` on `fog` still crashed to **1.29** (compared to frozen 6.04). Even from a clean start, the frozen model hallucinates fog points as tail classes, inflating the predicted tail ratio, falsely triggering the switch, and initiating a feedback loop that destroys performance.
- **Conclusion:** The `ratio >= 1.15` heuristic is dead, and moreover, the geometric domain-gap scalar hypothesis is also dead. A geometric scalar would rate `fog` as maximally far from the source, meaning it would positively green-light the exact domain that crashes the model. The real finding is that the prior mechanism itself is unstable under corruptions that induce tail-hallucination. `fog` and `crosstalk` do not have a "when to fire" problem - they have a "the prior correction is the wrong operation entirely" problem because the model's errors there are not precision failures that a prior can fix.

### 5.3 Stage 5: T-DRIFT Continual Run (Legacy Loose)
*The continual run on wet ground (4071 frames) over time with loose gating.*

- **Initial mIoU:** 48.73
- **Final mIoU:** 38.47 (Structural Collapse of -10.26)
- **Accuracy:** 91.18 -> 79.77
- **Firing Rate:** 74.03%

## 6. Previous Diagnostics (128D Prototype Implementations)

The `standard_t3a` and `conformalhdc` baselines evaluate prototypes in the 128D backbone latent space rather than the non-linearly projected 10,000D hypervector space on which the HDC classification head is trained. 

The diagnostic execution revealed a profound structural incompatibility: the 128D latent space features are not linearly separable via cosine similarity. As a result, the initial predictions using 128D centroids are effectively random chance (e.g., ~4.4% for fog, ~26% for wet ground) compared to the correct 10,000D space evaluations.

| Corruption | `standard_t3a` (128D) | `conformalhdc` (128D) |
|---|---|---|
| `fog` | 4.39 -> 5.08 | 4.39 -> 4.53 |
| `wet_ground` | 26.02 -> 22.64 | - |
| `snow` | 29.25 -> 31.09 | - |
| `motion_blur` | 28.85 -> 29.40 | - |
| `beam_missing` | 24.94 -> 28.16 | - |
| `crosstalk` | 13.69 -> 11.31 | - |
| `incomplete_echo` | 16.53 -> 20.02 | - |
| `cross_sensor` | 19.45 -> 21.29 | - |

*(Note: The `conformalhdc` run was cancelled after `fog` once the structural flaw was verified).*

### 6.1 `conformalhdc_10k` Prototype Normalization Failure

While `conformalhdc_10k` correctly maps features into the 10,000D hypervector space, the textbook Euclidean baseline implementations actively sabotage the HDC representation by explicitly L2-normalizing the learned prototypes (`W = F.normalize(W, dim=1)`). 

In an HDC network, the magnitudes of the learned `classify.weight` vectors are critical as they act as a learned class prior. Stripping this magnitude causes a massive drop in initial performance (e.g., dropping from the true frozen baseline of 42.02% down to 36.08% on `wet_ground`). 

Furthermore, the standard `0.90` softmax confidence threshold acts as a total blockade in high dimensions. The baseline scales the cosine similarity by a fixed temperature of `15.0`. In a 10,000-dimensional space, random cosine similarities are naturally extremely small ($\approx 0.01$). This yields tiny logits, capping the maximum softmax confidence around ~0.06. Because it never reaches the hard-coded 0.90 threshold, the gate rejects *every single point*, resulting in zero prototype updates during the sequence (the slight $\Delta$ in final mIoU is merely a side-effect of backbone BatchNorm running stat drift).

| Corruption | `conformalhdc_10k` |
|---|---|
| `fog` | 3.66 -> 4.07 |
| `wet_ground` | 36.08 -> 35.51 |
| `snow` | 37.97 -> 37.81 |

**Conclusion:** The standard Euclidean baselines (`ConformalHDC`, `StandardT3A`) are fundamentally incompatible with the mathematical mechanics of high-dimensional computing. Valid adaptation must occur using HDC-native mechanisms (e.g., Evidential Epistemic gating, Tau calibration) as developed in the M-series component ladder.
