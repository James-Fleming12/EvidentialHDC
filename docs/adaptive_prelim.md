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

*   **D1 — The Prior-Switch Signal:** The prior is a precision tool that suppresses rare classes (killing false positives on wet ground) but hurts when the model is already precise (deleting correct predictions on fog/snow/blur). We need to log the label-free *rare-class prediction rate* vs. the *source rare-class rate*. If this signal correlates with the prior's per-domain $\Delta$ (+helps vs -hurts), we have a zero-label controller for switching the prior on/off.
*   **D2 — Label Shift vs. Covariate Shift (The BBSE Trap):** BBSE assumes pure label shift ($p(x|y)$ is fixed, $p(y)$ changes). However, LiDAR corruptions like fog physically alter $p(x|y)$ (covariate shift). We must test if BBSE is viable by estimating the target confusion structure on a held-out labeled slice of corruption and comparing it to the source confusion matrix $C$. If they diverge sharply, BBSE will invert the wrong matrix and inject error.
*   **D3 — The Drift-Knee Sweep:** The 20° cap still lost -2.5 mIoU on wet ground. We must sweep $\Phi \in \{2, 5, 10, 20\}^\circ$ on wet ground alone to find the exact angle where the loss crosses zero. This identifies the "damage knee" and sets the target $\Phi_{\text{max}}$ for any future adaptive budget.
*   **D4 — Confirm M-B/M-C Saturation in Code:** `m_ab_gain` and `m_abc_loosen` were byte-identical to the cap-only run. We must log the actual gain multiplier and effective threshold per frame to guarantee they are constant. If they vary but simply don't matter, un-saturating the Dirichlet mapping won't help.

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
*Note: Due to a cache key collision in the `ablation_kitti-c.py` test runner (`fk` excluded the prior flags), the `m_d_prior_*` methods inadvertently loaded the cached `frozen` evaluation results instead of running their unique prior logic. These results are identically matched to the `frozen` baseline and will need to be re-run with a fixed cache key.*

### 5.2 Stage 3: Prior Switch (D5 Test)
*Because `frozen` was not evaluated in Stage 3, `m_d_prior_switch` successfully bypassed the cache and evaluated its prior logic across all 8 corruptions.*

| Method | Mean | fog | wet_ground | snow | motion_blur | beam_missing | crosstalk | incomplete_echo | cross_sensor |
|---|---|---|---|---|---|---|---|---|---|
| `frozen` | 33.68 | 6.04 | 42.02 | 50.22 | 50.52 | 39.44 | 10.71 | 43.90 | 26.56 |
| `m_d_prior_switch` | 32.24 | 1.29 | 47.84 | 49.80 | 49.42 | 39.15 | 5.57 | 38.00 | 26.84 |

**Observations:**
- **Wet Ground (+5.82):** The prior switch successfully turns on and yields massive gains!
- **Fog (-4.75) & Crosstalk (-5.14):** The switch was supposed to remain off for these domains, but performance cratered far below `frozen`. This indicates the simple `tail_ratio >= 1.15` heuristic is falsely triggering on these corruptions.

### 5.3 Stage 5: T-DRIFT Continual Run (Legacy Loose)
*The continual run on wet ground (4071 frames) over time with loose gating.*

- **Initial mIoU:** 48.73
- **Final mIoU:** 38.47 (Structural Collapse of -10.26)
- **Accuracy:** 91.18 -> 79.77
- **Firing Rate:** 74.03%
