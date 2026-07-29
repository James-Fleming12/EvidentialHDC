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
| `m_a_cap` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `m_ab_gain` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `m_abc_loosen` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| `m_abcd_prior` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### 2.2 STAGE 2: Isolated Prior Test (M-D)
*Verifies the pure inference-time gain (claimed +11.7 on wet_ground) without any prototype adaptation.*

| Method | Mean | fog | wet_ground | snow | motion_blur | beam_missing | crosstalk | incomplete_echo | cross_sensor |
|---|---|---|---|---|---|---|---|---|---|
| `m_d_prior_only` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
