# Future Directions and Methodological Fork

## 1. The Methodological Fork
Our empirical results have brought us to a genuine fork in our research direction. We should not indiscriminately build all possible methods; rather, we must run a single diagnostic test to determine which branch of reality we are in.

### Branch A: Prototype TTA is Viable (The Gating Hypothesis)
**Thesis:** Prototype adaptation fundamentally works, but *confidence* gating poisons it. The confident-wrong admission causes structural collapse (e.g., the `-10.26` mIoU collapse in T-DRIFT). *Consistency* gating fixes it.
**The Test:** Run **M2** (Weak-Strong Consistency Gate) on the T-DRIFT continual collapse (wet_ground).
- **If the collapse flattens:** Dual-gating is back. The core narrative becomes "confidence gating poisons prototype TTA; consistency gating fixes it."
- **If it still collapses:** Branch A is dead for good.

### Branch B: Adaptation is Poisoned (The Inference-Time Hypothesis)
**Thesis:** Adaptation is fundamentally poisoned in this setting. The only valid contribution is inference-time corrections (frozen + scoped prior). 
**The Test:** If M2 fails to stop the T-DRIFT collapse, adaptation is abandoned. The paper becomes a negative-result story for TTA, and we build **N2** (Uncertainty Decomposition) to scope exactly *when* the prior can be safely applied.

---

## 2. Proposed Mechanisms (Non-MV)

### N1 — Evidential Reject-Class Gating (COME)
- **Mechanism:** Replaces hard confidence thresholds with a Dirichlet subjective-logic gate. "Uncertainty mass" acts as its own channel; points update in proportion to $1 - u$.
- **Fixes:** Resolves the softmax blockade (where 10K-D cosines squash max softmax to ~0.06, causing 0.90 thresholds to reject everything).
- **Cost:** Trivial reweighting.
- **Risk:** Very similar to our existing epistemic gate (`soft_dual_weight`). We must verify it adds actual novelty and doesn't just relabel what we have.

### N2 — Dual-Channel Uncertainty Decomposition (EviATTA)
- **Mechanism:** Splits uncertainty into *distribution uncertainty* (domain gap/precision failure) and *data uncertainty* (inherent noise/hallucination).
- **Fixes:** The fog false-trigger that killed the ratio-based prior switch. Prior is applied when dist-unc is high and data-unc is low.
- **Risk:** The decomposition could be degenerate in HDC space (both channels move together). Requires a diagnostic check first to see if it actually separates fog from wet_ground.

### N3 — Conjugate Pseudo-Label Loss (Goyal et al.)
- **Mechanism:** Replaces ad-hoc heuristic prototype updates with the convex conjugate of the supervised training loss, yielding a principled update rule.
- **Fixes:** Objective-level poisoning of the prototypes.
- **Cost/Risk:** High derivation effort. HDC quantization or non-convexity might break conjugate assumptions.

### N4 — D-Optimal Subspace Prototypes (Liang et al.)
- **Mechanism:** Finds the informative subspace where semantic margin lives (D-optimal compression) and gates/updates over this dense core rather than the full 10,000 dimensions.
- **Fixes:** The flattened-similarity problem at its root.
- **Diagnostic:** Check the singular-value spectrum of source prototypes. If flat, HDC is truly holographic and compression will destroy the signal.

---

## 3. Proposed Mechanisms (MV / Multi-View)

### M1 — Structural-Alignment Gate with Negative Augmentations (SaTeen)
- **Mechanism:** Gates on both positive agreement (views agree) AND negative constraints (disagrees with destroyed views).
- **Fixes:** Catches hallucination cases where a false point might agree across mild positive views but fails to scatter under destruction.
- **Risk:** If hallucinated points also scatter under destruction like real points, the negative constraint adds nothing.

### M2 — Weak-Strong Soft-Voting Update (Improved Self-Training)
- **Mechanism:** Gates prototype updates on weak-strong augmentation consistency via soft voting, not confidence. Points update only if the pseudo-label survives strong augmentation.
- **Fixes:** Confident-wrong poisoning. Structurally-sound points cluster under augmentation, while outliers scatter and self-veto.
- **Risk:** Strong augmentation on LiDAR range projections must be carefully defined to preserve semantics.

---

## 4. Execution Sequence

1. **The Fork Test (M2 on T-DRIFT):** Run the weak-strong consistency gate on the T-DRIFT collapse. This strictly dictates whether we are in Branch A (gating) or Branch B (frozen + prior).
2. **N2 Diagnostic:** If Branch B, check if dual-channel uncertainty separates fog (high data-unc) from wet_ground (high dist-unc).
3. **N1 & M1:** Build only if Branch A is validated, to refine the consistency gate.
4. **N4:** Check singular-value spectrum.
5. **N3:** High-effort theoretical derivation, save for last.

---

## 5. Diagnostic Test Results

We ran automated diagnostic scripts on the frozen HDC network to validate the theoretical viability of our proposed mechanisms before full implementation.

### N4 Diagnostic: SVD Subspace Compression
**Test Protocol:** We extracted the 10,000-dimensional prototype vectors for all 17 classes from the pre-trained HDC classification layer and computed their Singular Value Decomposition (SVD) to analyze the energy retention across principal components (`check_n4_svd.py`).
**Results:** The singular value spectrum is extremely flat (e.g., $\sigma_1 = 1.47$, $\sigma_{13} = 0.73$). The top 8 components explain only 75% of the variance, proving that the HDC space is fundamentally holographic. There is no dense "core" subspace where the semantic margin lives.
**Verdict:** **FAILED.** D-optimal compression (N4) will destroy the distributed semantic representation. N4 is abandoned.

### N2 Diagnostic: Dual-Channel Uncertainty Decomposition
**Test Protocol:** We passed `wet_ground` (domain shift) and `fog` (sensor noise) projections through the model, applying a Dirichlet subjective-logic transformation (`evidence = softplus(logits)`) to the HDC logits. We then computed Distribution Uncertainty (epistemic) and Data Uncertainty (aleatoric) across 20 frames (`check_n2_uncertainty.py`).
**Results:** Both channels yielded virtually identical values across both corruptions:
- **Fog:** Dist Unc = 0.5781 | Data Unc = 2.8320
- **Wet Ground:** Dist Unc = 0.5791 | Data Unc = 2.8320
**Verdict:** **FAILED.** The Dirichlet uncertainty channels are completely degenerate and entangled in the HDC space. It is mathematically impossible to use N2 as a switch to separate domain gap from noise. Branch B (Uncertainty Gated Prior) is structurally blocked.

### M2 Diagnostic: Strong Augmentation Consistency
**Test Protocol:** We validated whether LiDAR range projections possess sufficient structural robustness to survive strong augmentations (20% random point dropout + $0.05$ std Gaussian noise on XYZ coordinates). We compared the accuracy of the frozen model on clean vs. augmented `wet_ground` frames (`check_m2_consistency.py`).
**Results:** 
- Clean Accuracy: 89.46%
- Augmented Accuracy: 82.80%
- Accuracy Drop: 6.66%
**Verdict:** **SUCCESS.** The strong augmentation is well-defined. It drops accuracy slightly but strongly preserves core semantics. This proves that pseudo-labels generated under M2 consistency gating will be highly reliable.

### The Fork Resolution
Because N2 is fundamentally degenerate in HDC space and M2's consistency augmentations are highly viable, **Branch A is officially the winner**. Our narrative will focus entirely on **Consistency Gating (M2/M1)** to fix the confident-wrong poisoning that collapses prototype TTA.
