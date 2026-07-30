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
