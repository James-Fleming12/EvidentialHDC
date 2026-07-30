# Future Directions and Methodological Fork

## 1. The Methodological Fork
Our empirical results have brought us to a genuine fork in our research direction. We should not indiscriminately build all possible methods; rather, we must run a single diagnostic test to determine which branch of reality we are in.

### Branch A: Prototype TTA is Viable (The Gating Hypothesis)
**Thesis:** Prototype adaptation fundamentally works, but *confidence* gating poisons it. The confident-wrong admission causes structural collapse (e.g., the `-10.26` mIoU collapse in T-DRIFT). *Consistency* gating fixes it.
**The Test:** Run **M2** (Weak-Strong Consistency Gate) on the T-DRIFT continual collapse (wet_ground).
- **If the collapse flattens:** Dual-gating is back. The core narrative becomes "confidence gating poisons prototype TTA; consistency gating fixes it."
- **If M2 fails to substantially reduce continual collapse, we reject the hypothesis that confidence gating is the primary failure mechanism and shift our focus away from prototype adaptation.**

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
**Test Protocol:** We passed `wet_ground` (domain shift) and `fog` (sensor noise) projections through the model. To break the softmax blockade of HDC cosines (which naturally squashes evidence to uniform), we scaled logits by a temperature of $\tau=15.0$ before applying the Dirichlet subjective-logic transformation (`evidence = softplus(logits * 15.0)`). We then masked out empty background pixels and computed Distribution Uncertainty (epistemic) and Data Uncertainty (aleatoric) across 20 frames (`check_n2_uncertainty.py`).
**Results:** Even after correcting the scaling and background artifacts, both channels remained heavily entangled and nearly identical across physically distinct corruptions:
- **Fog (Heavy Noise):** Dist Unc = 0.3855 | Data Unc = 2.5977
- **Wet Ground (Clean, Domain Shift):** Dist Unc = 0.3994 | Data Unc = 2.6328
**Verdict:** **FAILED.** The evidential decomposition used here does not separate domain shift from sensor noise in HDC space. Using N2 as a switch to gate a class prior (Branch B) is structurally unviable.

### M2 Diagnostic: Strong Augmentation Consistency
**Test Protocol:** We validated whether LiDAR range projections possess sufficient structural robustness to survive strong augmentations (20% random point dropout + $0.05$ std Gaussian noise on XYZ coordinates). We compared the accuracy of the frozen model on clean vs. augmented `wet_ground` frames (`check_m2_consistency.py`).
**Results:** 
- Clean Accuracy: 89.46%
- Augmented Accuracy: 82.80%
- Accuracy Drop: 6.66%
**Verdict:** **SUCCESS.** The strong augmentation is well-defined. It introduces enough variation to scatter outliers and causes a mild drop in clean accuracy without completely destroying core semantics. Consistency gating is structurally viable.

### D-COMPLEMENT Diagnostic: Consistency vs. Network Uncertainty Trap
**Test Protocol:** We tested whether view-disagreement provides a *complementary* signal to network uncertainty on severe hallucination corruptions (`fog` and `crosstalk`). We isolated points that the network was highly confident about (Dirichlet uncertainty < 0.5) and applied strong augmentation. We then measured the pseudo-label precision of points that *agreed* across views versus points that *disagreed* (`check_d_complement.py`).
**Results:** The consistency gate behaves in exactly the worst possible manner on structural hallucinations.
- **Fog (Confident):** Agree Precision = 9.67% | Disagree Precision = 45.21%
- **Crosstalk (Confident):** Agree Precision = 26.32% | Disagree Precision = 36.11%
**Verdict:** **FATAL FLAW.** Confident hallucinations are structurally robust to local noise/dropout. As a result, the views *agree* on the hallucination, meaning consistency gating preserves the poison (9.67% precision) and instead vetos the delicate, fragile structures that were actually more accurate (45.21% precision). Local geometric consistency is not sufficient to distinguish hallucinations from true positives in these corruptions.

### The Fork Resolution: Pivoting to the Physical Tether (Adaptive Budget)
We have now mathematically and empirically exhausted all three proposed branches for preventing TTA collapse:
1. **N4 (D-Optimal SVD):** Dead. HDC is holographic; it cannot be compressed.
2. **N2 (Dual Uncertainty):** Dead. The channels are entangled and cannot separate noise from domain gap.
3. **M2 (Consistency Gating):** Dead. View agreement perfectly correlates with robust structural hallucinations.

The filtering mechanisms investigated in this work were unable to reliably distinguish confident hallucinations from genuine predictions.

Instead of trying to filter the hallucinations, we **bound the damage**. In a 128D HDC hyperspace, angles are extremely rigid (orthogonal is 90 degrees). A domain shift (like `wet_ground`) only shifts the true semantic manifold by a small angle. A confident hallucination (like `fog` pulling the vegetation prototype into empty space) requires dragging the prototype across a massive angular distance.

The official narrative pivot is **M-A: Adaptive Budget (Rotation Cap)**. We enforce a hard physical tether (e.g., 20 degrees) on the prototypes. They are allowed to adapt to genuine domain shifts, but the moment a hallucination tries to hijack them, they hit the physical tether and stop, preserving the pre-trained semantics while preventing runaway collapse.

### Information Diagnostics (Tier 1)
Following the structural failure of the filtering gates, we mathematically audited the information geometry of the available TTA signals using a Logistic Regression **Complementary Information Test**. 
We fit a classifier to predict pseudo-label correctness using all available uncertainty signals concurrently, then performed a leave-one-out ablation to isolate unique information versus redundancy.

**Results:**
- **Fog (Full AUROC 0.8272):**
  - **Redundant:** Dirichlet Unc (+0.0004), Entropy (+0.0012), Margin (+0.0008), Feat Norm (-0.0006)
  - **Unique Info:** Max Cosine (+0.0061), Consistency (+0.0169)
- **Crosstalk (Full AUROC 0.6421):**
  - **Redundant:** Max Cosine (+0.0047), Entropy (+0.0005), Margin (-0.0008), Feat Norm (+0.0000)
  - **Unique Info:** Dirichlet Unc (+0.0562), Consistency (+0.0118)

**Verdict:** Within a linear decision model, nearly all network-derived confidence measures (Dirichlet Epistemic, Entropy, Margin, Max Cosine) are redundant. Adding one provides the maximum possible AUROC, and the others add ~0.0005 AUROC. This demonstrates that these uncertainty measures are largely redundant.
Furthermore, no linear combination of these uncertainty signals can reliably separate hallucinations from true points on `crosstalk` (maximum AUROC is only **0.6421**). This strongly implies the limitation lies in the available information rather than the choice of gating algorithm.

### Diagnostic M: Information Loss Through HDC
Following the mentor's recommendation to trace *where* information is destroyed, we completely bypassed the uncertainty metrics and trained Logistic Regression and MLP classifiers directly on the internal feature representations to predict correctness.
We tested two points in the pipeline:
1. **128D Continuous Backbone Features** (Before HDC projection)
2. **10,000D Binary HDC Embeddings** (After HDC binarization)

**Results (AUROC across Logistic/MLP):**
- **Fog:** 128D Backbone (~0.99) $\rightarrow$ 10,000D HDC (1.0000)
- **Crosstalk:** 128D Backbone (~0.99) $\rightarrow$ 10,000D HDC (1.0000)

**The Final Verdict:** This is a massive revelation. The information required to perfectly separate confident hallucinations from true points **exists** and is linearly decodable (AUROC 1.0000) in the 10,000D HDC space! The HDC projection/binarization *preserves* the information perfectly.
The information is **destroyed** at the final step: when the 10,000D vector is reduced to 17 prototype cosine similarities. Because hallucinations form dense, linearly separable clusters in arbitrary corners of the 10,000D space, they are easily detected by a hyper-plane. But because the network only measures distance to the 17 prototypes, a hallucination that happens to fall vaguely in the direction of the "vegetation" prototype is assigned high confidence, and its true spatial location is lost. 
Once reduced to logits, the maximum decodable AUROC plummets from 1.0000 to 0.6421. 

**The Contribution:** These experiments establish a clear causal chain explaining *why* prototype adaptation fails in HDC:
1. **128D Backbone $\rightarrow$ 10,000D HDC Embedding:** Discriminative information is perfectly preserved (AUROC 1.0000).
2. **HDC Embedding $\rightarrow$ Prototype Cosine Projection:** Information collapses as spatial structure is reduced to 17 scalar similarities.
3. **Information Collapse $\rightarrow$ Uncertainty Becomes Redundant:** Logit-based uncertainty measures collapse into a single dimension of redundant information.
4. **Redundancy $\rightarrow$ Filtering Fails:** Filtering mechanisms are structurally blind to hallucinations that happen to align with a prototype.
5. **Filtering Fails $\rightarrow$ Prototype Drift:** The prototypes are poisoned, leading to continual structural collapse.

**Conclusion:** Existing uncertainty signals are insufficient because the prototype projection discards discriminative information. This motivates methods that constrain prototype motion (such as an Adaptive Budget rotation cap, metric learning, or k-NN ensembles) rather than relying solely on logit-derived uncertainty.

### Diagnostic 3: Alternative Method Viability (Memory Banks, Soft Labels, Diffusion)
Following the realization that information is destroyed at the prototype projection layer, we evaluated alternative adaptation mechanisms that operate directly on the richer 10,000D representations or candidate sets (`check_alternatives.py`).

**1. PL-1 (Top-k Candidate Coverage)**
- **Fog:** Top-1 (13.08%) $\rightarrow$ Top-5 (66.51%)
- **Crosstalk:** Top-1 (27.87%) $\rightarrow$ Top-5 (75.13%)
**Verdict:** Approximately ~30% of the time, the true class is not even present in the Top-5 logits. Methods like PASLE that rely on iteratively refining soft candidate sets are fundamentally handicapped because the true answer has already been irrecoverably discarded by the logit bottleneck.

**2. DD-2 / MB-3 (Oracle Source Projection vs. Prototypes)**
We bypassed the class prototypes entirely and extracted a memory bank of 50,000 clean source 10,000D HDC vectors. We then classified corrupted target points based purely on their 1-Nearest Neighbor (1-NN) in the source bank.
- **Fog:** Prototype Accuracy (13.08%) $\rightarrow$ Source 1-NN Accuracy (**73.57%**)
- **Crosstalk:** Prototype Accuracy (27.87%) $\rightarrow$ Source 1-NN Accuracy (**53.38%**)
**Verdict:** The 10,000D feature representations remain vastly more informative than the 17 global prototypes. Memory bank methods, feature denoisers, or representation alignment methods have immense performance headroom.

**3. MB-2 (Target Neighborhood Purity)**
We queried the 10-Nearest Neighbors of each corrupted target point *within the corrupted target domain itself* and calculated how often the neighbors shared the same ground truth label.
- **Fog:** Target Neighborhood Purity = **90.54%**
- **Crosstalk:** Target Neighborhood Purity = **88.06%**
**Verdict:** **SUCCESS.** The semantic topology of the 10,000D HDC space remains pristine even under severe corruption! Hallucinated points do not scatter into random noise; they move together in dense, semantically pure clusters that simply no longer align with the global class prototypes. This provides incredibly strong theoretical justification for Contrastive Learning (AdaContrast), k-NN graphs, and Memory Bank methods, which can directly exploit this preserved local geometry.
