# Beyond Geometric Confidence: Multi-Source Uncertainty and Balanced Allocation for Backpropagation-Free LiDAR Test-Time Adaptation

---

## 1. What we established (and why it forced this pivot)

This project began by trying to build a better *geometric* confidence gate in
hyperdimensional space. That line of work is now closed, and its failure is
well-characterized. The following are **measured results**, not intuitions:

| Finding | Evidence |
|---|---|
| **Every geometric refinement loses to a first-order dot product.** Covariance ellipsoids, subspace reweighting, unions of balls, subcluster gating, and k-NN contrastive banks all underperform plain prototype cosine similarity as a gate. | rank sweep (all four subspace modes converge to ~0.78 AUROC vs 0.84 for the plain ball); precision–coverage (prototype ≥ k-NN at every operating point) |
| **The reason is `n ≪ d`.** ~5k samples/class in 10k dimensions. A mean is estimable; a covariance is not. Every second-order score is noise-dominated by construction. | spectrum diagnostic; monotone AUROC decline as more eigen-directions are used |
| **Set-valued conformal prediction is vacuous in HDC.** `E[\|C(x)\|] = 0.58` — prediction sets are always empty or singleton, never ambiguous, because HDC preserves inter-class separation. | calibration-drift run |
| **AUROC is the wrong metric for a gate.** k-NN had far higher AUROC (0.939 vs 0.81) yet *lost* on precision-at-coverage, which is what a gate actually uses. | knn_sweep vs precision_coverage |
| **THE DECISIVE ONE: no pseudo-label gate recovers any of the available headroom.** | overnight run, below |

### The overnight decision experiment

```
frozen (11 live classes)  47.93
oracle gate (GT labels)   50.47   (+2.73)      <- headroom EXISTS
supervised ceiling        59.03   (+11.10)     <- (with Bayesian shrinkage prior)
measurement noise (std)    1.00

best pseudo-label gate    ~47.98  (~0.00)      <- NONE of it is recoverable
   (macro-precision stalls at ~82%, zeroing the gain)
```

**There is massive headroom (up to +41 on some rare classes), and thresholding cannot reach any of it.**

### The diagnosis: a precision wall

The oracle and the best pseudo-gate move the prototypes by *comparable amounts* (drift ~0.71). Same update rule, same learning rate, similar magnitude of movement — and yet **+2.73 versus +0.00**. The only material difference is precision.

**At ~82% macro precision, the wrong 18% exactly cancel the right 82%.** The errors are not symmetric: a wrong pseudo-label drags a prototype toward *another class's* region — a far larger displacement than the small refinement a correct label buys.

And this is specifically a **rare-class** problem. *Pooled* precision at 10% coverage is 99.2%; **macro** precision at the same coverage is 81%. The megaclasses are easy and already correct. The classes mIoU actually weights are the ones the geometric gate cannot label reliably **at any threshold**.

> **Therefore: the bottleneck is pseudo-label QUALITY, not pseudo-label SELECTION — specifically for rare classes. No amount of clever filtering fixes a label that is wrong. We must make the labels better.**

Every design decision below follows from this.

---

## 2. The precision-requirement curve

We took the oracle gate and injected label noise at controlled rates to find the precision at which the gain disappears. Crucially, the injected noise mirrored the model's actual confusion matrix (e.g., adjacent class bleeding) rather than uniform random flips, perfectly simulating the systematic damage a real gate would permit.

```
oracle @ 100% target precision -> +2.73 mIoU
        @  98%                 -> +2.28
        @  95%                 -> +2.22
        @  90%                 -> +1.70
        @  85%                 -> +1.38
        @  82%                 -> +1.27
        @  80%                 -> +1.20
        @  75%                 -> +1.07
        @  70%                 -> +1.00    (hits the 2*noise floor)
```

This curve converts *"we need better pseudo-labels"* into a **concrete engineering target**. 
The result is highly encouraging: the wall is a gentle slope. We do not need a mythical 98% precision. If our uncertainty fusion can achieve just **85-90% pooled precision** (which corresponds to ~71-76% macro precision), we will comfortably capture measurable, statistically significant gains (+1.38 to +1.70 mIoU).

---

## 3. Contribution 1 — Multi-source uncertainty fusion

Every confidence signal used so far has been *geometric*: distance or similarity in HD space. That is demonstrably insufficient. Classes 0, 7, and 2 have terrible frozen IoUs, meaning their pre-trained prototypes are physically located in the wrong quadrant of hyperspace. **If the prototype is in the wrong place, a geometric gate measuring distance to that prototype is also wrong.**

We must fuse independent sources of uncertainty whose failure modes are uncorrelated with geometric distance.

**(a) Geometric uncertainty (HD space).** Prototype cosine similarity. Cheap, `O(K)`. Answers *"is this point near the current class manifold?"*

**(b) Epistemic uncertainty (the network).** Measured **without** the softmax.
- **Multi-RP ensemble** — project the same feature through several independent random projections.
- **Evidential Deep Learning / Feature-space density** in the *backbone's* latent space.

**(c) Spatial / temporal consistency.** A point whose neighbours and previous-frame correspondent agree with its label is far more likely correct. This acts as a **vote / veto on the LABEL**, not a smoother of the score. (Prior graph-Laplacian smoothing blurred errors across objects; consistency must act as a hard veto to reject errors, not smear them).

---

## 4. Contribution 2 — Balanced update allocation (Budgeting by Headroom)

### The problem, measured

A frequency-proportional update budget spends almost all of its capacity on classes that cannot improve. 
In our supervised ceiling tests, Class 11 (Road) had millions of points but **negative headroom (-0.57)**. Updating it actually hurts the model because it is already saturated. Meanwhile, Class 0 (Car) had only 98k points but **+41.07 headroom**. 

### The mechanism: a subcluster update ledger

We must allocate the adaptation budget by **headroom**, not by frequency.
- Maintain **K representative subclusters per class**, initialized from source.
- Route each admitted point to its nearest subcluster.
- **A subcluster contributes to the prototype update only if its update count is within a bounded range of its siblings'.** 

This equalizes adaptation signal *within* a class (no single dense mode dominates) and *across* classes (freezing saturated dense classes like Road, and funneling the update budget exclusively to high-headroom regions like Car/Motorcycle).

Crucially, **the subclusters only track updates; they never touch inference.** This prevents the Voronoi-shattering failure that destroyed previous subcluster-gating attempts.

---

## 5. Contribution 3 (variant) — Multi-view test-time augmentation

Generate augmented views (yaw roll, scale, dropout), bundle the resulting hypervectors, use cross-view soft agreement as a reliability signal. 
Positioned as a compute-scaling variant: *"when additional compute is available, multi-view agreement raises macro precision by X at Y× cost."*

---

## 6. Reading List

To execute this pivot, we must explore specific mechanisms for single-pass epistemic UQ and streaming buffer allocation.

1. **Efficient TTA UQ Estimation:** 
   We cannot afford Monte Carlo Dropout or Deep Ensembles (TTA must run strictly online). 
   - *Read:* Literature on **Evidential Deep Learning**, **Temperature Scaling**, or **Single-Pass Feature-Level Perturbations** to find the cheapest, most effective epistemic UQ method compatible with our frozen 3D backbone.
2. **Headroom-Based Resource Allocation:**
   - *Read:* Literature on estimating unsupervised headroom or predictive instability in streaming data. We need to validate metrics (e.g., tracking moving averages of entropy/confidence) to automatically identify and freeze saturated classes in the subcluster ledger.
3. **Category-Balanced Memory Banks:**
   - *Read:* **RoTTA (Robust Test-Time Adaptation).** This is the closest prior work (category-balanced memory bank for temporally correlated streams). Understand how they bound class updates, and clearly differentiate our backprop-free, intra-class balancing approach.

---

## 7. Order of work

1. **Identify the epistemic UQ method (Pillar 1).** Find a lightweight, single-pass method compatible with our frozen backbone.
2. **Draft the memory-safe subclustering ledger (Pillar 3).** Ensure it enforces the headroom-based budget.
3. **Uncertainty sources, one at a time.** Measure macro precision on each independent source.
4. **Fusion + full ablation.**
5. **mIoU validation**: error bars, live classes, interleaved eval, oracle reported alongside.
