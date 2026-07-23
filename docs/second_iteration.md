# Phase 2: Audit of Phase 1 Takeaways and Balancing Test Suite
**Date:** July 22-23, 2026 (v3)

## Objective
Establish mathematically rigorous diagnostics to audit the early Phase 1 hypotheses, and explicitly shift from heuristic fixes to explicit magnitude/rotation schedules and inter/intra-class balancing. 

---

## Part A: The True Phase 1 Takeaways & Overnight Diagnostics (July 23)

We implemented an overnight diagnostic suite (V1-V7) to formally test the takeaways from Phase 1. 

| Run | Snow-3 (Online / Frozen) | Wet Ground-3 (Online / Frozen) |
| :--- | :--- | :--- |
| **Run 1: Baseline** | 0.4114 / 0.4126 | 0.4324 / 0.4337 |
| **Run 2: Bug Reproduction** | **0.4362** / **0.4545** | **0.4592** / **0.4766** |
| **Run 3: Noise Floor (V7)** | 0.4112 / 0.4112 | 0.4329 / 0.4332 |

### A0. The PyTorch `.data` bug was a Per-Class, Time-Decaying Learning Rate
`model.classify.weight[c].data = F.normalize(...)` failed to overwrite the tensor in-place, meaning the un-normalized weight accumulated steps infinitely: `w_c += step`.
Because the classification layer always normalized on the forward pass, the unconstrained magnitude *only* affected the angular step size:
`rotation per step ≈ step_mag / ||w_c||`
As `||w_c||` grew linearly via accumulation, the effective angular learning rate decayed like **$1/t$**. 

**Overnight Proof:** By simply disabling the post-step normalization in Run 2 (`--reproduce_bug`), we completely restored the massive Phase 1 performance delta. Snow's frozen mIoU jumped from **0.4126 to 0.4545**, and Wet Ground jumped from **0.4337 to 0.4766**.

### A1. The 50–60° Rotation was a Prototype-Norm Artifact
The claim that a single phantom point dragged prototypes 60° was false. The massive rotations occurred because rare classes happened to have small initial prototype norms from pretraining, making their initial `1/||w_c||` multiplier massive. 

**Overnight Proof:** We updated the pretraining pipeline to correctly guarantee `||w_0|| = 1.0` for all classes right out of the gate. As a result, the Bug Reproduction run (Run 2) only saw a max rotation of ~11°, instead of 60°. The step sizes started reasonably small and decayed, rather than starting massive.

### A2. The Epistemic Anchor ($k=0$) does nothing
The regression from $k=0.0001$ to $k=0.0$ was only 0.0004 mIoU—well below the noise floor. The spring is neither the cause of the `0.4840` shortfall nor a necessary anchor. 

### A3. The `0.4840` vs `0.4129` Comparison is Confounded
The `0.4840` metric was reported as a *cumulative online* mIoU during the adaptation pass. The `0.4129` metric was reported as a *final frozen* mIoU post-adaptation.

**Overnight Proof:** Run 2 demonstrated that **Frozen** final mIoU (0.4545) is significantly higher than the **Online** cumulative mIoU (0.4362). As the model adapted with the $1/t$ decay, it eventually found a highly optimized geometry. The online metric was simply dragged down by early frames where the model was actively rotating.

### A4. The Noise Floor is Tight
Run 3 established that the variance of the baseline is extremely tight ($\pm 0.0014$ on Snow). The massive $+0.0419$ jump from the Bug Reproduction is purely structural, not noise.

---

## Part B: Verification Tests (Diagnostics)
*These diagnostics have been integrated into the codebase.*

| ID | Test | Status |
| :--- | :--- | :--- |
| **V1** | **Protocol equivalence** | Active. Logging explicitly tracks Initial / Final (Online) / Final (Frozen). |
| **V2** | **Log `\|\|w_c\|\|` per class** | Active. Added to `evaluate_and_adapt` before loop execution. |
| **V3** | **Class Index Fix** | Verified. Head/Tail groupings are correctly mapped. |
| **V4** | **Firing Rate Logging** | Active. Logs true boolean firings instead of mean weights. |
| **V5** | **Argparse Plumbing** | Active. |
| **V6** | **Bug-reproduction** | Active (`--reproduce_bug`). Confirmed the $1/t$ annealing hypothesis. |
| **V7** | **Noise Floor** | Evaluated via `--seed`. |

---

## Part C: The Step-Size Schedule (S-Series)

The evidence proves that the model *requires* a deep, early adaptation phase followed by a decay, rather than a continuous constant equilibrium. We must implement this mathematically rather than relying on a tensor accumulation bug.

1. **S1 (Global LR Schedules):** We explicitly tested constant, $1/t$, and cosine decay globally. They **failed** entirely (freezing the network). Why? Because a global schedule decays learning rates equally across all classes, meaning rare classes are frozen before they ever see enough points to adapt.
2. **S2 (Formalized Bug Reproduction / Norm-Driven Dynamic LR):** The PyTorch `.data` bug was actually an emergent, mathematically perfect **Per-Class Inverse-Frequency Schedule**. 
   * When a majority class fires constantly, its un-normalized norm $M_c = \|w_c\|$ inflates rapidly via accumulation (e.g., $1.0 \rightarrow 30.0$). Its effective learning rate ($\alpha = \text{step} / M_c$) plummets to $1/30$, freezing and protecting it.
   * When a rare class rarely fires, its norm stays close to $1.0$. When it finally does fire, its learning rate is un-attenuated ($1/1.0 = 1.0\times$), allowing it to adapt rapidly.
   * **Implementation:** We now explicitly track $M_c$ as a scalar for each class, update it via $M_c \leftarrow M_c + \text{step}$, apply the spring if applicable, and formally divide $\text{step} / M_c$ before applying it to the true, normalized prototypes.
3. **S3 (Explicit Early Stopping):** Adapt for first N frames, then freeze. 
4. **S4 (Soft Rotation Barrier):** Replace the 40° hard cap with a soft exponential barrier.
5. **S5 (Spring, Properly Evaluated):** Apply $k$ to all classes every frame, evaluated against the V7 noise floor.

---

## Part D & E: Inter/Intra-Class Balancing
* **IC1 (Per-class rotation budget):** Equalize angular displacement, not weights.
* **IC2 (Split $\gamma$):** Apply inverse-frequency to direction weighting only, vs magnitude only.
* **IC4 (Confusion-aware weighting):** Weight class $c$ by how many points it is actively losing in the confusion matrix.
* **XC1 (Per-subcluster Dirichlet calibration):** Compute calibration per $K$-means subcluster.
* **XC2 (Equal-weight-per-subcluster aggregation):** The non-restrictive replacement for the Subcluster Ledger.
