# The Representation and Adaptation Characterization Framework

## Goal
We will characterize the fundamental properties of the learning system across the entire adaptation pipeline:
`Input -> Backbone -> HDC Embedding -> Prototype Dictionary -> Decision Layer`

Each stage is evaluated according to five complementary questions:
1. **Information** – What task-relevant information is present?
2. **Geometry** – How is that information organized?
3. **Dynamics** – How does adaptation modify it over time?
4. **Stability** – How robust is the representation?
5. **Recoverability** – Can lost information be recovered?

Once these properties are understood, architecture design will be derived directly from the empirical findings.

---

## Phase I — Representation Audit
**Scientific Question:** Where does useful information disappear?
This phase is independent of any TTA algorithm.

**Information Diagnostics (Predictive Decodability)**
For every representation, characterize decodability using progressively stronger probes:
- Linear probe
- Logistic regression
- k-NN probe
- Small MLP probe

Target Prediction Variables:
- Semantic label
- Prediction correctness
- Corruption type
- Hallucination detection

**Information Ceilings (Oracle Studies)**
Establish the upper bounds for future methods:
- Oracle nearest neighbor
- Oracle graph propagation
- Oracle prototype
- Oracle adaptation

**Phase I Findings (Fog Corruption)**
- **Information is fully preserved in HDC Space:** The 10,000D HDC embedding is perfectly linearly separable for semantic classes (Linear Probe Acc: 0.9999).
- **Prototype Projection mangles linear geometry:** Linear Probe accuracy drops to 82.41% in the 17D prototype similarity space (and the global argmax drops to 13%). However, an MLP probe still achieves 91.12%, proving the semantic information physically survives but is geometrically distorted.
- **Uncertainty is perfectly decodable:** Prediction correctness (hallucination detection) is decodable with ~1.0000 AUROC at *every* stage, including the 17D similarities. The model *possesses* the knowledge that it is hallucinating, but the standard `max(logit)` metric fails to expose it.

**Deliverable:** A definitive mapping of which representation layers actually contain exploitable information. (Completed: HDC Space contains pristine linearly-separable information).

---

## Phase II — Geometry Characterization
**Scientific Question:** How is the representation organized?
This phase measures the geometry of the feature space.

**Global Geometry**
- Effective rank / Participation ratio / Intrinsic dimension
- Cluster covariance / Between-class overlap

**Local Geometry**
- Neighborhood purity
- Neighborhood overlap & Neighborhood evolution
- Neighborhood consensus
- Local density & Local outlier factor

**Geometry Preservation (Transformation Tracking)**
Track the topology across `Input -> Backbone -> HDC Projection -> Prototype` using:
- CKA (Centered Kernel Alignment) / CCA
- Neighborhood rank preservation (Spearman correlation)
- Trustworthiness & Continuity

**Phase II Findings (Fog Corruption)**
- **Global Dimensionality Collapse:** The effective rank (Participation Ratio) of the 10,000D HDC space is 13.3 dimensions, closely tracking the 17 semantic classes. After the prototype projection, the effective rank collapses to a devastating **2.8 dimensions**. The global geometric structure is crushed.
- **Topological Scrambling:** While global linear alignment (CKA) remains at 0.93, the local neighborhood topology is scrambled. The Spearman rank correlation of the 50 nearest neighbors between the HDC space and the Prototype space is only **0.3892**. 
- **Verdict:** The prototype projection severely destroys local geometric topology. Your closest neighbor in the pristine HDC space is arbitrarily shuffled when squashed into the 17D space. This mathematically guarantees that distance-based logic (like confidence margin gating) will fail in the 17D space, as the relative distances no longer reflect the true geometry of the data.

**Deliverable:** Identification of which geometric structures survive each projection and transformation. (Completed: Local ranking topology is destroyed by prototype projection).

---

## Phase III — Adaptation Dynamics
**Scientific Question:** Why does adaptation fail?
This phase models adaptation as a dynamical system.

**Prototype Dynamics**
For every prototype, record throughout adaptation:
- Position, Velocity, Acceleration
- Angular velocity, Angular acceleration

**Error Propagation (The Confirmation Bias Graph)**
Construct the dynamical chain: `Prediction -> Prototype update -> Future prediction -> Future update`.
Estimate the **reproduction number** of a hallucination to quantify confirmation bias.

**Influence Functions**
Estimate how much one sample changes the prototypes, logits, and neighboring predictions to measure the mathematical influence of specific points.

**Phase III Findings (Fog Corruption)**
- **Massive Confirmation Bias:** In the very first frame of adaptation, the system generates over **~91,000 Confident Hallucinations** (points with $>0.9$ confidence but incorrect labels). Since there are only ~130k points per frame, this implies nearly 70% of the point cloud is acting as a destructive gradient signal.
- **Instantaneous Drift:** Because these 91,000 points are treated identically to true positives, they forcefully drag the prototypes away from their ground-truth locations in a single frame (Prototype Velocity Spike: 0.2355). 
- **Verdict:** Adaptation fails as a dynamical system because the initial representation (in the 17D prototype space) is so geometrically distorted that it confidently triggers massive pseudo-label errors. The adaptation loop possesses no geometric checks to detect these outlier gradients, resulting in instantaneous confirmation-bias collapse.

**Deliverable:** A dynamic understanding of which samples and interactions dominate adaptation collapse. (Completed: Collapse is driven by a massive volume of highly-confident initial hallucinations).

---

## Phase IV — Stability Analysis
**Scientific Question:** What perturbations can the representation tolerate?

**Memory Capacity Scaling**
Sweep memory sizes (50, 100, 250, 500, 1k, 2.5k, 5k, 10k, 50k) and measure segmentation performance, hallucination detection, and adaptation quality to find saturation thresholds.

**Prototype Stability (Perturbation)**
Artificially perturb prototypes and measure recovery, divergence, and collapse thresholds.

**Temporal Stability**
Measure representation drift over sequences of 1, 5, 20, and 100 frames.

**Deliverable:** The precise stability margins and bounds of the representation.

---

## Phase V — Recoverability
**Scientific Question:** If information disappears, is it fundamentally gone?

**Representation Recovery (Inverse Problems)**
- Can prototype similarities reconstruct the HDC embedding?
- Can HDC reconstruct backbone features?

**Geometry Recovery**
- Can nearest-neighbor structure be reconstructed after projection?

**Oracle Recovery**
- Assume perfect graph, memory, contrastive, or diffusion methods. How much improvement is theoretically recoverable?

**Deliverable:** Proof of whether the informational bottlenecks are reversible or strictly destructive.

---

## Phase VI — Architecture Decision Matrix
Turn the diagnostics into an explicit design methodology.

| Observation | Architecture Implication |
| :--- | :--- |
| Information preserved before prototype projection | Operate in HDC space |
| Local geometry preserved | Memory banks / Graph methods |
| Global geometry preserved | Prototype refinement |
| Neighborhoods collapse | Contrastive relearning |
| Projection destroys geometry | Avoid centroid-based methods |
| Information unrecoverable after projection | Adapt before prototype layer |
| Strong temporal consistency | Temporal memory |
| Low memory saturation | Compact memory bank |
| Large oracle graph gain | Graph propagation |
| Large oracle diffusion gain | Input alignment |

---

## Experimental Protocol & Complexity
Every experiment must follow this unified protocol:

**1. Environments:** Fog, Wet Ground, Crosstalk, Snow, Beam Missing.
**2. Evaluation States:** Frozen model, Online adaptation, Oracle adaptation.
**3. Measured Metrics:** Representation, Geometry, Dynamics, Stability, Recoverability.

**Complexity Characterization**
To ensure the final selected architecture is viable for real-time robotic deployment, explicitly characterize:
- Memory complexity (MB/GB)
- Runtime / Latency / FLOPs
- Adaptation time
- Energy & Storage
