## Background

This project's biggest wins so far have come from training the feature extractor to be robust, so the natural next step is to focus there directly (which also lets us get away with cheap active-learning-style sampling for the hard conditions, instead of test-time adaptation). The two most closely related prior works are DGLSS and DGLSS++, which we attempted to adapt as trainers and which both caused the HDC model to collapse. This section states the problem formally, gives the exact losses of both papers and of our method, and documents how our implementation (`modules/DGLSS.py`, the `supcon_vib_dglss` / `supcon_vib_dglsspp` / `supcon_vib` methods in `modules/gen_trainers.py`) corresponds to those equations.

### Problem Setting

We train a LiDAR semantic segmentation model on a single labeled source domain and evaluate on unseen domains. Both papers model the domain gap as a combination of two factors: sensor-sparsity differences and scene-distribution differences. A scan is a range-projection tensor $X \in \mathbb{R}^{5 \times H \times W}$ (depth, xyz, remission) with a label map $Y \in \{0,\dots,C\}^{H \times W}$. The network is an encoder-decoder $\Phi$ producing a feature volume $F = \Phi_{\text{enc}}(X)$ and per-pixel predictions $\hat{y} = \Phi_{\text{dec}}(F)$.

Our setting specializes this in two ways.

1. The corruption regime is SemanticKITTI-C (fog, crosstalk, snow, wet-ground, incomplete-echo, beam-missing, motion-blur, cross-sensor), and the two conditions that destroy the representation are fog and crosstalk. The deployment path is the HDC decoder: a random binary projection $W \in \{\pm 1\}^{128 \times 10000}$ with $W_{ij} \sim 2\,\mathrm{Bernoulli}(0.5) - 1$, sign binarization $B = \operatorname{sign}(z W) \in \{\pm 1\}^{10000}$ of the 128D bottleneck feature $z$, class prototypes $P_c$ built as the mean binarized code of clean points of class $c$, and decode $\hat{c} = \arg\max_c \cos(B, P_c)$.
2. The representation of interest is therefore the 128D bottleneck $z$ (the exact input to the HDC projection), not the full encoder volume. All methods share the same SENet-to-128D backbone and the same training data and budget, and differ in the representation-consistency loss. The DGLSS arms are VIB-free (matching the papers); the $vib$ arm isolates VIB so its isotropy contribution is measured, not assumed. This is what makes the comparison controlled.

Notation used throughout: $F_i^s, F_i^a$ are the source and augmented feature volumes of scan $i$; $F_{i,p}$ / $F_{i,n}$ are the paired / unpaired voxel subsets (paired = occupied in both views); $Z_i \in \mathbb{R}^{C \times l}$ is the class-prototype matrix of scan $i$ with rows $z_{i,c}$; $\| \cdot \|_1$ is the entrywise $L_1$ norm.

### DGLSS

DGLSS (Kim, Kang, Oh, Yoon; CVPR 2023) is the first single-source domain generalization method for LiDAR semantic segmentation. It trains on a dense source (SemanticKITTI, 64-beam) and targets the two domain gaps above with two constraints, SIFC and SCC.

**Sparse augmentation.** At every iteration the source scan is randomly subsampled by dropping whole beam rows (dense-to-sparse), producing the augmented view $X_i^a$ with the same label map at the surviving positions.

**Sparsity Invariant Feature Consistency (SIFC).** Aligns the internal features of the two views. Paired positions are matched directly; unpaired source positions (occupied in the source but dropped in the augmented view) are matched to an affinity-filtered aggregation of augmented features. The aggregation weight for a neighbor $j$ of an unpaired source voxel with feature $f^s$ and coordinates $x^s$ is

$$
w_j = \frac{1}{\|x^s - x_j^s\|_2} \cdot \mathbb{1}\left[\frac{\langle f^s, f_j^s \rangle}{\|f^s\|\,\|f_j^s\|} \ge \tau\right], \qquad
f^{\text{agg}} = \frac{\sum_j w_j f_j^a}{\sum_j w_j},
$$

i.e. the affinity is computed in the source view and gated by the cosine threshold $\tau$, then combined with an inverse-spatial-distance falloff. The loss is

$$
L_{\text{SIFC}} = \frac{1}{N}\sum_{i=1}^{N}\Big( \|F_{i,p}^s - F_{i,p}^a\|_1 + \|F_{i,n}^s - F_{i,\text{agg}}^a\|_1 \Big).
$$

**Semantic Correlation Consistency (SCC).** Builds a per-scan class-prototype matrix from the metric-learner embedding $\Psi$,

$$
z_{i,c} = \frac{\sum_j \mathbb{1}[\tilde{y}_{i,j} = c]\, \Psi(\Phi_{\text{dec}}(F_i))_j}{\sum_j \mathbb{1}[\tilde{y}_{i,j} = c]},
$$

and constrains the class-correlation matrices to be equal across all pairs of scans in the batch,

$$
L_{\text{SCC}} = \frac{1}{L}\sum_{i} \sum_{j \neq i} \big( Z_i Z_i^T - Z_j Z_j^T \big),
$$

where $L$ is the number of valid pairs and only classes with nonzero voxel counts are included. The intuition is that inter-class relations (e.g. cars correlate with trucks and with road) are domain-invariant.

**Total loss.**

$$
L = L_{\text{sem}}^s + L_{\text{sem}}^a + \lambda_1 L_{\text{SIFC}} + \lambda_2 L_{\text{SCC}},
$$

with weighted cross-entropy for the semantic terms. Evaluation is standardized over SemanticKITTI, nuScenes-lidarseg, Waymo, and SemanticPOSS with 10 common classes.

### DGLSS++

DGLSS++ (Kim et al., TPAMI 2026) extends DGLSS to both generalization directions and refines both constraints.

**Bidirectional augmentation.** Dense-to-sparse uses the same beam subsampling. Sparse-to-dense aggregates consecutive scans into a dense scan, then subsamples beams from it.

**Generalized Masked SIFC (GMSIFC).** Extends SIFC to the case where either view can be sparser (so unpaired voxels can exist in both directions) and adds a masking strategy. The symmetric extension is

$$
L_{\text{GMSIFC}} = \frac{1}{N}\sum_{i}\Big( \|F_{i,p}^s - F_{i,p}^a\|_1 + \|F_{i,n}^s - F_{i,\text{agg}}^a\|_1 + \|F_{i,\text{agg}}^s - F_{i,n}^a\|_1 \Big),
$$

where $F_{i,\text{agg}}^s$ is the source-side aggregation obtained symmetrically. The mask excludes voxel features that are "semantically mixed": because multiple voxels map to one sparse feature, voxels of different classes can collapse onto the same feature, and aligning those against the consistency loss conflicts with the segmentation loss. The paper proves (Proposition 1: sparsification inflates class-proportion variance and the probability of majority flips, especially near class boundaries; Proposition 2: masking those voxels increases the expected margin gain) and implements the mask by keeping only features whose local neighborhood maps to a single class.

**Localized SCC (LSCC).** Replaces the scene-global correlation with local-region correlation, motivated by the fact that most inter-class interactions are local and that local point densities are more uniform. Each scan is partitioned into a uniform spatial grid; per-cell prototypes $Z_{i,j}$ are computed as in the SCC formula; and the loss combines all-pairs cell-correlation consistency with a per-scan contrastive term,

$$
L_{\text{LSCC}} = \frac{1}{L}\sum_{i,j} \sum_{(k,l) \neq (i,j)} \big( Z_{i,j} Z_{i,j}^T - Z_{k,l} Z_{k,l}^T \big)
- \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{M_i'} \log \frac{\sum_{k \in P(j)} \exp(\psi_i(j) \cdot \psi_i(k))}{\sum_{l \in A(j)} \exp(\psi_i(j) \cdot \psi_i(l))},
$$

where $\psi_i$ is the embedding of scan $i$, $P(j)$ the same-class indices of embedding $j$ (positives), and $A(j)$ all other indices (negatives). Note that LSCC therefore includes a within-scan InfoNCE term, not just correlation regularization.

**Total loss.** $L = L_{\text{sem}}^s + L_{\text{sem}}^a + \lambda_1 L_{\text{GMSIFC}} + \lambda_2 L_{\text{LSCC}}$.

### Our Robust HDC Method

Our method (`supcon_vib` family) follows the same high-level recipe: train on the source domain with corruption-simulating view augmentations plus a representation-consistency loss, so the 128D bottleneck stays robust and the HDC pathway survives.

**Augmentation.** The augmented view $X^a$ is a composition of generic physical degradations: beam dropout, anisotropic Gaussian depth jitter, random density decimation, and volumetric noise injection (fake returns into empty space). The hard-negative variant additionally builds an extreme view $X^e$ (mild view plus sparse wrong-beam injection).

**Base losses.** With the VIB reparameterization $z \sim \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$,

$$
L_{\text{sem}} = \frac{1}{2}\big( \text{CE}(\hat{y}, Y) + \text{CE}(\hat{y}^a, Y) \big), \qquad
L_{\text{KL}} = -\frac{1}{2}\sum_d \big( 1 + \log \sigma_d^2 - \mu_d^2 - \sigma_d^2 \big),
$$

the standard KL to $\mathcal{N}(0, I)$ which drags high-entropy corruption noise toward the origin.

**SupCon (the consistency mechanism).** Cross-view InfoNCE on the L2-normalized bottleneck:

$$
L_{\text{supcon}} = -\frac{1}{M}\sum_{i}\log \frac{\sum_{p \in P(i)} \exp(\tilde{z}_i \cdot \tilde{z}_p / \tau)}{\sum_{j} \exp(\tilde{z}_i \cdot \tilde{z}_j / \tau)}, \qquad \tilde{z} = \frac{z}{\|z\|}, \ \tau = 0.1,
$$

where $P(i)$ are the same-class anchors (across the clean and augmented views) and the denominator runs over all other anchors. The negative term is a uniformity objective: it is minimized only when the classes are spread over the unit sphere, which is the property that keeps the HDC sign-projection informative (see Important Differences).

**Variants.** The fragile-class variant reweights the anchors: $L_{\text{supcon}}$ is computed with a per-anchor weight $w_a = \lambda_{\text{frag}}$ for the casualty classes $\{2,7,13,14,15\}$ and $1$ otherwise. The hard-negative variant adds a same-class repulsion of the extreme view from its clean centroid,

$$
L_{\text{repel}} = \operatorname{mean}\Big( \operatorname{relu}\big( \cos(\tilde{z}^e, c_{\text{lbl}}) - m \big) \Big), \qquad m = 0.5,
$$

carving a distinct artifact sub-cluster instead of absorbing crosstalk artifacts into the class centroid.

**Total loss.**

$$
L_{\text{ours}} = L_{\text{sem}} + 0.01\, L_{\text{KL}} + 0.1\, L_{\text{supcon}},
$$

with the hardneg repulsion added (weighted) in the hard-negative variant. The key result is that this is the first trainer family where the HDC decode stays intact under the hard conditions: the 128D space is highly separable under heavy fog (49.4% linear probe) and survives the random projection + sign binarization (49.0% / 47.8%).

### Implementation Correspondence

The losses above are implemented in `modules/DGLSS.py` (DGLSS / DGLSS++) and in `modules/gen_trainers.py` (the `supcon_vib` branch), wired so all three methods share the same SENet-to-128D + VIB architecture and differ only in the consistency loss.

| Paper equation | Implementation |
| --- | --- |
| DGLSS sparse aug | `get_dglss_view`: drops whole beam rows at rate $p \sim U(0.3, 0.7)$ per sample |
| SIFC Eq. 2 (aggregation) | `dglss_sifc_loss`: `weights = aff * inv_dist`, `aff = 1[cos >= tau]`, `f_agg = weights @ paired_other_view`, L1 against the unpaired feature (aggregation subsampled to <= 1500 paired / unpaired per sample to bound the quadratic cost) |
| SIFC Eq. 3 | paired L1 + unpaired-source-to-aggregated-aug L1; the symmetric term is empty for dense-to-sparse, so `masked=False` reduces to Eq. 3 |
| GMSIFC Eq. 8 | `dglss_sifc_loss(masked=True)`: both unpaired directions plus the single-class neighborhood mask |
| SCC Eq. 5 | `dglss_scc_loss(local=False)`: all-pairs correlation consistency over the pooled 2B scans, shared classes only |
| LSCC Eq. 17 | `dglss_scc_loss(local=True)`: all-pairs per-cell correlation consistency + the per-scan InfoNCE term |
| Our SupCon / VIB / hardneg | the `supcon_vib` branch in `gen_trainers.py` |

The three DGLSS arms are **VIB-free by construction** (they route through a dedicated
branch that uses the plain bottleneck and the model's direct logits; no reparameterization,
no KL, no `logvar_head`). The DGLSS / DGLSS++ papers have no variational bottleneck, so
this matches them. VIB is a `supcon_vib`-only component, isolated by the `vib` arm.

**Known deviations (adaptations), kept on purpose:**

1. The primary DGLSS arms apply the consistency losses to the 128D bottleneck $z$ (the HDC-input space), not the encoder volume $F = \Phi_{\text{enc}}(X)$, because the research question is about that space specifically and the comparison must share an attachment point. The standard-implementation arm (`supcon_vib_dglss_enc`) addresses the fidelity question directly: it applies SIFC to the deepest encoder stage $x_4$ (1/8 resolution) and SCC to the decoded bottleneck, matching the paper's split of SIFC on $\Phi_{\text{enc}}(F)$ and SCC on $\Psi(\Phi_{\text{dec}}(F))$. The two attachment points are compared in the isotropy table, so the pressure-point concern (does constraining the bottleneck overstate the anisotropy?) is an empirical question, not an assumption.
2. There is no separate metric learner $\Psi$; the 128D bottleneck plays that role.
3. The GMSIFC voxel-level mask is realized as a 3x3 single-class neighborhood filter on the projection labels (the projection-domain analog of "voxels mapped to the same feature from one class only").
4. DGLSS++ dense augmentation (multi-scan aggregation for sparse-to-dense) is not implemented, since our source is dense; GMSIFC therefore reduces to the masked dense-to-sparse case.
5. VIB is present only in `supcon_vib` (and the `vib` control arm), not in the DGLSS arms. This is deliberate: it removes the confound where VIB's per-dimension $\mathcal{N}(0, I)$ prior (itself a mild isotropy force) could be what makes the space isotropic. The `vib`-only arm isolates VIB's contribution directly, and `supcon_vib` vs `vib` isolates SupCon's.

### Important Differences

1. **Where the consistency is enforced.** DGLSS aligns internal voxel features matched by coordinate between the two views and constrains scene-wise (SCC) or local (LSCC) class-correlation matrices. Our consistency lives on the projected 128D bottleneck as a pairwise angular contrastive objective with label-defined positives and negatives; no voxel correspondence is needed and no global correlation matrix is matched.

2. **The anisotropic vs isotropic hypothesis (the collapse mechanism).** SCC / LSCC require only that the class-correlation matrix $Z Z^T$ be consistent across scans or cells. A correlation matrix can be made consistent by collapsing the embedding into a low-dimensional subspace: a few dominant correlated directions satisfy the constraint without any requirement on directional coverage, so nothing stops the learned space from being anisotropic and low-rank. (Note also that even the LSCC contrastive term is within-scan, so it cannot enforce cross-class angular uniformity either.) SupCon, by contrast, is explicitly a uniformity objective: the denominator in the InfoNCE ratio is minimized only when directions are used roughly uniformly, so the resulting space is angularly isotropic with no single direction dominating.

   This matters because the HDC decode is a random projection followed by sign binarization. If the 128D features concentrate along one dominant direction with a strong shared mean, that direction dominates every random projection and the sign codes become near-constant across points: most of the 10000 coordinates saturate (dead-coordinate fraction goes to 1) and the pairwise Hamming distance of the codes goes to 0, so the prototypes collapse. An angularly uniform space lets each random projection receive a balanced mix of points and yields informative binary codes. This is the learned analog of the "isotropic smoothing" observed when random projection improved prototype accuracy at decode time (8.2% to 31.7%): DGLSS leaves the decode to fix the anisotropy, whereas our method bakes the isotropy into the representation. The isotropy diagnostic (`robust_diagnostic/isotropy_diag.py`) measures exactly this: participation ratio / top-5 variance / condition number for the spectrum, and dead-coordinate fraction + Hamming distance on the raw features for the collapse.

3. **Augmentation scope.** DGLSS augments sparsity only (dense-to-sparse); DGLSS++ adds dense augmentation for the sparse-to-dense direction. Our augmentation targets the specific corruption failure modes of the SemanticKITTI-C regime (fog and crosstalk) with depth jitter and volumetric fake-return injection, rather than cross-sensor sparsity.

4. **Supervision form.** DGLSS / DGLSS++ use weighted CE plus unsupervised consistency terms (L1 feature alignment, correlation matrices, and a within-scan InfoNCE in LSCC). Our SupCon term is label-informed (per-point same-class positives and all-other negatives, essentially free at dense LiDAR scale), paired with the VIB magnitude bottleneck and CE. The consistency is cross-view and pairwise-angular, not a correlation-matrix match.

5. **The evidence.** DGLSS / DGLSS++ as trainers collapsed our HDC decode, consistent with an anisotropic embedding saturating the sign-projection pathway. The `supcon_vib` family was the first configuration where the encoder robustness transferred to the HDC decode under fog and crosstalk, which is the empirical difference this project is built on.
