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

### Measured isotropy of the three frameworks

The measured results of the isotropy comparison (tables for clean-space isotropy
and the corrupted conditions, plus the interpretation) are tracked in
[`docs/robust_iterations.md`](robust_iterations.md), Iteration 1. In brief: the
plain DGLSS shows the collapse mechanism firing (10.2% dead HDC coordinates vs
0.8% for ours), and the dead fraction tracks mean-dominance, not rank, exactly as
the theoretical analysis predicts; but DGLSS++ decodes best on clean, so the
"harmful for HDC" claim is only partially confirmed, and the difference shows in
the binarized decode, not the continuous representation.

3. **Augmentation scope.** DGLSS augments sparsity only (dense-to-sparse); DGLSS++ adds dense augmentation for the sparse-to-dense direction. Our augmentation targets the specific corruption failure modes of the SemanticKITTI-C regime (fog and crosstalk) with depth jitter and volumetric fake-return injection, rather than cross-sensor sparsity.

4. **Supervision form.** DGLSS / DGLSS++ use weighted CE plus unsupervised consistency terms (L1 feature alignment, correlation matrices, and a within-scan InfoNCE in LSCC). Our SupCon term is label-informed (per-point same-class positives and all-other negatives, essentially free at dense LiDAR scale), paired with the VIB magnitude bottleneck and CE. The consistency is cross-view and pairwise-angular, not a correlation-matrix match.

5. **The evidence.** DGLSS / DGLSS++ as trainers collapsed our HDC decode, consistent with an anisotropic embedding saturating the sign-projection pathway. The `supcon_vib` family was the first configuration where the encoder robustness transferred to the HDC decode under fog and crosstalk, which is the empirical difference this project is built on.

## Theoretical Analysis

### HDC needs Isotropic Feature Spaces

The HDC pathway is a random projection followed by sign binarization. The claim
is that this pathway is destroyed precisely when the feature space is anisotropic
in a specific way: low-rank, with a strong shared mean direction. The theorem
below makes this rigorous, and the measured dead-coordinate fraction is the
empirical witness of the mechanism.

**Theorem 1 (sign-projection saturation).** Let $z_1, \dots, z_n \in \mathbb{R}^d$
be feature vectors with shared mean $\mu = \mathbb{E}[z_i]$ and covariance
$\Sigma = \mathbb{E}[(z_i - \mu)(z_i - \mu)^T]$. Let $W \in \{\pm 1\}^{d \times D}$
have independent Rademacher entries,
$\mathbb{P}(W_{jk} = +1) = \mathbb{P}(W_{jk} = -1) = 1/2$, the HDC random
projection. The binarized code of point $i$ is $b_i = \operatorname{sign}(W^T z_i)
\in \{\pm 1\}^D$. Call a coordinate $j$ **dead** if all $n$ points share the same
sign there, $b_{1j} = \dots = b_{nj}$, and let
$\operatorname{dead} = \frac{1}{D}\sum_{j=1}^{D} \mathbf{1}[\text{coordinate } j
\text{ is dead}]$. Then

$$
\mathbb{E}[\operatorname{dead}]
\;\ge\; \frac{1}{12}\Big(1 - \frac{2 n d\, \lambda_{\max}(\Sigma)}{\|\mu\|^2}\Big)_+,
$$

where $\lambda_{\max}(\Sigma)$ is the largest eigenvalue of $\Sigma$,
$(x)_+ = \max(x, 0)$, and the expectation is over $W$ and the $z_i$.

**Proof.** Write $z_i = \mu + \varepsilon_i$ with $\mathbb{E}[\varepsilon_i] = 0$
and $\operatorname{Cov}(\varepsilon_i) = \Sigma$. Fix one row $w \in \{\pm 1\}^d$
of $W$; it indexes one coordinate $j$ of the code, with
$w^T z_i = w^T \mu + w^T \varepsilon_i$.

*Step 1 (Chebyshev).* For each $i$,
$\operatorname{Var}(w^T \varepsilon_i) = w^T \Sigma w$, so

$$
\mathbb{P}\big(|w^T \varepsilon_i| \ge |w^T \mu|\big) \le \frac{w^T \Sigma w}{(w^T \mu)^2}.
$$

*Step 2 (union bound).* Over the $n$ points,

$$
\mathbb{P}\Big(\max_i |w^T \varepsilon_i| \ge |w^T \mu|\Big)
\le \sum_{i=1}^{n} \mathbb{P}\big(|w^T \varepsilon_i| \ge |w^T \mu|\big)
\le \frac{n\, w^T \Sigma w}{(w^T \mu)^2}.
$$

*Step 3 (dead coordinate).* If $|w^T \varepsilon_i| < |w^T \mu|$ for every $i$,
then $\operatorname{sign}(w^T z_i) = \operatorname{sign}(w^T \mu)$ for every $i$,
so coordinate $j$ is dead. Hence the probability (over $\varepsilon$) that
coordinate $j$ is dead is at least
$\big(1 - n\, w^T\Sigma w / (w^T\mu)^2\big)_+$.

*Step 4 (average over the projection).* Taking the expectation over the row $w$,

$$
\mathbb{E}_w[\operatorname{dead}] \;\ge\;
\mathbb{E}_w\Big[\Big(1 - \frac{n\, w^T\Sigma w}{(w^T\mu)^2}\Big)_+\Big].
$$

*Step 5 (Paley-Zygmund on the mean term).* Let $Z = (w^T\mu)^2$. Since the entries
of $w$ are independent Rademacher, $\mathbb{E}_w[Z] = \|\mu\|^2$ and
$\mathbb{E}_w[Z^2] = \mathbb{E}_w[(\sum_k w_k \mu_k)^4] \le 3\|\mu\|^4$ (the
fourth-moment bound for Rademacher sums). By the Paley-Zygmund inequality,

$$
\mathbb{P}_w\Big(Z \ge \tfrac12 \mathbb{E}_w[Z]\Big)
\;\ge\; \Big(1 - \tfrac12\Big)^2 \frac{\mathbb{E}_w[Z]^2}{\mathbb{E}_w[Z^2]}
\;\ge\; \frac14 \cdot \frac{\|\mu\|^4}{3\|\mu\|^4} = \frac{1}{12}.
$$

*Step 6 (spectral bound on the noise term).* On the event
$\{Z \ge \|\mu\|^2/2\}$, since $\|w\|^2 = d$,

$$
w^T\Sigma w \le \lambda_{\max}(\Sigma)\, \|w\|^2 = d\,\lambda_{\max}(\Sigma),
\qquad\text{so}\qquad
\frac{n\, w^T\Sigma w}{(w^T\mu)^2} \le \frac{2 n d\, \lambda_{\max}(\Sigma)}{\|\mu\|^2}.
$$

*Step 7 (combine).* Restricting the expectation in Step 4 to the measure-at-least
$1/12$ event from Step 5 and applying Step 6 on it,

$$
\mathbb{E}[\operatorname{dead}] \;\ge\;
\frac{1}{12}\Big(1 - \frac{2 n d\, \lambda_{\max}(\Sigma)}{\|\mu\|^2}\Big)_+.
\qquad\square
$$

**Corollary 1.** If $\|\mu\|^2 > 2nd\,\lambda_{\max}(\Sigma)$ then
$\mathbb{E}[\operatorname{dead}] > 0$; if
$\|\mu\|^2 \ge 4nd\,\lambda_{\max}(\Sigma)$ then
$\mathbb{E}[\operatorname{dead}] \ge 1/24$. In the regime where the shared mean
dominates the largest variance direction by a factor polynomial in $nd$, a
constant fraction (and in practice a near-total fraction) of the sign coordinates
are constant across the sample, so the binary codes collapse.

**Remark 1 (what the hypothesis of the theorem is).** Two conditions make the
right-hand side large, and they are exactly the two signatures of an anisotropic,
mean-dominated space: (i) a strong shared mean $\mu$ (all classes offset from the
origin by a common direction), and (ii) a covariance $\Sigma$ with a large spectral
concentration, $\lambda_{\max}(\Sigma)/\mathrm{Tr}(\Sigma)$ close to $1$, i.e. a
low effective rank. An angularly isotropic space has neither: $\mu \approx 0$ makes
the bound vacuous, and $\lambda_{\max}/\mathrm{Tr} \approx 1/d$ keeps it small. The
participation ratio, $\mathrm{PR} = \mathrm{Tr}(\Sigma)^2 / \|\Sigma\|_F^2$, is the
convenient measured proxy: $\mathrm{PR} \approx d$ for isotropic, and $\mathrm{PR}
\approx 1$ for the rank-1 collapse. The constant $1/12$ is a conservative
Paley-Zygmund bound, so the theorem is a sufficient condition, not an exact
prediction; the empirically measured dead-coordinate fraction is the ground truth
and moves toward $1$ in the collapse regime.

### Common Mechanisms in Previous Attempts (beyond just DGLSS) cause Anisotropy

The point of naming the mechanism abstractly is that the failure is not specific
to DGLSS. Any training objective whose constraints can be satisfied by a
low-rank embedding with a strong shared mean will tend to produce such an
embedding, because the constraints do not penalize directional concentration, and
Theorem 1 then applies. Three abstract families cover the harmful cases:

1. **Alignment / matching losses.** Feature-consistency constraints between two
   views (L1 matching, view-consistency, distillation, co-training) require the
   representations to be similar but never constrain where they live. Worse, L1
   alignment is a shrinkage toward the pairwise average: it systematically reduces
   variance in the unconstrained directions, which pushes toward rank collapse.
   SIFC and GMSIFC are instances.
2. **Correlation / Gram-consistency losses.** Objectives that make the
   class-correlation matrix $Z Z^T$ consistent across scenes (SCC, LSCC, and any
   "align the correlation structure" variant) constrain the Gram matrix, not the
   spectral distribution. A rank-$k$ embedding has a Gram matrix of rank at most
   $k$, and consistency of the Gram matrix imposes no requirement on its spread;
   in fact the low-rank solutions are the ones with the fewest parameters to keep
   consistent, so the constraint is biased toward collapse rather than opposed to
   it.
3. **Any objective without an angular-uniformity term.** Nothing that constrains
   only magnitudes, pairwise relations, or consistency prevents every class from
   concentrating in a low-dimensional subspace with a strong shared mean. The
   feature distribution is free to develop the dominant direction that Theorem 1
   flags.

Families 1 and 2 are not two separate failures; they are instances of one
abstract structure. Both are **Gram-consistency losses**: functions of the
pairwise inner-product structure of the features, minimized by making that
structure degenerate (features that coincide, or Gram matrices that are equal),
and hence blind to the angular and spectral distribution that the HDC pathway
depends on.

**Theorem 4 (Gram-consistency losses are isotropy-blind).** Let the alignment loss
be $\mathcal{L}_{\mathrm{align}}(f, g) = \mathbb{E}\,\varphi(\|f(x) - g(x)\|)$
for any non-negative increasing function $\varphi$ with $\varphi(0) = 0$ and
$\varphi(t) > 0$ for $t > 0$ (this covers both the $L_1$ matching of SIFC and
GMSIFC and the $L_2$ form), and let the correlation-consistency loss be
$\mathcal{L}_{\mathrm{corr}}(\{Z_s\}) = \frac{1}{L}\sum_{s \ne s'} \| Z_s Z_s^T -
Z_{s'} Z_{s'}^T \|_F^2$ over scene prototype matrices $Z_s \in \mathbb{R}^{C
\times d}$ (this is exactly the SCC loss, and LSCC is the same loss restricted to
local cells). Then:

(i) **Rank-1 global minimizers exist.** For alignment,
$\mathcal{L}_{\mathrm{align}} \ge 0$ and $\mathcal{L}_{\mathrm{align}} = 0$ iff
$f \equiv g$ almost surely, because $\varphi$ is non-negative and positive away
from zero. The minimizer set $\{f = g\}$ is closed under any common map applied to
both encoders, in particular under a rank-1 orthogonal projection $P = v v^T$,
so rank-1 minimizers exist. For correlation, any family with a common Gram matrix
is a global minimizer, and a rank-1 Gram matrix is realizable by setting
$Z_s = \sqrt{\alpha}\, u e_1^T$ for all $s$, so rank-1 minimizers exist.

(ii) **Rotation invariance (the inner-product forms).**
$\mathcal{L}_{\mathrm{align}}(U f, U g) = \mathcal{L}_{\mathrm{align}}(f, g)$
for the $L_2$ form and
$\mathcal{L}_{\mathrm{corr}}(\{Z_s U\}) = \mathcal{L}_{\mathrm{corr}}(\{Z_s\})$
for every orthogonal $U$, because $\|U x\| = \|x\|$ and
$(Z_s U)(Z_s U)^T = Z_s Z_s^T$. The losses cannot distinguish the feature space
from any rotation of it, so they cannot measure the angular distribution or the
spectral concentration.

**Proof.** (i) The minimizer characterizations follow from $\varphi$ being
non-negative and positive away from zero, and from the two loss functions being
zero exactly on the stated sets. The rank-1 construction is explicit. Note that
the proof does not use the specific form of $\varphi$: it uses only "minimized
iff the matched features coincide" and the closure of that set under common maps,
both of which hold for the $L_1$ and $L_2$ instances alike. (ii) Direct from the
norm and Gram identities above. $\square$

**Remark 3 (why DGLSS applies in full, and which choices are cosmetic).** The two
DGLSS constraints are instances of the theorem with no approximation. SCC and
LSCC are the correlation-consistency form verbatim. SIFC and GMSIFC are $L_1$
alignment losses, covered by part (i); the fact that the theorem is stated with
the general $\varphi$ rather than $L_1$ in particular is a presentation choice,
analogous to assuming a uniform mean in other proofs: it simplifies the rotation-
invariance statement (ii) but does not enter the proof of (i), which is the
essential content. The $L_1$ form is additionally covered by a second route: it
is a shrinkage toward the pairwise average, which reduces variance in the
unconstrained directions and pushes the embedding onto a lower-rank set. Either
way, the conclusions of the theorem hold for the exact DGLSS losses, not for a
simplified proxy.

**The contrastive uniformity term is the counter-example.** The SupCon uniformity
term is also a function of pairwise inner products, but it is not a
Gram-consistency loss: by Wang and Isola (2020, Proposition 1), the uniform
measure is the *unique* minimizer of the Gaussian-potential uniformity loss. Its
minimizer is the maximally non-degenerate angular distribution rather than a
degenerate manifold of low-rank embeddings, which is precisely the ingredient
that families 1 and 2 lack.

The common thread is the absence of any term that penalizes directional
concentration. That single missing ingredient is what all three families share,
and it is the reason they are all at risk regardless of the details of the
consistency loss.

### Our Method causes Isotropy

The SupCon objective is the one place the missing ingredient appears. Wang and
Isola (2020) prove that, as the number of negative samples grows, the contrastive
loss splits into an **alignment** term (positive pairs attracted) and a
**uniformity** term (the log-sum-exp over the denominator), and that the
uniformity term is minimized exactly by encoders whose features are uniform on the
unit hypersphere. In the language of that paper: the uniform measure is the unique
minimizer of the Gaussian-potential uniformity loss, and the finite minimizers
converge weak* to it. So "the normalized features are approximately uniform" is
not an assumption about our training; it is the known convergence target of the
SupCon uniformity term (up to the realizability caveats that paper states). What
remains for us to prove is the geometric step: what approximate uniformity
guarantees for the HDC sign-projection.

**Lemma 1 (uniformity is the convergence target of SupCon; Wang and Isola 2020).**
For the contrastive loss with temperature $\tau > 0$, as the number of negative
samples $M \to \infty$,
$\lim_{M\to\infty} [\mathcal{L}_{\mathrm{contrastive}}(f;\tau,M) - \log M] =
\tfrac{1}{\tau}\mathbb{E}_{p_{\mathrm{pos}}} [f(x)^T f(y)] +
\mathbb{E}_{p_{\mathrm{data}}}\log \mathbb{E}_{p_{\mathrm{data}}}
[e^{f(x)^T f(x^-)/\tau}]$. The first term is minimized iff $f$ is perfectly
aligned; if perfectly uniform encoders exist, they are the exact minimizers of the
second term (Theorem 1 of that paper). The uniform measure is the unique minimizer
of the associated Gaussian-potential uniformity loss (their Proposition 1), and
the $N$-point minimizers converge weak* to it (their Proposition 2).

**Theorem 2 (uniformity keeps the codes alive).** Let $u_1, \dots, u_n \in S^{d-1}$
be drawn from a distribution within total-variation distance $\delta$ of
the uniform measure on the sphere, and let $b_i = \operatorname{sign}(W^T u_i)$
for a Rademacher projection $W$. Then with probability at least $1 - \gamma$ over
the sample, every coordinate $j$ satisfies

$$
\Big| f_j - \tfrac12 \Big| \;\le\; C\Big(\delta + \sqrt{\tfrac{\log(D/\gamma)}{n}}\Big),
$$

where $f_j$ is the empirical fraction of $+1$ signs in coordinate $j$ and $C$ is
an absolute constant, and the expected pairwise Hamming distance between two codes
is at least $\tfrac12 - C(\delta + \sqrt{\log(1/\gamma)/n})$.

*Proof (the geometric step).* (1) By rotational invariance, the law of $X = w^T
U$ for uniform $U \in S^{d-1}$ is the same for every unit vector $w$; since $U$
and $-U$ are equidistributed, $\mathbb{P}(X > 0) = \mathbb{P}(X < 0) = 1/2$. (2)
The empirical fraction concentrates: Hoeffding over the $n$ iid signs gives
$|f_j - 1/2| = O(\sqrt{\log(D/\gamma)/n})$ uniformly over $D$ coordinates. (3)
$\delta$-closeness to uniform perturbs every such statement by $O(\delta)$. (4)
For two independent uniform points,
$\mathbb{P}(\operatorname{sign}(w^TU) \ne \operatorname{sign}(w^TU')) =
2 \cdot \tfrac12 \cdot \tfrac12 = \tfrac12$, giving the Hamming bound. Hence
uniformity produces balanced, maximally informative binary codes and a dead
fraction of essentially zero, the exact opposite of Theorem 1's regime.

The structure is therefore: Lemma 1 (cited) supplies the premise that Theorem 2
needs, and Theorem 2 (proved here) is only the geometric statement that uniform
features cannot saturate the HDC sign-projection. The $\delta$ in Theorem 2 is the
convergence residual of the SupCon uniformity term, not an arbitrary assumption.

**Theorem 3 (VIB removes the sign-projection saturation regime).** Let the 128D
bottleneck be reparameterized as $z = \mu(x) + \sigma(x) \odot \varepsilon$ with
$\varepsilon \sim \mathcal{N}(0, I_d)$, and let the training objective include the
VIB term
$\lambda \cdot \mathrm{KL}\big[q(z|x) \,\|\, \mathcal{N}(0, I_d)\big] =
\frac{\lambda}{2}\sum_{d}\big(\sigma_d(x)^2 + \mu_d(x)^2 - 1 - \log\sigma_d(x)^2\big)$
with $\lambda > 0$. Then:

(i) The KL's gradient with respect to $\mu_d(x)$ is $\lambda\,\mu_d(x)$, and with
respect to $\sigma_d(x)^2$ it is $\frac{\lambda}{2}\big(1 - 1/\sigma_d(x)^2\big)$.
It is the unique term in the objective whose gradient drives the per-dimension
mean toward zero and the per-dimension variance toward one.

(ii) The KL is minimized exactly at $\mu(x) = 0$, $\sigma(x) = \mathbf{1}$, where
the marginal of $z$ satisfies $\mathbb{E}[z] = 0$ and
$\mathrm{Cov}(z) = I_d$.

(iii) At that point the features are outside the Theorem-1 regime: with
$\|\mathbb{E}[z]\| = 0$ there is no shared mean to dominate the projections, and
with $\mathrm{Cov}(z) = I_d$ the spectral concentration is
$\lambda_{\max}/\mathrm{Tr} = 1/d$. Substituting into the Theorem-1 bound makes
the dead-coordinate lower bound vacuous (zero), matching the healthy-code regime
of Theorem 2.

**Proof.** (i) Differentiating the KL term gives
$\partial_{\mu_d}\mathrm{KL} = \mu_d(x)$ and
$\partial_{\sigma_d^2}\mathrm{KL} = \tfrac12(1 - 1/\sigma_d(x)^2)$. The other
losses in the objective do not contain these exact per-dimension gradient terms.
(ii) The KL is convex in $\mu_d^2$ and in $\sigma_d^2 - 1 - \log\sigma_d^2$
(minimized at $\sigma_d^2 = 1$), so its unique minimum is $\mu = 0$, $\sigma =
\mathbf{1}$; then $\mathbb{E}[z] = \mathbb{E}[\mu(x)] = 0$ (since
$\mathbb{E}[\varepsilon] = 0$) and, by the law of total variance,
$\mathrm{Cov}(z) = \mathrm{Cov}(\mu(x)) + \mathbb{E}[\sigma(x)\sigma(x)^T] = I_d$.
(iii) Direct substitution into the Theorem-1 bound. $\square$

**Remark 2 (why the balance stays empirical).** Theorem 3 justifies VIB's presence
on principle: it is the regularizer that removes exactly the mean-dominated,
spectrally-concentrated structure that Theorem 1 shows is harmful to HDC. It does
not say anything about the correct weight or about preserving class separation.
VIB alone, unopposed, also crushes the class structure (the measured 5x
over-collapse of the Phase 17 run), so SupCon's angular uniformity must counteract
that over-shrinkage. The counteraction, and the right weight, is an empirical
question to be settled by the ablations (the VIB-only arm in the isotropy
diagnostic), not by the theorem.

**What separates our method from the common mechanism.** The contrastive
uniformity term is precisely the angular-uniformity ingredient that families 1-3
lack: it penalizes the directional concentration that Theorem 1 shows is harmful,
and it is the hypothesis under which Theorem 2 holds. The VIB term (Theorem 3)
adds a second, provably complementary defense: it removes the mean-dominated and
spectrally-concentrated structure that Theorem 1 flags, while SupCon provides the
angular spread that VIB's over-shrinkage would otherwise crush. Nothing in the
alignment or correlation-consistency families has either property, which is the
formal content of the isotropic-vs-anisotropic claim: our losses contain the
anti-concentration and anti-collapse terms, theirs do not.
