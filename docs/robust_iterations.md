# Robust Feature-Extractor Comparison: Iterations

This tracks the empirical iterations of the robust-encoder comparison (the
isotropic-vs-anisotropic question and the DGLSS / DGLSS++ / supcon_vib shootout).
The framework and its theoretical analysis live in `docs/robust_details.md`; this
document records what was measured, condition by condition.

Throughout, bold marks the best value in each row; for the mIoU / linear-probe /
Hamming columns higher is better, for the dead-fraction and mean-fraction columns
lower is better.

## Background: DGLSS, DGLSS++, and our variants

This section states the two source methods as they appear in the papers, and the
changes made in our variants, in the mathematical form a final paper would use.
Implementation-stability details (e.g., gradient clipping) are noted but are not
method changes.

### DGLSS (Kim, Kang, Oh, Yoon; CVPR 2023)

**Setting.** A LiDAR scan is projected to a range image $X \in \mathbb{R}^{5
\times H \times W}$ with label map $Y$. The network is $\Phi = \Phi_{\text{dec}}
\circ \Phi_{\text{enc}}$ with internal features $F = \Phi_{\text{enc}}(X)$, and a
metric learner $\Psi$ operates on the decoded features. DGLSS trains on a single
dense source domain (SemanticKITTI) and targets unseen domains that differ in
sparsity and scene distribution.

**Sparse augmentation.** Each iteration the source scan is subsampled by dropping
whole beam rows (dense-to-sparse), producing the augmented view $X^a$ with the
same label map at the surviving positions.

**Sparsity Invariant Feature Consistency (SIFC).** For each source voxel with
feature $f^s$ and coordinates $x^s$, an aggregated augmented feature is built from
its neighbors with affinity and inverse-distance weights,

$$
w_j = \frac{1}{\|x^s - x_j^s\|_2}\, \mathbb{1}\Big[
\frac{\langle f^s, f_j^s \rangle}{\|f^s\|\,\|f_j^s\|} \ge \tau \Big],
\qquad
f^{\text{agg}} = \frac{\sum_j w_j f_j^a}{\sum_j w_j},
$$

and the loss aligns the source and augmented internal features,

$$
L_{\text{SIFC}} = \frac{1}{N}\sum_{i=1}^{N}
\Big( \|F_{i,p}^s - F_{i,p}^a\|_1 + \|F_{i,n}^s - F_{i,\text{agg}}^a\|_1 \Big),
$$

where $F_{i,p}$ / $F_{i,n}$ are the paired / unpaired (in the augmented view)
subsets.

**Semantic Correlation Consistency (SCC).** Per-scan class prototypes from the
metric-learner embedding,

$$
z_{i,c} = \frac{\sum_j \mathbb{1}[\tilde{y}_{i,j} = c]\, \Psi(\Phi_{\text{dec}}(F_i))_j}
{\sum_j \mathbb{1}[\tilde{y}_{i,j} = c]},
$$

and the class-correlation matrices are constrained to be equal across scans,

$$
L_{\text{SCC}} = \frac{1}{L}\sum_{i}\sum_{j \ne i} \big( Z_i Z_i^T - Z_j Z_j^T \big).
$$

**Total.** $L = L_{\text{sem}}^s + L_{\text{sem}}^a + \lambda_1 L_{\text{SIFC}} + \lambda_2 L_{\text{SCC}}$, with weighted cross-entropy for the semantic terms.

### DGLSS++ (Kim et al.; TPAMI 2026)

**Bidirectional augmentation.** Dense-to-sparse uses the beam subsampling above;
sparse-to-dense aggregates consecutive scans into a dense scan before
subsampling.

**Generalized Masked SIFC (GMSIFC).** Extends SIFC to either direction and adds a
mask that excludes voxel features mapped from multiple inconsistent semantic
classes,

$$
L_{\text{GMSIFC}} = \frac{1}{N}\sum_{i}\Big( \|F_{i,p}^s - F_{i,p}^a\|_1
+ \|F_{i,n}^s - F_{i,\text{agg}}^a\|_1 + \|F_{i,\text{agg}}^s - F_{i,n}^a\|_1 \Big),
$$

with the aggregation applied symmetrically and the mask applied to all three terms.

**Localized SCC (LSCC).** Prototypes are computed per spatial cell $Z_{i,j}$, and
the loss adds all-pairs cell-correlation consistency to a per-scan contrastive
term,

$$
L_{\text{LSCC}} = \frac{1}{L}\sum_{i,j}\sum_{(k,l) \ne (i,j)}
\big( Z_{i,j} Z_{i,j}^T - Z_{k,l} Z_{k,l}^T \big)
- \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{M_i'}
\log \frac{\sum_{k \in P(j)} \exp(\psi_i(j) \cdot \psi_i(k))}
{\sum_{l \in A(j)} \exp(\psi_i(j) \cdot \psi_i(l))},
$$

where $P(j)$ / $A(j)$ are the same-class / all-other embedding indices of $j$.

**Total.** $L = L_{\text{sem}}^s + L_{\text{sem}}^a + \lambda_1 L_{\text{GMSIFC}} + \lambda_2 L_{\text{LSCC}}$.

### Our variants (the changes for the final paper)

Our implementations of the two methods are applied to the same SENet-to-128D
architecture as our reference method, trained on the same data and budget, and
differ from the papers in the following respects.

1. **Correlation on normalized prototypes.** The class prototypes in SCC and
   LSCC are computed on the normalized class means
   $p_{i,c} = \bar{z}_{i,c} / \|\bar{z}_{i,c}\|_2$, where $\bar{z}_{i,c}$ is the
   raw mean of the 128D bottleneck features of class $c$. The Gram entries are
   then cosine similarities in $[-1, 1]$, so the correlation-consistency loss is
   bounded and scale-free. The unnormalized form (raw means) scales as
   $\|z\|^4$ and diverges under VIB-free training; this normalization is the
   numerical fix and corresponds to evaluating the correlation on a unit
   embedding, as the paper's metric-learner setup implies.
2. **Attachment to the HDC-input space.** SIFC / GMSIFC and SCC / LSCC act on the
   128D bottleneck $z$ (the input to the HDC random projection), not on the
   encoder volume $\Phi_{\text{enc}}(X)$; the bottleneck plays the role of the
   metric-learner output $\Psi(\Phi_{\text{dec}}(X))$. All compared methods share
   this attachment point, so the comparison is controlled.
3. **GMSIFC mask realization.** The voxel-level "single semantic class" mask is
   realized as a $3 \times 3$ neighborhood purity filter on the projection labels:
   a position is kept iff its local neighborhood contains exactly one class. This
   is the projection-domain proxy for "voxels mapped from a single semantic
   class."
4. **VIB-free.** The DGLSS / DGLSS++ arms carry no VIB KL term; the papers have
   no variational bottleneck. (Our reference method includes VIB; the DGLSS arms
   deliberately do not, matching the papers.)
5. **Dense augmentation omitted.** The source is dense, so GMSIFC reduces to the
   masked dense-to-sparse form and the symmetric unpaired-augmented term is empty.
6. **Sparse augmentation.** Beam-row dropout at rate $p \sim U(0.3, 0.7)$,
   matching the papers' dense-to-sparse setting.

**Note (training stability, not a paper change).** The prototype normalization in
(1) is the numerical fix for the divergence described below; the raw-form SCC
reproduces the flatline of the earlier DGLSSTrainer. Empirically confirmed at
equal budget (4 epochs, 10% data): the raw form collapses to a 0.0 HDC decode on
clean, fog and crosstalk, while the normalized form trains and decodes (clean
0.317, fog 0.036, crosstalk 0.051). Any further measures, such as gradient
clipping or the subsampling of the SIFC affinity aggregation to bound cost, are
implementation details and do not change the loss.

**Theoretical framing.** The isotropy hypothesis, the dead-coordinate saturation
theorem, the cosine-ranking preservation result (why anisotropic spaces can still
decode in HDC), and the unified "Gram-consistency losses are isotropy-blind"
result are in `docs/robust_details.md`. The measurements below test these claims.

## Iteration 0: the diagnostic battery

One battery of diagnostics on the same three trained feature extractors
(`supcon_vib`, `supcon_vib_dglss`, `supcon_vib_dglsspp`, all at 12 epochs and 10%
data, saved to `robust_diagnostic/logs/<method>/`). Three views of the same
checkpoints: the isotropy of the clean and corrupted spaces (0.1), the full
8-condition HDC robustness sweep (0.2), and the labeled decoding ceiling (0.3).

### 0.1 Isotropy of the three frameworks

Evaluated on the 128D bottleneck: participation ratio (PR, effective rank of 128),
dead sign-coordinate fraction (the collapse mechanism), mean-fraction (how dominant
the shared mean direction is), code Hamming distance, and HDC prototype mIoU.

**Clean-space isotropy (the decisive comparison):**

| method | PR | dead-frac | mean-frac | Hamming | clean HDC mIoU | clean LP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib (ours) | 3.9 | 0.008 | **0.47** | 0.416 | 0.429 | **0.866** |
| supcon_vib_dglss | 4.1 | 0.102 | 0.68 | 0.340 | 0.389 | 0.853 |
| supcon_vib_dglsspp | 2.7 | **0.003** | 0.51 | **0.417** | **0.456** | 0.860 |

**Corrupted conditions preview (fog, crosstalk, snow; deadF = lower is better,
HDC mIoU and LP = higher is better):**

| method | fog deadF | fog HDC | fog LP | xtalk deadF | xtalk HDC | xtalk LP | snow HDC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib (ours) | 0.185 | 0.062 | 0.159 | **0.026** | 0.097 | 0.252 | 0.336 |
| supcon_vib_dglss | 0.257 | 0.056 | 0.184 | 0.079 | **0.099** | **0.315** | 0.329 |
| supcon_vib_dglsspp | **0.107** | **0.077** | **0.191** | **0.026** | 0.098 | 0.279 | **0.340** |

**What the isotropy view shows:**

1. **The collapse mechanism fires for DGLSS.** The plain DGLSS has a dead-coordinate
   fraction of 10.2% on clean vs 0.8% for ours and 0.3% for DGLSS++ (a 13-34x
   difference), and the lowest clean HDC mIoU (0.389 vs 0.429 for ours). This is
   the first direct confirmation that DGLSS saturates the HDC sign-projection more
   than our method does.
2. **The dead fraction tracks the mean-dominance, not the rank, exactly as Theorem
   1 of the theoretical analysis predicts.** DGLSS has the strongest shared mean
   (mean-fraction 0.68) and the highest dead fraction; DGLSS++ is the most low-rank
   (PR 2.7) yet has almost no dead coordinates (0.003) because it is not
   mean-dominated (0.51). The theorem's bound is vacuous without the shared mean,
   and the data follows that: low rank alone does not saturate, the combination of
   mean and low rank does.
3. **The story is not uniform.** DGLSS++ decodes the best of the three on clean
   (0.456 HDC mIoU) despite the lowest rank, so the claim "DGLSS / DGLSS++ are
   harmful for HDC" is only confirmed for the plain DGLSS in this run, and its harm
   is modest.

### 0.2 Full-condition robustness sweep

The three checkpoints evaluated on all 8 conditions, loaded without retraining
(`isotropy_diag.py --eval_only`). The headline metric is the HDC prototype mIoU per
condition; the dead-fraction is the mechanism reference.

**HDC prototype mIoU per condition (higher is better):**

| condition | supcon_vib (ours) | supcon_vib_dglss | supcon_vib_dglsspp |
| :--- | :--- | :--- | :--- |
| clean | 0.429 | 0.389 | **0.456** |
| fog | 0.062 | 0.056 | **0.077** |
| crosstalk | 0.097 | **0.099** | 0.098 |
| snow | 0.336 | 0.329 | **0.340** |
| wet_ground | 0.380 | 0.377 | **0.406** |
| incomplete_echo | 0.364 | 0.326 | **0.393** |
| beam_missing | 0.406 | 0.372 | **0.434** |
| motion_blur | 0.380 | 0.345 | **0.406** |
| cross_sensor | **0.333** | 0.312 | 0.322 |
| **mean (8 corrupted)** | 0.295 | 0.277 | **0.310** |

**HDC dead-coordinate fraction per condition (the collapse mechanism; lower is
better):**

| condition | supcon_vib (ours) | supcon_vib_dglss | supcon_vib_dglsspp |
| :--- | :--- | :--- | :--- |
| clean | 0.008 | 0.102 | **0.003** |
| fog | 0.186 | 0.257 | **0.107** |
| crosstalk | **0.026** | 0.079 | **0.026** |
| snow | 0.005 | 0.066 | **0.001** |
| wet_ground | 0.008 | 0.082 | **0.005** |
| incomplete_echo | 0.010 | 0.104 | **0.006** |
| beam_missing | **0.003** | 0.051 | 0.006 |
| motion_blur | 0.021 | 0.099 | **0.003** |
| cross_sensor | 0.015 | 0.031 | **0.014** |
| **mean (8 corrupted)** | 0.034 | 0.096 | **0.021** |

**What the full-condition sweep shows:**

1. **The collapse mechanism is confirmed on every condition.** The plain DGLSS has
   the highest dead-coordinate fraction on all 9 sets (mean 0.096 over the
   corrupted conditions, vs 0.034 for ours and 0.021 for DGLSS++, a 3-5x gap), and
   the highest mean-fraction throughout (the shared-mean dominance of Theorem 1).
   The mean-dominated form of DGLSS consistently saturates the HDC sign-projection
   more than the other two.
2. **But DGLSS++ is the best HDC decoder overall.** Its mean HDC mIoU over the 8
   corrupted conditions (0.310) beats ours (0.295) and plain DGLSS (0.277); it wins
   clean and 5 of 7 geometric conditions, ours wins fog, and plain DGLSS narrowly
   wins crosstalk. This is a direct empirical instance of Remark 2 in the theory:
   DGLSS++ is the most low-rank (PR about 2.4-2.9 everywhere) yet has almost no
   dead coordinates, because its anisotropy is structured (the dominant directions
   carry the classes) rather than mean-dominated. Low effective dimensionality does
   not predict HDC failure.
3. **The "isotropy is unique to our approach" claim is not supported.** Ours and
   DGLSS++ have comparable dead-fractions (0.008 vs 0.003 clean), and DGLSS++ is at
   least as good at the HDC decode. What the data does distinguish is the
   mean-dominated form (plain DGLSS): it is the one that saturates the codes and
   decodes worst. The operative distinction is mean-dominance, not the method
   family.
4. **The continuous space does not separate them.** Linear-probe accuracy is
   comparable across the three on every condition, with DGLSS++ often highest
   (fog 0.189, beam_missing 0.824, incomplete_echo 0.900). The differences show in
   the binarized decode, not in how separable the 128D features are.

### 0.3 Frozen labeled ceiling

Each extractor evaluated with a LABELED decoder to see the recoverable ceiling per
condition (`frozen_ceiling_diag.py`): a logistic probe fit on clean labels (the
continuous-space ceiling, mIoU), and HDC prototypes re-estimated on the corrupted
points with true labels (the binarized ceiling).

**LP mIoU (continuous labeled ceiling; higher is better):**

| condition | supcon_vib (ours) | supcon_vib_dglss | supcon_vib_dglsspp |
| :--- | :--- | :--- | :--- |
| fog | 0.044 | 0.061 | **0.069** |
| crosstalk | 0.062 | **0.098** | 0.079 |
| snow | 0.369 | **0.395** | 0.384 |
| wet_ground | 0.458 | **0.474** | 0.445 |
| incomplete_echo | **0.497** | 0.490 | 0.470 |
| beam_missing | 0.488 | **0.504** | 0.484 |
| motion_blur | **0.478** | 0.439 | 0.446 |
| cross_sensor | **0.346** | 0.345 | 0.320 |
| **mean** | 0.343 | **0.351** | 0.337 |

**HDC oracle mIoU (binarized labeled ceiling; higher is better):**

| condition | supcon_vib (ours) | supcon_vib_dglss | supcon_vib_dglsspp |
| :--- | :--- | :--- | :--- |
| fog | 0.113 | 0.113 | **0.118** |
| crosstalk | **0.216** | 0.176 | 0.197 |
| snow | 0.357 | 0.340 | **0.367** |
| wet_ground | 0.441 | 0.426 | **0.455** |
| incomplete_echo | 0.359 | 0.329 | **0.390** |
| beam_missing | 0.407 | 0.367 | **0.430** |
| motion_blur | 0.386 | 0.346 | **0.406** |
| cross_sensor | 0.337 | 0.302 | **0.350** |
| **mean** | 0.327 | 0.300 | **0.339** |

Zero-shot HDC mIoU for reference (means over the 8 corrupted conditions from 0.2):
ours 0.295, DGLSS 0.277, DGLSS++ 0.310.

**What the labeled ceiling shows:**

1. **The ceiling ordering differs by pathway.** In the continuous space (LP mIoU),
   plain DGLSS is the best (0.351), ours second (0.343), DGLSS++ third (0.337). In
   the binarized space (HDC oracle), DGLSS++ is the best (0.339), ours second
   (0.327), plain DGLSS the worst (0.300). The plain DGLSS's sign-saturation costs
   it specifically in the HDC labeled ceiling, even with perfect labels: its
   continuous ceiling is the best, its binarized ceiling the worst.
2. **No single extractor dominates.** The spread across methods is small (about 1-4
   points in either mean), and ours is consistently second or tied-best in both
   pathways. The DGLSS family is not clearly worse than ours at the frozen-feature
   ceiling.
3. **The fog/crosstalk recoverable targets are low for all three.** The labeled
   HDC ceiling is about 0.11-0.12 on fog and 0.18-0.22 on crosstalk for every
   extractor; the continuous LP mIoU ceiling is lower still on fog (0.04-0.07)
   because the rare classes dominate the mIoU. These are the numbers a label
   budget (active learning) can actually reach, and they are far below the healthy
   conditions (0.34-0.50), consistent with the earlier 17-class oracle findings.
   Note the crosstalk HDC oracle (0.18-0.22) is much higher than the crosstalk LP
   mIoU (0.06-0.10): the binarized pathway is the higher ceiling on the collapsed
   conditions.

### 0.4 Consolidated takeaways

Across the three views of the same extractors:

1. The sign-saturation mechanism (dead fraction) fires for the plain,
   mean-dominated DGLSS on every condition; ours and DGLSS++ are near-zero.
2. But DGLSS++ is the best HDC decoder in both the zero-shot sweep and the labeled
   ceiling, and the plain DGLSS is worst in the binarized pathway even with labels.
   Structured low-rank anisotropy decodes fine, confirming the theoretical Remark 2.
3. "Isotropy is unique to our approach" is not supported: the operative
   distinction is mean-dominance, not the method family.
4. The labeled fog/crosstalk ceilings are low for all three (~0.11-0.12 fog,
   ~0.18-0.22 crosstalk HDC oracle), and the binarized pathway holds the higher
   ceiling on the collapsed conditions. These are the active-learning targets.
5. Caveat: the extractors are small-scale and under-converged (12 epochs at 10%
   data), and the DGLSS losses were fixed for a divergence immediately before these
   diagnostics. The ordering (plain DGLSS most sign-saturated and worst in the
   binarized path; DGLSS++ strongest decoder) is the robust takeaway to re-test at
   larger scale.

## Iteration log

### Iteration 1: do the previous supcon_vib TTA difficulties still hold on the DGLSS / DGLSS++ extractors?

The first research iteration is the systematic recheck, on the new feature
extractors, of the difficulties we hit with the supcon_vib model: the space
diagnostics and the whole battery of TTA methods that were developed against the
labeled ceiling. The question is which of those difficulties are properties of
the corrupted features (and so recur across extractors) and which were specific to
the supcon_vib space (and so may not hold for DGLSS or DGLSS++).

For each extractor (supcon_vib, supcon_vib_dglss, supcon_vib_dglsspp) and each
condition (fog, crosstalk, snow control), the frozen 128D features are passed
through the same battery (`tta_ceiling_diag.py`):

- **the "no label-free path" space diagnostics** (the checks that established the
  assignment wall on supcon_vib): rec@3 (is the true class even in the top-3 clean
  prototypes for the zs-wrong points), cosine-to-true, rank-of-true-class for the
  recoverable points, LP accuracy on the recoverable set, and the assignment gap
  (oracle-assigned vs LP-assigned re-estimate);
- **the TTA methods** developed against the ceiling: naive EMA, the
  confidence-gated and distance-gated weighted updates, BN-statistic alignment,
  and kNN reassignment.

Zero-shot and oracle are not re-run: they are already measured in the background
(Iteration 0.3, frozen labeled ceiling), and are used here only to frame the
gap-closed fraction.

The decisive comparisons, mirroring what we learned on supcon_vib:

1. Does the assignment wall still bind on the DGLSS / DGLSS++ features (rec@3 at
   or below the random baseline, low LP accuracy on the recoverable set, a small
   oracle-vs-LP assignment gap), or does one of the extractors make the labeled
   ceiling more reachable label-free?
2. Does the distance-gated update stay flat (the supcon_vib result), or do the
   DGLSS features let the uncertainty-gated update move the prototypes?
3. Does the kNN reassignment keep recovering some of the oracle gap, and is its
   gap-closing fraction similar across extractors?

**Results** (`tta_ceiling_diag.py`, frozen features, pool 500k / val 100k).

Assignment-wall diagnostics (rec@3 = true class in top-3 clean prototypes for the
zs-wrong points, random baseline ~0.19; rankT = mean true-class rank for the
recoverable points; LPrec = LP accuracy on the recoverable set; gorc / glp =
gated re-estimate with oracle- vs LP-assigned labels, R = 0.25):

| extractor | cond | rec@3 | cosT | rankT | LPrec | gorc | glp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib | fog | 0.240 | 0.124 | 3.07 | 0.025 | 0.113 | 0.087 |
| supcon_vib | crosstalk | 0.467 | 0.165 | 2.33 | 0.116 | 0.159 | 0.131 |
| supcon_vib | snow | 0.484 | 0.535 | 1.27 | 0.462 | 0.341 | 0.341 |
| supcon_vib_dglss | fog | 0.210 | 0.283 | 2.68 | 0.081 | 0.106 | 0.090 |
| supcon_vib_dglss | crosstalk | 0.477 | 0.265 | 2.50 | 0.246 | 0.140 | 0.138 |
| supcon_vib_dglss | snow | 0.459 | 0.674 | 1.30 | 0.452 | 0.339 | 0.339 |
| supcon_vib_dglsspp | fog | 0.205 | 0.152 | 1.78 | 0.044 | 0.127 | 0.121 |
| supcon_vib_dglsspp | crosstalk | 0.457 | 0.135 | 2.18 | 0.124 | 0.154 | 0.149 |
| supcon_vib_dglsspp | snow | 0.482 | 0.540 | 1.31 | 0.437 | 0.350 | 0.350 |

TTA methods (full-scene mIoU, with zero-shot and oracle for reference):

| extractor | cond | zero-shot | naive | conf | dist | bn | knn | oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib | fog | 0.082 | 0.092 | 0.094 | 0.094 | 0.096 | 0.083 | 0.110 |
| supcon_vib | crosstalk | 0.119 | 0.136 | 0.137 | 0.136 | 0.147 | 0.128 | 0.243 |
| supcon_vib | snow | 0.348 | 0.358 | 0.358 | 0.358 | 0.354 | 0.338 | 0.363 |
| supcon_vib_dglss | fog | 0.076 | 0.106 | 0.104 | 0.107 | 0.108 | 0.087 | 0.116 |
| supcon_vib_dglss | crosstalk | 0.130 | 0.164 | 0.161 | 0.164 | 0.179 | 0.141 | 0.211 |
| supcon_vib_dglss | snow | 0.349 | 0.328 | 0.325 | 0.329 | 0.359 | 0.336 | 0.355 |
| supcon_vib_dglsspp | fog | 0.101 | 0.129 | 0.126 | 0.127 | 0.131 | 0.119 | 0.127 |
| supcon_vib_dglsspp | crosstalk | 0.124 | 0.161 | 0.161 | 0.160 | 0.186 | 0.147 | 0.222 |
| supcon_vib_dglsspp | snow | 0.355 | 0.351 | 0.351 | 0.351 | 0.359 | 0.348 | 0.374 |

**What Iteration 1 shows:**

1. **The assignment wall holds on every extractor.** Fog rec@3 is 0.21-0.24, at or
   just above the ~0.19 random baseline, on all three extractors; the LP accuracy
   on the recoverable set is 0.03-0.08 on fog and 0.12-0.25 on crosstalk; and the
   gated oracle-vs-LP gap is tiny (about 0.01-0.03) everywhere. Detection is
   solvable, assignment is not, identically across DGLSS, DGLSS++ and supcon_vib.
   No extractor makes the labeled ceiling reachable label-free.
2. **The TTA methods are mostly flat, and slightly less so than before.** On fog
   the naive / confidence / distance / BN updates move a few points above zero-shot
   (dglsspp 0.101 to 0.131), and the fog oracle headroom is small (~0.03-0.04) so
   they nearly reach it. On crosstalk the oracle gap is large (0.09-0.12), and the
   methods close only part of it, with BN alignment the best label-free
   (0.147-0.186). The distance-gated update is no flatter than naive EMA, so the
   earlier "flat" result is not reproduced sharply here.
3. **kNN is not the best method on these features.** On fog and crosstalk the kNN
   reassignment sits at or below the simple weighted updates (e.g. ours crosstalk
   kNN 0.128 vs naive 0.136, BN 0.147), and its oracle-gap fraction is small
   (fog 0.06-0.71, crosstalk 0.07-0.20). This differs from the earlier harness
   where kNN was the best re-estimate; the budget split and feature scale here
   favor the weighted updates.
4. **The control behaves.** On snow there is no headroom (oracle about equals
   zero-shot), all methods are flat, and kNN is the one that drifts below zero-shot.
5. **Caveat.** Same under-converged extractors as Iteration 0; the ordering (wall
   persists everywhere; no label-free method approaches the crosstalk oracle) is
   the robust takeaway.

### Iteration 2: gate-signal structure (is there a usable gate, or exploitable density structure?)

The Iteration-1 TTA methods were near-flat, and the weighted updates were all
near-identical, which suggested the weight signals were weak. This diagnostic
(`gate_structure_diag.py`) measures, per extractor and condition, the correct-vs-
wrong AUROC of every candidate gate signal (LP confidence, entropy, distance to
the nearest clean prototype, feature norm, top-2 margin, and local density = mean
distance to k=20 neighbors), a logistic-regression fusion over all of them, the
recoverability of the confident-but-wrong points (recCW), and the centroid
separation between confident-correct and confident-wrong in signal space.

Per-extractor correct-vs-wrong AUROC (fog / crosstalk / snow; fusion = all
signals combined; recCW = true class in top-3 clean prototypes for the
confident-wrong points):

| extractor | cond | conf | entr | dist | norm | marg | dens | fusion | recCW | C/W gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib | fog | 0.243 | 0.757 | 0.144 | 0.155 | 0.160 | **0.912** | **0.923** | 0.328 | 1.77 |
| supcon_vib | crosstalk | 0.205 | 0.792 | 0.153 | 0.140 | 0.129 | **0.914** | **0.918** | 0.574 | 0.20 |
| supcon_vib | snow | 0.266 | 0.749 | 0.152 | 0.540 | 0.481 | 0.688 | **0.911** | 0.564 | 0.47 |
| supcon_vib_dglss | fog | 0.549 | 0.417 | 0.679 | **0.865** | 0.212 | 0.688 | **0.865** | 0.292 | 2.94 |
| supcon_vib_dglss | crosstalk | 0.255 | 0.733 | 0.487 | 0.648 | 0.295 | 0.735 | **0.812** | 0.499 | 0.95 |
| supcon_vib_dglss | snow | 0.363 | 0.643 | 0.239 | 0.761 | 0.373 | 0.672 | **0.864** | 0.559 | 2.07 |
| supcon_vib_dglsspp | fog | 0.571 | 0.406 | 0.472 | **0.839** | 0.166 | 0.611 | **0.909** | 0.194 | 6.71 |
| supcon_vib_dglsspp | crosstalk | 0.388 | 0.608 | 0.334 | **0.722** | 0.132 | 0.728 | **0.889** | 0.532 | 5.68 |
| supcon_vib_dglsspp | snow | 0.312 | 0.685 | 0.273 | 0.706 | 0.293 | 0.690 | **0.855** | 0.518 | 1.50 |

**What Iteration 2 shows:**

1. **There IS a strong label-free signal.** On our extractor, local density
   separates correct from wrong points at AUROC 0.91 on fog and crosstalk; on the
   DGLSS and DGLSS++ extractors, feature norm is the strong signal (fog 0.87 and
   0.84). The fusion over all signals is 0.81-0.92 everywhere. The near-flat
   Iteration-1 TTA came from testing only the weak weight signals (confidence and
   distance-to-prototype, AUROC 0.14-0.68); density and norm were never used as
   update weights.
2. **The best gate differs by extractor.** Density for supcon_vib; norm for the
   DGLSS arms. This matches the earlier finding that feature norm was the dominant
   gate on the robust encoder.
3. **The assignment wall holds even for the confident-wrong points.** Their true
   class is in the top-3 clean prototypes only 19-33% of the time on fog, at or
   near the random baseline. We can identify WHICH points are wrong (density /
   norm / fusion rank them at 0.81-0.92), but not WHAT class they are, exactly the
   wall Iteration 1 established.
4. **The DGLSS / DGLSS++ extractors show a larger correct-vs-wrong separation** in
   signal space (C/W gap 2.9-6.7 vs 0.2-1.8 for ours), so their structure is more
   separable, not less.
5. **Actionable next step.** Test the untried gates as weighted-update weights:
   a density-gated and a norm-gated prototype update, which the Iteration-1
   battery did not run (it used confidence and distance weights). Given the 0.84-0.91
   point-level AUROCs, these should move the update further toward the oracle than
   naive EMA, which is the missing lever the earlier flatness obscured.

### Iteration 3: the Iteration-2 gates as update weights

The density- and norm-gated prototype updates, run per extractor with the gate
Iteration 2 identified as strong for it (`ttagate_diag.py`): norm for the DGLSS
arms, density for supcon_vib. Same 100k pool / 100k val split; zero-shot and
oracle recomputed as the gap references. mIoU and fraction of the oracle gap
closed:

| extractor | cond | zero-shot | gate mIoU | oracle | gap-closed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib | fog | 0.082 | dens 0.093 | 0.123 | **0.28** |
| supcon_vib | crosstalk | 0.119 | dens 0.136 | 0.244 | **0.13** |
| supcon_vib_dglss | fog | 0.076 | norm 0.103 | 0.135 | **0.45** |
| supcon_vib_dglss | crosstalk | 0.130 | norm 0.144 | 0.209 | **0.17** |
| supcon_vib_dglsspp | fog | 0.101 | norm 0.125 | 0.143 | **0.58** |
| supcon_vib_dglsspp | crosstalk | 0.124 | norm 0.159 | 0.223 | **0.36** |

**What Iteration 3 shows:**

1. **The norm-gated update is a real, non-flat TTA lever on the DGLSS extractors.**
   It closes 45-58% of the fog gap and 17-36% of the crosstalk gap (DGLSS++ fog
   0.101 to 0.125, gap 0.58). This is the first weighted update that moves
   meaningfully on fog, unlike the near-flat confidence / distance gates.
2. **It is comparable to, not clearly better than, the naive weighted update.** On
   the same budget the naive EMA closes a similar fraction (Iteration 1: DGLSS
   fog 0.106, DGLSS++ fog 0.129). The value of the norm gate is that it is a
   principled weight (correct points carry higher norms on the DGLSS arms) that
   behaves no worse than naive, and the fog gap is genuinely closable at this
   scale.
3. **The density gate now works for supcon_vib** (after a monotone-shift fix): fog
   0.082 to 0.093 and crosstalk 0.119 to 0.136, closing 28% and 13% of the oracle
   gaps. This is comparable to the naive update on the same extractor (Iteration
   1: fog gap ~0.36, crosstalk ~0.14). So the density signal is a usable weight,
   consistent with its 0.91 correct-vs-wrong AUROC, but it does not beat naive
   EMA on ours either.
4. **Caveat.** Same under-converged extractors; and the gap fractions are on a
   100k-pool split, so compare within this table rather than to the 500k-pool
   Iteration-1 numbers.

### Iteration 4: medium-scale validation

The Iteration 0-3 measurements were all on under-converged micro extractors
(12 epochs at 10% data). This iteration re-runs the battery on a medium-scale
DGLSS++ checkpoint and the medium supcon_vib pretrain to check which findings
scale. Setup (`--med` flags; outputs `..._results_med.json`):

- **supcon_vib**: the existing medium pretrain `logs/med_pretrain_supcon_vib`
  (saved at epoch 25);
- **supcon_vib_dglsspp**: the medium run trained for this validation
  (24 epochs at 100% of sequence 08, `robust_diagnostic/logs/supcon_vib_dglsspp`);
- **supcon_vib_dglss**: no medium run exists; stays at the micro checkpoint
  (12 epochs, 10% data). Treat its column as the under-converged reference, not a
  scale comparison.

The isotropy file contains only the medium DGLSS++ run; the frozen-ceiling and
TTA-ceiling files contain all three extractors.

#### 4.1 Isotropy of the medium DGLSS++ space

128D-bottleneck isotropy per condition (dead-fraction / mean-fraction lower is
better; LP and HDC mIoU higher is better):

| set | PR | deadF | meanF | Hamming | LP | HDC mIoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| clean | 4.25 | 0.000 | 0.394 | 0.425 | 0.885 | 0.530 |
| fog | 3.49 | 0.221 | 0.830 | 0.248 | 0.524 | 0.068 |
| crosstalk | 2.30 | 0.147 | 0.708 | 0.323 | 0.224 | 0.115 |
| snow | 3.80 | 0.000 | 0.351 | 0.450 | 0.794 | 0.396 |
| wet_ground | 3.18 | 0.000 | 0.508 | 0.406 | 0.848 | 0.483 |
| incomplete_echo | 3.72 | 0.000 | 0.411 | 0.426 | 0.909 | 0.449 |
| beam_missing | 3.58 | 0.000 | 0.273 | 0.461 | 0.865 | 0.506 |
| motion_blur | 3.63 | 0.000 | 0.397 | 0.442 | 0.868 | 0.502 |
| cross_sensor | 4.19 | 0.000 | 0.266 | 0.474 | 0.786 | 0.434 |
| **mean (8 corrupted)** | 3.49 | 0.046 | 0.468 | 0.404 | 0.727 | 0.369 |

#### 4.2 Frozen labeled ceiling at medium scale

Each extractor evaluated with a labeled decoder (`frozen_ceiling_diag.py --med`):
LP = logistic probe fit on clean labels (continuous ceiling), HDC prototypes
re-estimated with true labels on the corrupted points (binarized ceiling). Bold =
best per row.

**LP mIoU (continuous labeled ceiling):**

| condition | supcon_vib (med) | supcon_vib_dglss (micro) | supcon_vib_dglsspp (med) |
| :--- | :--- | :--- | :--- |
| fog | 0.036 | **0.061** | 0.049 |
| crosstalk | 0.040 | **0.098** | 0.061 |
| snow | **0.429** | 0.395 | 0.423 |
| wet_ground | 0.516 | 0.474 | **0.603** |
| incomplete_echo | **0.565** | 0.490 | 0.554 |
| beam_missing | 0.580 | 0.505 | **0.613** |
| motion_blur | 0.555 | 0.440 | **0.622** |
| cross_sensor | **0.501** | 0.344 | 0.476 |
| **mean** | 0.403 | 0.351 | **0.425** |

**Zero-shot HDC mIoU (reference):**

| condition | supcon_vib (med) | supcon_vib_dglss (micro) | supcon_vib_dglsspp (med) |
| :--- | :--- | :--- | :--- |
| fog | **0.078** | 0.056 | 0.068 |
| crosstalk | 0.101 | 0.099 | **0.115** |
| snow | 0.384 | 0.329 | **0.396** |
| wet_ground | 0.446 | 0.377 | **0.483** |
| incomplete_echo | 0.412 | 0.326 | **0.449** |
| beam_missing | 0.470 | 0.372 | **0.506** |
| motion_blur | 0.450 | 0.345 | **0.502** |
| cross_sensor | 0.395 | 0.312 | **0.434** |
| **mean** | 0.342 | 0.277 | **0.369** |

**HDC oracle mIoU (binarized labeled ceiling):**

| condition | supcon_vib (med) | supcon_vib_dglss (micro) | supcon_vib_dglsspp (med) |
| :--- | :--- | :--- | :--- |
| fog | **0.156** | 0.113 | 0.151 |
| crosstalk | **0.221** | 0.176 | 0.214 |
| snow | 0.404 | 0.340 | **0.410** |
| wet_ground | 0.489 | 0.426 | **0.514** |
| incomplete_echo | 0.408 | 0.329 | **0.448** |
| beam_missing | 0.474 | 0.367 | **0.506** |
| motion_blur | 0.454 | 0.346 | **0.503** |
| cross_sensor | 0.433 | 0.302 | **0.451** |
| **mean** | 0.380 | 0.300 | **0.399** |

(LP-accuracy means over the 8 corrupted conditions for reference: supcon_vib
0.77, dglss 0.72, dglsspp 0.73.)

#### 4.3 TTA battery at medium scale

Same battery as Iteration 1 (`tta_ceiling_diag.py --med`, frozen features,
pool 500k / val 100k). Assignment-wall diagnostics (rec@3 = true class in top-3
clean prototypes for the zs-wrong points, random baseline ~0.19; rankT = mean
true-class rank for the recoverable points; LPrec = LP accuracy on the
recoverable set; gorc / glp = gated re-estimate with oracle- vs LP-assigned
labels):

| extractor | cond | rec@3 | cosT | rankT | LPrec | gorc | glp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib (med) | fog | 0.081 | 0.118 | 3.69 | 0.061 | 0.099 | 0.092 |
| supcon_vib (med) | crosstalk | 0.129 | 0.126 | 4.82 | 0.078 | 0.138 | 0.114 |
| supcon_vib (med) | snow | 0.431 | 0.581 | 1.27 | 0.508 | 0.386 | 0.386 |
| supcon_vib_dglss (micro) | fog | 0.210 | 0.283 | 2.68 | 0.080 | 0.106 | 0.090 |
| supcon_vib_dglss (micro) | crosstalk | 0.477 | 0.265 | 2.50 | 0.245 | 0.141 | 0.138 |
| supcon_vib_dglss (micro) | snow | 0.459 | 0.674 | 1.30 | 0.456 | 0.339 | 0.339 |
| supcon_vib_dglsspp (med) | fog | 0.137 | 0.006 | 2.50 | 0.542 | 0.111 | 0.109 |
| supcon_vib_dglsspp (med) | crosstalk | 0.272 | 0.024 | 2.37 | 0.120 | 0.154 | 0.147 |
| supcon_vib_dglsspp (med) | snow | 0.321 | 0.355 | 1.21 | 0.352 | 0.415 | 0.414 |

TTA methods (full-scene mIoU, with zero-shot and oracle for reference):

| extractor | cond | zero-shot | naive | conf | dist | bn | knn | oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib (med) | fog | 0.101 | 0.084 | 0.084 | 0.083 | **0.107** | 0.093 | 0.164 |
| supcon_vib (med) | crosstalk | 0.120 | 0.098 | 0.098 | 0.098 | **0.151** | 0.107 | 0.262 |
| supcon_vib (med) | snow | 0.394 | **0.400** | 0.399 | 0.400 | 0.400 | 0.381 | 0.407 |
| supcon_vib_dglss (micro) | fog | 0.076 | 0.106 | 0.104 | 0.106 | **0.108** | 0.087 | 0.116 |
| supcon_vib_dglss (micro) | crosstalk | 0.130 | 0.165 | 0.163 | 0.165 | **0.179** | 0.141 | 0.211 |
| supcon_vib_dglss (micro) | snow | 0.349 | 0.328 | 0.325 | 0.329 | **0.359** | 0.336 | 0.355 |
| supcon_vib_dglsspp (med) | fog | 0.092 | 0.114 | 0.114 | 0.117 | **0.127** | 0.109 | 0.200 |
| supcon_vib_dglsspp (med) | crosstalk | 0.141 | 0.150 | 0.151 | 0.151 | **0.174** | 0.148 | 0.250 |
| supcon_vib_dglsspp (med) | snow | 0.418 | **0.425** | 0.424 | 0.425 | 0.412 | 0.412 | 0.429 |

#### 4.4 The gated update at medium scale

The norm-gated prototype update on the medium DGLSS++ extractor
(`ttagate_diag.py --methods supcon_vib_dglsspp --med`, 100k pool / 100k val):

| extractor | cond | zero-shot | gate mIoU | oracle | gap-closed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib_dglsspp (med) | fog | 0.092 | norm 0.114 | 0.198 | 0.20 |
| supcon_vib_dglsspp (med) | crosstalk | 0.141 | norm 0.149 | 0.249 | 0.08 |

#### 4.5 What Iteration 4 shows

1. **The medium DGLSS++ extractor decodes better and stays structured.** Clean HDC
   mIoU rises from 0.456 (micro) to 0.530 (medium) and the 8-condition corrupted
   mean from 0.310 to 0.369; the space is also less anisotropic at scale (clean PR
   2.7 to 4.3, mean-fraction 0.51 to 0.39). The medium run does not degrade the
   DGLSS++ decoder — it improves it.
2. **The mean-dominance mechanism localizes to the collapsed conditions.** The dead
   fraction is zero on every healthy condition but fires on fog (0.221) and
   crosstalk (0.147), the only conditions with an elevated mean-fraction (0.830 fog,
   0.708 crosstalk) and the conditions where HDC mIoU collapses to 0.068 / 0.115.
   The theory's mechanism (shared-mean dominance saturating the HDC sign codes) is
   exactly where the decode fails.
3. **Where both arms are medium scale, DGLSS++ is the best decoder.** On the frozen
   labeled ceiling, medium DGLSS++ beats medium supcon_vib on the corrupted mean in
   the continuous (LP-mIoU 0.425 vs 0.403), zero-shot HDC (0.369 vs 0.342), and
   binarized oracle (0.399 vs 0.380) pathways; the micro DGLSS column is worst on
   all three (scale gap in zero-shot HDC: 0.277 to 0.369). Caveat: the two medium
   checkpoints are not identically budgeted (supcon_vib is the earlier
   `med_pretrain` run, DGLSS++ is 24 ep at 100% data), so the head-to-head is
   indicative, not controlled.
4. **The assignment wall persists at scale.** Medium DGLSS++ fog rec@3 is 0.137
   (below the ~0.19 random baseline) and the gated oracle-vs-LP gap is ~0.002-0.007
   — detection without assignment, exactly as at micro scale. One structural change:
   on fog the recoverable set is smaller (rec@3 0.14 vs 0.21 micro) but far more
   cleanly labeled (LPrec 0.54 vs 0.04), so what is recoverable is better
   recoverable.
5. **The label-free TTA methods stay flat, and the best lever remains BN
   alignment.** Naive / confidence / distance gates close ~0.08-0.32 of the
   fog/crosstalk gaps on the DGLSS++ and DGLSS arms, while on supcon_vib they sit
   *below* zero-shot on the collapsed conditions (fog 0.101 to 0.084); BN alignment
   is the best label-free update everywhere (fog 0.107-0.127, crosstalk
   0.151-0.179).
6. **The norm-gate lever does not scale its gap fraction.** The Iteration-3 headline
   (norm gate closes 58% of the DGLSS++ fog gap) was on the under-converged micro
   extractor with a small oracle (0.143). At medium scale the oracle is larger (fog
   0.198, crosstalk 0.249) and the norm gate closes 0.20 / 0.08 of the gap — exactly
   matching naive EMA (0.20 / 0.08) and below BN alignment (0.32 / 0.31). The
   absolute gain is similar (fog +0.022), but relative to the larger oracle gap its
   fraction halves. The "norm gate as the superior lever" result was a small-scale
   artifact; at scale it is a comparable-but-not-better principled weight,
   consistent with the Iteration-3 caveat.

### Iteration 5: per-class autopsy of the micro-vs-medium label-free gap

Iteration 4 asked why the label-free update closes less of the labeled ceiling at
medium scale. Working hypothesis: at medium scale the training data leans so
heavily on the majority classes that minority-class features become less robust
and sit farther from their class prototypes, so the update helps them less.
`scale_gap_diag.py` runs the re-trained micro checkpoint (12 ep / 10% data, the
original having been overwritten by the medium run) and the medium checkpoint
(24 ep / 100% data) on the SAME 100k pool / 100k val split, reporting per class:
feature proximity to the clean class mean (`feat_cos`, the "close to the
prototype" measure), HDC code proximity (`hdc_cos`), zero-shot-correct fraction,
LP pseudo-label recall, and the zs / naive / oracle per-class IoU. Class ids
(all17 map): 2 bicycle, 4 car, 7 pedestrian, 11 road, 12 other-ground,
13 sidewalk, 14 terrain, 15 building, 16 vegetation (bus / other-object absent
from seq 08).

**Aggregate gap-closed, same split both scales:**

| cond | scale | zs | naive | oracle | gap-closed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | micro | 0.088 | 0.074 | 0.120 | -0.43 |
| fog | med | 0.082 | 0.101 | 0.176 | **0.20** |
| crosstalk | micro | 0.100 | 0.132 | 0.209 | **0.30** |
| crosstalk | med | 0.125 | 0.130 | 0.222 | 0.05 |

**Per-class feature proximity on fog** (`feat_cos`, mean 128D cosine of the
class's fog points to its CLEAN class mean; higher = closer to the prototype):

| class | pool freq | micro | med |
| :--- | :--- | :--- | :--- |
| road (11) | 24923 | 0.587 | **0.856** |
| sidewalk (13) | 14584 | 0.210 | **0.022** |
| terrain (14) | 7733 | -0.281 | -0.221 |
| vegetation (16) | 5243 | 0.431 | **0.112** |
| car (4) | 3305 | 0.847 | **0.321** |
| building (15) | 1455 | 0.451 | **0.205** |
| pedestrian (7) | 288 | 0.156 | -0.005 |

**LP pseudo-label recall on fog** (fraction of a class's points the logistic probe
assigns to it; the quality the naive update depends on):

| class | micro | med |
| :--- | :--- | :--- |
| road (11) | 0.774 | **0.897** |
| sidewalk (13) | 0.093 | 0.071 |
| terrain (14) | 0.001 | 0.000 |
| vegetation (16) | 0.116 | **0.049** |
| car (4) | 0.734 | **0.149** |
| building (15) | 0.323 | **0.006** |
| pedestrian (7) | 0.000 | 0.000 |

Spearman rho(freq, feat_cos) — distance-to-prototype becomes frequency-dependent
at scale: fog 0.04 (micro) to **0.48** (med); crosstalk 0.33 to 0.29.

**Per-class naive-update effect on fog** (zs -> naive per-class IoU, micro vs med):

| class | micro zs->naive | med zs->naive | micro gap | med gap |
| :--- | :--- | :--- | :--- | :--- |
| road (11) | 0.483 -> 0.205 (destroyed) | 0.433 -> 0.524 (fixed) | - | **0.82** |
| car (4) | 0.054 -> 0.070 | 0.122 -> 0.102 (hurt) | 0.60 | -0.11 |
| sidewalk (13) | 0.155 -> 0.192 | 0.040 -> 0.132 | 0.52 | 0.46 |
| terrain (14) | 0.001 -> 0.051 | 0.010 -> 0.007 | 0.36 | -0.01 |
| vegetation (16) | 0.067 -> 0.127 | 0.083 -> 0.093 | 1.08 | 0.09 |
| building (15) | 0.031 -> 0.020 | 0.050 -> 0.049 | -0.42 | -0.03 |

#### 5.1 Is the hypothesis correct?

**The feature-space half is confirmed, sharply.** At medium scale the model
becomes a majority-class specialist: road's fog points sit far closer to the road
prototype than at micro (feat_cos 0.856 vs 0.587) while every other class's points
drift away (car 0.847 to 0.321, vegetation 0.431 to 0.112, building 0.451 to
0.205, sidewalk 0.210 to 0.022). The Spearman rho between class frequency and
distance-to-prototype jumps from ~0 (micro: everyone equally far) to 0.48 (med) on
fog. The same pattern holds on crosstalk. This is exactly the claimed
"minority-class features are less robust at medium scale."

**The consequence for the naive EMA is real but indirect, and it is the LP
pseudo-labels that mediate it.** The update only helps a class if the logistic
probe still finds that class in the pooled corrupted points. At medium scale the
LP's per-class recall polarizes to the majority (road 0.897, car 0.734 -> 0.149,
building 0.323 -> 0.006, terrain -> 0.000): the update concentrates its gains in
road (fog gap 0.82) and leaves the drifting classes flat or worse (car 0.122 ->
0.102, terrain 0.010 -> 0.007, vegetation gap 1.08 -> 0.09). At micro, the LP is a
weaker but more even classifier (car 0.734, building 0.323, vegetation 0.116), so
the update spreads real gains across car / sidewalk / terrain / vegetation.

**The twist: "micro gets closer to the ceiling" is not a clean statement.** On
crosstalk it is true (0.30 vs 0.05). On fog at this 100k split the micro naive
update is actually NEGATIVE (-0.43): it destroys the road prototype (0.483 ->
0.205) because micro's fog space is far more collapsed (LP acc 0.159 vs 0.524), so
the dominant class's 24.9k pooled points carry noisy pseudo-labels and the
unit-weighted mean drags its prototype. The Iteration-4 fog "micro reaches the
ceiling" was a 500k-pool artifact of the same instability. So the honest
statement is: medium-scale polarization (minority features drifting, LP recall
collapsing off the majority) removes the label-free update's ability to help the
minority classes, and the micro model's apparent success is partly the inverse
instability (its update helps minority classes but erases the majority).

Caveat: the "micro" column here is the re-trained model (fresh seed, fog zs 0.088
vs 0.101 for the original), so absolute values differ slightly from Iterations 0-4;
the per-class comparison is controlled within this run.

#### 5.2 What to test next

1. **Class-balanced update (the README Pillar-3 lever).** Weight the pool per class
   so each class carries equal total weight in the prototype re-estimation instead
   of road's 24.9k points swamping every other class. This directly tests whether
   the drifting minority prototypes are recoverable when the update is not
   majority-dominated.
2. **Pseudo-label vs pool-mass ablation.** Compare, at medium scale: (a) unit
   weights + LP labels (naive, measured), (b) unit weights + oracle labels (oracle,
   measured), (c) LP labels with a per-class minimum-support threshold, (d)
   class-balanced weights + LP labels. Separates assignment noise from
   majority-dominance.
3. **Per-class signal-to-noise (SNIR).** Measure per class the corrupted-feature
   cosine to its own clean centroid vs to the nearest other centroid, at each scale,
   to quantify "minority less robust" as a loss of class structure rather than just
   a shift.
4. **Pool-size sensitivity.** Sweep the pool (50k-500k) at micro scale to confirm
   whether the fog sign flip (negative at 100k, near-ceiling at 500k) is a
   real property or an artifact of pool mass / pseudo-label noise.

#### 5.3 Update variants at medium scale (support threshold + HDC pseudo-labels)

Before committing to a 5h class-balanced training run, the working assumption that
"minority features sit far from their prototypes, so the update can't help them"
was checked against the medium supcon_vib extractor (`logs/med_pretrain_supcon_vib`)
with the same per-class autopsy. The two structural alternatives were also tested,
eval-only, on both medium checkpoints: a per-class minimum-support threshold in the
update (`weighted_mean_update`, `min_support=256`: keep the base prototype when a
class's assigned pool weight is below threshold) and the zero-shot HDC decode as the
pseudo-label source instead of the logistic probe. Aggregate gap-closed
(zs -> variant -> oracle) per condition:

| extractor | cond | naive (LP) | LP+support | HDC labels | HDC+support | oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib (med) | fog | -0.25 | -0.38 | -0.13 | -0.13 | 0.146 |
| supcon_vib (med) | crosstalk | -0.16 | -0.26 | -0.09 | -0.09 | 0.233 |
| dglsspp (med) | fog | +0.19 | +0.20 | +0.15 | +0.15 | 0.176 |
| dglsspp (med) | crosstalk | +0.02 | +0.08 | +0.06 | +0.04 | 0.222 |

Key per-class numbers (fog): sv_med car zs 0.157 -> naive 0.026 / HDC-labels 0.114 /
oracle 0.200; dg_med car zs 0.122 -> naive 0.102 / oracle 0.303 (and on crosstalk
dg_med car naive 0.252 -> LP+support 0.313, the one variant that clearly helps).

**What this shows:**

1. **No variant rescues supcon_vib's update.** All four stay negative
   (-0.09 to -0.38) on both conditions. Even the HDC-decode labels only reduce the
   damage (sv_med fog car 0.026 -> 0.114) without going positive.
2. **The binding limit for supcon_vib is the labeled ceiling, not the update
   noise.** sv_med fog oracle is 0.146 vs zs 0.090 — only 0.056 of headroom to work
   with; dg_med fog has 0.176 vs 0.082 (0.094). supcon_vib's corrupted features are
   less recoverable *even with true labels* (fog car oracle 0.200 vs 0.303, terrain
   0.161 vs 0.204, vegetation 0.173 vs 0.185). With almost no gap to close, any
   label-free update noise pushes the result below zero-shot.
3. **This answers the noise-robustness question:** supcon_vib med is not "less
   robust to update noise" in a way a better update fixes — its medium-scale fog
   space has a low recoverable ceiling, so the LP's noise (fog LP recall for car
   0.005) has nothing to cancel and the update net-negative. dglsspp med, with real
   headroom, is the extractor on which the naive update works at all (+0.19).
4. **The support threshold helps only where it protects, and it is imperfect.** It
   recovers dg_med's crosstalk car (0.252 -> 0.313), but for sv_med's fog car it
   fails (LP+support 0.007, worse than naive 0.026): the logistic probe assigns
   >= 256 (mostly wrong) pool points to car on fog, so a total-weight threshold
   counts assignments, not correct ones, and lets the low-precision class through.
   A precision-aware or confidence-gated threshold would be needed.
5. **The class-balance training idea loses its premise at this evidence.** supcon_vib
   med has no feature polarization (rho(freq, feat_cos) = -0.36 fog) and still fails;
   therefore "minority features far from prototypes" is not what breaks the naive
   EMA at medium scale. What breaks it is the low labeled ceiling on the collapsed
   conditions. Balancing the extractor to tighten minority classes cannot raise a
   ceiling that is set by how little of the minority structure survives fog.

**Caveat.** supcon_vib's med checkpoint is the earlier `med_pretrain` run (epoch 25),
not a budget-matched rerun, so the exact oracle ordering (dg_med fog 0.176 vs sv_med
0.146) is split-dependent and indicative, not controlled. The robust conclusion is
the ceiling-boundedness itself, which both medium extractors show.

#### 5.4 The supcon_vib micro-to-medium trajectory (settles the "majority overemphasis" question)

To test whether supcon_vib's lower fog ceiling came from majority-class
overemphasis in the larger dataset, the same autopsy was run on the existing MICRO
supcon_vib checkpoint (`robust_diagnostic/logs/supcon_vib`, 12 ep / 10% data):

| cond | scale | zs | naive | oracle | gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fog | micro | 0.073 | 0.083 | 0.109 | **+0.28** |
| fog | med | 0.090 | 0.076 | 0.146 | **-0.25** |
| crosstalk | micro | 0.106 | 0.120 | 0.217 | **+0.13** |
| crosstalk | med | 0.107 | 0.087 | 0.233 | **-0.16** |

**The ceiling ROSE with scale, so majority overemphasis is not what lowers it.**
supcon_vib's fog oracle goes 0.109 (micro) to 0.146 (med), crosstalk 0.217 to 0.233,
mirroring dglsspp (fog 0.120 to 0.176). The larger dataset raises the recoverable
ceiling; it does not depress it. dglsspp's consistently higher fog oracle at BOTH
scales (0.120 vs 0.109, 0.176 vs 0.146) marks the gap as an extractor property of
the corrupted features, not a scale effect.

**What scale actually breaks is the pseudo-label assignment, not the ceiling.** The
naive EMA flips from +0.28 (micro) to -0.25 (med) for supcon_vib on fog, and the
per-class driver is the logistic probe's car recall collapsing from 0.838 (micro,
where the LP is an "even" classifier and feat_cos is 0.819) to 0.005 (med, where the
LP polarizes to road-only and car feat_cos drops to 0.351): the micro update lifts
car 0.057 -> 0.066, the medium update destroys it 0.157 -> 0.026. So the medium
scale simultaneously raises the ceiling and removes the update's ability to reach
it, decoupling the two: the ceiling is a feature property, the naive-EMA failure is
an assignment problem.

**Consequence for the plan.** The class-balance training idea is now refuted on both
of its premises: supcon_vib shows no polarization yet fails, and its (and
dglsspp's) ceiling rises with scale rather than falling. The label-free TTA failure
at scale is the tightening assignment wall (pseudo-label recall collapse), bounded
additionally by how little ceiling the extractor has on the collapsed conditions.

## Iteration 6: DGLSS++ with corruption-targeted augmentation (micro gate)

Iterations 4-5 established that (a) the label-free TTA failure at scale is the
assignment wall plus a low recoverable ceiling, and (b) the minority-class drift is
a corruption-shift under fog/crosstalk. The natural training lever is the one
mechanism supcon_vib has that the DGLSS arms lack: the corruption-targeted
augmented view. This iteration adds a **`supcon_vib_dglsspp_cor`** arm — DGLSS++
(GMSIFC + LSCC + CE, still VIB-free) but with the augmented view set to supcon_vib's
`get_augmented_view` (beam-drop + fog depth jitter + 20% density sparsity) plus a
crosstalk fake-return injection, so the consistency constraints learn invariance to
the exact corruptions that collapse the minority classes.

**Micro gate (12 ep / 10% data), same-split aggregate vs the two micro baselines:**

| extractor | cond | zs | naive | oracle | gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| dglsspp (micro) | fog | 0.088 | 0.074 | 0.120 | -0.43 |
| supcon_vib (micro) | fog | 0.073 | 0.083 | 0.109 | +0.28 |
| **dglsspp_cor (micro)** | fog | 0.081 | 0.093 | 0.114 | **+0.37** |
| dglsspp (micro) | crosstalk | 0.100 | 0.132 | 0.209 | +0.30 |
| supcon_vib (micro) | crosstalk | 0.106 | 0.120 | 0.217 | +0.13 |
| **dglsspp_cor (micro)** | crosstalk | 0.104 | 0.118 | 0.202 | **+0.15** |

**Isotropy (`isotropy_results.json`, micro cor run):** clean HDC mIoU 0.432 (vs
0.427 baseline), fog 0.069 (vs 0.075), crosstalk 0.095 (vs 0.090), 8-condition mean
0.300 (vs 0.297). The fog dead-coordinate fraction **halves** (0.101 vs 0.183) and
crosstalk dead-fraction halves (0.020 vs 0.041). Training is the best of the three
micro runs (best-val IoU 0.303 vs dg 0.283 / sv 0.289).

**What the micro gate shows:**

1. **The corruption augmentation flips the fog naive-EMA gap from strongly negative
   to the best of the three micro models** (-0.43 baseline, +0.28 supcon_vib,
   +0.37 cor). The mechanism is visible per class: the baseline's negative fog gap
   came from the update destroying the ROAD prototype (0.483 -> 0.205); cor protects
   it (0.495 -> 0.460), and car's LP recall stays high (0.893) so the update helps
   car (0.057 -> 0.063).
2. **It produces a cleaner fog space.** The dead-coordinate fraction under fog is
   halved (0.183 -> 0.101) and the 8-condition HDC mean is at least tied with the
   baseline, so the augmentation does not trade robustness for a decode loss.
3. **The minority feat_cos are roughly unchanged at micro** (car 0.825 vs 0.847,
   vegetation 0.450 vs 0.431, building 0.409 vs 0.451) — the micro effect is a
   cleaner overall fog space and a protected majority prototype, not yet a
   per-minority tightening. The minority-specific claim is only decidable where the
   baseline polarizes, i.e. at medium scale.

**Next.** The decisive test is the medium-lite cor run (12 ep / 100% data, ~5h):
whether at the scale where dglsspp polarizes (dg_med rho(freq, feat_cos) +0.48, fog
oracle 0.176) the cor variant holds minority classes closer to their prototypes
(lower rho, higher fog oracle) and keeps the naive-EMA gap positive.

### 6.1 Medium-lite cor run (12 ep / 100% data)

The medium-lite corruption-augmented run completed cleanly (12 epochs, train IoU
0.161 -> 0.432). Isotropy, compared with the plain DGLSS++ medium run (dg_med, 24
ep / 100% data). Budget caveat up front: cor_med used HALF the epochs of dg_med, so
this is not a same-budget comparison — it shows the cor variant is at least as
robust as the baseline at half the cost.

**HDC mIoU per condition (higher is better):**

| condition | dg_med (24ep) | cor_med (12ep) |
| :--- | :--- | :--- |
| clean | 0.530 | **0.532** |
| fog | 0.068 | **0.073** |
| crosstalk | **0.115** | 0.108 |
| snow | 0.396 | **0.415** |
| wet_ground | **0.483** | 0.450 |
| incomplete_echo | 0.448 | **0.458** |
| beam_missing | **0.506** | 0.500 |
| motion_blur | **0.502** | 0.495 |
| cross_sensor | **0.434** | 0.425 |
| **mean (8 corrupted)** | **0.369** | 0.365 |

**Space structure (lower dead-fraction is better):**

| condition | dg_med | cor_med |
| :--- | :--- | :--- |
| fog deadF | 0.221 | **0.208** |
| crosstalk deadF | 0.147 | **0.059** |
| crosstalk PR | 2.30 | **2.95** |
| fog LP | **0.524** | 0.518 |
| crosstalk LP | 0.224 | **0.231** |

**What the medium-lite isotropy shows:**

1. **The cor variant matches the full-budget baseline at half the cost.** cor_med
   (12 ep) lands on the same 8-condition HDC mean (0.365 vs 0.369) and clean decode
   (0.532 vs 0.530) as dg_med (24 ep), with fog better (0.073 vs 0.068) and the
   healthy conditions roughly tied. The corruption augmentation is at least as
   effective per epoch as the plain DGLSS++ training, likely more.
2. **The crosstalk space is far healthier.** The dead-coordinate fraction drops
   0.147 -> 0.059 (a 2.5x reduction), participation ratio rises 2.30 -> 2.95 (less
   saturated / less anisotropic), and the linear probe rises 0.224 -> 0.231. The
   corruption-augmented model saturates the HDC sign-projection far less under
   crosstalk, the same direction the micro run showed for fog.
3. **The fog mean-fraction stays high** (0.830, matching dg_med) and the fog
   dead-fraction only improves slightly (0.221 -> 0.208), so the augmentation has
   not eliminated fog's shared-mean dominance at this budget — but it no longer
   needs to, because the decode is at least as good.
4. **The decisive per-class test is still pending.** The isotropy does not measure
   the minority-class proximity (rho(freq, feat_cos)), the fog/crosstalk labeled
   ceiling, or the naive-EMA gap. Those need the `scale_gap_diag` eval on the
   cor_med checkpoint, which is the next diagnostic before deciding on a full-budget
   cor run (24 ep / 100%) for the same-budget head-to-head.

### 6.2 Per-class autopsy of cor_med (`scale_gap_diag`, same 100k/100k split)

Same-split aggregate vs the plain medium DGLSS++ baseline. Budget caveat: cor_med
is 12 ep / 100%, dg_med is 24 ep / 100%, so the ceiling comparison below is
confounded by training budget (the ceiling rises with epochs).

| extractor | cond | zs | naive | oracle | gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| dg_med (24ep) | fog | 0.082 | 0.100 | **0.176** | +0.19 |
| cor_med (12ep) | fog | 0.084 | 0.094 | 0.147 | +0.16 |
| dg_med (24ep) | crosstalk | 0.125 | 0.127 | 0.222 | +0.02 |
| cor_med (12ep) | crosstalk | 0.109 | 0.121 | 0.200 | **+0.14** |

Spearman rho(freq, feat_cos): dg_med fog +0.48 / crosstalk +0.29; cor_med fog +0.57
/ crosstalk +0.33.

Key per-class differences (car = class 4):

| metric | dg_med fog | cor_med fog | dg_med xtalk | cor_med xtalk |
| :--- | :--- | :--- | :--- | :--- |
| car feat_cos | 0.321 | **0.599** | 0.521 | **0.834** |
| car LP recall | 0.149 | **0.372** | 0.401 | **0.758** |
| car zs -> naive | 0.122 -> 0.102 | 0.075 -> 0.090 | 0.352 -> 0.252 (hurt) | 0.269 -> 0.391 (gap 0.98) |
| car oracle | **0.303** | 0.128 | 0.335 | 0.393 |

**What the medium-lite per-class autopsy shows:**

1. **The labeled ceiling was NOT raised at this budget.** cor_med fog oracle 0.147
   is below dg_med's 0.176 (crosstalk 0.200 vs 0.222). Because the ceiling rises
   with training and cor_med trained half as long, this is not conclusive, but it
   does not support the "augmentation raises the ceiling" hypothesis either.
2. **The majority polarization was NOT reduced.** rho(freq, feat_cos) is +0.57 on
   cor_med fog, higher than dg_med's +0.48 (crosstalk +0.33 vs +0.29). The
   augmentation tightens road's features (0.856 -> 0.936) at least as much as the
   minority classes, so the frequency-dependence of prototype distance does not
   shrink.
3. **What the augmentation clearly does is improve the assignment.** The car class
   shows it on both conditions: fog LP recall 0.149 -> 0.372 and feat_cos 0.321 ->
   0.599; crosstalk LP recall 0.401 -> 0.758 with the naive update now lifting car
   (0.269 -> 0.391, gap 0.98) instead of destroying it (dg_med 0.352 -> 0.252).
   This is exactly the assignment-side lever: the pseudo-labeler can now find the
   mid-frequency classes.
4. **The naive-EMA gap is positive on both conditions, with crosstalk much better**
   (+0.14 vs +0.02). Combined with the aggregate decode matching dg_med at half the
   budget, the augmentation is a real improvement in TTA-relevant structure even
   though it does not move the fog ceiling.
5. **The open question is the budget.** cor_med at 12 ep cannot separate "the
   augmentation raises the ceiling" from "it only improves assignment/efficiency".
   The two decisive runs are (a) plain DGLSS++ at 12 ep / 100% (~5h) for a
   matched-budget comparison with cor_med, or (b) the full 24 ep cor run (~10h) for
   the same-budget head-to-head with dg_med. If the goal is "raise the fog/crosstalk
   ceiling", the full cor run is the direct test; if the goal is the cheaper
   marginal effect of the augmentation, (a) resolves the confound at half the cost.




