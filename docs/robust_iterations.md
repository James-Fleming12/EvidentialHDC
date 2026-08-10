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

**Total.** $L = L_{\text{sem}}^s + L_{\text{sem}}^a + \lambda_1 L_{\text{SIFC}}
+ \lambda_2 L_{\text{SCC}}$, with weighted cross-entropy for the semantic terms.

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

**Total.** $L = L_{\text{sem}}^s + L_{\text{sem}}^a + \lambda_1 L_{\text{GMSIFC}}
+ \lambda_2 L_{\text{LSCC}}$.

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

