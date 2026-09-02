# Domain-Invariant Representation Learning (DIRT-GAN): applying it to the robust feature extractor

Status: assessment + naive overnight test design. Source: `thirdparty/domain_inv_rep_learning.pdf`
(Nguyen et al., "Domain Invariant Representation Learning with Domain Density Transformations", NeurIPS 2021).

## 1. Context of the method

The paper tackles **domain generalization**: train on a set of source domains, generalize to an
unseen target domain, without any target data. The standard recipe is an invariant representation
`z = g(x)` shared across domains, but most methods only align the marginal `p(z)`, which is
insufficient (Figure 1: aligned marginal, unaligned conditional -> 0% accuracy).

**Core idea.** If the domains are related by **density transformation functions** `f_{d,d'}` that map
the data distribution of domain `d` to that of `d'`, then a representation that is **invariant under
all such transformations** is domain-invariant in both the marginal and the conditional sense:

```
min E_{d,d' in Ds, p(x,y|d)} [ l(y, g_theta(x)) + dis(g_theta(x), g_theta(f_{d,d'}(x))) ]
```

**Theorem 1** (the theoretical result we care about): if `f_{d,d'}` is an invertible, differentiable
density transform and `p(z|x) = p(z|f_{d,d'}(x))` for all x, then the representation aligns BOTH
`p(z)` and `p(y|z)` across domains. The invariance condition is the sufficient (and under the
label-invariance assumption, necessary) condition.

**Implementation.** In their setting the domains are unaligned (different datasets), so the
transformations are learned with a StarGAN (domain-classification + adversarial + cycle-consistency
losses). After training, the generator serves as `f`, and the representation is trained with Eq. 7.
**Distance ablation** (Rotated MNIST): squared error 97.1 best, contrastive 95.8, cosine 90.1.
Squared error has a known side effect of shrinking the representation norm.

**Connection to contrastive learning** (Section 2, relevant to us): contrastive learning enforces
similarity under *hand-crafted augmentations*, which are NOT learned and do NOT target real domain
transformations. The paper's contribution is learning `f` and being invariant under the *real* shift.
This is the crux for our application: our DGLSS++ already enforces consistency under hand-crafted
synthetic corruption proxies (GMSIFC/LSCC/SupCon), i.e., the weaker "contrastive" version.

## 2. Applying the theory to our setting: where it is harder, and what to improve

### 2.1 The harder problem setting (vs the paper's)

| | paper | ours |
|---|---|---|
| task | image classification, 7-10 classes | dense LiDAR semantic segmentation, 19 classes per-point |
| domains | rotation / style (near-invertible appearance shifts) | fog / crosstalk (severe, information-destroying) |
| target effect | accuracy drops modestly | the representation COLLAPSES (linear probe ~0.08 mIoU) |
| transform | learnable, near-invertible (GAN) | fog deletes the remission channel (var -> 1.3e-5); crosstalk injects wrong-beam returns |

Two direct consequences:

- **The invertibility assumption of Theorem 1 is violated.** The theorem needs `f` invertible and
  differentiable so `p(x|d)` and `p(x'|d')` are related by its Jacobian. Fog and crosstalk
  are information-destroying operators, not invertible density transforms. A representation that is
  invariant under an information-destroying transform must drop the destroyed information, and that
  information may be exactly what the class boundary needs. The existence premise of a useful
  invariant representation is genuinely in question for fog/crosstalk, which is consistent with the
  arc's finding that the recoverable ceiling on the destroyers is low even with labels.
- **The label-invariance assumption (Remark 1) DOES hold for us.** `p(y|d)` is unchanged across
  corruptions (the corruptions are applied to the same scans with the same labels). So the
  necessary condition for a domain-invariant representation is satisfied; only the transform
  invertibility is in question.

### 2.2 Marginal-only invariance risks over-collapse

The paper aligns both marginal and conditional because the invariance is combined with the
supervised loss. If we add ONLY the marginal term `||z8_clean - z8_corrupt||^2`, the safest
trivial solution is a constant representation (perfectly invariant, zero class information). Our
pipeline already has the conditional machinery (SupCon pull toward clean class anchors, LSCC
class-correlation consistency), so the DIRT term should be ADDED to the full supervised DGLSS++
objective, not replace it. The naive test keeps the entire existing loss and adds the invariance term.

### 2.3 The norm-shrinking side effect conflicts with the HDC magnitude structure

The paper acknowledges squared-error invariance shrinks the representation norm. Our HDC pipeline
is magnitude-sensitive in a specific way: the ARC measured that the feature magnitude is load-bearing
under fog/crosstalk (the sign-binarization discards it, which is why the raw 128-d probe beats the
code probe on the destroyers). Forcing `||z8_clean - z8_aug||^2` may pull the augmented view's
magnitude toward clean, potentially eroding the very signal the raw probe uses. Cosine or a
whitened distance may be safer for the dense/HDC setting even though squared error won the paper's
classification ablation. This is a first thing to test.

### 2.4 The synthetic proxy is the weak version (the paper's own criticism)

The naive test uses our EXISTING synthetic corruption-targeted augmentation (beam drop + depth
jitter + density sparsity + fake returns) as the proxy for `f`. That is exactly the hand-crafted
augmentation the paper argues is weaker than a learned transformation. Moreover, DGLSS++ already
enforces consistency under these proxies via GMSIFC/LSCC/SupCon, so a naive `||z8 - z8_aug||^2`
term may be partly REDUNDANT with the existing losses. The honest expectation is that the naive
test shows little or no gain, because the proxy is neither the real corruption nor a learned
transformation.

### 2.5 What we can improve on (the strongest application)

Our setting has something the paper's does not: **exact aligned clean-to-corrupt pairs** (the same
scan in clean, fog, and crosstalk form, with the same labels). That makes the transformation `f`
known and free, so we do NOT need the GAN, and we can test the theory's cleanest form:
`||z8(clean) - z8(fog)||^2` and `||z8(clean) - z8(crosstalk)||^2` on aligned pairs. This is a
strictly better application of Theorem 1 than the paper's GAN approximation. The caveat: it uses
REAL corrupted scans, which raises the test-leak question. If we restrict the pairs to the TRAINING
sequences (0-7, 9, 10) and evaluate on 08, the instances are unseen (the corruption TYPE is seen,
not the test instances). But to keep the first test leak-free by construction, the naive run below
uses only synthetic views.

## 3. The naive overnight test (no fog/crosstalk leak)

**Question.** Does adding the direct representation-invariance term `L_inv = ||z8 - z8_aug||^2`
on the EXISTING synthetic clean-to-augmented views improve the extractor over the DGLSS++-19
baseline, at no protocol cost?

**Why no leak.** The training data is clean KITTI (seq 0-7, 9, 10) + the existing synthetic
corruption-targeted augmented views. No real fog/crosstalk scan is used anywhere. Evaluation is on
seq 08 fog/crosstalk/others. This is the same protocol as the DGLSS++-19 baseline, so the comparison
is apples-to-apples and the risk of testing on training data is zero by construction.

**Method.** `supcon_vib_dglsspp` on the 19-class map (`semantic-kitti-19.yaml`), 24 epochs at 100%,
6.8M params (the established recipe), with the additional loss term

```
L_total = L_dglsspp + DIRT_W * mean((z8 - z8_aug).pow(2))
```

over the valid masked points, where `z8_aug` is the bottleneck of the existing corruption-targeted
augmented view. Implemented as a `DIRT_W` env weight on the DGLSS++ trainer (0.0 = exact baseline,
so a sweep is cheap). Start with `DIRT_W=1.0` for the overnight; consider a cosine variant in a
follow-up (Section 2.3).

**Evaluation.** The established map19 protocol: `--map19 --ceiling`, heavy, all 8 conditions
(optionally 3-sev), compared directly against the DGLSS++-19 zero-shot/ceiling numbers already
recorded in `twenty_cls_iterations.md`.

**Decisive reads.**

1. Healthy-condition zero-shot: already near the ceiling for DGLSS++-19 (beam_missing 50.8 vs
   ceiling 50.9), so a gain here is unlikely; a LOSS would indicate the invariance term erodes
   class info (over-collapse, Section 2.2).
2. Fog/crosstalk zero-shot: if the synthetic proxy is a good enough stand-in for the real
   corruption, the invariance could push the 7.5/8.8 up. The honest expectation is little movement,
   because (a) the proxy is weaker than the real collapse, and (b) the existing GMSIFC/LSCC/SupCon
   already enforce consistency under the same proxy.
3. The ceiling (labeled oracle): must not drop. If it does, the norm-shrinking (Section 2.3) is
   hurting the feature structure, and the aligned-real-pair or cosine variant is the next step.

**Runner sketch.**

```
DIRT_W=1.0 bash run_retrain_dglsspp_19cls.sh 3        # trains supcon_vib_dglsspp + L_inv, 19-class
CKPT="robust_diagnostic/logs/supcon_vib_dglsspp_19cls" MAP19=1 CEILING=1 \
  OUTJSON="robust_diagnostic/logs/lp_three_decoder_dglsspp_19cls_dirt.json" \
  bash run_lp_three_decoder.sh 3                      # the map19 zero-shot + ceiling eval
```

**Follow-ups if the naive test is inconclusive.**

1. Aligned-real-pair invariance on TRAINING sequences (0-7, 9, 10): `||z8(clean) - z8(fog)||^2`
   and the crosstalk version, the cleanest test of Theorem 1 and the most likely to move
   fog/crosstalk. Restrict to training sequences and confirm the eval (08) instances are unseen.
2. Distance metric: cosine or a whitened/scale-robust distance instead of squared error, to avoid
   the norm-shrinking interaction with the HDC magnitude structure.
3. Lambda scheduling (warm up the invariance weight after the supervised loss has formed class
   structure, to reduce the over-collapse risk in Section 2.2).
