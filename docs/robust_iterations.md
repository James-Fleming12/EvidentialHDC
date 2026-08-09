# Robust Feature-Extractor Comparison: Iterations

This tracks the empirical iterations of the robust-encoder comparison (the
isotropic-vs-anisotropic question and the DGLSS / DGLSS++ / supcon_vib shootout).
The framework and its theoretical analysis live in `docs/robust_details.md`; this
document records what was measured, condition by condition.

Throughout, bold marks the best value in each row; for the mIoU / linear-probe /
Hamming columns higher is better, for the dead-fraction and mean-fraction columns
lower is better.

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

## Iteration log (future)

(Subsequent iterations appended here as the encoder thread progresses.)
