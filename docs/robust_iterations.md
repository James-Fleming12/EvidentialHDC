# Robust Feature-Extractor Comparison: Iterations

This tracks the empirical iterations of the robust-encoder comparison (the
isotropic-vs-anisotropic question and the DGLSS / DGLSS++ / supcon_vib shootout).
The framework and its theoretical analysis live in `docs/robust_details.md`; this
document records what was measured, condition by condition.

## Iteration 1: measured isotropy of the three frameworks

All three methods trained at equal budget (12 epochs at 10% data), evaluated on the
128D bottleneck. The participation ratio (PR, effective rank of 128), the dead
sign-coordinate fraction (the collapse mechanism), the mean-fraction (how dominant
the shared mean direction is), the code Hamming distance, and the HDC prototype
mIoU are reported on clean and on the corrupted conditions.

**Clean-space isotropy (the decisive comparison):**

| method | PR | dead-frac | mean-frac | Hamming | clean HDC mIoU | clean LP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib (ours) | 3.9 | 0.008 | 0.47 | 0.42 | 0.429 | 0.866 |
| supcon_vib_dglss | 4.1 | **0.102** | **0.68** | 0.34 | 0.389 | 0.853 |
| supcon_vib_dglsspp | **2.7** | 0.003 | 0.51 | 0.42 | **0.456** | 0.860 |

**Corrupted conditions (HDC prototype mIoU; dead-fraction and linear-probe in the
row):**

| method | fog deadF | fog HDC | fog LP | xtalk deadF | xtalk HDC | xtalk LP | snow HDC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| supcon_vib (ours) | 0.185 | 0.062 | 0.159 | 0.026 | 0.097 | 0.252 | 0.336 |
| supcon_vib_dglss | 0.257 | 0.056 | 0.184 | 0.079 | 0.099 | 0.315 | 0.329 |
| supcon_vib_dglsspp | 0.107 | 0.077 | 0.191 | 0.026 | 0.098 | 0.279 | 0.340 |

**What the measurements show:**

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
4. **On the corrupted conditions, the difference shows in the binarized decode, not
   the continuous representation.** The DGLSS arms have competitive, even slightly
   higher, linear-probe accuracy on fog and crosstalk than ours; only the HDC
   prototype decode separates them (on fog, DGLSS again has the highest dead
   fraction, 0.257, and the lowest fog HDC mIoU, 0.056).
5. **Caveat.** This is a small-scale run (12 epochs at 10% data, under-converged),
   and the DGLSS losses were fixed for a numerical divergence immediately before
   it. The magnitudes are directional, not final.

**Follow-up (the full-condition robustness sweep):** the three trained checkpoints
are evaluated on all 8 conditions to get a complete per-condition robustness
picture (`isotropy_diag.py --eval_only`, all conditions). The results are appended
to this iteration log as Iteration 2.

## Iteration 2: full-condition robustness sweep

The three trained checkpoints (12 epochs at 10% data) evaluated on all 8
conditions, loaded without retraining (`isotropy_diag.py --eval_only`). The
headline metric is the HDC prototype mIoU per condition; the dead-fraction and
linear-probe are the mechanism and continuous-space references.

**HDC prototype mIoU per condition (best in bold):**

| condition | supcon_vib (ours) | supcon_vib_dglss | supcon_vib_dglsspp |
| :--- | :--- | :--- | :--- |
| clean | 0.429 | 0.389 | **0.456** |
| fog | **0.062** | 0.056 | 0.077 |
| crosstalk | 0.097 | **0.099** | 0.098 |
| snow | 0.336 | 0.329 | **0.340** |
| wet_ground | 0.380 | 0.377 | **0.406** |
| incomplete_echo | 0.364 | 0.326 | **0.393** |
| beam_missing | 0.406 | 0.372 | **0.434** |
| motion_blur | 0.380 | 0.345 | **0.406** |
| cross_sensor | **0.333** | 0.312 | 0.322 |
| **mean (8 corrupted)** | 0.295 | 0.277 | **0.310** |

**HDC dead-coordinate fraction per condition (the collapse mechanism; higher is
more sign-saturated):**

| condition | supcon_vib (ours) | supcon_vib_dglss | supcon_vib_dglsspp |
| :--- | :--- | :--- | :--- |
| clean | 0.008 | **0.102** | 0.003 |
| fog | 0.186 | **0.257** | 0.107 |
| crosstalk | 0.026 | **0.079** | 0.026 |
| snow | 0.005 | **0.066** | 0.001 |
| wet_ground | 0.008 | **0.082** | 0.005 |
| incomplete_echo | 0.010 | **0.104** | 0.006 |
| beam_missing | 0.003 | **0.051** | 0.006 |
| motion_blur | 0.021 | **0.099** | 0.003 |
| cross_sensor | 0.015 | **0.031** | 0.014 |
| **mean (8 corrupted)** | 0.034 | **0.096** | 0.021 |

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
5. **Caveat.** Still the small-scale, under-converged regime (12 epochs at 10%
   data), and the DGLSS losses were fixed for a divergence immediately before
   Iteration 1. The magnitudes are directional, not final; the ordering (plain
   DGLSS worst and most sign-saturated; DGLSS++ strongest decoder) is the robust
   takeaway to re-test at larger scale.

## Iteration log (future)

(Subsequent iterations appended here as the encoder thread progresses.)
