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

## Iteration 2: full-condition robustness sweep (pending)

(To be filled from `isotropy_diag.py --eval_only` on the three checkpoints.)

## Iteration log (future)

(Subsequent iterations appended here as the encoder thread progresses.)
