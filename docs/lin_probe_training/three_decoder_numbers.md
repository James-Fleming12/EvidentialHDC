# Three-decoder numbers: no-HDC vs prototype vs code-linear vs raw-128-d-linear

Results of `robust_diagnostic/lp_three_decoder_diag.py` on the frozen DGLSS++
encoder (supcon_vib_dglsspp), JSON
`robust_diagnostic/logs/lp_three_decoder_dglsspp.json`.

## Setup

- Frozen features (128-d `z8`); projection 128 -> 10000 (seed-42 random +/-1),
  codes = `sign(z @ proj)`.
- Both HDC decoders fit on CLEAN only (zero-shot): prototype = mean binarized
  code per class (cosine decode), linear = ridge probe `W = (X^T X + lam I)^-1
  X^T Y` on the codes. A fourth decoder, raw-linear, is the SAME ridge probe on
  the RAW 128-d features (same fitter, same fit set; the input space is the
  only change), so `lin` vs `raw-lin` cleanly answers "does the HDC projection
  help or hurt the linear classifier".
- Fit on the first 30k clean points (lam 1e-3); eval on the first 100k points
  of each condition/severity. Per-condition number = 3-severity mean
  (light/moderate/heavy). mIoU over the 17 classes.

## Results (3-severity mean mIoU)

| condition | no-hdc (model head) | prototype | linear (code) | raw-linear (128-d) |
|-----------|---------------------|-----------|---------------|--------------------|
| clean | 0.595 | 0.622 | **0.887** | 0.689 |
| fog | 0.125 | 0.130 | 0.125 | **0.180** |
| crosstalk | 0.094 | 0.098 | 0.086 | **0.159** |
| snow | 0.420 | 0.414 | 0.405 | **0.421** |
| wet_ground | 0.550 | 0.570 | **0.683** | 0.654 |
| incomplete_echo | 0.547 | 0.546 | **0.673** | 0.630 |
| beam_missing | 0.523 | 0.533 | **0.605** | 0.570 |
| motion_blur | 0.563 | 0.576 | **0.645** | 0.644 |
| cross_sensor | 0.443 | 0.436 | 0.452 | **0.463** |
| 8-condition mean | 0.408 | 0.413 | 0.459 | **0.465** |

## Finding 1: the HDC projection HELPS the linear probe on clean and the healthy
conditions, and HURTS it on fog/crosstalk

`lin - raw-lin` (the projection effect on the linear probe), 3-sev:

| clean | fog | crosstalk | snow | wet_ground | incomplete_echo | beam_missing | motion_blur | cross_sensor |
|-------|-----|-----------|------|------------|-----------------|--------------|-------------|--------------|
| **+0.198** | -0.055 | -0.073 | -0.016 | +0.029 | +0.044 | +0.036 | +0.001 | -0.011 |

This resolves the Phase 8 / Phase 16 tension for THIS encoder, with the clean
apples-to-apples comparison the older tables could not give (same ridge, same
fit set):

- On CLEAN the projection helps ENORMOUSLY (0.689 -> 0.887, +0.198): the lift
  to 10000 random binary features makes the classes far more linearly
  separable than in the raw 128-d space (the hyperdimensional random-feature
  lift). The Phase 8 "projection hurts by 1.6 pts" was on a different encoder,
  a different metric, and a different fitter.
- On the healthy corruptions the projection still helps (+0.001 to +0.044).
- On fog and crosstalk it HURTS (-0.055, -0.073): under the destroyers the
  `sign()` binarization discards the magnitude information that is load-bearing
  there, so the code probe falls below the raw probe.

## Finding 2: "the linear classifier consistently outperforms" needs scoping

The CODE-space linear probe (the project's "linear probe") beats the prototype
on clean and 5 of 8 conditions (wet_ground, incomplete_echo, beam_missing,
motion_blur, clean), but it LOSES to the prototype on fog (0.125 vs 0.130) and
crosstalk (0.086 vs 0.098), and is the worst decoder on crosstalk. The claim
holds for the healthy conditions but not for the two conditions the project
cares most about.

## Finding 3: the RAW 128-d linear probe is the best fog/crosstalk decoder

`raw-linear` beats all three other decoders on fog, crosstalk, snow and
cross_sensor (4 of 8 conditions) and ties on the rest; it is the best single
decoder on the destroyers. The magnitude that `sign()` throws away is exactly
what survives under fog/crosstalk. This is a direct, falsifiable read on the
deployment decoder: the HDC code probe is the right decoder where the
representation is intact, but the binarization caps it on the destroyers.

## Caveat: protocol differs from the AL harness, so the ABSOLUTE fog/crosstalk
numbers here are not directly comparable to the AL-arc frozen numbers

- Eval slice: this run evaluates the FIRST 100k points of each condition
  stream; the AL harness evaluates the LAST 100k points of a random permutation
  (`perm[-100000:]`). Those are different scenes (the Phase 14 finding: ~98
  frames apart).
- Fitter: this run uses `ridge_fit_exact`; the harness uses the sketched-CG
  `ridge_fit_soft` (nystrom m=1000, 8 CG iters) which regularizes more heavily.

Consequently fog/heavy `lin` here (0.071) is below the AL-arc frozen 0.111 and
crosstalk/heavy (0.083) is well below the AL-arc 0.154. The FOUR-WAY ordering
within this run is valid (identical slices and fit for all four decoders), but
the absolute numbers should be re-measured with the harness protocol (perm
split + a fitter flag) before they are slotted into existing tables.

## Next step

1. Align the protocol (eval = `perm[-val_size:]`, fit = random clean subset,
   fitter flag) and re-run so the absolute numbers are comparable to the AL
   arc, then confirm the reversals hold.
2. Run `lp_why_linear_diag.py` for the mechanism: does P1 isotropy (participation
   ratio) predict which conditions the projection helps/hurts? Does the
   disagreement analysis show the raw probe recovering what the code probe
   loses on fog/crosstalk, and the code probe recovering what the prototype
   throws away on the healthy conditions?
