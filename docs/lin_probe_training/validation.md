# Linear Probe Validation: what the linear probe is, why it is the reference decoder, and the evidence

This file validates the value of adding the linear probe to the HDC mechanism
over the basic alternatives: the model's own trained head (no HDC), the
prototype / nearest-centroid decoder (R1), the linear probe on the HDC code
(R4), and the same linear probe on the raw 128-d features. It is also the
context for the `lin_probe_training` efficiency work (HyperLiDAR buffer
selection as an anchor for training efficiency; Hamming / popcount-style and
near-orthogonality-based decode for inference efficiency). It documents what
the linear probe does, the prior consensus across the old docs and README, the
initial measurement on the current DGLSS++ encoder, and the decisive test still
pending.

## 1. The frozen feature the classifiers see

The encoder (DGLSS++ / cov-shift DGLSS++) maps each input scan to per-point
128-d features `z` (the bottleneck volume `z8`, the last element the model
forward returns: `out = model(x)` gives `(pred, [aux,] out)` where `out` is the
128-d feature map; the AL diagnostics read it as `z8 = out[2] if len(out)==3
else out[1]`). Every decoder below is a function of these FROZEN 128-d features;
the encoder is never fine-tuned.

## 2. The HDC encoding

The linear probe and the prototype classifier do not operate on the raw 128-d
vectors; they operate on binarized hypervectors:

```
proj : 128 x 10000 random bipolar matrix,  proj_ij ~ 2*Bernoulli(0.5) - 1  (seed 42)
code = sign(z @ proj) in {+1, -1}^10000
```

`get_hdc_projection` (modules/oracle_core.py) builds `proj`; `hdc_codes`
(robust_diagnostic/al_full_dataset_diag.py) computes the codes. The codes are
BINARY (+1/-1), so they admit Hamming-distance / popcount-style decode instead
of float dot products, and the 10000 rows live in a near-orthogonal random
subspace (relevant to the inference-efficiency goal). The encoding is a fixed
random projection: it is label-free and never trained.

## 3. The four decoders being compared

| decoder | what it does | decode |
|---------|--------------|--------|
| **No HDC** | the model's own trained head, a 1x1 conv `semantic_output: 128 -> 17` learned jointly with the encoder (`pred = softmax(semantic_output(z))`) | `argmax(pred)` |
| **Prototype HDC (R1)** | per-class mean of the CLEAN codes, L2-normalized: `P_c = mean_{i: y_i=c} sign(z_i proj) / ||.||` (build_prototypes) | `argmax_c cos(code, P_c)` |
| **Linear HDC (R4)** | ridge-regularized least squares on the CLEAN codes with one-hot targets: `W = (X^T X + lam I)^-1 X^T Y`, `W` is 10000 x 17 (ridge_fit_exact) | `argmax(code @ W)` |
| **Raw-128-d linear** | the SAME ridge probe on the RAW 128-d features (same fitter, same fit set; the input space is the only change) | `argmax(z @ W_raw)` |

The "no HDC" number is the plain semantic-segmentation result of the encoder
(the head trained end-to-end). The R4-vs-R1 pair differ only in the DECODING
rule: a single centroid per class vs a full linear map, both on the same
binarized codes. The raw-128-d linear isolates the value of the PROJECTION
itself: it answers "does the HDC projection help or hurt the linear
classifier" with the cleanest possible comparison.

## 4. The prior consensus (old docs and README)

### 4.1 The projection preserves separability; the prototype throws it away (Phase 7/8)

`docs/gen_iterations.md` Phase 7/8, heavy fog, supcon_vib encoder:

| representation stage | linear probe | prototype |
|----------------------|--------------|-----------|
| raw 128-d | **49.4%** | 8.2% |
| random projection (continuous) | **49.0%** | 31.7% |
| sign binarized (the HDC code) | **47.8%** | 24.1% |

Two things established the reference-decoder status of the linear probe:

1. **The HDC encoding preserves the information.** Random projection loses ~0.4
   points and binarization loses another ~1.2 points; the linear-separability
   limit survives the full HDC pipeline nearly intact (49.4 -> 47.8).
2. **The prototype decoder is the bottleneck, not the encoding.** A single
   class-mean centroid throws away ~20 points of accuracy that the linear probe
   recovers. This is the README's "the linear-probe decoder on the HDC code
   recovers the structure the distance-to-prototype rule throws away".

### 4.2 Prototype -> linear is a strict win everywhere (README C10)

The README's full-harness DGLSS++ columns (zero-shot HDC-zs R1, lines 136-143,
vs zero-shot R4, lines 182-189) show the linear probe beating the prototype on
ALL 8 conditions:

| condition | proto (R1) | linear (R4) | gain |
|-----------|------------|-------------|------|
| fog | 6.8 | 9.7 | +2.9 |
| crosstalk | 11.5 | 11.8 | +0.3 |
| snow | 39.6 | 54.3 | +14.7 |
| wet_ground | 48.3 | 49.2 | +0.9 |
| incomplete_echo | 44.9 | 48.2 | +3.3 |
| beam_missing | 50.6 | 58.3 | +7.7 |
| motion_blur | 50.2 | 54.8 | +4.6 |
| cross_sensor | 43.4 | 46.9 | +3.5 |

The C10 section states it verbatim: the linear-probe decoder "raises the
ceiling over distance-to-prototype on every condition" (fog 30.1->36.9% ep-10,
wet_ground 25.4->41.9%). The AL arc treats this as closed: `new_iters.md` says
"R1 prototype decoder closed (R1 < R4 on every condition)", and prototype-based
methods were removed from consideration because they do not hold for the linear
classifier.

### 4.3 The linear probe is at the code space's information limit

`docs/cov_shift/cov_full_scale.md` (decoder-ceiling probe,
`probe_decoder_ceiling_diag.py`, cov_ep10, pool-fit): no more expressive decoder
beats R4 on the code:

- **kNN (code and raw) < R4 linear** on every condition (kNN1-code 0.393-0.558
  vs R4 0.448-0.603). The decision boundary is effectively linear in the code
  space.
- **linear (raw 128) < R4 linear (code) everywhere** (0.37-0.52 vs 0.45-0.60):
  "the HDC binarization HELPS the linear probe, it is not the cap."
- RFF kernel ridge collapses (0.04-0.06, a conditioning artifact); the balanced
  per-class-lam probe is essentially equal to R4.

Consequence: the recoverable gaps (frozen->ceiling) are real headroom, not a
decoder artifact, and the R4 ceiling is the correct upper bound for the TTA/AL
story.

### 4.4 The projection vs the linear probe: depends on the space, but the consensus is "helps"

- **Phase 8 (supcon_vib, healthy space)**: projection slightly hurt the linear
  probe (49.4 -> 47.8, -1.6, accuracy, LogReg-vs-Ridge). The docs frame this as
  "information survives projection+binarization", roughly neutral.
- **Phase 16 (supcon_vib_strongvib, over-collapsed/anisotropic space)**: the
  binarized-10kD linear probe (38.2%) was nearly DOUBLE the raw 128-d probe
  (20.8%) , the projection HELPED, the same "isotropic smoothing over the
  extreme VIB anisotropic rays" mechanism that lifted the prototype 8.2 ->
  31.7.
- **Decoder-ceiling probe (cov_ep10)**: code > raw on every condition.

Net consensus in the docs: the projection helps (or is at worst roughly
neutral for) the linear probe; the prototype->linear change is a strict,
across-the-board improvement; and the linear probe on the code is the
reference decoder and the code-space information limit. Caveats on the older
numbers: Phase 8/16 used different fitters (LogReg vs Ridge) and fit sizes (50k
vs 10k), so directions are trustworthy but magnitudes are not precise.

## 5. Initial four-decoder measurement on the current DGLSS++ encoder (slice run)

`lp_three_decoder_diag.py` (first version) measured the four decoders on the
frozen plain DGLSS++ encoder (`supcon_vib_dglsspp`), zero-shot (clean fit),
3-severity mean. Caveat: this first run used a BIASED protocol (fit on the
first 30k clean points, eval on the first 100k points of each condition
stream), so its absolute numbers are not comparable to the README and its
reversals contradict the consensus (Section 6). Reported here as the
measurement that motivated the full-protocol re-run.

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

Projection effect on the linear probe, `lin - raw-lin` (3-sev):

| clean | fog | crosstalk | snow | wet_ground | incomplete_echo | beam_missing | motion_blur | cross_sensor |
|-------|-----|-----------|------|------------|-----------------|--------------|-------------|--------------|
| **+0.198** | -0.055 | -0.073 | -0.016 | +0.029 | +0.044 | +0.036 | +0.001 | -0.011 |

Three tentative findings from this slice run (all to be confirmed on the full
protocol):

1. **The projection HELPS the linear probe on clean and the healthy conditions,
   HURTS it on fog/crosstalk.** On clean the lift to 10000 random binary
   features made the classes far more linearly separable (0.689 -> 0.887,
   +0.198, the hyperdimensional random-feature lift). On fog and crosstalk it
   HURTS (-0.055, -0.073): the `sign()` binarization discards magnitude, which
   may be load-bearing under the destroyers.
2. **"The linear classifier consistently outperforms" needs scoping.** The code
   probe beats the prototype on clean + 5 of 8 conditions but LOSES to it on
   fog (0.125 vs 0.130) and crosstalk (0.086 vs 0.098), and is the worst decoder
   on crosstalk.
3. **The raw 128-d linear probe was the best decoder on fog/crosstalk** (and
   snow/cross_sensor): it beat all three others on 4 of 8 conditions. The
   magnitude that `sign()` discards may be what survives under the destroyers.

## 5b. Full-protocol confirmation (fog/crosstalk, heavy): the reversal SURVIVED

Full-harness protocol (`lp_three_decoder_diag.py` v2): 200k clean reservoir fit,
full streaming eval of EVERY frame (~300M points/condition), spectral-exact
ridge, severity heavy. Plain DGLSS++ (`supcon_vib_dglsspp`), JSON
`robust_diagnostic/logs/lp_three_decoder_dglsspp.json` (fog/crosstalk only).

| condition | no-hdc (model head) | prototype | code-linear | raw-128d-linear |
|-----------|---------------------|-----------|-------------|-----------------|
| clean | 0.549 | 0.520 | **0.638** | 0.523 |
| fog/heavy | 0.104 | 0.096 | 0.096 | **0.114** |
| crosstalk/heavy | 0.107 | 0.111 | 0.117 | **0.138** |

- **Protocol validated against the README.** fog/heavy code-linear 0.096 vs the
  README DGLSS++ zs R4 9.7%, crosstalk 0.117 vs 11.8%: nearly exact. The
  full-protocol numbers are now directly comparable to the README tables.
- **The projection effect on the linear probe (`lin - raw`)** is clean +0.115
  (helps), fog -0.018, crosstalk -0.021 (hurts). The slice-run reversal SURVIVED
  the full protocol.
- **Code vs prototype**: the code probe beats proto on crosstalk (+0.006) but
  TIES on fog (0.096 vs 0.096). "Linear > proto on every condition" does NOT
  hold on fog for plain DGLSS++ zero-shot.
- **The raw 128-d probe is the best decoder on fog/crosstalk** (0.114 / 0.138).
- **Likely mechanism**: the 10000-d code probe has ~80x the capacity of the
  128-d probe, overfits the clean fit (best on clean 0.638), and generalizes
  worse OOD under the collapse; `sign()` discards the magnitude that is
  load-bearing there. This is a capacity/regularization trade-off, not an
  encoding defect. (`lp_why_linear_diag.py` P1-P6 tests this.)

## 6. Reconciliation: the decisive test is RESOLVED for fog/crosstalk (zero-shot hurt, ceiling help)

The full-protocol run CONFIRMED the slice-run reversal for plain DGLSS++ zero-shot
on the destroyers (Section 5b): raw > code on fog/crosstalk, code ties proto on
fog. So the slice run was NOT an artifact of the biased slice.

This does NOT contradict the cov_ep10 ceiling probe, because it is a different
regime: the cov_ep10 probe was a POOL-FIT (ceiling), where the projection HELPS
even on fog/crosstalk (code 0.448 vs raw 0.373 fog; 0.544 vs 0.480 crosstalk).
The refined story:

- **At the CEILING (in-distribution pool fit), the projection helps everywhere,
  including the destroyers** (cov_ep10, Section 4.3).
- **At ZERO-SHOT (clean fit), the projection helps on clean and healthy
  conditions** (code > raw: clean +0.115, and the healthy-condition slice data),
  **but HURTS specifically on fog/crosstalk** (raw > code), because the
  clean-fit 10000-d probe overfits clean and generalizes poorly under the
  collapse, and `sign()` discards the magnitude that is load-bearing there.
- **The healthy conditions are near-ceiling from zero-shot** (README C10
  DGLSS++ zs->ceil R4 gaps 0.4-7.8) vs fog/crosstalk (15.6 / 17.3). So the
  projection's benefit is realized at ZERO-SHOT on the healthy conditions and
  only at the CEILING on the destroyers.

Remaining evidence gaps before the story is fully nailed:

1. **Full-protocol healthy-condition zero-shot raw-vs-code**: the 8-condition
   sweep of `lp_three_decoder` (only fog/crosstalk confirmed so far). Expect
   code > raw on the healthy conditions, matching the slice run and the
   cov_ep10 ceiling probe.
2. **DGLSS++ ceiling (pool-fit) raw-vs-code on fog/crosstalk**: directly tests
   whether the projection still helps at the ceiling for plain DGLSS++, whose
   code space collapses (cov_ep10 says yes, but it has the input-IN rescue that
   plain DGLSS++ lacks). Requires fitting the probes on a corrupted-pool
   reservoir (a `--pool_fit` variant of `lp_three_decoder_diag.py`).

The mechanism is the next measurement: `lp_why_linear_diag.py` (same full
protocol) tests whether P1 isotropy / participation ratio predicts the
help/hurt split, and whether the disagreement analysis (P6) shows the raw probe
recovering what the code probe loses on the destroyers (the
capacity/regularization story) vs the code probe recovering what the prototype
throws away on the healthy conditions.

## 7. Validation protocol

Two protocols are in use:

- **README / final-paper protocol** (the one to match): clean fit = reservoir
  over ALL clean frames (seed 7, cap 200k), spectral-exact ridge
  (`ridge_fit_exact`, lam 1e-3); eval = FULL streaming decode of EVERY point of
  EVERY frame of seq 08 (~300M points/condition), default severity heavy; the
  clean-fit reservoir is excluded from the clean eval. This is what
  `lp_three_decoder_diag.py` and `lp_why_linear_diag.py` (v2) implement.
- **3-severity average** (AL-arc / GeoID-style reporting): `--sevs
  light,moderate,heavy` for the per-condition mean across severities.

Common: SemanticKITTI sequence 08, 17-class map (semantic-kitti-all.yaml),
conditions fog, crosstalk, snow, wet_ground, incomplete_echo, beam_missing,
motion_blur, cross_sensor (fog and crosstalk are the two destroyers). Zero-shot
protocol: the HDC decoders are fit on CLEAN features only, no labels from the
target condition. Metric: per-class mIoU (ConfAccum), class 0 and absent
classes excluded.

## 8. Efficiency motivation (why this matters)

- **Training efficiency.** HyperLiDAR (thirdparty/HyperLiDAR.pdf) reports that
  buffer selection for the HDC classifiers gives 3-4x mIoU gains from the same
  label budget. The analogous question here: can a smarter choice of WHICH
  points build the prototypes / fit the probe (rather than all clean points)
  raise mIoU at fixed compute or fixed labels, and can it be done more cheaply
  (subsampled X^T X, incremental least squares, etc.).
- **Inference efficiency.** The codes are +1/-1. A linear decode `code @ W` is a
  matrix-vector product that could become Hamming/popcount-style operations if W
  is also binarized (a ternary/binary probe) or if the decode is prototype-only
  (cosine over binary vectors = Hamming distance). The near-orthogonality of the
  random projection rows means the 10000-d space is highly redundant: a smaller
  code dimension, or a sparse probe, may preserve the linear-probe mIoU at a
  fraction of the inference cost. The validation numbers in Sections 4-6 anchor
  these experiments: the projection's value (Section 5) and the R4 reference
  ceiling (Section 4.3) define what a cheaper decoder must not lose.

## 9. Relevant code paths

- `modules/oracle_core.py`: `get_hdc_projection`, `build_hdc_prototypes`,
  `compute_miou`.
- `robust_diagnostic/al_full_dataset_diag.py`: `hdc_codes`, `onehot`,
  `ridge_fit_exact`, `ridge_fit_balanced`, `build_prototypes`, `ConfAccum`,
  `knn_predict` (the prototype cosine decode), `CONDS_ALL`, the
  `--kittic_sev` / 3-severity harness.
- `robust_diagnostic/lp_three_decoder_diag.py` + `run_lp_three_decoder.sh`: the
  four-decoder full-protocol evaluation (v2).
- `robust_diagnostic/lp_why_linear_diag.py` + `run_lp_why_linear.sh`: the
  mechanism diagnostics (P1-P6).
- `modules/network/ResNet.py`: the model forward; `out` (z8) is the last return,
  `pred = softmax(semantic_output(out))` is the no-HDC classifier.
- Prior reference results: `docs/gen_iterations.md` Phase 7/8 and Phase 16;
  `README.md` C10 section; `docs/cov_shift/cov_full_scale.md` decoder-ceiling
  probe (`probe_decoder_ceiling.json`).
