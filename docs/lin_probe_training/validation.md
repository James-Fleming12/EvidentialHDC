# Linear Probe Validation: what the linear probe is, and why it is the reference decoder

This file is the context for the `lin_probe_training` work: more efficient ways to
train the classifiers, ways to raise the trained-classifier mIoU (HyperLiDAR's
buffer selection as an external anchor), and cheaper inference that exploits the
near-orthogonality / binary nature of the hypervectors. It documents what the
linear probe actually does, where it lives in the pipeline, how it compares to the
other decoders, and the validation protocol the experiments will use.

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

## 3. The three decoders being compared

| decoder | what it does | decode |
|---------|--------------|--------|
| **No HDC** | the model's own trained head, a 1x1 conv `semantic_output: 128 -> 17` learned jointly with the encoder (`pred = softmax(semantic_output(z))`) | `argmax(pred)` |
| **Prototype HDC** | per-class mean of the CLEAN codes, L2-normalized: `P_c = mean_{i: y_i=c} sign(z_i proj) / ||.||` (build_prototypes) | `argmax_c cos(code, P_c)` |
| **Linear HDC** | ridge-regularized least squares on the CLEAN codes with one-hot targets: `W = (X^T X + lam I)^-1 X^T Y`, `W` is 10000 x 17 (ridge_fit_exact) | `argmax(code @ W)` |

The "no HDC" number is the plain semantic-segmentation result of the encoder
(the head trained end-to-end). The two HDC decoders both read the binarized
codes; they differ only in the DECODING rule: a single centroid per class
(prototype) vs a full linear map (probe).

## 4. The measured gap: why the linear probe is the reference

The linear probe consistently beats the prototype on every condition; this was
established and isolated in the Phase 7/8 headroom diagnostics
(docs/gen_iterations.md). On heavy fog, on the supcon_vib encoder:

| representation stage | linear probe | prototype |
|----------------------|--------------|-----------|
| raw 128-d | **49.4%** | 8.2% |
| random projection (continuous) | **49.0%** | 31.7% |
| sign binarized (the HDC code) | **47.8%** | 24.1% |

Two things follow:

1. **The HDC encoding preserves the information.** Random projection loses ~0.4
   points and binarization loses another ~1.2 points; the linear-separability
   limit survives the full HDC pipeline nearly intact (49.4 -> 47.8).
2. **The prototype decoder is the bottleneck, not the encoding.** A single
   class-mean centroid throws away ~20 points of accuracy that the linear probe
   recovers. This is the "the linear-probe decoder on the HDC code recovers the
   structure the distance-to-prototype rule throws away" claim from the README.

So the linear HDC probe is the reference decoder: its mIoU is the practical
ceiling of the frozen-feature family, and the prototype decoder is the
efficiency target (it is cheaper but leaves points on the table). The
experiments in this directory are anchored on this gap.

## 5. Validation protocol

- **Data.** SemanticKITTI sequence 08 (the standard held-out sequence for these
  evaluations; the clean train/val split used by the diagnostics). Clean parser
  from `kitti_dir`; corrupted from `kittic_dir/<cond>/<sev>`.
- **Conditions.** fog, crosstalk, snow, wet_ground, incomplete_echo, beam_missing,
  motion_blur, cross_sensor (CONDS_ALL in robust_diagnostic/al_full_dataset_diag.py).
  fog and crosstalk are the two conditions that destroy the representation.
- **Severity.** Each condition has light/moderate/heavy subdirs. The reported
  per-condition number is the 3-severity average (light/moderate/heavy) to match
  GeoID-style reporting; the harness can also run a single severity
  (`--kittic_sev`).
- **Fit.** The no-HDC head is frozen (trained with the encoder). The prototype
  and linear decoders are fit on CLEAN features only (zero-shot protocol: the
  README's "HDC-zs"), then evaluated on each corrupted condition. No labels from
  the target condition are used.
- **Metric.** per-class mIoU (ConfAccum in the diagnostics) and, where useful,
  per-class IoU / accuracy. 17-class map (semantic-kitti-all.yaml).

## 6. Efficiency motivation (why this matters)

- **Training efficiency.** HyperLiDAR (thirdparty/HyperLiDAR.pdf) reports that
  buffer selection for the HDC classifiers gives 3-4x mIoU gains from the same
  label budget. The analogous question here: can a smarter choice of WHICH points
  build the prototypes / fit the probe (rather than all clean points) raise mIoU
  at fixed compute or fixed labels, and can it be done more cheaply (subsampled
  X^T X, incremental least squares, etc.).
- **Inference efficiency.** The codes are +1/-1. A linear decode `code @ W` is a
  matrix-vector product that could become Hamming/popcount-style operations if W
  is also binarized (a ternary/binary probe) or if the decode is prototype-only
  (cosine over binary vectors = Hamming distance). The near-orthogonality of the
  random projection rows means the 10000-d space is highly redundant: a smaller
  code dimension, or a sparse probe, may preserve the linear-probe mIoU at a
  fraction of the inference cost. These are the questions the lin_probe_training
  experiments will measure against the Phase 7/8 reference numbers above.

## 7. Relevant code paths

- `modules/oracle_core.py`: `get_hdc_projection`, `build_hdc_prototypes`,
  `compute_miou`.
- `robust_diagnostic/al_full_dataset_diag.py`: `hdc_codes`, `onehot`,
  `ridge_fit_exact`, `ridge_fit_balanced`, `build_prototypes`, `ConfAccum`,
  `knn_predict` (the prototype cosine decode), `CONDS_ALL`, the
  `--kittic_sev` / 3-severity harness.
- `modules/network/ResNet.py`: the model forward; `out` (z8) is the last return,
  `pred = softmax(semantic_output(out))` is the no-HDC classifier.
- Prior reference results: `docs/gen_iterations.md` Phase 7/8; the Phase 8
  degradation table (Section 4 above).
