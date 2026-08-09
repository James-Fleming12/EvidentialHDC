# Seven-Class Setting: Robust Encoder + Prototype TTA

This tracks the pivot to the D3CTTA / GIPSO 7-class SemanticKITTI setting
(`config/labels/semantic-kitti-7.yaml`), carrying forward the two threads the
17-class setting could not resolve: **more robust feature extractor training** and
**test-time adaptation that updates prototypes when necessary**.

## Why the 7-class setting

- **Phase 24.12** showed the 17-class (12-evaluated) label space was hurting the
  HDC pipeline: with the exact D3CTTA 7-class map, clean HDC prototype mIoU jumped
  +19 points (0.444 -> 0.632), the absolute fog/crosstalk numbers were higher, and
  the LP robustness gap was smaller on both conditions. The 17-class space carries
  fragile rare classes (bicycle ~0.0002, pedestrian ~0.0005, truck ~0.0022 of
  points) that collapse under corruption and poison the prototypes; the 7-class
  space folds them into well-supported superclasses.
- It is the protocol the thirdparty papers (D3CTTA, GIPSO) actually use, so it is
  the fair comparison space. Part of their apparent fog/crosstalk advantage was
  this label granularity, not just their mechanism or features.
- **Phase 24.13** established that the standard uncertainty signal (distance to the
  nearest clean prototype) works mechanically in this setting: on 7-class fog the
  distance gate exactly reaches the full-label oracle (0.165), so the gating
  mechanism is already at the label-free ceiling. The remaining gap to clean
  (0.771 at 10% retention) is the feature-structure limit, which is the encoder
  problem, not a decode problem.

## The two threads

### Thread 1: robust feature extractor training

The encoder thread carried over from the 17-class work: the `supcon_vib` family
(hard-negative SupCon, fragile-class weighting, VIB), the DGLSS / DGLSS++ /
supcon_vib isotropy comparison (the `robust_diagnostic` suite), and the
class-count-aware regime. In the 7-class setting the encoder has far fewer,
well-supported classes, so:

- the artifact-separation and isotropy questions are cleaner (no dead rare classes
  dragging the prototypes),
- the class-conditional collapse (Phase 24.2) is expressed on classes with real
  support instead of vanishing ones,
- and the hardneg finding still applies: the artifact-separation training is what
  makes the distance signal informative (25.7 hardneg distance AUROC 0.96 / 0.88
  vs plain supcon_vib 0.76 / 0.71).

The class-count models (`classcount_seven`) are the plain `supcon_vib` baseline in
this setting.

### Thread 2: TTA that updates prototypes when necessary

The decode thread: use the uncertainty signal to decide WHEN and WHICH points
update the prototypes, then re-decode. In the 17-class setting every update variant
poisoned the decode (Phases 14, 22.x) because the prototypes were corrupted and the
confident pseudo-labels were wrong. The 7-class setting changes the preconditions:

- prototypes are healthy (clean proto mIoU 0.632 vs 0.444),
- the distance signal is a decent correct-vs-wrong ranker (AUROC 0.71-0.76 on
  fog/crosstalk),
- the full-label oracle for crosstalk (0.275) is well above the clean-centroid
  gate (0.204), so re-estimating prototypes can genuinely help there.

The open question this thread tracks: does a LABEL-FREE gated prototype update
(weight points by distance-to-clean-prototype, re-estimate, re-decode) beat the
clean-centroid zero-shot on fog/crosstalk, and how close to the oracle? On fog the
24.13 gate already equals the oracle, so the expectation is that updates add little
there; on crosstalk the oracle gap (0.275 vs 0.204) is the headroom.

## Established evidence (pointer table)

| Phase / doc | Finding relevant to the 7-class pivot |
| :--- | :--- |
| 24.11 | D3CTTA mechanism failed on our 17-class features (sel-acc ~0.1); D3CTTA uses this 7-class space |
| 24.12 | 7-class clean HDC proto mIoU +19 pts; smaller fog/crosstalk LP gaps |
| 24.13 | distance gate reaches the 7-class fog oracle (0.165); crosstalk gate 0.204 vs oracle 0.275 |
| 25.7 / evi_iterations | hardneg artifact separation sharpens the distance signal (AUROC 0.96 / 0.88) |
| robust_diagnostic | isotropy comparison (DGLSS / DGLSS++ / supcon_vib) and the D3CTTA recoverability check |

## Plan

1. **Encoder baseline in 7 classes**: `classcount_seven` (plain `supcon_vib`,
   10 epochs at 50% data) is the reference. Extend to the hardneg variant and the
   DGLSS / DGLSS++ arms, measured with the isotropy diagnostics. The isotropy
   comparison is the decisive encoder test (both baselines sit at PR ~4 and the
   collapse mechanism is not yet triggered).
2. **Prototype-TTA in 7 classes**: the label-free gated weighted prototype update
   is CLOSED (Iteration 1: flat on fog/crosstalk, same assignment wall as
   iterations 0-8). The oracle gap on crosstalk (0.275 vs 0.204 gate) is real but
   requires true labels; a label-free version would need an assignment signal the
   distance gate does not provide. The remaining TTA question is whether any
   label-free update (not gate + re-estimate) can move the centroids toward the
   corruption.
3. **Adopt the 7-class space for all comparisons** going forward (it is the
   thirdparty protocol).

## Iteration log

### Iteration 1: per-class breakdown, label-free gated update, isotropy (`seven_cls_diag.py`)

**Per-class nearest-centroid IoU (7-class model):**

| class | clean | fog | crosstalk |
| :--- | :--- | :--- | :--- |
| vehicle (1) | 0.84 | 0.08 | 0.30 |
| pedestrian (2) | 0.10 | 0.00 | 0.00 |
| road (3) | 0.91 | 0.46 | 0.40 |
| sidewalk (4) | 0.81 | 0.01 | 0.05 |
| terrain (5) | 0.68 | 0.03 | 0.03 |
| manmade (6) | 0.42 | 0.05 | 0.18 |
| vegetation (7) | 0.67 | 0.11 | 0.07 |

**Findings:**

1. **Pedestrian is the universal casualty**: the one rare class in the 7-class map
   (person + moving-person, ~0.0003 of points) is 0.10 clean and 0.00 under both
   corruptions. The identical pattern holds in the 17-class map (person 0.11 ->
   0.00). Road is the universal survivor (0.46 / 0.40). The 7-class mIoU is carried
   by road plus the partial survivors; the class-conditional collapse (Phase 24.2)
   is not fixed by the coarse map, only hidden for the classes folded into
   background. Pedestrian is the standing casualty to target (the fragile-class
   weighting thread).

2. **The label-free gated prototype update is FLAT** (fog 0.104 -> 0.113 @ 50% ->
   0.105 @ 10%; crosstalk 0.146 -> 0.152 @ 50% -> 0.146 @ 10%; all17 the same).
   Re-estimating the centroids on the distance-confident subset just reproduces the
   clean centroids, because those points already decode correctly; the far points
   that would move the centroids toward the corruption are exactly the ones the
   label-free gate excludes. The true-label oracle reaches 0.165 / 0.275 (seven),
   so the headroom is real but requires the assignment the gate cannot provide (the
   iterations 0-8 wall: assignment accuracy is not re-estimate quality). "Update
   prototypes when necessary" via distance-gating does not recover fog/crosstalk.

3. **Isotropy: PR ~4 for both models** (seven 3.7, all17 4.5), with near-identical
   HDC diversity (deadF 0.014 / 0.009, hamming 0.40 / 0.41). The clean features are
   highly anisotropic in both, yet the HDC pathway is healthy (deadF ~1%) and the
   7-class model decodes better on clean (0.632 vs 0.443). So the Phase 24.12 +19
   clean mIoU is a LABEL-SPACE effect (fewer, well-supported classes), not a
   feature-geometry effect. The anisotropy-collapse mechanism (dead-coordinate
   saturation) is NOT triggered at the supcon_vib baseline, so the DGLSS / DGLSS++
   comparison (isotropy_diag) is the test that must show whether those regimes push
   the space below this PR ~4 baseline.

**Implications:** the label-free gated-update TTA is closed as a mechanism on these
features (same assignment wall as iterations 0-8); the 7-class decode advantage is
label-space, not encoder geometry; the encoder-geometry question is unchanged and
the isotropy comparison remains the decisive encoder test.

## Scripts and configs

- `config/labels/semantic-kitti-7.yaml` - the exact D3CTTA map, reindexed 0 = background
- `robust_diagnostic/class_count_diag.py` - the class-granularity comparison
- `robust_diagnostic/proto_distance_diag.py` - the distance-gate retention curves
- `robust_diagnostic/seven_cls_diag.py` - the background diagnostics (per-class
  breakdown, label-free gated update, isotropy of the class-count models)
