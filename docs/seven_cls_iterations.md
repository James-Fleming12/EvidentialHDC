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
   DGLSS / DGLSS++ arms, measured with the isotropy diagnostics.
2. **Prototype-TTA in 7 classes**: the gated weighted prototype update
   (`robust_diagnostic/seven_cls_diag.py`), then the oracle ladder, and whether a
   label-free update beats zero-shot on fog/crosstalk. On crosstalk the oracle gap
   (0.275 vs 0.204) is the headroom to chase.
3. **Adopt the 7-class space for all comparisons** going forward (it is the
   thirdparty protocol).

## Iteration log

(To be filled as the two threads are exercised in the 7-class setting.)

## Scripts and configs

- `config/labels/semantic-kitti-7.yaml` - the exact D3CTTA map, reindexed 0 = background
- `robust_diagnostic/class_count_diag.py` - the class-granularity comparison
- `robust_diagnostic/proto_distance_diag.py` - the distance-gate retention curves
- `robust_diagnostic/seven_cls_diag.py` - the background diagnostics (per-class
  breakdown, label-free gated update, isotropy of the class-count models)
