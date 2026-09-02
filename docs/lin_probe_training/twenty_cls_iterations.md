# Twenty-class iterations: DGLSS++-19 zero-shot vs GeoID, on the honest 19-class map

Status: first full-protocol comparison. The overnight DGLSS++-19 retrain
(training the DGLSS++ extractor on GeoID's exact 19-class map) finished; this
doc records the numbers against GeoID's reported SemanticKITTI-C results.

## Background: why a 20-class (19-class) setting

The project's earlier decoder comparisons used our `semantic-kitti-all.yaml`
17-class map, whose classes are MERGED: manmade = building+fence+pole+traffic-
sign+structure, driveable = road+parking+lane-marking, vegetation+trunk,
pedestrian = person+bicyclist+motorcyclist. That map inflates mIoU (a coarser
task) and made the "competitive with GeoID" reading unreliable.

GeoID evaluates SemanticKITTI-C on the standard 19-class map (car, bicycle,
motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, road,
parking, sidewalk, other-ground, building, fence, vegetation, trunk, terrain,
pole, traffic-sign). To compare honestly we:
1. Built `config/labels/semantic-kitti-19.yaml` from GeoID's own
   `semantic-kitti-c.yaml` learning map (no dead classes; every class mapped).
2. Added `--map19` to the decoder harness: evaluate on that exact map with
   GeoID's mIoU convention (the mean over the FIXED 19 classes, absent classes
   count as 0; matches `thirdparty/GeoID/utils/eval.py`).
3. Retrained the DGLSS++ extractor with a 20-class head on that map
   (24 epochs at 100% of the standard train split; sequence 08 never trained)
   so the extractor actually learns the fine classes (the frozen 17-class-
   trained features collapsed on the merged sub-classes: parking 0.06 vs road
   0.91, trunk 0.09 vs vegetation 0.82, etc.).

## The comparison table (mIoU %, 19-class map, 3-severity average)

Our numbers are the code-space linear probe (R4) fit on a 200k clean reservoir
(spectral-exact ridge, lam 1e-3), full-dataset streaming eval, averaged over
light/moderate/heavy (the fair severity convention, matching GeoID's reporting;
JSON `robust_diagnostic/logs/lp_three_decoder_dglsspp_19cls_3sev.json`). Our labeled
ceiling is the same probe fit on a 400k corrupted-pool reservoir (the supervised
oracle upper bound, NOT a method). GeoID numbers are the paper's Table 1
(MinkowskiNet): source-only = no adaptation, adapted = its gradient test-time
training.

| condition | DGLSS++-19 zs (ours) | DGLSS++-19 ceil (ours, oracle) | GeoID source-only | GeoID adapted |
|-----------|----------------------|--------------------------------|-------------------|---------------|
| fog | **16.8** | 25.8 | 33.20 | 40.14 |
| crosstalk | **8.9** | 21.9 | 24.46 | 40.82 |
| snow | **43.1** | 46.7 | 32.04 | 40.63 |
| wet_ground | **47.3** | 51.4 | 45.53 | 47.66 |
| incomplete_echo | **47.6** | 47.2 | 50.54 | 50.79 |
| beam_missing | **52.5** | 52.8 | 51.05 | 52.10 |
| motion_blur | **47.9** | 49.3 | 54.09 | 54.59 |
| cross_sensor | **46.6** | 48.0 | 47.76 | 48.97 |
| mean | 38.8 | 42.9 | 42.33 | 46.96 |

## CRITICAL caveats (read the table carefully)

1. **Severity convention is now MATCHED.** Both sides are 3-severity averages
   (light/moderate/heavy), so the head-to-head is apples-to-apples on severity.
   (Heavy-only numbers are available in the JSON; the 3-sev mean is higher on most
   conditions, most notably fog 7.5 -> 16.8, and barely on crosstalk 8.8 -> 8.9.)
2. **Label space and metric DO match.** Both use the same 19-class map and the
   same fixed-19 mIoU mean. That part is comparable.
3. **Backbone / capacity differ.** Ours: range-view SENet (DGLSS++-19,
   6.8M params, 24 epochs). GeoID: point-based MinkowskiNet (MinkUNet34,
   ~37.9M params, 200k training steps). So this compares a 6.8M range-view
   extractor to a ~38M point-based one.
4. Our ceiling is a labeled oracle (fit on corrupted-pool labels GeoID does
   not have); report it only as headroom, not a method-vs-method number.

## Initial reads

- **The healthy conditions are competitive at zero-shot, now severity-fair.** On
  snow, wet_ground, beam_missing, incomplete_echo, motion_blur, cross_sensor, our
  3-sev zero-shot (43.1-52.5) exceeds GeoID's source-only on three conditions
  (snow 43.1 vs 32.04, beam_missing 52.5 vs 51.05, wet_ground 47.3 vs 45.53) and
  is within 2-7 points on the rest. On beam_missing we are at parity with GeoID's
  ADAPTED (52.5 vs 52.10).
- **Fog/crosstalk still collapse, and the ceiling is the damning number.**
  Our fog/crosstalk zero-shot (16.8/8.9) is far below GeoID source-only
  (33.2/24.5). Even our LABELED CEILING on fog (25.8) is below GeoID's
  SOURCE-ONLY (33.2). That means: with every label, the DGLSS++-19 features
  cannot match what GeoID's MinkowskiNet does with no adaptation on fog. This
  is the clearest evidence yet that the fog/crosstalk representation
  collapse is the binding constraint, and it is the direct motivation for the
  geoid-cenet38 run (is the GeoID signal / capacity transferable to a
  range-view network, or is it specific to the point-based MinkowskiNet?).
- **The retrain helped the fine classes.** The 19-class retrained zero-shot
  (e.g., wet_ground heavy 44.0) is well above the frozen-17-class-features map19
  number (wet_ground heavy 34.1), confirming the merged-sub-class collapse was
  largely a training-map issue.

## Next steps

1. ~~Confirm the severity effect~~ (DONE: the table above is the 3-sev mean,
   apples-to-apples with GeoID's reporting; heavy-only is in the JSON).
2. The geoid-cenet38 run (GeoID loss on a range-view CENET scaled to
   MinkUNet34's ~37.9M params, 19-class map) tests whether a capacity-matched
   range-view extractor with the GeoID objective produces the reliable feature
   space the TTA signal needs, or whether that is specific to MinkowskiNets.
3. If fog/crosstalk collapse persists across all range-view variants, the
   comparison story is: healthy-condition parity with GeoID's zero-shot/adapted
   at a fraction of the capacity, with the destroyers as the open problem.
