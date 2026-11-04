# GraphCritic v0.2 graph-only results

These are graph construction and target diagnostics, not a QC or FQL learning
comparison. The v0.2 build ran on the same OGBench Scene task1 training dataset
as the saved v0.1 radius-0.85 artifact.

## Artifact comparison

| Metric | v0.1 node graph, raw standardized state, radius 0.85 | v0.2 occurrence graph, default Scene config |
|---|---:|---:|
| Dataset ID | `ogbench:scene-play-singletask-task1-v0:66625e0cd9f2fcf92f5988f4d1bedeb2aa8b1e7481662a79751ca02dc0423cb2` | same |
| Graph content hash | `c1062f7924d6263ecd3854a0372ef0440dbfde327db6dae5ae88eb7369308e35` | `847e13a048b4d2513cfc9d391e2dcdda5847349d493ad5ceb499e976db4eff65` |
| Graph schema / kind | 0.2 / node | 0.3 / state occurrence |
| States / nodes | 1,001,000 / 643,058 | 1,001,000 / 1,001,000 |
| Temporal / bridge edges | merged transition graph; 675,471 total edges | 1,000,000 / 0 |
| Reachable fraction | 0.591504 | 0.034345 |
| Valid `h=5` target fraction | 0.996 | 0.996 |
| Valid non-success rows at cap | 0.996982 | 0.968358 |
| Target cap | 1006 steps | 1006 steps |
| Monotonicity / rank correlation | 0.981400 / 0.853955 | 0.999968 / 1.000000 |
| Graph build time / peak RSS | not recorded in saved artifact | 521.15 s / 3310.7 MiB |

The v0.1 row is read from the saved artifact at
`outputs/graphs/scene-play-singletask-task1-v0/radius-0.85/`. Its graph hash
and dataset ID are recorded in that artifact's `graph.json`. The v0.2 row is
from `outputs/scene-v02/`.

The v0.2 target provenance mix over valid rows is 0.2933% chunk-success,
3.1549% finite graph cost, and 96.5518% cap. The 96.8358% cap rate uses the
gate denominator: valid, non-success rows only. The two percentages therefore
have different denominators.

## Why v0.2 retained no bridges

The default configuration uses `data_calibrated`, five distance bins, and a
200-step bridge-penalty cap. It found 2,596,862 bridge candidates. Every
candidate was removed because its distance-bin lower-confidence penalty
exceeded that cap:

| Distance bin | Action-matched evidence (+ / −) | Lower bound `ell_lcb` | Penalty `lambda_steps` |
|---:|---:|---:|---:|
| 1 | 26,738 / 213,618 | 0.109759 | 219.84 |
| 2 | 7,655 / 241,008 | 0.029988 | 348.94 |
| 3 | 3,711 / 232,414 | 0.015131 | 417.00 |
| 4 | 2,218 / 217,047 | 0.009629 | 461.97 |
| 5 | 1,348 / 200,861 | 0.006257 | 504.86 |

Thus v0.2 currently fails the training gate: `reachable_frac=0.034345` is
below 0.5 and the valid non-success cap rate `0.968358` exceeds 0.5. The
finite-cost paths are temporal replay within recorded trajectories, not
cross-episode stitching. `bridge_use_frac` and `stitch_gain` are both zero.
The gate returned exit code 2 after saving the graph, targets, and diagnostics;
no QC/FQL training was started from this artifact.

The graph has 32 Scene modes. Success states occur in modes 15 (2,408 states)
and 31 (154 states). It contains 37,360 contracted temporal segments (median
length 33, 95th percentile 47, maximum 349). The nearest-neighbor backend was
`jax_gpu`; `d_step=0.848614` and the absolute bridge radius was 1.697228.

## Simulator checks and remaining evidence

The saved Scene sidecar is aligned to the 1,001,000 state occurrences. A
200-row replay check did **not** meet the specified `1e-4` observation tolerance:
17 rows exceeded it, with maximum absolute feature error 1.476283. Two rows
have large errors in object-yaw features 26–27; the largest is row 525285
(feature 27 error 1.476283). The other 15 failing rows have smaller errors
(0.000158–0.011804) across robot joints, gripper, contact, and object features.
The replay checker now prints failed row IDs and dimensions to make follow-up
diagnosis repeatable. Simulator-side calibration and replay results must be
treated as unresolved until the saved observations and simulator observation
conventions agree.

The build also sampled 500 path replays and reported 0.926 success, all with
zero bridges. This measures replay inside already recorded trajectory paths;
it is not evidence for bridge reliability or a successful stitching policy.
The replay implementation was subsequently changed to stop on truncation as
well as termination, so this earlier path rate is exploratory and should not
be used as the final path-execution gate result.

The following guide criteria remain unmet or not yet measured:

- Default Scene graph gate: cap fraction, reachability, and positive stitching
  gain do not pass.
- R1 deterministic replay: the 200-row sidecar check fails the `1e-4` threshold.
- R3 comparison: the Scene `sim_verified` bridge rollout table has not been
  run and compared with the data-calibrated bins.
- R3 toy confidence coverage: the specified 200-repeat, 10,000-rollout
  calibration coverage experiment has not been run.
- R4 path gate: the recorded path result predates the truncation fix and uses
  no bridge edges, so it does not establish the required switched-path rate.
- X1/X2 radius, feature, `k`, and cost-policy selection sweeps have not been
  run. No graph configuration is currently cleared for QC/FQL training.
- R6 TDR encoder training remains optional and has not been started.

The available NumPy 1.26 and NumPy 2.x core suites each passed 71 tests during
implementation. This does not replace the failed replay check or the remaining
simulator and graph-selection experiments above.
