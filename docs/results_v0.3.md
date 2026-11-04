# Graph Critic v0.3 Scene results

Dataset: `scene-play-singletask-task1-v0`, train split, 1,000,000 transitions. Dataset ID is `ogbench:scene-play-singletask-task1-v0:66625e0cd9f2fcf92f5988f4d1bedeb2aa8b1e748316e507e89f4a9afb87c341`.

## R1′: Scene position features and replay

The installed OGBench Scene layout is 40 observation values and 25 qpos values. The named layout and qpos reconstruction were checked on 1,000 rows. Maximum absolute qpos-to-observation errors were `1.79e-7` for cube xyz and `2.09e-7` for drawer and window positions (limit `1e-5`).

`ogbench_scene_pos` has 13 dimensions: eef (5), gripper (1), cube position plus 90-degree-invariant yaw (5), drawer (1), and window (1). Group weights are `1/√5, 1, 1/√5, 1, 1`. Its measured `d_step` is `0.1428919`; every `rho_j` is at most `0.1110`, below the requested `0.2` threshold. Raw identity features had `d_step=0.8486140`; X0's selected slow dimensions are `[0–5, 12–17, 19–21, 28–39]`.

The Scene `rho_j` vector (dimensions 0–12) is `[0.04010, 0.02934, 0.11100, 0.01459, 0.01359, 0.05003, 0, 0, 0, 0, 0, 0, 0]`.

`robomimic_pos` was inspected on QC's Lift MH low-dimensional v1.5 dataset (31,127 transitions, 19 raw observation values). It yields 12 features: eef position/quaternion (7), gripper (2), and object xyz (3), with group weights `1/√7, 1/√2, 1/√3`. The measured `d_step` is `0.0469800`, and the `rho_j` vector (dimensions 0–11) is `[0.02334, 0.02898, 0.02929, 0.00675, 0.00635, 0.02741, 0.02332, 0.00233, 0.00214, 0, 0, 0]`; all are below `0.2`. The quaternion sign-invariance regression test passes.

The 1,000-row replay check passes. `q99/d_step=0.03626` and `max/d_step=0.38620`, below `0.1` and `0.5`. Raw observation errors are reference-only: q99 `0.1022`, max `1.8944`. The repeated-reset test passes bitwise for 100 starts over 10 steps after clearing MuJoCo's unsaved `qacc_warmstart` during sidecar restores. At row `525285`, replay and sidecar quaternion yaw differ by `0` radians while raw yaw observations differ by `1.4763`; the raw yaw discontinuity hypothesis is supported. See [fork and dataset notes](forks.md).

## R2′: geometric graph build

Built with [`scene_v03_geometric.yaml`](../configs/graph/scene_v03_geometric.yaml): `radius=1.5 d_step`, `rho=1`, `Λ₀=0`. Graph hash: `ba32a9a77223f04730d33a84879144abb92394217ff12ec25e8dd6a792fdd1ff`.

| Metric | Result |
| --- | ---: |
| States / goals | 1,001,000 / 2,562 |
| Temporal / bridge edges | 1,000,000 / 724,969 |
| Candidate pairs / retained bridges | 1,009,091 / 724,969 |
| `d_step` / bridge radius | 0.1428919 / 0.2143378 |
| Bridge `τ` counts | 1: 284,390; 2: 440,579 |
| Bridge distance (`d_step` units) | p50 1.108; p95 1.467; max 1.500 |
| Reachable rows / failed-episode reachable | 0.5953 / 0.5819 |
| Non-success target cap fraction | 0.4161 |
| Reachable paths using ≥1 bridge | 0.9793 |
| Successful-row stitch gain >1 step / median gain | 0.6838 / 57 steps |
| Goal bridge edges | 82, all to goal mode 15 |
| Build time / peak RSS | 58.8 s / 2.18 GiB |

All four graph-only v0.3 gates pass (`cap≤0.5`, reachable `≥0.5`, failed-episode reachable `≥0.3`, positive bridge use and stitch gain). This graph used no simulator rollouts. The full diagnostics are in `outputs/scene-v03/geometric-r15-rho1-l0/diagnostics.json`.

The planned X0 prior-cost build also passed the graph-only gates. With the selected standardized raw dimensions and `prior` cost, it produced 3,671,126 bridges; reachability was 0.8222, failed-episode reachability 0.8185, non-success cap fraction 0.1800, bridge use 0.9907, and stitch gain fraction 0.8463. Build time was 66.7 seconds. Hash: `e5ebb5c2c0094356f2c31d0ee9b6c16aaf1e3414b3f48370e0546ee1a27509ad`. It has no direct goal bridges and is an X0 diagnostic, not the v0.3 time policy.

A second X1 geometric point, `r=1.5 d_step`, `ρ=1`, `Λ₀=10`, also passes graph-only gates. Its hash is `b2ea185f12de0018c29b63a654dbf5f955ab4c3cb2d3d8b436397a2a7959317d`; it has 0.5953 reachability, 0.5819 failed-episode reachability, and 0.4235 non-success cap fraction. Its 500-start πG success is 0.116 overall and 0.257 for routes with at most three bridges, so it does not pass the simulation gate.

The learned `r_H`, q=0.8 graph uses the held-out-calibrated model in [`scene_v03_learned_rh_q80.yaml`](../configs/graph/scene_v03_learned_rh_q80.yaml). Hash: `6298c991d24ce0cfa8c9c6e2b3cb5c66d8d4b3ac804ed0bbf69e88ca0322f700`. It has 7,436,179 bridges, `τ` p50=8 / p90=40, bridge use 0.9931, reachability 0.9995, failed-episode reachability 0.9997, and non-success cap fraction 0.00149. Build time was 354.4 seconds; graph-only gates pass.

## R6: learned frame-gap model

The Scene run uses 800/100/100 episode splits and early-stopped after 26,000 of 200,000 maximum steps (best calibration loss at step 16,000). The held-out results are:

| Check | Result | Gate |
| --- | ---: | ---: |
| Spearman(median `τ`, `Δ`) | 0.9382 | ≥0.8, pass |
| Far-pair AUROC for `F(H)` | 0.99992 | ≥0.95, pass |
| Test coverage, q=0.8 (`q′=0.6533`) | 0.7713 | 0.80±0.02, fail |
| Test coverage, q=0.9 (`q′=0.8272`) | 0.8724 | 0.90±0.02, fail |
| Calibration coverage, q=0.8 / 0.9 | 0.80005 / 0.90005 | calibration split only |

The label-collision diagnostic sampled 20,000 cross-episode far pairs; 13,635 fell in `r < d_pos ≤ r_H` (`r_H=3.4324`). For the `r`-exclusion model, 0.132% of those pairs were predicted near (`F(H)>0.5`). A conformal calibration direction bug was caught by a regression test and corrected before recording these metrics. The `r`-exclusion model's calibrated test-coverage gate remains open; calibration-split coverage alone is not sufficient.

The `r_H` ablation was then trained as a separate model. It has Spearman `0.9349`, far AUROC `0.99925`, and test coverage `0.8017` at q=0.8 (`q′=0.7023`), which passes the R6 thresholds. At q=0.9 it gives `0.9202` coverage (`q′=0.9`), 0.0002 outside the ±0.02 limit. Its collision set contains 13,612 pairs; 1.84% are predicted near. [`scene_v03_learned_rh_q80.yaml`](../configs/graph/scene_v03_learned_rh_q80.yaml) selects the passing q=0.8 checkpoint; its graph passes graph-only gates but its πG result is reported below.

## R3′: graph-tracking simulation check

`extract_path` and `GraphTrackingPolicy` are implemented. The synthetic noisy point-control test reaches the stitched goal while open-loop replay of the failed fragment does not. Scene evaluation used 500 starts and a destination tube radius of `1×d_step = 0.142892`.

| Graph | πG success | πG success, ≤3 bridges | Pessimism hold | Successful `T/(C_G/c)` p50 / p90 | Completed bridge ratios p50 / p90 (n) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Geometric, `r=1.5`, `ρ=1`, `Λ₀=0` | 0.100 | 0.352 (n=128) | 0.066 | 1.000 / 1.018 | 0.5 / 1.0 (n=11) |
| Geometric, `r=1.5`, `ρ=1`, `Λ₀=10` | 0.116 | 0.257 (n=214) | 0.076 | 1.000 / 1.015 | 0.5 / 1.4 (n=5) |
| Learned, q=0.9, `r` exclusion, `Λ₀=5` | 0.050 | 0.127 (n=189) | 0.036 | 1.000 / 1.012 | 0.5 / 0.9 (n=3) |
| Learned, q=0.8, `r_H` exclusion, `Λ₀=5` | 0.036 | 0.073 (n=246) | 0.026 | 1.000 / 1.052 | 1.0 / 1.0 (n=1) |

None passes the πG gate (`success≥0.6` for routes with at most three bridges). The q=0.9 learned graph reached only 3 of 1,233 bridge destination tubes; the q=0.8/`r_H` graph reached 1 of 1,200. These bridge-ratio quantiles are too sparse for a reliable calibration claim. Most failures ended after tracking distance exceeded `2r` for ten consecutive steps. The low πG rates block the QC pilot even though graph-only gates pass.

## Core acceptance tests

The core suite passes under both supported NumPy environments: `core` and `core-numpy2` each report `81 passed, 1 skipped`. The skipped check is the JAX parity test, which passes in the benchmark environment; `tests/test_v03.py` reports `11 passed` there.
