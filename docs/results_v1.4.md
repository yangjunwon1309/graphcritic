# Graph Critic v1.4 — 4×4 P13 auxiliary experiment and cube-double statistics

The P13 4×4 task 2 data below are directly generated custom data, not a standard benchmark.

Date: 2026-11-13  
Branch: `quasi_develop`; QC fork branch: `graph-critic`

## Commits, tests, and execution state

| Item | Value |
|---|---|
| Cube-double statistics implementation | `e11fd8d0683cb03013e1bb27dacd7a2b64c6a7d4` |
| 4×4 robot-window candidate-cache fix | `00d50a2a19342df539bf9b7de80eddcff8d05701` |
| Root execution commit for A2–A6 | `00d50a2a19342df539bf9b7de80eddcff8d05701` |
| A1 execution commit | `e11fd8d0683cb03013e1bb27dacd7a2b64c6a7d4` |
| QC fork commit | `f12e624ff62a23c77366e751832361fe8a017f79` |
| `git_dirty` at successful A1–A5 and B experiment starts | `false` |
| `git_dirty` at A6 TD run start | `true` (the untracked `docs/results_v1.4.md` draft existed) |
| Allowed test command | `tests/test_cube_double_stats.py tests/test_v14.py tests/test_v12.py` |
| Allowed test result | **9 passed**, one upstream deprecation warning, 0.46 s |
| Push to `origin/quasi_develop` | Failed: `fatal: could not read Username for 'https://github.com': No such device or address` |

The first A2 candidate-cache build stopped with `ValueError: flattened windows do not match the Puzzle frame schema`. The existing helper assumed a 28-dimensional 3×3 frame. The committed fix derives the frame width from the dataset while retaining the 19-dimensional robot feature prefix. A2 then completed and its 50,000-source parity check had 0 mismatches and 0 excluded ties.

The first test invocation had one failure in the new synthetic stationary-layout fixture because its stable plateaus were shorter than the script's minimum. The fixture was lengthened; the final allowed test invocation passed all 9 tests.

| Stage | Elapsed time |
|---|---:|
| A1 object-change set | 11.71 s |
| A2 candidate cache, successful run | Not recorded |
| A3 B-max sweep | 156.53 s |
| A4 graph and discounted targets | 76.92 s internal; wall time not recorded |
| A4 board audit | Not recorded |
| A5 Lsame discounted-cost QC, 300k | 20 min 58 s |
| A6 TD QC, 300k | 1 h 14 min 26 s |
| B cube-double statistics | Not recorded |

## A — P13 generated 4×4 task 2

### A1–A3: change patterns, candidates, and B-max sweep

| Metric | Value |
|---|---:|
| Dataset episodes / transitions | 1,321 / 1,000,000 |
| Episodes reaching the generated goal | 1,321 / 1,321 |
| Aligned state rows | 1,001,321 |
| Observed change patterns (including unchanged) | 17 |
| Omitted patterns | 0 |
| Change-pattern occurrences | 19,749,010 |
| Candidate pairs | 11,483,911 |
| Same-object / unchanged candidates | 5,372,711 |
| Changed-object candidates | 6,111,200 |
| Boundary source nodes | 190,975 |
| Candidate destination nodes | 235,437 |
| Parity sources / compared groups / mismatches / excluded ties | 50,000 / 100,000 / 0 / 0 |

| `b` | Reachability `R(b)` | Cost p50 (steps) | Cost p90 (steps) |
|---:|---:|---:|---:|
| 0 | 0.713326 | 226.000 | 465.000 |
| 1 | 0.997696 | 129.626 | 245.392 |
| 2 | 0.997696 | 122.138 | 193.234 |
| 3 | 0.997696 | 121.231 | 186.458 |
| 4 | 0.997696 | 121.220 | 185.327 |
| 5 | 0.997696 | 121.220 | 185.146 |
| 6–32 | 0.997696 | 121.220 | 185.145 |

| Sweep item | Value |
|---|---:|
| `R_sat` | 0.997696 |
| `B*` | 1 |
| Gate A (`R_sat ≥ 0.90`) | Pass |

### A4: Lsame graph, B*=1

| Metric | Value |
|---|---:|
| Graph hash | `6bd8f696e65e362770facddea65caca03f57abdb78927f054a730f32afbc01aa` |
| Candidate pairs judged | 5,372,711 |
| C0-optimistic candidates | 2,168,002 |
| Accepted bridges | 1,822,810 |
| Retained bridges | 59,780 |
| Successful-goal bridge destinations | 0 |
| Reachability | 0.997696 |
| Reference G-R ρ=0.9 reachability | 0.997616 |
| Loss-mask fraction | 0.994716 |
| Valid target fraction | 0.994716 |
| Discounted target mean / p50 / p90 / max | 66.3024 / 74.0460 / 91.8635 / 100.0000 |
| Capped fraction among loss-mask rows | 0.002314 |
| Originally reachable states whose cost decreased | 0.740473 |
| Reachable states with 0 / 1 / ≥2 bridges in shortest path | 185,372 / 813,642 / 0 |
| Unreachable states | 2,307 |
| Retained bridge `n_p=0 / 1 / 2 / 3 / ≥4 / ∞` | 59,780 / 0 / 0 / 0 / 0 / 0 |
| Board audit: retained bridges with `n_p≥1` | 0 |
| Gate B | Pass |

### A5–A6: QC

Each checkpoint evaluation used 50 episodes. Wrong-button count is the mean Hamming distance from the task 2 goal board, computed from the per-button final state means (equivalent to the sign-reversed `episode.final_reward`).

| Method | Success 50k | 100k | 150k | 200k | 250k | 300k | Final mean wrong buttons | Final critic loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lsame discounted-cost graph critic | 0.68 | 0.64 | 0.80 | 0.78 | 0.82 | 0.78 | 1.38 | 1.6411144 |
| QC TD critic | 0.04 | 0.56 | 0.14 | 0.10 | 0.60 | 0.46 | 2.38 | 0.5952642 |
| BC | Not enabled | — | — | — | — | — | — | — |
| G0 regret, prior method | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | — | — |
| G-R ρ=0.9 regret, prior method | 0.04 | 0.08 | 0.14 | 0.12 | 0.16 | 0.14 | — | — |
| D1 oracle-filter regret, prior method | 0.08 | 0.08 | 0.08 | 0.16 | 0.06 | 0.18 | — | — |

`--eval_bc_also` was not used: the QC CLI permits it only in `graph_direct` mode. A5 artifact directory: `outputs/puzzle/p13-4x4-task2-100-generated/v14/qc/Lsame_discounted/qc/v14-4x4p13-Lsame-discounted/puzzle-4x4-play-singletask-task2-v0/sd00020261112_225028/`. A6 artifact directory: `outputs/puzzle/p13-4x4-task2-100-generated/v14/qc/td/qc/v14-4x4p13-td/puzzle-4x4-play-singletask-task2-v0/sd00020261112_231151/`.

The A6 run used root execution commit `00d50a2a19342df539bf9b7de80eddcff8d05701`; its only dirty item was the untracked results draft. No source-code file changed after the execution commit.

## B — cube-double standard training data statistics

Dataset: OGBench `cube-double-play-v0` train split, `/home/url/.ogbench/data/cube-double-play-v0.npz`. The task success condition is both cubes within 4 cm of the two goal positions, with either cube-to-goal assignment accepted.

### B1: dataset and observation layout

| Metric | Value |
|---|---:|
| Episodes / transitions | 1,000 / 1,000,000 |
| State frames per episode | 1,001 (all episodes) |
| Transition length per episode | 1,000 (all episodes) |
| Observation dimension | 37 |
| Robot feature indices | `[0, 19)` |
| Cube 0 feature indices / XYZ indices | `[19, 28)` / `[19, 20, 21]` |
| Cube 1 feature indices / XYZ indices | `[28, 37)` / `[28, 29, 30]` |
| Cube XYZ decoding | `raw_xyz_m = obs_xyz / 10 + [0.425, 0, 0]` |

### B2: movement and stationary layouts

| Metric | Value |
|---|---:|
| Valid five-frame windows | 996,000 |
| Five-frame displacement q99.9, cube 0 XYZ (m) | 0.096557 / 0.119862 / 0.109195 |
| Five-frame displacement q99.9, cube 1 XYZ (m) | 0.096908 / 0.119015 / 0.108850 |
| Movement threshold (5% of q99.9), cube 0 XYZ (m) | 0.004828 / 0.005993 / 0.005460 |
| Movement threshold (5% of q99.9), cube 1 XYZ (m) | 0.004845 / 0.005951 / 0.005442 |
| Moving-frame-window fraction, cube 0 / cube 1 | 0.234381 / 0.236507 |
| Merged movement bouts | 11,379 |
| Movement-bout length p10 / p50 / p90 (frames) | 44 / 47 / 49 |
| Transport path speed p50 (cm/frame) | 1.423678 |
| Stationary layouts | 12,013 |
| Stacked stationary layouts / fraction | 1,609 / 0.133938 |

Stacked: XY separation `<2 cm` and vertical separation within `1 cm` of the `4 cm` cube edge. Stationary layout: stable interval between merged five-frame movement bouts, retaining at least 5 stable frames; representative coordinates are coordinate-wise medians.

### B3: task goal states

Success tolerance is 4 cm in 3D; cube assignment is interchangeable.

| Task | Goal XYZs (m) | Successful frames | Episodes with a successful frame | Successful stationary layouts | Stationary layouts with exactly one cube near a goal | Fraction |
|---:|---|---:|---:|---:|---:|---:|
| 1 | (0.425,0,0.02), (0.425,0.1,0.02) | 1,113 | 24 | 21 | 1,348 | 0.112212 |
| 2 | (0.35,0.1,0.02), (0.5,0.1,0.02) | 1,004 | 25 | 19 | 1,376 | 0.114543 |
| 3 | (0.425,-0.2,0.02), (0.425,0.2,0.02) | 1,529 | 29 | 28 | 1,425 | 0.118621 |
| 4 | (0.425,0.1,0.02), (0.425,-0.1,0.02) | 1,557 | 35 | 28 | 1,398 | 0.116374 |
| 5 | (0.425,0,0.02), (0.425,0,0.06) | 2,452 | 50 | 43 | 631 | 0.052526 |

### B4: cross-episode stationary-layout matches

Counts are unordered layout pairs, excluding same-episode pairs. A match uses the better of the two cube assignments; every matched cube pair must be within the listed 3D radius.

| Radius | Both cubes match: pairs | Mean both-cube matches per layout | At least one cube matches: pairs | Mean one-or-more matches per layout | Individual cube matches over layout pairs |
|---:|---:|---:|---:|---:|---:|
| 1 cm | 2,064 | 0.343628 | 498,365 | 82.970948 | 500,429 |
| 2 cm | 14,852 | 2.472655 | 1,941,398 | 323.216182 | 1,956,250 |
| 4 cm | 131,171 | 21.838175 | 7,247,146 | 1,206.550570 | 7,378,317 |

### B5: stationary-layout graph connectivity

Nodes are stationary layouts; edges are same-episode consecutive layouts plus bidirectional cross-episode matches where both cubes match within radius.

| Radius | Nodes | Same-episode directed edges | Cross-episode bidirectional pairs | Directed edge count | Task 1 reachable | Task 2 reachable | Task 3 reachable | Task 4 reachable | Task 5 reachable |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 cm | 12,013 | 11,013 | 2,064 | 15,141 | 6,108 / 0.508449 | 6,140 / 0.511113 | 6,132 / 0.510447 | 6,160 / 0.512778 | 6,043 / 0.503038 |
| 2 cm | 12,013 | 11,013 | 14,852 | 40,717 | 11,603 / 0.965870 | 11,603 / 0.965870 | 11,603 / 0.965870 | 11,604 / 0.965954 | 11,603 / 0.965870 |
| 4 cm | 12,013 | 11,013 | 131,171 | 273,355 | 12,011 / 0.999834 | 12,011 / 0.999834 | 12,011 / 0.999834 | 12,011 / 0.999834 | 12,011 / 0.999834 |

Cube-double JSON: `outputs/cube-double/stats_v14.json`. P13 v1.4 outputs: `outputs/puzzle/p13-4x4-task2-100-generated/v14/`.

The results commit is local on `quasi_develop`; the branch could not be pushed because this environment has no interactive GitHub HTTPS credentials.

