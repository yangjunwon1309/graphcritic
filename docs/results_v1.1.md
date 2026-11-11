# Graph Critic v1.1 — execution results

Date: 2026-11-12
Branch: `quasi_develop`
Task: `puzzle-3x3-play-singletask-task3-v0`

## Commits, tests, and execution state

| Item | Result |
|---|---|
| Root implementation / execution commit | `e7767834893c65bc47fa39922e73a389480b6d9b` |
| QC submodule execution commit | `32244b4f4d6e1dae1803b3d9d861901a28c188bb` |
| Root `git_dirty` during S1–S4 | `false` |
| QC submodule `git_dirty` during S1–S4 | `false` |
| Requested tests | `tests/test_v11.py`, `tests/test_v10.py`, `tests/test_quasimetric_v08.py`: **38 passed**, one upstream deprecation warning, 49.81 s |
| Push status | Root `quasi_develop` and QC `graph-critic` pushes failed: HTTPS GitHub requested a username but this shell has no interactive credential device |

| Stage | Status | Elapsed time |
|---|---|---:|
| S0 tests | Passed | 49.81 s |
| S1 observed object-change set | Completed | 5 s |
| S2 candidate cache and 50k parity | Completed; parity passed | 95 s |
| S3 candidate audit / Gate A | Gate passed | 304 s |
| S4 Lobj graph and bridge audit / Gate B | Gate failed | 92 s |
| S5 action heads and offline G1/G2/G3 | Not run | — |
| S6 direct-Q + BC QC | Not run | — |
| S7 discounted-cost QC | Not run | — |

## S1 — observed object-change set

Observed 19,810,000 within-episode pairs across horizons 1–20. Ten patterns met `min_count=10`; zero patterns or observations were omitted.

| Changed button indices | Count | Minimum `k` |
|---|---:|---:|
| `[]` | 13,369,067 | 1 |
| `[0, 1, 3]` | 717,283 | 1 |
| `[0, 1, 2, 4]` | 735,165 | 1 |
| `[1, 2, 5]` | 711,360 | 1 |
| `[0, 3, 4, 6]` | 707,608 | 1 |
| `[1, 3, 4, 5, 7]` | 718,884 | 1 |
| `[3, 6, 7]` | 707,438 | 1 |
| `[2, 4, 5, 8]` | 685,815 | 1 |
| `[5, 7, 8]` | 724,328 | 1 |
| `[4, 6, 7, 8]` | 733,052 | 1 |

| Summary | Value |
|---|---:|
| Observed / retained patterns | 10 / 10 |
| Patterns omitted below `min_count` | 0 |
| Omitted occurrences | 0 |
| Change-set SHA256 | `12eead96ded7013b9388cb99532e8cbd802a2c77a8d03697bc16f0192ce89679` |

## S2/S3 — candidate cache and `d_theta` audit

Candidate nodes: 201,876; boundary sources: 199,532. Candidate rule: `cross_episode_observed_object_change_v1_robot_window5_same32_changed32`. Each source produced 32 candidates in each bundle.

| Bundle | GF(2) `n_p` | Candidate count | `d_theta` p10 | p50 | p90 | Fraction `d_theta≤20` |
|---|---:|---:|---:|---:|---:|---:|
| `Δ=0` | 0 | 6,385,024 | 1.0632 | 18.4124 | 47.4465 | 0.5170 |
| `Δ=0` | 1 | 0 | — | — | — | — |
| `Δ=0` | 2 | 0 | — | — | — | — |
| `Δ=0` | 3 | 0 | — | — | — | — |
| `Δ=0` | `≥4` | 0 | — | — | — | — |
| `Δ=0` | `∞` | 0 | — | — | — | — |
| `Δ≠0` | 0 | 0 | — | — | — | — |
| `Δ≠0` | 1 | 6,385,024 | 2.8244 | 49.9319 | 81.8830 | 0.1666 |
| `Δ≠0` | 2 | 0 | — | — | — | — |
| `Δ≠0` | 3 | 0 | — | — | — | — |
| `Δ≠0` | `≥4` | 0 | — | — | — | — |
| `Δ≠0` | `∞` | 0 | — | — | — | — |

| Candidate metric | Value |
|---|---:|
| Total pairs | 12,770,048 |
| `n_p≥2` count / fraction | 0 / 0.0000 |
| Gate A (`≤1%`) | Pass |
| GPU/cKDTree parity | 50,000 sources; 100,000 bundle comparisons; 0 mismatches; 0 ties excluded; pass |
| Frozen `d_theta` checkpoint SHA256 | `9adaf65cb96691dbece0448ebe32535edfe5fecf3c2125c6cddada77557965cd` |

## S4 — Lobj graph and lower-bound audit

Graph hash: `380fd43abe3a9bb25c1cc1a1afbfd8bab973051197decf8f1c7ecbf692a4f5b4`.

| Metric | Value |
|---|---:|
| G0 optimistic candidate pairs | 376,398 |
| Accepted bridges | 119,149 |
| Retained bridges | 27,346 |
| Retained success-destination bridges | 684 |
| Reachability | 0.968517 |
| Loss-mask fraction | 0.996000 |
| Bridge paths with 0 / 1 / 2 bridges, count | 8,469 / 304,751 / 656,266 |
| Bridge paths with 0 / 1 / 2 bridges, reachable fraction | 0.008736 / 0.314343 / 0.676922 |
| Impossible retained bridges | 3,711 / 27,346 |
| Impossible retained-bridge fraction | 0.135705 |
| Impossible bridge `n_p` distribution | `n_p=0`: 0; `n_p=1`: 3,711; `n_p≥2` or `∞`: 0 |
| Gate B (`≤5%`) | **Fail** (0.135705) |
| Button-change interval min / p1 / p5 | 26 / 28 / 29 frames |
| Discounted target mean / p50 / p90 / max | 80.3773 / 85.2462 / 97.4687 / 100 |
| Discounted target capped-row fraction | 0.031631 |

| Reference graph | Retained bridges | Reachability | Impossible bridge fraction |
|---|---:|---:|---:|
| L, v0.8.4 | 30,209 | 0.982 | 0.858 |
| O upper-bound, v0.8.4 | 7,707 | 0.861 | 0.018 |
| G-R `ρ=0.9`, v0.6 | — | 0.997 | — |

## Stop point and scope

| Item | Result |
|---|---|
| Last completed stage | S4 |
| Stop gate and measured value | Gate B failed: impossible retained-bridge fraction `0.135705` > `0.05` |
| S5 action-head training / G1–G3 | Not run |
| S6 direct-Q + BC QC | Not run |
| S7 discounted-cost QC | Not run |
| Out-of-scope experiments run | None |
| Candidate artifacts | `outputs/puzzle/standard-3x3-task3/v11/` |
| Graph and targets | `outputs/puzzle/standard-3x3-task3/v11/Lobj/` |
| Board audit | `outputs/puzzle/standard-3x3-task3/v11/Lobj_board_check.json` |
| Candidate audit | `outputs/puzzle/standard-3x3-task3/v11/candidates_audit.json` |
