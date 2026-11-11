# Graph Critic v1.1 — execution results

Date: 2026-11-12  
Branch: `quasi_develop`

## Commits, tests, and execution state

| Item | Result |
|---|---|
| Root implementation commit | Pending |
| QC submodule commit | Pending |
| Execution root commit | Pending |
| Execution QC commit | Pending |
| Root `git_dirty` at execution | Pending |
| QC `git_dirty` at execution | Pending |
| Requested tests | Pending |
| Push status | Pending |

| Stage | Status | Elapsed time |
|---|---|---:|
| S0 tests | Pending | — |
| S1 observed object-change set | Pending | — |
| S2 candidate cache and 50k parity | Pending | — |
| S3 candidate audit / Gate A | Pending | — |
| S4 Lobj graph and bridge audit / Gate B | Pending | — |
| S5 action heads and offline G1/G2/G3 / Gate C | Pending | — |
| S6 direct-Q + BC QC | Pending | — |
| S7 discounted-cost QC | Pending | — |

## S1 — observed object-change set

| Changed button indices | Count | Minimum `k` |
|---|---:|---:|
| Pending | — | — |

| Summary | Value |
|---|---:|
| Observed patterns | Pending |
| Retained patterns | Pending |
| Patterns omitted below `min_count` | Pending |
| Omitted occurrences | Pending |
| Change-set SHA256 | Pending |

## S2/S3 — candidate cache and `d_theta` audit

Candidate cache: `outputs/puzzle/standard-3x3-task3/v11/candidates_obj.npz`  
Candidate rule: `cross_episode_observed_object_change_v1_robot_window5_same32_changed32`

| Bundle | GF(2) `n_p` | Candidate count | `d_theta` p10 | p50 | p90 | Fraction `d_theta≤20` |
|---|---:|---:|---:|---:|---:|---:|
| `Δ=0` | 0 | Pending | — | — | — | — |
| `Δ=0` | 1 | Pending | — | — | — | — |
| `Δ=0` | 2 | Pending | — | — | — | — |
| `Δ=0` | 3 | Pending | — | — | — | — |
| `Δ=0` | `≥4` | Pending | — | — | — | — |
| `Δ=0` | `∞` | Pending | — | — | — | — |
| `Δ≠0` | 0 | Pending | — | — | — | — |
| `Δ≠0` | 1 | Pending | — | — | — | — |
| `Δ≠0` | 2 | Pending | — | — | — | — |
| `Δ≠0` | 3 | Pending | — | — | — | — |
| `Δ≠0` | `≥4` | Pending | — | — | — | — |
| `Δ≠0` | `∞` | Pending | — | — | — | — |

| Candidate metric | Value |
|---|---:|
| Total pairs | Pending |
| `n_p≥2` count | Pending |
| `n_p≥2` fraction | Pending |
| Gate A (`≤1%`) | Pending |
| 50k GPU/cKDTree parity | Pending |

## S4 — Lobj graph and lower-bound audit

| Metric | Value |
|---|---:|
| G0 optimistic candidate pairs | Pending |
| Accepted bridges | Pending |
| Retained bridges | Pending |
| Retained success-destination bridges | Pending |
| Reachability | Pending |
| Loss-mask fraction | Pending |
| Bridge path count 0 / 1 / 2 | Pending |
| Impossible retained bridges | Pending |
| Impossible retained bridge fraction | Pending |
| Gate B (`≤5%`) | Pending |
| Impossible bridge GF(2) count distribution | Pending |
| Discounted target mean / p50 / p90 / max | Pending |
| Discounted target capped-row fraction | Pending |

## S5 — action heads and offline direct-Q checks

| Seed | C1 violation | C2 violation | `d` median k=5 | k=10 | k=15 | k=20 | F3 alarm | Gate C |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | Pending | Pending | — | — | — | — | Pending | Pending |
| 1 | Pending | Pending | — | — | — | — | Pending | Pending |
| 2 | Pending | Pending | — | — | — | — | Pending | Pending |
| 3 | Pending | Pending | — | — | — | — | Pending | Pending |

| Offline direct-Q metric | Value |
|---|---:|
| G1 | Pending |
| G2 | Pending |
| G3 | Pending |
| Fallback fraction | Pending |
| Fallback reasons | Pending |
| Mean neighborhood size | Pending |
| Candidate-Q standard deviation | Pending |
| Decision seconds | Pending |

## QC metrics

| Target / policy | 50k | 100k | 150k | 200k | 250k | 300k | Mean wrong buttons at final eval | Final loss / graph-target Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Lobj direct-Q | Pending | Pending | Pending | Pending | Pending | Pending | Pending | N/A |
| Same-checkpoint BC | Pending | Pending | Pending | Pending | Pending | Pending | Pending | N/A |
| Lobj discounted cost critic | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| O cost critic, v0.8.4b | 0.04 | 0.24 | 0.26 | 0.16 | 0.28 | 0.30 | 2.98 | — |
| L cost critic, v0.8.4b | 0.00 | 0.06 | 0.02 | 0.04 | 0.02 | 0.06 | 2.96 | — |
| TD QC, v0.6 | 0.10 | 1.00 | 0.98 | 1.00 | N/A | N/A | N/A | — |

| QC artifact | Path |
|---|---|
| Lobj direct-Q and BC | `outputs/puzzle/standard-3x3-task3/v11/qc/Lobj_direct/` |
| Lobj discounted cost critic | `outputs/puzzle/standard-3x3-task3/v11/qc/Lobj_discounted/` |

## Stop point and scope

| Item | Result |
|---|---|
| Last completed stage | Pending |
| Stop gate and value, if any | Pending |
| S4/S5/S6/S7 not run, if stopped | Pending |
| Out-of-scope experiments run | None |
