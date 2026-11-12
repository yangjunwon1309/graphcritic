# Graph Critic v1.3 — FQL versus FQL with graph critic

Date: 2026-11-12  
Branch: `quasi_develop`; QC fork branch: `graph-critic`

## Commits, tests, and execution state

| Item | Value |
|---|---|
| Root implementation and runbook commit | `39055bc8b54559eaa170309acf5e5e9c9e61c3d3` |
| QC fork test commit | `f12e624ff62a23c77366e751832361fe8a017f79` |
| Execution commit, root / QC fork | `39055bc8b54559eaa170309acf5e5e9c9e61c3d3` / `f12e624ff62a23c77366e751832361fe8a017f79` |
| `git_dirty` at S1, S2, S3 start | `false` / `false` / `false` |
| Allowed tests | `third_party/qc/tests/test_graph_discounted_fql.py tests/test_v12.py`: **7 passed**, two upstream deprecation warnings (10.28 s) |
| Unfinished stages | None |

| Stage | Elapsed time |
|---|---:|
| S1, h=1 target and graph build | 57.85 s wall; 56.87 s recorded by builder |
| S2, FQL with TD critic, 300k | 1,218.66 s |
| S3, FQL with graph critic, 300k | 1,253.45 s |

The QC code path already performed graph-target regression using `softplus(Q)`, routed the FQL actor Q term through `_actor_q`, and skipped target-critic updates in graph mode. The added one-update test loads an h=1 discounted artifact, performs a graph-mode FQL update without bootstrap fields, confirms critic parameters change, and confirms target-critic parameters remain unchanged. The TD training path was not modified.

## S1 — h=1 target artifact

| Metric | Value |
|---|---:|
| Target kind / horizon / gamma | discounted / 1 / 0.99 |
| Graph hash | `3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa` |
| Expected graph hash match | Yes |
| Target content hash | `5e8e4d99a56a944f041e1c2b94d91182c34c7e4f32bc330f4a0bf6cbea7ca04d` |
| Valid fraction | 1.000000 |
| Loss-mask fraction | 1.000000 |
| Capped loss-mask fraction (`source == 2`) | 0.054730 |
| Target mean / p50 / p90 / max | 87.849926 / 94.133892 / 99.768791 / 100.000000 |
| `targets.json` config horizon | 1 |

## S2/S3 — QC results

Each checkpoint evaluation used 50 episodes. Wrong-button count is the sign-reversed `episode.final_reward`.

| Method | Success 50k | 100k | 150k | 200k | 250k | 300k | Final mean wrong buttons | Final critic loss | Final graph-target Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FQL, TD critic | 0.00 | 0.00 | 0.38 | 0.66 | 0.86 | 0.94 | 0.26 | 0.536636 | N/A |
| FQL, discounted graph critic | 0.16 | 0.42 | 0.38 | 0.54 | 0.60 | 0.52 | 1.90 | 6.429309 | 0.960522 |
| QC + graph critic, v1.2b reference | 0.44 | 0.78 | 0.96 | 1.00 | 0.98 | 0.94 | 0.24 | — | — |
| QC TD, v0.6 reference | 0.10 | 1.00 | 0.98 | 1.00 | — | — | — | — | — |
| BC, v1.2b reference | 0.00 | 0.00 | 0.02 | 0.00 | 0.02 | 0.02 | — | — | — |

Final actor loss terms at 300k:

| Method | `bc_flow_loss` | `distill_loss` | Q loss term |
|---|---:|---:|---:|
| FQL, TD critic | 0.1466754 | 0.005410172 | 55.2761594 |
| FQL, discounted graph critic | 0.14747274 | 0.018665627 | 85.91917456 |

The Q loss term is computed from the logged actor-loss identity `actor_loss − bc_flow_loss − 100 × distill_loss`; it is not a separately emitted CSV column. The graph run's `critic/graph_target_spearman` is the final `offline_agent.csv` value.

## Artifact paths

- h=1 graph and targets: `outputs/puzzle/standard-3x3-task3/v13/Lsame_h1/`
- FQL TD run: `outputs/puzzle/standard-3x3-task3/v13/qc/fql_td/qc/v13-3x3t3-fql-td/puzzle-3x3-play-singletask-task3-v0/sd00020261112_214832/`
- FQL graph-critic run: `outputs/puzzle/standard-3x3-task3/v13/qc/fql_graph/qc/v13-3x3t3-fql-graph/puzzle-3x3-play-singletask-task3-v0/sd00020261112_220927/`

## Push status

Pushes to both configured remotes were attempted and failed because this environment has no interactive GitHub HTTPS credentials (`fatal: could not read Username for 'https://github.com': No such device or address`). The root and QC commits remain local; both worktrees are clean.
