# Graph Critic v1.4b — P13 4×4 task 2 FQL results

**Dataset note:** This experiment uses directly generated P13 data (1,321 episodes, 1,000,000 transitions; all generated episodes reached the goal). It is not a standard benchmark.

Date: 2026-11-13  
Branch: `quasi_develop`; QC fork branch: `graph-critic`

## Commits and execution state

| Item | Value |
|---|---|
| Root runbook / execution commit | `b98208dee273cf3086bf71931122d0052b720954` |
| QC execution commit | `f12e624ff62a23c77366e751832361fe8a017f79` |
| FQL worktree at execution | Clean, detached at `e8cd16eb490332924dfa2492097219f181765933`; unchanged |
| `git_dirty` at S1 / S2 / S3 | `false` / `false` / `false` for root, QC, and FQL |
| Code changes | None; only the runbook and this report were added to the root repository |
| Tests | Not run; no implementation code changed |

| Stage | Elapsed time |
|---|---:|
| S1, h=1 graph and target build | 78.83 s |
| S2, FQL TD, 300k and six 50-episode evaluations | 22 min 08 s (training progress timer) |
| S3, FQL graph critic, 300k and six 50-episode evaluations | ≈23 min 04 s (run-directory timestamp to final `eval.csv` update) |

## S1 — h=1 discounted target

| Metric | Value |
|---|---:|
| Target kind / horizon / gamma | discounted / 1 / 0.99 |
| Graph hash | `6bd8f696e65e362770facddea65caca03f57abdb78927f054a730f32afbc01aa` |
| Expected graph hash match | Yes |
| Target content hash | `95df1cd20b635ce604bdc88ced291805c10012ad8f3f1a0cef4af141e186bf36` |
| Valid fraction | 1.000000 |
| Loss-mask fraction | 1.000000 |
| Capped loss-mask fraction (`source == 2`) | 0.002306 |
| Target mean / p50 / p90 / max | 65.330266 / 73.124138 / 91.649787 / 100.000000 |
| `targets.json` dataset ID | `ogbench:puzzle-4x4-play-singletask-task2-v0:p13-4x4-task2-mix1-d0.3-c0.5-seed0-script9ab803500c6b:6f73eb1d6bfe8d950e567055edc3f4a55e88f115885945efac21640924817de6` |

## S2/S3 — FQL results

Each checkpoint evaluation used 50 episodes. Mean wrong buttons is the Hamming distance from the task-2 goal board, computed from the final per-button state means. Final actor Q loss is derived as `actor_loss - bc_flow_loss - 100 × distill_loss`.

| Method | Success 50k | 100k | 150k | 200k | 250k | 300k | Final mean wrong buttons | Final critic loss | Final graph-target Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FQL, TD critic | 0.08 | 0.40 | 0.32 | 0.82 | 0.48 | 0.82 | 0.92 | 0.6135143 | N/A |
| FQL, discounted graph critic | 0.02 | 0.32 | 0.32 | 0.30 | 0.38 | 0.34 | 4.12 | 2.2126908 | 0.9970107 |
| QC + graph critic, best-of-N, v1.4 reference | 0.68 | 0.64 | 0.80 | 0.78 | 0.82 | 0.78 | 1.38 | 1.6411144 | N/A |
| QC TD, v1.4 reference | 0.04 | 0.56 | 0.14 | 0.10 | 0.60 | 0.46 | 2.38 | 0.5952642 | N/A |
| FQL TD, alpha=100, v1.3 reference (3×3) | 0.00 | 0.00 | 0.38 | 0.66 | 0.86 | 0.94 | — | — | N/A |
| FQL graph critic, alpha=100, v1.3 reference (3×3) | 0.16 | 0.42 | 0.38 | 0.54 | 0.60 | 0.52 | — | — | 0.960522 |

| Method | Final `bc_flow_loss` | Final `distill_loss` | Final Q loss term |
|---|---:|---:|---:|
| FQL, TD critic | 0.15724154 | 0.0035515733 | 45.31133513 |
| FQL, discounted graph critic | 0.15715562 | 0.0039401916 | 62.22487522 |

## Artifact paths

- h=1 graph and targets: `outputs/puzzle/p13-4x4-task2-100-generated/v14b/Lsame_h1/`
- FQL TD run: `outputs/puzzle/p13-4x4-task2-100-generated/v14b/qc/fql_td/qc/v14b-4x4p13-fql-td/puzzle-4x4-play-singletask-task2-v0/sd00020261113_154202/`
- FQL graph-critic run: `outputs/puzzle/p13-4x4-task2-100-generated/v14b/qc/fql_graph/qc/v14b-4x4p13-fql-graph/puzzle-4x4-play-singletask-task2-v0/sd00020261113_160500/`
- Reused v1.4 graph and candidate cache: `outputs/puzzle/p13-4x4-task2-100-generated/v14/`

## Push commands

- QC fork (`graph-critic`): 변경 없음.
- FQL fork: 변경 없음.
- Root (`quasi_develop`): `b98208d Add v1.4b P13 FQL runbook` and this results-report commit (`Record v1.4b P13 FQL results`). The command below prints the exact commit hashes before pushing.

```bash
cd /home/url/graphcritic && git status -sb && git log --oneline origin/quasi_develop..HEAD && git push origin quasi_develop
```
