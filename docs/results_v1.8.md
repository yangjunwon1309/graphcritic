Dataset: OGBench cube-double-play 표준 데이터(100만), singletask task 1–5.

# Graph Critic v1.8 — cube-double implementation and gated execution

Date: 2026-11-14  
Branch: `quasi_develop`; QC / DQC fork branches: `graph-critic`

## Commits, clean state, and tests

| Item | Result |
|---|---|
| Root implementation | `2d9e3f2` — Implement v1.8 cube-double graph workflow |
| DQC compatibility | `341c529` — Support cube-double DQC single-task graph data |
| Root follow-up fixes | `59c1c65` — Add cube position decoder for v1.8 audit; `65a9224` — Fix v1.8 terminal chunk endpoint mapping; `4ed62a6` — Use the installed JAX CUDA backend in v1.8 runbook |
| S1 execution commit | `59c1c65129dbfa384211b49f74744959541f5172` |
| Successful S2 execution commit | `4ed62a6f2bb217b34eb1d950d245c9ab233e4c0b` |
| `git_dirty` at S1 / S2 start | `false` / `false` for root, QC, FQL, and DQC |
| Tests | `tests/test_v18.py`: **7 passed**; `third_party/dqc/tests/test_v18_cube_double.py`: **1 passed**; one upstream pytest deprecation warning per invocation |
| Additional checks | `compileall`, `git diff --check`: passed |

| Stage | Elapsed time |
|---|---:|
| S1 data/loader alignment and movement audit | ≈5 s |
| S2 robot-only dθ training, 20k | 839.26 s |
| S2 final gate checker | <1 s |
| Failed pre-training attempts | S1 import failure: ≈5 s; S2 terminal endpoint mapping: <1 s; S2 unsupported `JAX_PLATFORMS=gpu`: <1 s |

The S1 import issue was fixed in `59c1c65`. The first S2 attempt found that a
five-step chunk ending at an episode terminal must use the terminal-state index
of the episode's final transition row. The training adapter and offset helper
were corrected in `65a9224`. A subsequent command failed because this Pixi
environment registers the GPU backend as `cuda`; the runbook was updated and
the successful run used `JAX_PLATFORMS=cuda`.

## S1 — data and row alignment

| Metric | Value |
|---|---:|
| Dataset SHA256 | `a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9` |
| Episodes / state frames / transitions | 1,000 / 1,001,000 / 1,000,000 |
| Observation dimension | 37 (robot `[0,19)`, cube 0 `[19,28)`, cube 1 `[28,37)`) |
| QC regular-loader rows aligned | 1,000,000 / 1,000,000 |
| DQC compact-loader valid rows aligned | 1,000,000 / 1,000,000 |
| Next observations, actions, terminals, and h=5 validity | All matched |
| Alignment gate | Pass |

| Task | Successful state rows |
|---:|---:|
| 1 | 392 |
| 2 | 583 |
| 3 | 1,004 |
| 4 | 724 |
| 5 | 1,379 |

The movement audit uses the v0.8.2b detector: a five-frame absolute XYZ
displacement above 5% of the per-cube/per-axis p99.9 displacement is moving.
There were 996,000 valid five-frame windows. The moving-window fraction was
0.234381 for cube 0, 0.236507 for cube 1, and 0.466738 for either cube moving.
The p99.9 displacement thresholds (meters, cube × XYZ) were
`[[0.004828, 0.005993, 0.005460], [0.004845, 0.005951, 0.005442]]`.

## S2 — shared robot-only dθ model

The model used only the 19D robot observation. Five-frame windows were flattened
in time order to 95D; the episode-level split was 90/10 with seed 0. The saved
20k checkpoint records dataset ID
`ogbench:cube-double-play-v0:a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9`.

| Checkpoint | Collapse alarm | Dead-unit fraction | λ | Held-out 5-step distance p90 | `d > 6.25` rate | F3 gate |
|---|---:|---:|---:|---:|---:|---|
| 10k | false | 0 for all layers | 24.8073 | 5.8192 | 0.066576 | Not passed |
| 20k | false | 0 for all layers | 31.8353 | 5.7380 | 0.059378 | **Fail** (`≤0.05` required) |

The final checkpoint's encoder/projector output variances were 6.7648 / 9.6812.
The 20k triangle diagnostic reported 60 violations among 100,000 sampled
triangles (0.000600). The collapse check passed; the held-out F3 chunk-violation
gate failed.

## Stopped stages

Execution stopped after S2 as required by the F3 gate. S3 δ sweep, S4 G2 audit,
S5 task graphs and targets, and all QC/DQC runs in S6–S8 were not run. No
benchmark success or comparison values were measured. No QC-paper cube-double
reference row is included because the workflow stopped before policy runs.

Artifacts are under ignored `outputs/cube-double-play/v18/`, including
`data_alignment.json`, `dtheta_robot_window5/step_020000/`, and
`dtheta_gate.json`. The gate JSON records `gate_passed=false`.

## Push 명령

- QC (`graph-critic`): 변경 없음.
- FQL: 변경 없음.
- DQC (`graph-critic`): `341c529 Support cube-double DQC single-task graph data`.
- Root (`quasi_develop`): `2d9e3f2 Implement v1.8 cube-double graph workflow`; `59c1c65 Add cube position decoder for v1.8 audit`; `65a9224 Fix v1.8 terminal chunk endpoint mapping`; `4ed62a6 Use the installed JAX CUDA backend in v1.8 runbook`. This report commit is also pending; the command prints its hash and title.

```bash
cd /home/url/graphcritic/third_party/dqc && git status -sb && git log --oneline origin/graph-critic..HEAD && git push origin graph-critic
cd /home/url/graphcritic && git status -sb && git log --oneline origin/quasi_develop..HEAD && git push origin quasi_develop
```
