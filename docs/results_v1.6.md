# Graph Critic v1.6 — DQC TD versus graph critic results

**Dataset note:** P13 4×4 task 2 uses directly generated data (1,321 episodes, 1,000,000 transitions; generated episodes reached the goal). It is not a standard benchmark.

Date: 2026-11-13
Branch: `quasi_develop`; DQC fork branch: `graph-critic`

## Commits, clean state, and tests

| Repository | Execution commit | Branch / state | `git_dirty` | Tests |
|---|---|---|---|---|
| Root | `ef92576f5df310785820a737fe27bb88778f8f44` | `quasi_develop` | `false` | `tests/test_v16.py`: 2 passed |
| QC fork | `f12e624ff62a23c77366e751832361fe8a017f79` | `graph-critic` | `false` | No v1.6 changes; not run |
| FQL fork | `e8cd16eb490332924dfa2492097219f181765933` | detached | `false` | No v1.6 changes; not run |
| DQC fork | `39d41719a344928f0ba8890afd02975d569c5169` | `graph-critic` | `false` | `tests/test_v16_graph_critic.py`: 4 passed |

Both graph builds ran clean at root commit `14bfdfb52c1dbde1812bbe00ca0aff04c4965a95`. R1–R4 ran with root `ef92576f5df310785820a737fe27bb88778f8f44` and DQC `39d41719a344928f0ba8890afd02975d569c5169`; the root, QC, FQL, and DQC worktrees were clean at each run start.

| Stage | Elapsed time |
|---|---:|
| 3×3 gamma-pow target build | 61.73 s |
| P13 4×4 gamma-pow target build | 72.13 s |
| Speed probe, batch 4096 | 87.23 s / 1,000 updates |
| Speed probe, batch 1024 | 72.68 s / 1,000 updates |
| Speed probe, batch 256 | 60.79 s / 1,000 updates |
| R1, 3×3 TD | 1,773 s |
| R2, 3×3 graph | 1,688 s |
| R3, P13 4×4 TD | 1,963 s |
| R4, P13 4×4 graph, clean full rerun | 1,631 s (27 min 11 s) |

| Speed-probe batch | Updates/s | Estimated 300k duration |
|---:|---:|---:|
| 4096 | 11.46 | 7.27 h |
| 1024 | 13.76 | 6.06 h |
| 256 | 16.45 | 5.07 h |

All estimates exceeded the 3-hour threshold; batch size **256** was selected for all four runs. The completed runs' logged training `time/total_time` values were 1,716.83 s (R1), 1,617.93 s (R2), 1,900.15 s (R3), and 1,575.93 s (R4).

The first R4 attempt has an evaluation CSV only through 250k and no 300k checkpoint. Its `train.csv` logged step 300k, but the final evaluation/checkpoint were absent. It was left intact; R4 was restarted from step 0 into a separate timestamped run directory and completed there.

## T1 — gamma-pow targets

| Task / target | Expected graph hash | Generated graph hash | Match | Valid / loss-mask fraction | Unreachable fraction | y mean | p50 | p90 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 3×3 task 3 | `3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa` | `3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa` | Yes | 0.976 / 0.976 | 0.055525 | 0.694431 | 0.735766 | 0.876025 |
| P13 4×4 task 2 | `6bd8f696e65e362770facddea65caca03f57abdb78927f054a730f32afbc01aa` | `6bd8f696e65e362770facddea65caca03f57abdb78927f054a730f32afbc01aa` | Yes | 0.968296 / 0.968296 | 0.002357 | 0.858550 | 0.859080 | 0.956933 |

Both targets use `kind=gamma_pow`, `h=25`, and `gamma=0.999`; maximum y is 1.000000 for both. The capped (`source == 2`) fraction among loss-mask rows is 0.056019 for 3×3 and 0.002357 for P13 4×4.

## T3 — DQC results

Each evaluation used 50 episodes. Wrong-button values are the mean final `episode.final_wrong_buttons`. Success threshold steps use strict `>0.5` and `>0.9` comparisons.

| Task / method | Success at 50k / 100k / 150k / 200k / 250k / 300k | Wrong buttons at 0 / 50k / 100k / 150k / 200k / 250k / 300k | First `>0.5` | First `>0.9` | Final chunk critic loss | Final partial/action critic loss |
|---|---|---|---:|---:|---:|---:|
| 3×3 task 3, DQC TD | 0.00 / 0.34 / 0.98 / 1.00 / 1.00 / 0.98 | 6.80 / 4.14 / 2.64 / 0.10 / 0.00 / 0.00 / 0.08 | 150k | 150k | 0.558706 | 0.280708 |
| 3×3 task 3, DQC graph | 0.34 / 0.40 / 0.64 / 0.54 / 0.62 / 0.64 | 7.18 / 2.96 / 2.32 / 1.60 / 1.98 / 1.70 / 1.62 | 150k | Not reached | 0.516619 | 0.260797 |
| P13 4×4 task 2, DQC TD | 0.70 / 0.80 / 0.78 / 0.84 / 0.88 / 0.96 | 5.22 / 1.42 / 1.08 / 1.02 / 0.88 / 0.64 / 0.14 | 50k | 300k | 0.414634 | 0.207010 |
| P13 4×4 task 2, DQC graph | 0.70 / 0.78 / 0.88 / 0.90 / 0.88 / 0.94 | 4.94 / 2.00 / 1.40 / 0.52 / 0.54 / 0.66 / 0.32 | 50k | 300k | 0.370939 | 0.185131 |

## Historical reference rows

| Task / method | Success at 50k / 100k / 150k / 200k / 250k / 300k |
|---|---|
| 3×3 task 3, QC + graph critic (best-of-N) | 0.44 / 0.78 / 0.96 / 1.00 / 0.98 / 0.94 |
| 3×3 task 3, QC TD | 0.10 / 1.00 / 0.98 / 1.00 / — / — |
| 3×3 task 3, FQL TD (`alpha=100`) | 0.00 / 0.00 / 0.38 / 0.66 / 0.86 / 0.94 |
| P13 4×4 task 2, QC + graph critic | 0.68 / 0.64 / 0.80 / 0.78 / 0.82 / 0.78 |
| P13 4×4 task 2, QC TD | 0.04 / 0.56 / 0.14 / 0.10 / 0.60 / 0.46 |
| P13 4×4 task 2, FQL TD (`alpha=100`) | 0.08 / 0.40 / 0.32 / 0.82 / 0.48 / 0.82 |
| P13 4×4 task 2, FQL graph critic (`alpha=100`) | 0.02 / 0.32 / 0.32 / 0.30 / 0.38 / 0.34 |

## Artifact paths

- 3×3 target: `outputs/puzzle/standard-3x3-task3/v16/Lsame_h25_gp/`
- P13 4×4 target: `outputs/puzzle/p13-4x4-task2-100-generated/v16/Lsame_h25_gp/`
- R1: `outputs/puzzle/standard-3x3-task3/v16/qc/td/dqc/v16-3x3t3-dqc-td/sd00020261113_173502/`
- R2: `outputs/puzzle/standard-3x3-task3/v16/qc/graph/dqc/v16-3x3t3-dqc-graph/sd00020261113_180457/`
- R3: `outputs/puzzle/p13-4x4-task2-100-generated/v16/qc/td/dqc/v16-4x4p13-dqc-td/sd00020261113_183325/`
- R4 complete rerun: `outputs/puzzle/p13-4x4-task2-100-generated/v16/qc/graph/dqc/v16-4x4p13-dqc-graph/sd00020261113_193502/`
- R4 incomplete first attempt retained: `outputs/puzzle/p13-4x4-task2-100-generated/v16/qc/graph/dqc/v16-4x4p13-dqc-graph/sd00020261113_190626/`

## Push commands

- QC fork: 변경 없음.
- FQL fork: 변경 없음.
- DQC fork (`graph-critic`): `f1085a7 Add singletask graph critic to DQC`; `39d4171 Handle offline W&B URL on shutdown`. The remote does not yet have `graph-critic`; create it with:

```bash
cd /home/url/graphcritic/third_party/dqc && git status -sb && git log --oneline -2 && git push -u origin graph-critic
```

- Root (`quasi_develop`): `14bfdfb Add v1.6 DQC graph critic workflow`; `94ba06c Pin DQC offline shutdown fix`; `ef92576 Fix DQC speed log field in v1.6 runbook`; results-report commit (this document).

```bash
cd /home/url/graphcritic && git status -sb && git log --oneline origin/quasi_develop..HEAD && git push origin quasi_develop
```
