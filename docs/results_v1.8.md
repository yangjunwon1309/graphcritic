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
| Resume/gate/diagnostic implementation | `7f1f5d3` — Resume v1.8 dtheta training through 40k |
| S1 execution commit | `59c1c65129dbfa384211b49f74744959541f5172` |
| Successful S2 execution commit | `4ed62a6f2bb217b34eb1d950d245c9ab233e4c0b` |
| Resume and S3/S4 execution commit | `7f1f5d34a452d0e7fd435824d816c87679c4fad0` |
| `git_dirty` at S1 / S2 start | `false` / `false` for root, QC, FQL, and DQC |
| `git_dirty` at resumed S2 / S3-S4 start | `false` / `false` for root, QC, FQL, and DQC |
| Tests | `tests/test_v18.py`: **9 passed**; `third_party/dqc/tests/test_v18_cube_double.py`: **1 passed** in the initial v1.8 implementation; one upstream pytest deprecation warning per invocation |
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

## S2 continuation — 20k to 40k

The continuation resumed from
`outputs/cube-double-play/v18/dtheta_robot_window5/step_020000/` and wrote to
`outputs/cube-double-play/v18/dtheta_robot_window5_r40k/`. It did not overwrite
the source checkpoints. The 10k/20k train-split measurements below were computed
afterward from those saved checkpoints with the same deterministic 100,000-pair
sample (`seed=1953`) used by the new training report. No weights were updated
during those measurements.

| Resume check | Value |
|---|---|
| Restored step | 20,000 |
| Primal optimizer state restored | `true` |
| Dual λ and dual optimizer state restored | `true` |
| Source train-state SHA256 | `67164adc262c04dbee24123af94e6f98c2dbb12de738c45e291567246c0d8ba0` |
| Source / restored starting λ | 31.83527374 / 31.83527374 |
| Checkpoint at 30k: params SHA256 | `739849125e20287b7e5f68a583af90bc08c7f49057b884883ee30f08a2dc1aac` |
| Checkpoint at 40k: params SHA256 | `457edaaf0c4aed26963f130a414fd3de3862f45da0b1e2a07a7a01dd5c6f74da` |
| `git_dirty` at continuation start | `false` |

| Checkpoint | Held-out violation `d>6.25` | Train violation `d>6.25` | Held-out p90 | λ | Dead units | Collapse | Triangle violation | F3 / checker |
|---:|---:|---:|---:|---:|---:|---|---:|---|
| 10k | 0.066576 | 0.058230* | 5.8192 | 24.8073 | 0 | false | 0.000790 | F3 threshold fail |
| 20k | 0.059378 | 0.046040* | 5.7380 | 31.8353 | 0 | false | 0.000600 | stop gate fail at initial S2 check |
| 30k | 0.044076 | 0.029560 | 5.4818 | 48.1910 | 0 | false | 0.000150 | F3 threshold pass; checker not run |
| 40k | 0.038876 | 0.017370 | 5.4217 | 64.8483 | 0 | false | 0.000070 | stop gate passed |

*Train violation for the old checkpoints is the post-hoc fixed sample described
above. The 30k and 40k values were emitted by `train_chunk_constraint` in the
checkpoint evaluation log. At 40k, `check_v18_model.py --step 40000
--violation-gate stop` returned `gate_passed=true` and
`F3_violation_reported=false`; report mode was not used. The 40k checkpoint is
the required S3+S8 model regardless of the 30k value.

## Held-out 40k violation distribution

The diagnostic used 99,600 held-out five-step pairs. Joint velocity is the L2
norm of the six raw robot joint-velocity observation fields, maximized over the
source model's trailing five-frame window. Deciles contain 9,960 pairs each.
Cube movement uses the S1 five-frame displacement rule.

| Joint-velocity decile | Speed p10–p90 | Violation `d>6.25` | d p50 | d p90 |
|---:|---:|---:|---:|---:|
| 1 | 0.199–0.426 | 0.008032 | 4.3312 | 5.1884 |
| 2 | 0.515–0.708 | 0.011847 | 4.1290 | 5.1875 |
| 3 | 0.738–0.842 | 0.013554 | 4.1387 | 5.1785 |
| 4 | 0.864–0.948 | 0.019478 | 4.1886 | 5.2099 |
| 5 | 0.969–1.053 | 0.023293 | 4.2163 | 5.3028 |
| 6 | 1.075–1.170 | 0.030221 | 4.2228 | 5.3500 |
| 7 | 1.197–1.310 | 0.051908 | 4.2623 | 5.5485 |
| 8 | 1.341–1.482 | 0.079217 | 4.2964 | 5.9074 |
| 9 | 1.524–1.739 | 0.076606 | 4.2612 | 5.8918 |
| 10 | 1.811–2.342 | 0.074598 | 4.2512 | 5.9088 |

| Cube window class | Pairs | Violation `d>6.25` | d p50 | d p90 |
|---|---:|---:|---:|---:|
| Moving | 46,678 | 0.035349 | 4.3073 | 5.4216 |
| Stationary | 52,922 | 0.041986 | 4.1558 | 5.4220 |
| All held-out pairs | 99,600 | 0.038876 | 4.2320 | 5.4217 |

## S3 — δ sweep

The sweep selected δp = 4 cm. Its five-task mean reachability at b=32 was
0.043899; the 3 cm mean was 0.023751, which is 2.015 percentage points below
the 4 cm result and outside the 0.5-point selection band. Counts below are
candidate pairs and pairs accepted by the dθ≤20 judge before the task graph
optimistic-cost filter.

| δp | Candidate pairs | dθ-accepted pairs | Mean reachability at b=32 | Per-task B* |
|---:|---:|---:|---:|---|
| 0.5 cm | 2,413 | 538 | 0.012029 | 0 / 0 / 0 / 0 / 0 |
| 1 cm | 19,324 | 3,162 | 0.012322 | 0 / 0 / 0 / 0 / 0 |
| 2 cm | 155,306 | 21,964 | 0.014384 | 0 / 0 / 0 / 0 / 0 |
| 3 cm | 524,880 | 73,796 | 0.023751 | 0 / 1 / 1 / 1 / 1 |
| 4 cm | 1,279,713 | 188,896 | 0.043899 | 1 / 1 / 1 / 1 / 1 |

For selected δp=4 cm, the sweep's per-task saturation and cost summaries were:

| Task | B* | Reachability at B* | C p50 steps | C p90 steps |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.018016 | 420.887 | 911.960 |
| 2 | 1 | 0.040977 | 552.000 | 1,081.295 |
| 3 | 1 | 0.042930 | 575.310 | 1,091.133 |
| 4 | 1 | 0.056354 | 566.707 | 1,064.033 |
| 5 | 1 | 0.061216 | 410.504 | 980.141 |

## S4 — G2 gate and stop

G2 used 11,575 accepted bridge occurrences at the selected δp. Position and
rotation tolerance checks had no violations. The contact/elevated-cube mismatch
gate failed at 3,111/11,575 = 0.268769, above the 0.05 limit.

| Task | Accepted occurrences | Contact/elevated mismatch fraction | `D_pos > δp` | `D_rot > 15°` |
|---:|---:|---:|---:|---:|
| 1 | 412 | 0.456311 | 0 | 0 |
| 2 | 2,247 | 0.397419 | 0 | 0 |
| 3 | 2,268 | 0.332892 | 0 | 0 |
| 4 | 2,214 | 0.248419 | 0 | 0 |
| 5 | 4,434 | 0.163509 | 0 | 0 |
| **Total** | **11,575** | **0.268769** | **0** | **0** |

The workflow stopped at S4. S5 task graphs/targets were not built; no G3 check,
QC/DQC training, or seed-1/2 run was started. Consequently, no policy success
curves, threshold-crossing steps, AUC, or five-task method means are available.

## Timings and artifacts added in this continuation

| Stage | Elapsed time |
|---|---:|
| v1.8 root tests, `tests/test_v18.py` | 0.43 s (9 passed) |
| Resume 20k→40k training | 837.11 s training timer; 14 min 12 s wall |
| Historical 10k/20k train-pair diagnostics | 24.10 s |
| 40k stop-mode gate | <1 s |
| 40k violation distribution diagnostic | 23.28 s |
| S3 candidate/score sweep and budget sweep | 829.27 s |
| S4 G2 audit | 2.12 s |

- 30k/40k checkpoints and resume logs: `outputs/cube-double-play/v18/dtheta_robot_window5_r40k/`
- Historical train-split diagnostics: `outputs/cube-double-play/v18/dtheta_robot_window5_r40k/historical_train_constraint.json`
- Speed/movement diagnostic: `outputs/cube-double-play/v18/dtheta_robot_window5_r40k/violation_diagnostic.json`
- δ candidates, score caches, sweep, and G2 result: `outputs/cube-double-play/v18/bridge_sweep/`
- No task graph or target artifact was produced because the S4 gate failed.

## Push 명령

- QC (`graph-critic`): 변경 없음.
- FQL (`master`): 변경 없음 (worktree remains detached at `e8cd16eb490332924dfa2492097219f181765933`, equal to `origin/master`).
- DQC (`graph-critic`): 변경 없음.
- Root (`quasi_develop`): `7f1f5d3 Resume v1.8 dtheta training through 40k`, plus this results-report commit (the command lists its exact hash and title).

```bash
cd /home/url/graphcritic/third_party/qc && git status -sb && git log --oneline origin/graph-critic..HEAD && git push origin graph-critic
cd /home/url/graphcritic/third_party/fql && git status -sb && git log --oneline origin/master..HEAD && git push origin master
cd /home/url/graphcritic/third_party/dqc && git status -sb && git log --oneline origin/graph-critic..HEAD && git push origin graph-critic
cd /home/url/graphcritic && git status -sb && git log --oneline origin/quasi_develop..HEAD && git push origin quasi_develop
```
