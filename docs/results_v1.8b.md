Dataset: OGBench cube-double-play 표준 데이터(100만), singletask task 1–5.

# Graph Critic v1.8b — cube-double bridge revision results

Date: 2026-11-14
Branch: `quasi_develop`; QC / DQC fork branches: `graph-critic`

## Commits, clean state, and tests

| Item | Value |
|---|---|
| Implementation commit | `812007adfd2399cd94d6e0a2037221dc5d7e9b22` |
| Root worktree at every run start | clean (`git_dirty=false`) |
| QC / FQL / DQC worktrees | clean at run start; no source changes |
| Tests | `tests/test_v18.py`: 13 passed; 1 upstream AnyIO pytest deprecation warning |
| Fixed dθ checkpoint | `outputs/cube-double-play/v18/dtheta_robot_window5_r40k/step_040000/` |
| Fixed dθ params SHA256 | `457edaaf0c4aed26963f130a414fd3de3862f45da0b1e2a07a7a01dd5c6f74da` |

## S3′ — δxy × K_free sweep

Candidate count is shared across K_free for each δxy. C summaries are over reachable graph nodes at the selected B*. `dθ free p50/p90` covers accepted bridges whose two endpoints are non-contact; `>20` reports the subset admitted only by K_free > 20.

| Setting | Candidates | Accepted | Mean R(b=32) | Task B* (1/2/3/4/5) | C p50 (1/2/3/4/5) | C p90 (1/2/3/4/5) | Free dθ p50/p90 | Free dθ>20 p50/p90 | Sweep time (s) |
|---|---:|---:|---:|---|---|---|---|---|---:|
| v1.8 isotropic 4cm, C0 | 1,279,713 | 412 | 0.0439 | 1/1/1/1/1 | 420.9/552.0/575.3/566.7/410.5 | 912.0/1081.3/1091.1/1064.0/980.1 | 11.87/18.01 | —/— | 2.11 |
| v1.8 isotropic 4cm, no C0 | 1,279,713 | 188,896 | 0.7316 | 7/7/7/6/6 | 909.2/876.6/934.5/886.0/715.9 | 1425.2/1338.2/1364.0/1340.2/1161.2 | 11.28/18.09 | —/— | 10.85 |
| δxy=2cm, Kfree=20 | 122,877 | 17,747 | 0.0158 | 0/0/1/1/0 | 105.0/209.0/213.0/317.9/57.0 | 545.6/679.4/862.0/856.9/643.2 | 10.92/17.95 | —/— | 1.64 |
| δxy=2cm, Kfree=40 | 122,877 | 32,443 | 0.0170 | 0/0/1/1/1 | 105.0/209.0/240.0/342.0/146.0 | 545.6/679.4/899.3/853.0/812.0 | 18.52/35.70 | 29.95/38.04 | 1.64 |
| δxy=2cm, Kfree=80 | 122,877 | 65,489 | 0.0230 | 0/1/1/4/2 | 105.0/351.0/292.2/644.0/282.0 | 545.6/962.8/996.7/1432.7/1054.6 | 40.85/72.30 | 52.39/74.28 | 1.71 |
| δxy=3cm, Kfree=20 | 393,950 | 57,180 | 0.2230 | 28/27/19/22/22 | 4039.8/3236.0/2316.2/2161.0/1937.6 | 6078.3/6138.8/3928.5/4891.8/3689.0 | 11.18/18.04 | —/— | 3.17 |
| δxy=3cm, Kfree=40 | 393,950 | 107,061 | 0.3999 | 22/22/13/18/22 | 2785.0/2325.8/2043.6/1802.2/1544.7 | 3982.9/4054.3/2983.8/3351.5/2558.7 | 19.09/35.85 | 29.96/38.03 | 5.10 |
| δxy=3cm, Kfree=80 | 393,950 | 213,249 | 0.5538 | 13/9/9/9/9 | 1927.4/1324.5/1461.7/1261.9/1119.8 | 2662.3/2046.8/2217.4/2005.9/1794.9 | 40.30/71.89 | 51.22/73.91 | 7.62 |
| δxy=4cm, Kfree=20 | 931,001 | 143,219 | 0.6671 | 9/8/7/7/7 | 1115.0/1095.4/1184.0/1117.3/878.3 | 1754.2/1717.6/1762.6/1710.6/1461.1 | 11.33/18.10 | —/— | 8.61 |
| δxy=4cm, Kfree=40 | 931,001 | 270,663 | 0.7391 | 8/7/7/6/6 | 935.5/891.6/923.0/893.5/726.3 | 1395.9/1375.7/1349.0/1360.9/1230.0 | 19.29/35.87 | 29.92/38.02 | 10.19 |
| δxy=4cm, Kfree=80 | 931,001 | 519,769 | 0.7995 | 6/6/6/5/5 | 801.5/728.0/706.1/725.0/599.2 | 1187.5/1066.9/1067.2/1107.8/968.3 | 38.77/71.48 | 49.75/73.66 | 12.01 |

Selection: `xy_4cm_kfree_80`; δxy=4 cm, δz=1 cm, δrot=15°, K_free=80; mean R(b=32)=0.799517; B*=6/6/6/5/5. Gate passed: `True`.
Legacy C0 reproduction: 0.043899 (v1.8 reference 0.043899). Legacy match with `bridge_filter=none`: 0.731592.

## S4′ — G2 audit

Accepted bridge occurrences: 2,598,845; structural gate passed: `True`; support gate passed: `True`; overall: `True`.

| Task | Accepted occurrences | Dxy violations | Dz violations | Drot violations | Contact mismatches | Support-state mismatch | Prior G2 (contact/elevated) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 519,769 | 0 | 0 | 0 | 0 | 0.356% (1,851) | 0.000% (0) |
| 2 | 519,769 | 0 | 0 | 0 | 0 | 0.356% (1,851) | 0.000% (0) |
| 3 | 519,769 | 0 | 0 | 0 | 0 | 0.356% (1,851) | 0.000% (0) |
| 4 | 519,769 | 0 | 0 | 0 | 0 | 0.356% (1,851) | 0.000% (0) |
| 5 | 519,769 | 0 | 0 | 0 | 0 | 0.356% (1,851) | 0.000% (0) |
| **All** | **2,598,845** | **0** | **0** | **0** | **0** | **0.356% (9,255)** | **0.000% (0)** |

Definitions: Dxy≤selected δxy; Dz≤1 cm; Drot≤15° under cube symmetry; contact is `obs[18]>0.5`; support classes are table (`z<3 cm`), stacked (horizontal distance <2 cm and upper cube is 3–5 cm above the other), or lifted.

## S5 — task graphs, targets, and G3 measurements

| Task | B* | Accepted / retained | Graph hash | Reachability | Valid / loss mask | Capped | Target mean / p50 / p90 |
|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 6 | 519,769 / 33,691 | `733e17c18d34aa724e74f380e361b535531a56f84ca474187d8ec0abf063a574` | 0.7973 | 0.9960 / 0.9960 | 0.2036 | 99.50 / 99.99 / 100.00 |
| 2 | 6 | 519,769 / 34,932 | `ca959d003217efc682a9a644725aa6e37b38de29096e5331cdc6d0ddb2495af7` | 0.8007 | 0.9960 / 0.9960 | 0.2002 | 99.39 / 99.97 / 100.00 |
| 3 | 6 | 519,769 / 34,880 | `edacc8d320b5190a082ff1b729a729cdf201170c706d9d31b49e6b2649405668` | 0.8000 | 0.9960 / 0.9960 | 0.2009 | 99.14 / 99.97 / 100.00 |
| 4 | 5 | 519,769 / 31,924 | `51703e4a0486d5a14e699827cd07eb49867680dae8ca651f4a46612cc66daa3e` | 0.7976 | 0.9960 / 0.9960 | 0.2033 | 99.25 / 99.97 / 100.00 |
| 5 | 5 | 519,769 / 32,047 | `817f05745c5f2bd5e42942628d48da48859dd413f444c4f04c8d4db014b36fb7` | 0.7963 | 0.9960 / 0.9960 | 0.2046 | 98.09 / 99.90 / 100.00 |

Target/graph verification: all five graph hashes recomputed and all ten h=5 discounted/gamma-power artifacts loaded with matching dataset IDs and graph hashes; `valid=loss_mask=0.996` for every task.

## S6–S8 — QC and DQC performance

Success curves list evaluations at 25k, 50k, …, 300k (50 episodes each). AUC is the trapezoidal area normalized over 25k–300k. Threshold steps use strict `success > 0.5` and `success > 0.9`.

| Method | Seed | Task | Success 25k→300k | Final (225k–300k mean) | First >0.5 | First >0.9 | AUC | Elapsed wall, incl. eval (s) | Run directory |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| DQC graph-action (γ=0.99) | 0 | 1 | 0.00/0.02/0.06/0.14/0.24/0.22/0.16/0.18/0.18/0.14/0.30/0.16 | 0.195 | — | — | 0.1564 | 1563 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/DQC_graph/task1/seed0/dqc/v18b-cube-t1-dqc-graph-s0/sd00020261114_205247` |
| DQC TD (γ=0.99) | 0 | 1 | 0.02/0.02/0.02/0.08/0.14/0.10/0.10/0.16/0.12/0.22/0.08/0.00 | 0.105 | — | — | 0.0955 | 2094 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/DQC_td/task1/seed0/dqc/v18b-cube-t1-dqc-td-s0/sd00020261114_201754` |
| DQC graph-action (γ=0.99) | 0 | 2 | 0.00/0.00/0.00/0.00/0.00/0.02/0.00/0.00/0.00/0.00/0.00/0.00 | 0.000 | — | — | 0.0018 | 1660 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/DQC_graph/task2/seed0/dqc/v18b-cube-t2-dqc-graph-s0/sd00020261114_215418` |
| DQC TD (γ=0.99) | 0 | 2 | 0.00/0.00/0.00/0.00/0.00/0.02/0.02/0.00/0.02/0.02/0.00/0.02 | 0.015 | — | — | 0.0082 | 2127 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/DQC_td/task2/seed0/dqc/v18b-cube-t2-dqc-td-s0/sd00020261114_211851` |
| QC graph | 0 | 1 | 0.02/0.00/0.02/0.04/0.06/0.14/0.00/0.10/0.12/0.14/0.12/0.20 | 0.145 | — | — | 0.0773 | 1472 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/QC_graph/task1/seed0/qc/v18b-cube-t1-qc-graph-s0/cube-double-play-singletask-task1-v0/sd00020261114_131909` |
| QC TD | 0 | 1 | 0.08/0.24/0.32/0.58/0.64/0.72/0.72/0.82/0.80/0.64/0.80/0.78 | 0.755 | 100k | — | 0.6100 | 4142 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/QC_td/task1/seed0/qc/v18b-cube-t1-qc-td-s0/cube-double-play-singletask-task1-v0/sd00020261114_121009` |
| QC graph | 0 | 2 | 0.00/0.00/0.00/0.00/0.00/0.02/0.00/0.00/0.00/0.00/0.00/0.00 | 0.000 | — | — | 0.0018 | 1498 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/QC_graph/task2/seed0/qc/v18b-cube-t2-qc-graph-s0/cube-double-play-singletask-task2-v0/sd00020261114_145642` |
| QC TD | 0 | 2 | 0.00/0.02/0.04/0.02/0.04/0.18/0.08/0.28/0.20/0.38/0.44/0.60 | 0.405 | 300k | — | 0.1800 | 4380 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/QC_td/task2/seed0/qc/v18b-cube-t2-qc-td-s0/cube-double-play-singletask-task2-v0/sd00020261114_134342` |
| QC graph | 0 | 3 | 0.00/0.00/0.00/0.00/0.00/0.00/0.00/0.00/0.00/0.02/0.02/0.00 | 0.010 | — | — | 0.0036 | 1498 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/QC_graph/task3/seed0/qc/v18b-cube-t3-qc-graph-s0/cube-double-play-singletask-task3-v0/sd00020261114_163502` |
| QC TD | 0 | 3 | 0.02/0.00/0.06/0.02/0.08/0.14/0.14/0.28/0.12/0.38/0.50/0.22 | 0.305 | — | — | 0.1673 | 4402 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/QC_td/task3/seed0/qc/v18b-cube-t3-qc-td-s0/cube-double-play-singletask-task3-v0/sd00020261114_152140` |
| QC graph | 0 | 4 | 0.00/0.00/0.00/0.00/0.02/0.00/0.00/0.00/0.00/0.00/0.00/0.00 | 0.000 | — | — | 0.0018 | 1506 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/QC_graph/task4/seed0/qc/v18b-cube-t4-qc-graph-s0/cube-double-play-singletask-task4-v0/sd00020261114_181440` |
| QC TD | 0 | 4 | 0.00/0.00/0.02/0.00/0.02/0.02/0.04/0.06/0.08/0.14/0.12/0.10 | 0.110 | — | — | 0.0500 | 4480 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/QC_td/task4/seed0/qc/v18b-cube-t4-qc-td-s0/cube-double-play-singletask-task4-v0/sd00020261114_170000` |
| QC graph | 0 | 5 | 0.00/0.00/0.00/0.00/0.00/0.00/0.00/0.00/0.02/0.00/0.00/0.06 | 0.020 | — | — | 0.0045 | 1493 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/QC_graph/task5/seed0/qc/v18b-cube-t5-qc-graph-s0/cube-double-play-singletask-task5-v0/sd00020261114_195303` |
| QC TD | 0 | 5 | 0.02/0.00/0.02/0.10/0.12/0.04/0.26/0.24/0.26/0.32/0.40/0.44 | 0.355 | — | — | 0.1809 | 4397 | `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/QC_td/task5/seed0/qc/v18b-cube-t5-qc-td-s0/cube-double-play-singletask-task5-v0/sd00020261114_183946` |

### Method means

Curves show the per-step mean success. QC means use all five tasks; DQC means use only completed tasks 1–2. The matching-task QC rows also use tasks 1–2. Final success is the mean of the last four evaluations (225k–300k).

| Method | Included tasks | Mean curve 25k→300k | Final success (last 4 mean) | Mean AUC | Mean elapsed wall, incl. eval (s) |
|---|---|---|---:|---:|---:|
| QC TD | 1/2/3/4/5 | 0.02/0.05/0.09/0.14/0.18/0.22/0.25/0.34/0.29/0.37/0.45/0.43 | 0.386 | 0.2376 | 4360.2 |
| QC graph | 1/2/3/4/5 | 0.00/0.00/0.00/0.01/0.02/0.03/0.00/0.02/0.03/0.03/0.03/0.05 | 0.035 | 0.0178 | 1493.4 |
| DQC TD (γ=0.99) | 1/2 | 0.01/0.01/0.01/0.04/0.07/0.06/0.06/0.08/0.07/0.12/0.04/0.01 | 0.060 | 0.0518 | 2110.5 |
| DQC graph-action (γ=0.99) | 1/2 | 0.00/0.01/0.03/0.07/0.12/0.12/0.08/0.09/0.09/0.07/0.15/0.08 | 0.098 | 0.0791 | 1611.5 |
| QC TD | 1/2 | 0.04/0.13/0.18/0.30/0.34/0.45/0.40/0.55/0.50/0.51/0.62/0.69 | 0.580 | 0.3950 | 4261.0 |
| QC graph | 1/2 | 0.01/0.00/0.01/0.02/0.03/0.08/0.00/0.05/0.06/0.07/0.06/0.10 | 0.073 | 0.0395 | 1485.0 |

## Stage timings and execution boundary

| Stage | Elapsed |
|---|---:|
| candidate_xy_2cm_seconds | 92.28 s |
| candidate_xy_3cm_seconds | 206.86 s |
| candidate_xy_4cm_seconds | 338.46 s |
| score_xy_2cm_seconds | 29.90 s |
| score_xy_3cm_seconds | 6.52 s |
| score_xy_4cm_seconds | 6.98 s |
| task_graph_target_builds_seconds | 46.75 s |
| S4′ G2 audit | 72.99 s |
| DQC graph task 1 seed 0 | 1563 s (exit 0) |
| DQC td task 1 seed 0 | 2094 s (exit 0) |
| DQC graph task 2 seed 0 | 1660 s (exit 0) |
| DQC td task 2 seed 0 | 2127 s (exit 0) |
| QC graph task 1 seed 0 | 1472 s (exit 0) |
| QC td task 1 seed 0 | 4142 s (exit 0) |
| QC graph task 2 seed 0 | 1498 s (exit 0) |
| QC td task 2 seed 0 | 4380 s (exit 0) |
| QC graph task 3 seed 0 | 1498 s (exit 0) |
| QC td task 3 seed 0 | 4402 s (exit 0) |
| QC graph task 4 seed 0 | 1506 s (exit 0) |
| QC td task 4 seed 0 | 4480 s (exit 0) |
| QC graph task 5 seed 0 | 1493 s (exit 0) |
| QC td task 5 seed 0 | 4397 s (exit 0) |

Runs completed successfully: 14/30; failed completed runs: 0; started runs: 15; ledgered runs: 14/30.
Interrupted run (excluded from results): DQC TD task 3 seed 0; `/home/url/graphcritic/outputs/cube-double-play/v18/v18b/runs/DQC_td/task3/seed0/dqc/v18b-cube-t3-dqc-td-s0/sd00020261114_222158`; last evaluation step `25000`.
Stop reason: 방향 확인용 실험이라 seed 0, QC 5 task + DQC 일부 task로 충분하다고 사용자가 판단해 중단. S7의 나머지 task와 S8 seed 1·2는 실행하지 않았다.
Outputs and logs: `outputs/cube-double-play/v18/v18b/`. The original v1.8 artifacts and fixed 40k dθ model were reused; no dθ retraining was performed.

## Push 명령

- QC (`graph-critic`): 변경 없음.
- FQL (`master`): 변경 없음.
- DQC (`graph-critic`): 변경 없음.
- Root (`quasi_develop`): `812007a Implement v1.8b cube bridge rules and sweep`; the report commit is also pending and its exact hash/title is shown by `git log` below.

```bash
cd /home/url/graphcritic/third_party/qc && git status -sb && git log --oneline origin/graph-critic..HEAD && git push origin graph-critic
cd /home/url/graphcritic/third_party/fql && git status -sb && git log --oneline origin/master..HEAD && git push origin master
cd /home/url/graphcritic/third_party/dqc && git status -sb && git log --oneline origin/graph-critic..HEAD && git push origin graph-critic
cd /home/url/graphcritic && git status -sb && git log --oneline origin/quasi_develop..HEAD && git push origin quasi_develop
```
