Dataset: OGBench puzzle-3x3-play 표준 데이터(100만), singletask task 3.

# Graph Critic v1.7 — direct action-chunk graph targets in DQC

Date: 2026-11-13  
Branch: `quasi_develop`; DQC fork branch: `graph-critic`

## Commits, clean state, and tests

| Stage | Root commit | DQC commit | `git_dirty` at start |
|---|---|---|---|
| Target builds, R5 and R6 | `3a1971d410bb8fb599879148a919885fcc621cbb` | `86f6af5` | `false` |
| Final checkpoint diagnostic | `0089edb` | `1f06e2e` | `false` |
| v1.7 tests | `0089edb` | `1f06e2e` | `false` |

QC was clean at `f12e624ff62a23c77366e751832361fe8a017f79`; FQL was clean,
detached at `e8cd16eb490332924dfa2492097219f181765933`. Neither was changed.

| Check | Result |
|---|---|
| `tests/test_v17_action_graph.py` | **4 passed**, 3 upstream deprecation warnings, 6.43 s |
| Action graph loss | Direct masked BCE on `graph_target`; no expectile weighting or value bootstrap |
| Unused modules in action mode | Chunk critic is not created; value gradients are zero; target action critic is unchanged |
| Default graph-target level | `chunk`; numerical loss matches the v1.6 formula in the regression test |

R5/R6 retained the v1.6 R2 settings: `q_agg=mean`, `num_qs=2`, batch 256,
best-of-N 32, flow steps 10, learning rate `3e-4`, seed 0, and 300k updates.
R5 changed `backup_horizon=5`, disabled the chunk critic, selected
`graph_target_level=action`, and used the h=5 target. R6 also changed discount
and target gamma to `0.99`, because R5's final success was `0.74` (≤0.90).

## T1 — h=5 `gamma_pow` targets

| Target | Gamma | Graph hash matches Lsame | Valid fraction | Loss-mask fraction | Capped (`source=2`), all rows / masked rows | y mean / p50 / p90 / max | Build time | Target content hash |
|---|---:|---|---:|---:|---|---|---:|---|
| `Lsame_h5_gp` | 0.999 | Yes | 0.996000 | 0.996000 | 0.054722 / 0.054942 | 0.709233 / 0.750928 / 0.893727 / 1.000000 | 71.31 s | `7556ac9c8c28fdc5f1af6bb661c82e8be9e639e06765b17366fd4fa93823ad37` |
| `Lsame_h5_gp_g099` (conditional R6) | 0.99 | Yes | 0.996000 | 0.996000 | 0.054722 / 0.054942 | 0.116812 / 0.056279 / 0.323474 / 1.000000 | 55.70 s | `35038a90cefe7e3c2021923f02a2bea3153b69d1f9d18e5f56816be40ede5462` |

Both builds produced graph hash
`3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa` and
dataset ID
`ogbench:puzzle-3x3-play-singletask-task3-v0:bca9b1775ea5c01af5ac3f9b652709d341bddd17fe4b6368056566ac3c5fbb0f`.
The y statistics are over loss-mask rows. `source=2` accounts for 54,722 of
1,000,000 rows and 54,722 of 996,000 loss-mask rows.
The 0.999 build ran at root commit `3a1971d410bb8fb599879148a919885fcc621cbb`
with `git_dirty=false`.

## Results

Success is evaluated on 50 episodes at each checkpoint. Wrong-button values
are the mean final wrong-button count at step 0 and each saved checkpoint.
Training duration is the final `time/total_time` from `train.csv`, in seconds.
“First >” uses strict comparisons.

| Run | Success 50k / 100k / 150k / 200k / 250k / 300k | Wrong buttons 0 / 50k / 100k / 150k / 200k / 250k / 300k | First >0.5 | First >0.9 | Final chunk / action critic loss | Training time |
|---|---|---|---:|---:|---:|---:|
| R1 — DQC TD (v1.6) | 0.00 / 0.34 / 0.98 / 1.00 / 1.00 / 0.98 | 6.80 / 4.14 / 2.64 / 0.10 / 0.00 / 0.00 / 0.08 | 150k | 150k | 0.558706 / 0.280708 | 1,716.83 s |
| R2 — DQC graph, distilled (v1.6) | 0.34 / 0.40 / 0.64 / 0.54 / 0.62 / 0.64 | 7.18 / 2.96 / 2.32 / 1.60 / 1.98 / 1.70 / 1.62 | 150k | Not reached | 0.516619 / 0.260797 | 1,617.93 s |
| R5 — DQC graph-action, γ=0.999 | 0.50 / 0.66 / 0.68 / 0.66 / 0.64 / 0.74 | 6.88 / 1.98 / 1.32 / 1.14 / 1.36 / 1.56 / 1.04 | 100k | Not reached | — / 0.496603 | 1,275.86 s |
| R6 — DQC graph-action, γ=0.99 | 0.86 / 0.96 / 0.98 / 0.94 / 0.96 / 0.94 | 7.08 / 0.50 / 0.20 / 0.08 / 0.24 / 0.20 / 0.22 | 50k | 100k | — / 0.290449 | 1,209.83 s |

| Reference | Success 50k / 100k / 150k / 200k / 250k / 300k |
|---|---|
| QC graph critic, v1.2b | 0.44 / 0.78 / 0.96 / 1.00 / 0.98 / 0.94 |

## Fixed-row action-critic diagnostic

The same 10,000 rows were selected with seed 0 from h=5 valid loss-mask rows.
Scores are `sigmoid(q_agg(action_critic_logits))`, using each run's configured
`q_agg=mean`. R2 and R5 are compared with the γ=0.999 targets; R6 is compared
with the γ=0.99 targets. This restored checkpoints and evaluated network
outputs only; it did not train or step an environment.

| Checkpoint | Target gamma | Rows | Spearman | MAE |
|---|---:|---:|---:|---:|
| R2 — distilled action critic | 0.999 | 10,000 | 0.886900 | 0.043477 |
| R5 — direct action critic | 0.999 | 10,000 | 0.991271 | 0.012033 |
| R6 — direct action critic | 0.99 | 10,000 | 0.995651 | 0.006264 |

## Artifact paths

- Runbook: [RUN_v1.7.md](RUN_v1.7.md)
- γ=0.999 graph/targets: `outputs/puzzle-3x3-task3/v17/Lsame_h5_gp/`
- γ=0.99 graph/targets: `outputs/puzzle-3x3-task3/v17/Lsame_h5_gp_g099/`
- R5 run: `outputs/puzzle-3x3-task3/v17/runs/R5_dqc_graph_action/dqc/v17-3x3t3-dqc-graph-action/sd00020261113_213623/`
- R6 run: `outputs/puzzle-3x3-task3/v17/runs/R6_dqc_graph_action_g099/dqc/v17-3x3t3-dqc-graph-action-g099/sd00020261113_220018/`
- Diagnostic JSON: `outputs/puzzle-3x3-task3/v17/action_target_diagnostic.json`
- Reused R2 checkpoint: `outputs/puzzle-3x3-task3/runs/A4_dqc_graph_distill/params_300000.pkl`

## Push 명령

- QC (`graph-critic`): 변경 없음.
- FQL: 변경 없음.
- DQC (`graph-critic`) commits pending push:
  - `86f6af5 Add direct action graph targets to DQC`
  - `aa40eb3 Fix DQC checkpoint diagnostic loader`
  - `142b2a5 Fix DQC diagnostic dataset index lookup`
  - `1f06e2e Match DQC action-score aggregation in diagnostic`
- Root (`quasi_develop`) commits pending push:
  - `3a1971d Add v1.7 DQC action-target runbook`
  - `fdbffee Pin DQC diagnostic loader fix for v1.7`
  - `f3bdb0a Pin DQC diagnostic index fix for v1.7`
  - `0089edb Pin DQC action-score diagnostic update`
  - This results-report commit is also pending; its hash/title appears in the command output below.

```bash
cd /home/url/graphcritic/third_party/dqc && git status -sb && git log --oneline origin/graph-critic..HEAD && git push origin graph-critic
cd /home/url/graphcritic && git status -sb && git log --oneline origin/quasi_develop..HEAD && git push origin quasi_develop
```
