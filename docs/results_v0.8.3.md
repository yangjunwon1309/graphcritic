# Graph Critic v0.8.3 D1 oracle QC upper-bound results

Date: 2026-11-10

This is the approved oracle upper-bound ablation. The target uses the exact GF(2) `n_p≤1` filter and is not a learned Graph Critic result. The oracle provenance was written into the graph, target metadata, QC flags, W&B config, and CSV logs.

## Approved checks

- `tests/test_quasimetric_v08.py -q`: **18 passed**.
- `JAX_PLATFORM_NAME=cpu pixi run -e bench pytest -q third_party/qc/tests/test_graph_targets.py`: **4 passed**.
- No other tests were run for v0.8.3.

## R2: D1 oracle graph and targets

The new output was built with `n_p≤1`, Λ=10.5, the shared window-5 candidate cache, and the v0.8 checkpoint.

| Metric | Result |
|---|---:|
| Retained bridges | 52,351 |
| Reachable states | 99.2854% |
| G-R ρ=0.9 reference reachability | 99.7616% |
| Reachability difference | −0.4762 percentage points |
| h=5 target valid fraction | 99.4716% |
| h=5 loss-mask fraction | 98.9403% |
| Shortest-path bridge count among reachable states (0 / 1 / 2) | 19.3273% / 35.3777% / 45.2951% |
| h=5 regret p90, loss-masked rows | 10.8396 steps |
| Zero-regret fraction, loss-masked rows | 76.9364% |

The target contains one value per transition row. The loss mask excludes both-capped regret rows from QC loss, so its coverage is lower than the target valid fraction.

Artifacts are under `outputs/puzzle/p13-4x4-task2-100-generated/v08/obj_oracle_np1_qc/`:

- Target provenance: `oracle_upper_bound`, `oracle_np_max=1`.
- Graph hash: `50418c09a28f075847ace0f4f7944609510f5d58f18d5eac4361d853f658e1b0`.
- Target content hash: `460b47c669a686c3ff9c90ca0376f39be944293a6f1994d6ad05eeebfafd15b4`.
- Build commit: `5ad306c5e8df50b093587eb0113cb42d4dae0135`, `git_dirty=false`.

## R3: QC results

QC ran P13 task 2 with seed 0, best-of-32, h=5, sparse rewards, 300k offline steps, zero online steps, 50k checkpoint/evaluation intervals, and 50 evaluation episodes. W&B was offline.

| Updates | Success |
|---:|---:|
| 50k | 8% (4/50) |
| 100k | 8% (4/50) |
| 150k | 8% (4/50) |
| 200k | 16% (8/50) |
| 250k | 6% (3/50) |
| 300k | 18% (9/50) |

The final training-batch regret Spearman was **0.5358**. Against the v0.6 G-R ρ=0.9 reference (14% success and 0.676 final Spearman), D1 ended 4 percentage points higher in success and lower in batch rank correlation. The success difference is under the guide's 10-point interpretation threshold; this single-seed, 50-episode run does not establish a performance improvement. The oracle graph gives an upper-bound comparison point, not evidence that the learned event network works.

Checkpoints and logs are at `outputs/puzzle/p13-4x4-task2-100-generated/v08/qc/qc/v083-p13-d1-oracle/puzzle-4x4-play-singletask-task2-v0/sd00020261110_175753/`. All six checkpoints from 50k to 300k were saved. `eval.csv`, `offline_agent.csv`, and `flags.json` contain `graph_target_provenance`; the provenance value is `oracle_upper_bound|oracle_np_max=1` in CSV and the full JSON object in flags.

The QC implementation commit is `2027a60b1ec8c149cf7bd10ede9377bd788c1b16`. The main run checkout was commit `5ad306c5e8df50b093587eb0113cb42d4dae0135`, clean at graph build time. R3b and R3c were not run because they remain approval-gated.

## Repository publication state

The QC commit was created locally on `graph-critic`, but pushing it failed because this environment has no GitHub HTTPS credential and no usable SSH key. The parent `quasi_develop` commits and updated submodule pointer are also local and have not been pushed. Existing staged edits to `docs/results_v0.7.md` and `graphcritic/servo.py` were preserved and were not included in the v0.8.3 commits.
