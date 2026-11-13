# Outputs cleanup and migration report

## 0. Inventory

Inventory written to `outputs/_inventory.csv` with 191 run/artifact rows. `st_birthtime` is unavailable on this filesystem, so `created` is the earliest file mtime proxy. Initial `outputs/` size was 77.98 GiB; the retained unique footprint is 24.23 GiB and the delete-pending footprint is 53.77 GiB. The detailed dry-run move plan is `outputs/_plan.md`.

## 1. Run selection

Success curves were read from the run `eval.csv` files and compared after rounding to two decimal places. The most recent completed exact match was selected where duplicates existed.

- **A1_qc_td:** not found. No run was inferred.
- **A2_qc_graph:** `puzzle/standard-3x3-task3/v12/qc/Lsame_discounted/qc/v12-3x3t3-Lsame-discounted/puzzle-3x3-play-singletask-task3-v0/sd00020261112_172749`.
- **A3_dqc_td:** `puzzle/standard-3x3-task3/v16/qc/td/dqc/v16-3x3t3-dqc-td/sd00020261113_173502`.
- **A4_dqc_graph_distill:** `puzzle/standard-3x3-task3/v16/qc/graph/dqc/v16-3x3t3-dqc-graph/sd00020261113_180457`.
- **A5_fql_td:** `puzzle/standard-3x3-task3/v13/qc/fql_td/qc/v13-3x3t3-fql-td/puzzle-3x3-play-singletask-task3-v0/sd00020261112_214832`.
- **A6_fql_graph:** `puzzle/standard-3x3-task3/v13/qc/fql_graph/qc/v13-3x3t3-fql-graph/puzzle-3x3-play-singletask-task3-v0/sd00020261112_220927`.
- **A7_qcfql_td:** `puzzle/standard-3x3-task3/v13/qc/qcfql_td/qc/v13b-3x3t3-qcfql-td/puzzle-3x3-play-singletask-task3-v0/sd00020261113_013218`.
- **A8_qcfql_graph:** `puzzle/standard-3x3-task3/v13/qc/qcfql_graph/qc/v13b-3x3t3-qcfql-graph/puzzle-3x3-play-singletask-task3-v0/sd00020261113_015603`.
- **A9_ablation_L_cost:** `puzzle/standard-3x3-task3/v08/qc/L_lam10.5_cost/qc/v084b-3x3t3-L-cost/puzzle-3x3-play-singletask-task3-v0/sd00020261110_220336`.
- **A10_ablation_O_cost:** `puzzle/standard-3x3-task3/v08/qc/O_np1_cost/qc/v084b-3x3t3-oracle-np1-cost/puzzle-3x3-play-singletask-task3-v0/sd00020261110_222508`.
- **A11_bc:** not found. No run was inferred.
- **B1_qc_td:** `puzzle/p13-4x4-task2-100-generated/v14/qc/td/qc/v14-4x4p13-td/puzzle-4x4-play-singletask-task2-v0/sd00020261112_231151`.
- **B2_qc_graph:** `puzzle/p13-4x4-task2-100-generated/v14/qc/Lsame_discounted/qc/v14-4x4p13-Lsame-discounted/puzzle-4x4-play-singletask-task2-v0/sd00020261112_225028`.
- **B3_dqc_td:** `puzzle/p13-4x4-task2-100-generated/v16/qc/td/dqc/v16-4x4p13-dqc-td/sd00020261113_183325`.
- **B4_dqc_graph_distill:** `puzzle/p13-4x4-task2-100-generated/v16/qc/graph/dqc/v16-4x4p13-dqc-graph/sd00020261113_193502`.
- **B5_fql_td:** `puzzle/p13-4x4-task2-100-generated/v14b/qc/fql_td/qc/v14b-4x4p13-fql-td/puzzle-4x4-play-singletask-task2-v0/sd00020261113_154202`.
- **B6_fql_graph:** `puzzle/p13-4x4-task2-100-generated/v14b/qc/fql_graph/qc/v14b-4x4p13-fql-graph/puzzle-4x4-play-singletask-task2-v0/sd00020261113_160500`.

A11 note: the `eval_bc.csv` paired with a direct-Q run is not a standalone BC run and was not counted as A11.

## 2. Dependency closure

Required 3×3 and P13 graph, target, dθ, candidate, dataset, and sweep inputs were present and retained. The complete C1 `cube-double` stats directory was retained. DQC offline W&B directories were moved into their corresponding retained run directories; the incomplete first R4 attempt and its W&B folder are pending review.

## 3–4. Plan and moves

The dry-run plan is `outputs/_plan.md`. Moves used `mv` on the same filesystem. Retained run folders moved intact. JSON path references were rewritten from the relocation map; graph hashes and dataset IDs were preserved. Unselected outputs were moved to `_delete_pending` without deletion.

## 5. Verification

Verification records are written below from the completed read-only checks.
### Verification results

- Repository cleanliness at apply/verification: DQC `false` at `39d41719a344928f0ba8890afd02975d569c5169` (graph-critic), FQL `false` at `e8cd16eb490332924dfa2492097219f181765933` (detached), QC `false` at `f12e624ff62a23c77366e751832361fe8a017f79` (graph-critic), root `false` at `4e722f4f7b3291391b4cfbd9fc11b596ab786c0e` (quasi_develop).
- Retained graph/target pairs loaded: 12; all stored graph hashes, dataset IDs, and target graph links validated.
- Checkpoint restores: QC 11/11; DQC 4/4.
- INDEX success curves matched relocated eval CSVs: 15/15; all selected curves matched.


## 6. Delete-pending

Pending tree: `outputs/_delete_pending/`; measured bytes: 53.77 GiB. The data remains recoverable. After reviewing the pending list, the requested deletion command is `rm -rf /home/url/graphcritic/outputs/_delete_pending`.


### Delete-pending item list

| Original path | Size | Classification |
|---|---:|---|
| `data` | 330.93 MiB | other intermediate |
| `graphs` | 671.38 MiB | other intermediate |
| `lift-v02` | 773.00 B | other intermediate |
| `puzzle/inspect_puzzle.json` | 16.33 KiB | other intermediate |
| `puzzle/inspect_v06.json` | 17.22 KiB | other intermediate |
| `puzzle/p0-qc-task3-restart-2.log` | 172.73 KiB | other intermediate |
| `puzzle/p0-qc-task3-save50k.log` | 1.57 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/gr-rho1` | 1.30 GiB | regret |
| `puzzle/p13-4x4-task2-100-generated/qc-g0-regret.log` | 554.32 KiB | regret |
| `puzzle/p13-4x4-task2-100-generated/qc-gr-rho09-regret.log` | 552.22 KiB | regret |
| `puzzle/p13-4x4-task2-100-generated/qc-gr-rho1-regret.log` | 558.29 KiB | regret |
| `puzzle/p13-4x4-task2-100-generated/v07` | 3.50 GiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/board_check.json` | 16.20 KiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/board_check_v082-event-embeddings-d63bfec66d11a7f7.npy` | 261.95 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/board_check_v082-event-embeddings-d63bfec66d11a7f7.npy.json` | 554.00 B | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/board_check_v082.json` | 27.73 KiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/candidates_window5.json` | 1.36 KiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/candidates_window5.npz` | 20.93 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/check/a2_fix123` | 72.81 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/check/window5_chunk/data.json` | 15.08 KiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/check/window5_chunk/embedding_cache` | 1.80 GiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/check/window5_chunk/step_010000` | 74.96 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/check/window5_chunk/train_metrics.jsonl` | 15.68 KiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/check/window5_chunk/training.jsonl` | 10.38 KiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/e_phi_check_v082b-event-embeddings-22293a768d7314cb.npy` | 455.78 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/e_phi_check_v082b-event-embeddings-22293a768d7314cb.npy.json` | 554.00 B | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/e_phi_check_v082b.json` | 2.06 KiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/e_phi_check_v082b.stdout.log` | 1.82 KiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/embeddings` | 2.69 GiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/obj_learned` | 484.09 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/obj_oracle_np1` | 456.59 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/obj_oracle_np1_qc` | 464.22 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/object2_events` | 10.73 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/object2_motion` | 10.73 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/object_events` | 10.70 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/object_labels_check_v082b.json` | 3.58 KiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/object_motion` | 10.70 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/qc` | 350.76 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/quasimetric` | 761.31 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/quasimetric_lam0` | 470.68 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/quasimetric_lam10.5` | 468.75 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v08/shared` | 86.99 MiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v14/Lsame_board_check.json` | 3.29 KiB | other intermediate |
| `puzzle/p13-4x4-task2-100-generated/v16/qc/graph/dqc/v16-4x4p13-dqc-graph/sd00020261113_190626` | 1.77 GiB | R4 incomplete |
| `puzzle/p13-4x4-task2-100-generated/witness` | 1.65 MiB | other intermediate |
| `puzzle/p13-4x4-task5-100-generated` | 4.09 GiB | other intermediate |
| `puzzle/p13-mix-smoke` | 26.01 MiB | other intermediate |
| `puzzle/p13-resume-interrupt` | 27.44 MiB | other intermediate |
| `puzzle/p13-resume-smoke` | 25.51 MiB | other intermediate |
| `puzzle/p13-smoke` | 26.53 MiB | other intermediate |
| `puzzle/p13-smoke-v2` | 26.54 MiB | other intermediate |
| `puzzle/standard-3x3-task3/calibration.json` | 4.05 KiB | other intermediate |
| `puzzle/standard-3x3-task3/g0-cost` | 922.94 MiB | other intermediate |
| `puzzle/standard-3x3-task3/gr-rho1` | 528.72 MiB | regret |
| `puzzle/standard-3x3-task3/v08/L_lam10.5` | 334.21 MiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/O_np1` | 331.29 MiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/board_check_v084.json` | 16.07 KiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/check/window5_chunk/data.json` | 11.89 KiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/check/window5_chunk/embedding_cache` | 1.54 GiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/check/window5_chunk/step_010000` | 74.42 MiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/check/window5_chunk/train_metrics.jsonl` | 15.70 KiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/check/window5_chunk/training.jsonl` | 10.38 KiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/e_phi_check_v084-event-embeddings-61ff34156c8f526d.npy` | 394.27 MiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/e_phi_check_v084-event-embeddings-61ff34156c8f526d.npy.json` | 500.00 B | other intermediate |
| `puzzle/standard-3x3-task3/v08/e_phi_check_v084.json` | 2.04 KiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/embeddings/embeddings-75eef2e22e0130f8-d8e00c609a016926.npy` | 1.54 GiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/embeddings/embeddings-75eef2e22e0130f8-d8e00c609a016926.npy.json` | 501.00 B | other intermediate |
| `puzzle/standard-3x3-task3/v08/object_events` | 9.58 MiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/object_labels_check_v084.json` | 2.80 KiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/object_motion` | 9.58 MiB | other intermediate |
| `puzzle/standard-3x3-task3/v08/qc/L_lam10.5` | 344.85 MiB | regret |
| `puzzle/standard-3x3-task3/v08/qc/O_np1` | 344.85 MiB | regret |
| `puzzle/standard-3x3-task3/v09` | 0.00 B | other intermediate |
| `puzzle/standard-3x3-task3/v10` | 5.09 GiB | v1.0 d+ |
| `puzzle/standard-3x3-task3/v11/Lobj` | 334.65 MiB | v1.1 Lobj |
| `puzzle/standard-3x3-task3/v11/Lobj_board_check.json` | 8.19 KiB | v1.1 Lobj |
| `puzzle/standard-3x3-task3/v11/embeddings-9adaf65cb96691db-426289f1c15e0d43.npy` | 1.54 GiB | v1.1 Lobj |
| `puzzle/standard-3x3-task3/v11/embeddings-9adaf65cb96691db-426289f1c15e0d43.npy.json` | 501.00 B | v1.1 Lobj |
| `puzzle/standard-3x3-task3/v12/Lsame_board_check.json` | 3.27 KiB | other intermediate |
| `puzzle/standard-3x3-task3/v12/action_heads` | 37.42 MiB | direct Q / action head |
| `puzzle/standard-3x3-task3/v12/action_heads_res` | 149.68 MiB | direct Q / action head |
| `puzzle/standard-3x3-task3/v12/direct_q_Lsame_res.json` | 1.66 KiB | direct Q / action head |
| `puzzle/standard-3x3-task3/v12/offline_gate_res.json` | 1.27 KiB | other intermediate |
| `puzzle/standard-3x3-task3/v12/qc/Lsame_direct_res` | 344.85 MiB | other intermediate |
| `puzzle/standard-3x3-task3/v13c` | 621.91 MiB | FQL a1000 |
| `puzzle/standard-3x3-task3/v16/speed` | 897.24 MiB | other intermediate |
| `puzzle/standard-3x3-task3/witness-free.json` | 37.62 KiB | other intermediate |
| `puzzle/standard-3x3-task4` | 1.93 GiB | other intermediate |
| `puzzle/standard-4x4-task2` | 3.20 GiB | other intermediate |
| `puzzle/standard-4x4-task5` | 2.49 GiB | other intermediate |
| `qc` | 170.85 MiB | other intermediate |
| `scene-v02` | 775.97 MiB | other intermediate |
| `scene-v03` | 4.69 GiB | other intermediate |
| `scene-v04` | 738.51 MiB | other intermediate |
| `scene-v05` | 3.93 GiB | other intermediate |
| `tau` | 6.53 MiB | other intermediate |
| `v08` | 721.00 B | other intermediate |
| `/tmp/tmp082_5i0o/wandb/offline-run-20261113_190626-2uhfrpe9` | 1.62 MiB | R4 incomplete W&B offline data |

### Hard-coded output path references

- `/home/url/graphcritic/docs/RUN_v0.8.md:32:D=$ROOT/outputs/puzzle/p13-4x4-task2-100-generated`
- `/home/url/graphcritic/docs/RUN_v0.8.md:341:`outputs/puzzle/standard-3x3-task3/{g0,gr-rho09}`. Their graph metadata records`
- `/home/url/graphcritic/docs/RUN_v0.8.md:366:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.2.md:20:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.2.md:42:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.2.md:66:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.2.md:92:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.2.md:114:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.3c.md:25:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.3c.md:49:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.3c.md:76:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.3c.md:100:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.4b.md:13:D=$ROOT/outputs/puzzle/p13-4x4-task2-100-generated`
- `/home/url/graphcritic/docs/RUN_v1.3.md:33:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.3.md:65:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.3.md:90:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.6.md:19:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.6.md:20:D4=$ROOT/outputs/puzzle/p13-4x4-task2-100-generated`
- `/home/url/graphcritic/docs/RUN_v1.0.md:23:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.1.md:24:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.1.md:49:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.1.md:65:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.1.md:83:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.1.md:102:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.1.md:126:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.1.md:153:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v1.1.md:176:D3=$ROOT/outputs/puzzle/standard-3x3-task3`
- `/home/url/graphcritic/docs/RUN_v0.9.md:22:D3=$ROOT/outputs/puzzle/standard-3x3-task3`

## Push 명령

- QC fork (`graph-critic`): 변경 없음.
- FQL fork: 변경 없음.
- DQC fork (`graph-critic`): 변경 없음.
- Root (`quasi_develop`) commits pending push:
  - `5cafb87 Add safe outputs reorganization workflow`
  - `4e722f4 Fix migration report string formatting`
  - `dd737ee Align graph targets for DQC checkpoint restore`
  - `9ae8593 Fix DQC success range in outputs index`
  - The cleanup report and results index are committed with this report; the command below prints their commit hash and title too.

```bash
cd /home/url/graphcritic && git status -sb && git log --oneline origin/quasi_develop..HEAD && git push origin quasi_develop
```
