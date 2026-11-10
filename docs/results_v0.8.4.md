# Graph Critic v0.8.4 — Puzzle 3×3 task 3 execution results

Date: 2026-11-10  
Branch: `quasi_develop`

## Execution state and elapsed time

| Item | Result |
|---|---|
| Execution commit | `631931bc0237d0aac1f5c01bd3e627eb73c4ef2f` |
| `git_dirty` at execution start | `false` |
| Final working tree before this report commit | clean |
| Existing artifacts | `candidates_window5.npz`, `L_lam10.5`, `O_np1`, and `window5_chunk/step_020000` reused |

| Stage | Elapsed time |
|---|---:|
| S1, S2/S3, L graph, O graph, L QC | Reused; not timed in this execution |
| D0 object label check | ≈33 s |
| D2 event model, 20k | 238.63 s |
| D2 motion model, 20k | 243.63 s |
| D2.5 candidate board check | 49.6 s wall time; embedding 11.11 s; pair distances 4.92 s |
| D3 / D4 | Not run |
| O_np1 QC, 300k with six 50-episode evaluations | 20 min 36 s |

## Previously completed stages and reused values

| Stage/artifact | Recorded values |
|---|---|
| S1 | One-step constraint violation 0.0293; changed-button 0.0101; unchanged-button 0.0328; `P(d≤5)=0.5285`; dead units 0; triangle violations 0 |
| S2/S3 | Parity 100%; candidates 6,385,024; successful destinations 54,873; S3 pass |
| L, `L_lam10.5` | Accepted bridges 96,657; retained 30,209; reachability 0.9823; loss-mask fraction 0.9819; zero-regret fraction 0.860; regret p90 7.32; impossible-bridge fraction 0.858 |
| O, `O_np1` | Accepted bridges 16,585; retained 7,707; reachability 0.8612; loss-mask fraction 0.8589; zero-regret fraction 0.957; regret p90 ≈0; impossible-bridge fraction 0.018 |
| L QC success, 50k–300k | 0.00 / 0.02 / 0.00 / 0.00 / 0.00 / 0.04 |
| L QC final `regret_batch_spearman` | 0.4103991 |

L QC run directory: `outputs/puzzle/standard-3x3-task3/v08/qc/L_lam10.5/qc/v084-3x3t3-L_lam10.5/puzzle-3x3-play-singletask-task3-v0/sd00020261110_202848/`

## D0 — object label check

| Metric | Value |
|---|---:|
| Contact intervals | 30,967 |
| Board changes | 30,897 |
| Relative count error | 0.0022656 |
| Single-toggle intervals | 30,788 / 30,788 (1.0000) |
| Uncovered board changes | 109 / 30,897 (0.003528) |
| `T_I` minimum / p1 / p5 | 26 / 28 / 29 frames |
| Moved fraction | 0.383819 (382,284 / 996,000 chunks) |
| Gate | Pass (`gate_passed=true`) |

## D2 — object quasimetric models

Both models have `step_020000`; `collapse_alarm=false` for both. Dead-unit fraction was 0.0 in each recorded layer.

| Model | Held-out constraint pairs | Violation at cost | Violation beyond cost + ε | Encoder output variance | Projector output variance | Motion-time Spearman |
|---|---:|---:|---:|---:|---:|---:|
| Events | 3,055 | 0.007856 | 0.001309 | 10.995489 | 20,457.711047 | N/A |
| Motion | 99,600 | 0.670914 | 0.283092 | 2.057748 | 0.462489 | 0.874339 |

## D2.5 — event score on cached candidates

Candidate cache: 6,385,024 pairs; pair-cache SHA256 `0c13ccd87c99d3c45d873808e6cd120a39528c71e937db700493eb9cbcf49a2b`.

| Minimum press count `n_p` | Candidate count | `e_φ` p50 |
|---|---:|---:|
| 0 | 944,245 | 0.097399 |
| 1 | 43,397 | 1,090.355347 |
| ≥2 | 5,397,382 | 7.557895 |
| ∞ | 0 | N/A |

`e_φ ≤ 1.5` against `n_p ≤ 1`:

| Actual `n_p` class | `e_φ ≤ 1.5` | `e_φ > 1.5` | Total |
|---|---:|---:|---:|
| `n_p ≤ 1` | 579,645 (TP) | 407,997 (FN) | 987,642 |
| `n_p > 1` | 2,693,157 (FP) | 2,704,225 (TN) | 5,397,382 |

| Gate metric | Value |
|---|---:|
| Median order `eφ(n_p=0) < eφ(n_p=1) < eφ(n_p≥2)` | False |
| False-positive rate for `n_p≥2` at `e_φ≤1.5` | 0.498975 |
| D2.5 gate | Fail (`d2_5_gate_passed=false`) |

## D3/D4

D3 and D4 were not run. D2.5 gate value: `false`.

## QC — O_np1, 300k

Graph target provenance: `oracle_upper_bound`, `oracle_np_max=1`. Evaluation used 50 episodes per checkpoint.

| Target | Success 50k | 100k | 150k | 200k | 250k | 300k | Final `regret_batch_spearman` | Mean wrong buttons at final evaluation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L, `L_lam10.5` | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.04 | 0.4103991 | 4.04 |
| O, `O_np1` | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.22155233 | 4.16 |
| TD QC v0.6 reference | 0.10 | 1.00 | 0.98 | 1.00 | N/A | N/A | N/A | N/A |

Wrong-button counts are the sign-reversed `episode.final_reward` values at each target's last evaluation.

O_np1 QC run directory: `outputs/puzzle/standard-3x3-task3/v08/qc/O_np1/qc/v084-3x3t3-oracle-np1/puzzle-3x3-play-singletask-task3-v0/sd00020261110_212353/`

## Execution boundary

- D branch stopped after D2.5; D3 and D4 were not generated or evaluated.
- O_np1 QC completed at 300,000 offline steps; no D_learned QC was run.
- No TD QC, BC run, 250k-data run, or simulator diagnostic was run.
