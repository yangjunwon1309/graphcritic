# Graph Critic v0.8.4b — cost target QC results

Date: 2026-11-10  
Branch: `quasi_develop`

## Commits, tests, and elapsed time

| Item | Result |
|---|---|
| Implementation commit | `199314afa34bc3d39ecad1a11b8687ee3716cd6d` |
| Execution commit | `199314afa34bc3d39ecad1a11b8687ee3716cd6d` |
| `git_dirty` during graph builds | `false` for L and O |
| `--target-kind` default | `regret` |
| Allowed tests, final run | `tests/test_oracle_targets_v083.py` + `tests/test_quasimetric_v08.py`: 26 passed in 41.73 s |
| Allowed tests, initial run | 25 passed, 1 failed in 41.90 s; the existing regret test fixture was updated with `kind="regret"` |

| Stage | Elapsed time |
|---|---:|
| L_cost graph and target build | 60.14 s |
| O_np1_cost graph and target build | 59.60 s |
| L_cost QC, 300k and six 50-episode evaluations | 20 min 45 s |
| O_np1_cost QC, 300k and six 50-episode evaluations | 20 min 21 s |

## Graph hash gate

| Graph | Expected hash | Generated hash | Match |
|---|---|---|---|
| L_lam10.5_cost | `67f003725a1095f9b3bdf1e863ecc8382852f63b8d884a1d669ee83c1b2ddcfd` | `67f003725a1095f9b3bdf1e863ecc8382852f63b8d884a1d669ee83c1b2ddcfd` | Yes |
| O_np1_cost | `bb5b26bf1776785d7196745d97ee1b98e177feeb7112c69ef6344c9c3b3548f9` | `bb5b26bf1776785d7196745d97ee1b98e177feeb7112c69ef6344c9c3b3548f9` | Yes |

Both `result.json` files record `target_kind=cost`, `git_commit=199314afa34bc3d39ecad1a11b8687ee3716cd6d`, and `git_dirty=false`. Both `targets.json` files record `config.kind=cost`.

## Cost target summaries

Fractions and cost statistics use the valid, loss-masked target rows. Capped fraction is the fraction of loss-mask rows with `source == 2`.

| Target artifact | Valid fraction | Loss-mask fraction | Capped fraction | Cost mean | p50 | p90 | Max | Target content hash |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| L_lam10.5_cost | 0.996 | 0.996 | 0.0178153 | 114.430249 | 95.545090 | 164.382483 | 1,006 | `b8cb7a1b21b1f17b3f97b9be3e4dd15d77428fa530b9653f6d35881f31b8e883` |
| O_np1_cost | 0.996 | 0.996 | 0.1423976 | 435.763845 | 352.057831 | 1,006 | 1,006 | `86ad73777c5c091b5194a841b4af64b9e527fea126e2de97e290b7211d35fb62` |

## QC results

Each checkpoint evaluation used 50 episodes. Wrong-button count is the sign-reversed final `episode.final_reward`.

| Target | Success 50k | 100k | 150k | 200k | 250k | 300k | Final graph-target Spearman | Final critic loss | Mean wrong buttons, final eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L_cost | 0.00 | 0.06 | 0.02 | 0.04 | 0.02 | 0.06 | 0.8900673 | 610.52856 | 2.96 |
| O_np1_cost | 0.04 | 0.24 | 0.26 | 0.16 | 0.28 | 0.30 | 0.9795043 | 2,978.6704 | 2.98 |
| L regret, v0.8.4 reference | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.04 | 0.410 | N/A | 4.04 |
| O regret, v0.8.4 reference | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.222 | N/A | 4.16 |
| TD QC, v0.6 reference | 0.10 | 1.00 | 0.98 | 1.00 | N/A | N/A | N/A | N/A | N/A |

L_cost run directory: `outputs/puzzle/standard-3x3-task3/v08/qc/L_lam10.5_cost/qc/v084b-3x3t3-L-cost/puzzle-3x3-play-singletask-task3-v0/sd00020261110_220336/`

O_np1_cost run directory: `outputs/puzzle/standard-3x3-task3/v08/qc/O_np1_cost/qc/v084b-3x3t3-oracle-np1-cost/puzzle-3x3-play-singletask-task3-v0/sd00020261110_222508/`

## Artifact paths

- L graph and targets: `outputs/puzzle/standard-3x3-task3/v08/L_lam10.5_cost/`
- O graph and targets: `outputs/puzzle/standard-3x3-task3/v08/O_np1_cost/`
- L QC: `outputs/puzzle/standard-3x3-task3/v08/qc/L_lam10.5_cost/`
- O QC: `outputs/puzzle/standard-3x3-task3/v08/qc/O_np1_cost/`

## Execution boundary

- Graph builds used the existing v0.8.4 L/O graph inputs, `window5_chunk/step_020000`, and `candidates_window5.npz`.
- No additional target/graph build, D, TD, BC, alternate-lambda, or other-task experiment was executed.
