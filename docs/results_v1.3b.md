# Graph Critic v1.3b — QC-FQL versus QC-FQL with graph critic

Date: 2026-11-13  
Branch: `quasi_develop`; QC fork branch: `graph-critic`

## Commits and execution state

| Item | Value |
|---|---|
| Root execution commit | `3cc634d0db50ef02eb3bed5a8819c870deca22f6` |
| QC fork execution commit | `f12e624ff62a23c77366e751832361fe8a017f79` |
| `git_dirty` at S1, root / QC | `false` / `false` |
| `git_dirty` at S2, root / QC | `false` / `false` |
| Code changes | None |
| Tests | Not run (no code changes; not requested) |
| S1 elapsed time | 23 min 19 s |
| S2 elapsed time | 22 min 02 s |

Both runs used the existing `v12/Lsame` graph and target files. Graph content hash: `3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa`; target configuration: `kind=discounted`, `h=5`, `gamma=0.99`. No graph or target was rebuilt.

## S1/S2 — QC-FQL results

Each evaluation used 50 episodes. Wrong-button count is the mean `episode.final_wrong_buttons` from the final evaluation. Actor losses are the last `offline_agent.csv` row at 300k. The Q loss term is derived from the logged actor-loss identity:

`q_loss = actor_loss - bc_flow_loss - 100 × distill_loss`.

| Method | Success 50k | 100k | 150k | 200k | 250k | 300k | Final mean wrong buttons | Final critic loss | Final graph-target Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QC-FQL, TD critic | 0.00 | 0.02 | 0.08 | 0.12 | 0.08 | 0.04 | 4.08 | 0.97235376 | N/A |
| QC-FQL, discounted graph critic | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 5.44 | 3.008918 | 0.9751438 |

| Method | Final `bc_flow_loss` | Final `distill_loss` | Final Q loss term |
|---|---:|---:|---:|
| QC-FQL, TD critic | 0.1728703 | 0.013067648 | 48.8119599 |
| QC-FQL, discounted graph critic | 0.17294136 | 0.17164434 | 56.44942464 |

## Reference success rates

| Reference | 50k | 100k | 150k | 200k | 250k | 300k |
|---|---:|---:|---:|---:|---:|---:|
| FQL TD critic, v1.3 | 0.00 | 0.00 | 0.38 | 0.66 | 0.86 | 0.94 |
| FQL graph critic, v1.3 | 0.16 | 0.42 | 0.38 | 0.54 | 0.60 | 0.52 |
| QC + graph critic, best-of-N, v1.2b | 0.44 | 0.78 | 0.96 | 1.00 | 0.98 | 0.94 |
| QC TD, v0.6 | 0.10 | 1.00 | 0.98 | 1.00 | — | — |

## Artifact paths

- QC-FQL TD run: `outputs/puzzle/standard-3x3-task3/v13/qc/qcfql_td/qc/v13b-3x3t3-qcfql-td/puzzle-3x3-play-singletask-task3-v0/sd00020261113_013218/`
- QC-FQL graph-critic run: `outputs/puzzle/standard-3x3-task3/v13/qc/qcfql_graph/qc/v13b-3x3t3-qcfql-graph/puzzle-3x3-play-singletask-task3-v0/sd00020261113_015603/`
- Reused graph and targets: `outputs/puzzle/standard-3x3-task3/v12/Lsame/`

## Push status

Pending.
