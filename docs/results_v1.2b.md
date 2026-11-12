# Graph Critic v1.2b — residual action head and direct-Q results

Date: 2026-11-12  
Branch: `quasi_develop`; QC fork branch: `graph-critic`

## Commits, tests, and execution state

| Item | Value |
|---|---|
| Residual action-head implementation | `ef80a10d2fde7e57ef87c194c9ede9d50cd7d50f` |
| QC direct-Q optimizer fix | `b0ff1be5552bf2a98dce3d98624264882b1fd383` |
| Root commit pinning QC fix | `3d539149dee8d755f78cbb79091df17a81508f06` |
| Execution commit for S3/S4 | `3d539149dee8d755f78cbb79091df17a81508f06` |
| `git_dirty` at successful experiment starts | `false` |
| Allowed tests | `tests/test_v12b.py tests/test_v12.py tests/test_v11.py`: **16 passed**, one upstream deprecation warning (11.44 s) |

| Stage | Elapsed time |
|---|---:|
| S1, four residual action heads, 20k each | 2,434.82 s |
| S2, config generation and 20k offline gate | 147 s wall; evaluator reports 130.25 s |
| S3, first attempt stopped at update 1 | 57 s |
| S3, direct-Q plus BC, 300k | 2,065 s |
| S4, discounted-cost QC, 300k | 1,142 s |

The first S3 attempt failed during optimizer initialization because the graph-direct parameter tree used a `FrozenDict` while the optimizer expected a plain dictionary. The QC fork now normalizes the parameter tree before optimizer initialization. The failed attempt produced no completed checkpoint; S3 was rerun successfully from step 0 after committing the fix. S4 then completed normally.

## S1 — residual action heads

All heads were evaluated on 20,000 held-out chunks. The reference (v_{ref}) violation rate was 0.02870 for each seed. C2 maximum was 0, the collapse alarm was false, and maximum dead-unit fraction was 0 for every seed; all four passed Gate C′.

| Seed | C1 violation | C2 max | d median k=5 / 10 / 15 / 20 | softplus(h) mean / p90 | Alt-action distance greater / median increase | λ₁ |
|---:|---:|---:|---|---|---|---:|
| 0 | 0.03075 | 0 | 5.0172 / 9.7641 / 14.4980 / 19.1721 | 0.008677 / 0.000296 | 0.8777 / 1.1328 | 21.6900 |
| 1 | 0.03275 | 0 | 5.0410 / 9.7847 / 14.5154 / 19.1878 | 0.009050 / 0.000426 | 0.8710 / 1.2616 | 21.3812 |
| 2 | 0.03240 | 0 | 5.0254 / 9.7722 / 14.5032 / 19.1728 | 0.009145 / 0.000468 | 0.8738 / 1.2525 | 20.5297 |
| 3 | 0.03195 | 0 | 5.0406 / 9.7830 / 14.5123 / 19.1901 | 0.009315 / 0.000287 | 0.8758 / 1.1699 | 22.3786 |
| v1.2 free reference, seed 0 | 0.5878 | C2 violation 0.9998 | 6.37 / 10.77 / 15.23 / 19.71 | — | — | — |

## S2 — direct-Q offline gate

The Lsame graph hash remained `3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa`.

| Gate / diagnostic | Result |
|---|---:|
| G1: ensemble d median to (x_{t+5}), n=18,781 | 5.0893 |
| G1: fraction with d > 6.25 | 0.03711; pass |
| G2: median |Q − (5 + C(xₜ₊₅))|, n=18,748 | 79.3980; fail |
| G2: Q > y+5 / Q < y−5 | 0.06501 / 0.82211 |
| G3: alternative action distance greater fraction / median increase | 0.90086 / 0.53450 |
| Fallbacks | 33 / 18,781 = 0.001757; all `no_route_within_K` |
| Mean neighborhood size |N(s)| | 112.3007 |
| Q versus y Spearman | 0.57790 |

At the final S3 evaluation (300k), direct-Q diagnostics were 15 fallbacks (fraction 0.003; `no_route_within_K`), mean neighborhood size 115.039, mean candidate-Q standard deviation 188.08, and mean decision time 0.01181 s.

## S3/S4 — QC results

Each checkpoint evaluation used 50 episodes. Wrong-button count is the sign-reversed `episode.final_reward`.

| Method | Success 50k | 100k | 150k | 200k | 250k | 300k | Final mean wrong buttons | Final critic loss | Final graph-target Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lsame direct-Q, residual heads | 0.02 | 0.02 | 0.04 | 0.02 | 0.10 | 0.00 | 4.26 | 0.0 | N/A |
| BC, paired with direct-Q run | 0.00 | 0.00 | 0.02 | 0.00 | 0.02 | 0.02 | 4.38 | N/A | N/A |
| Lsame discounted cost critic | 0.44 | 0.78 | 0.96 | 1.00 | 0.98 | 0.94 | 0.24 | 3.008918 | 0.975144 |
| O cost critic reference, v0.8.4b | 0.04 | 0.24 | 0.26 | 0.16 | 0.28 | 0.30 | 2.98 | — | — |
| L cost critic reference, v0.8.4b | 0.00 | 0.06 | 0.02 | 0.04 | 0.02 | 0.06 | — | — | — |
| TD QC reference, v0.6 | 0.10 | 1.00 | 0.98 | 1.00 | — | — | — | — | — |

For the discounted cost critic, final graph-target Spearman and critic loss are the 300k `offline_agent.csv` values. For the direct-Q run, critic loss is 0 because no learned critic is used; graph-target Spearman is not emitted for this mode.

## Artifact paths

- Residual action heads: `outputs/puzzle/standard-3x3-task3/v12/action_heads_res/`
- Direct-Q config and offline gate: `outputs/puzzle/standard-3x3-task3/v12/direct_q_Lsame_res.json`, `outputs/puzzle/standard-3x3-task3/v12/offline_gate_res.json`
- Direct-Q and BC run: `outputs/puzzle/standard-3x3-task3/v12/qc/Lsame_direct_res/qc/v12b-3x3t3-Lsame-direct/puzzle-3x3-play-singletask-task3-v0/sd00020261112_165256/`
- Discounted-cost QC run: `outputs/puzzle/standard-3x3-task3/v12/qc/Lsame_discounted/qc/v12-3x3t3-Lsame-discounted/puzzle-3x3-play-singletask-task3-v0/sd00020261112_172749/`

## Push status

Report commit: `c24e71b` (updated below with the push outcome). Pushes to both configured remotes were attempted and failed because this environment has no interactive GitHub HTTPS credentials (`fatal: could not read Username for 'https://github.com': No such device or address`). The root and QC commits remain local; both worktrees are clean.
