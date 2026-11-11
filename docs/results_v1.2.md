# Graph Critic v1.2 — execution results

Date: 2026-11-12  
Branch: `quasi_develop`; QC fork branch: `graph-critic`

## Commits, tests, and execution state

| Item | Value |
|---|---|
| v1.2 implementation/runbook commit | `41d76367299867f7a52db63f13b57dc73bf79c51` |
| Sweep import fix commit | `316ddbc2e2bc3c954f673a586def0f54a14080b0` |
| Action-head endpoint lookup fix commit | `a05aaaf324537c5dd06a265e9406104d21bbc3f4` |
| QC fork commit | `af33db55b9440927181c4a25e1b6d2bca08560f2` |
| Tests | `test_v12.py`, `test_v11.py`, `test_v10.py`, `test_quasimetric_v08.py`: **43 passed**, one upstream deprecation warning |
| `git_dirty` for S1/S2 | `false`, execution commit `316ddbc2e2bc3c954f673a586def0f54a14080b0` |
| `git_dirty` for S3 | `false`, execution commit `a05aaaf324537c5dd06a265e9406104d21bbc3f4` |

| Stage | Elapsed time |
|---|---:|
| S1 same-object sweep | 112.60 s |
| S1 any-object sweep | 112.46 s |
| S2 `Lsame` graph and target build | 59.14 s internal; 61 s wall |
| S2 board audit | 14 s wall |
| S3 seed 0 action head, 20k updates and held-out evaluation | 884.46 s |

The first S1 invocation stopped before processing data because the new CLI did
not add the repository root to `sys.path`. The import fix was committed before
the successful S1 runs. The first S3 invocation stopped at the first update
because endpoint row IDs were passed as state vectors; the row lookup fix was
committed before the successful seed-0 run.

## S1 — bridge-budget sweep

`C` quantiles are over reachable states and are in steps. Gate A threshold was
`R_sat >= 0.90`.

| `b` | Same `R(b)` | Same C p50 | Same C p90 | Any `R(b)` | Any C p50 | Any C p90 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.029844 | 222.000 | 618.000 | 0.029844 | 222.000 | 618.000 |
| 1 | 0.945805 | 268.906 | 523.419 | 0.968517 | 225.542 | 452.650 |
| 2 | 0.945805 | 227.697 | 411.485 | 0.968517 | 181.153 | 336.353 |
| 3 | 0.945805 | 221.929 | 397.999 | 0.968517 | 174.785 | 315.700 |
| 4 | 0.945805 | 221.466 | 394.783 | 0.968517 | 173.896 | 312.272 |
| 6 | 0.945805 | 221.457 | 393.428 | 0.968517 | 173.853 | 310.582 |
| 8 | 0.945805 | 221.457 | 393.428 | 0.968517 | 173.853 | 310.581 |
| 12 | 0.945805 | 221.457 | 393.428 | 0.968517 | 173.853 | 310.581 |
| 16 | 0.945805 | 221.457 | 393.428 | 0.968517 | 173.853 | 310.581 |
| 32 | 0.945805 | 221.457 | 393.428 | 0.968517 | 173.853 | 310.581 |

| Sweep | `R_sat` | `B*` | Gate A |
|---|---:|---:|---|
| Same object | 0.945805 | 1 | Pass |
| Any object | 0.968517 | 1 | Comparison only |

## S2 — `Lsame` graph and board audit

| Metric | Value |
|---|---:|
| Candidate cache pairs before same-object filter | 12,770,048 |
| Same-object candidates | 6,385,024 |
| C0-optimistic candidates | 165,286 |
| Accepted bridges | 85,275 |
| Retained bridges at `B*=1` | 16,561 |
| Graph hash | `3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa` |
| Reachability | 0.945805 |
| Discounted target valid fraction | 0.996 |
| Discounted target loss-mask fraction | 0.996 |
| Bridge count 0 / 1 | 14,207 / 932,544 reachable states |
| Unreachable states | 54,570 |
| Retained bridge `n_p=0 / 1 / 2 / 3 / ≥4 / ∞` | 16,561 / 0 / 0 / 0 / 0 / 0 |
| Retained bridges with `n_p >= 1` | 0 |
| Board-audit impossible bridges | 0 / 16,561 (0.000000) |
| Discounted target mean / p50 / p90 / max | 88.3188 / 94.3721 / 99.7792 / 100.0000 |
| Capped loss-mask fraction | 0.054942 |
| Gate B | Pass |

Reference reachability: L `0.982`, O `0.861`, v1.1 Lobj `0.969`, G-R `ρ=0.9` `0.997`.

## S3 — action heads and offline gate

Only seed 0 ran. Seeds 1–3 were not started after seed 0 failed Gate C.

| Seed | Step | C1 violation | C2 violation | F3/collapse | Dead units | `d` median k=5 | k=10 | k=15 | k=20 |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 20,000 | 0.5878 | 0.9998 | No | 0.0000 | 6.3689 | 10.7685 | 15.2280 | 19.7092 |
| 1–3 | — | — | — | Not run | — | — | — | — | — |

| Gate C item | Required | Result |
|---|---:|---:|
| Seed 0 held-out C1 violation | ≤ 0.05 | 0.5878 — Fail |
| Seed 0 collapse alarm | None | None |

The run stopped at Gate C. `write_direct_q_config.py`, offline G1/G2/G3,
direct-Q QC, discounted-cost QC, and BC evaluation were not run. No QC results
are available for v1.2. The v1.2 QC comparison rows are therefore:

| Method | Success at 50k, 100k, 150k, 200k, 250k, 300k | Final mean wrong buttons |
|---|---|---:|
| `Lsame` direct Q | Not run | Not run |
| `Lsame` discounted cost critic | Not run | Not run |
| BC | Not run | Not run |
| O cost critic reference | 0.04 / 0.24 / 0.26 / 0.16 / 0.28 / 0.30 | 2.98 |
| L cost critic reference | 0.00 / 0.06 / 0.02 / 0.04 / 0.02 / 0.06 | — |
| TD QC v0.6 reference | 0.10 / 1.00 / 0.98 / 1.00 / — / — | — |

## Artifact paths

- Sweeps: `outputs/puzzle/standard-3x3-task3/v12/sweep_same.json`, `sweep_any.json`
- `Lsame` graph and targets: `outputs/puzzle/standard-3x3-task3/v12/Lsame/`
- Board audit: `outputs/puzzle/standard-3x3-task3/v12/Lsame_board_check.json`
- Seed-0 action head and held-out evaluation: `outputs/puzzle/standard-3x3-task3/v12/action_heads/`
- QC runs: not created

## Push status

Push attempts failed for both branches because this environment has no
interactive GitHub HTTPS credentials (`fatal: could not read Username for
'https://github.com': No such device or address`). The commits remain local:
`quasi_develop` and the QC fork's `graph-critic` branch are ahead of their
respective origins.
