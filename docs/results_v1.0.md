# Graph Critic v1.0 — execution error fix and S2 audit

Date: 2026-11-11  
Branch: `quasi_develop`

## Commits, tests, and elapsed time

| Item | Result |
|---|---|
| v1.0 implementation commit | `2b7c698` |
| S2 / d⁺ provenance fixes | `e1ef34b`, `7cb2677`, `118e35f` |
| Execution commit | `7cb2677af12ebd714623daeb521c22eb9fbe5d3d` |
| QC submodule commit | `779c721` |
| Root `git_dirty` during final audit | `false` |
| QC submodule `git_dirty` during final audit | `false` |
| Requested tests | `tests/test_v10.py` + `tests/test_quasimetric_v08.py`: **31 passed**, one upstream pytest deprecation warning, 49.59 s |
| Final S2 audit elapsed time | Approximately 94 s |
| Push status | Local commits are ready; `git push` could not authenticate because this shell has no GitHub HTTPS credentials. |

The prescribed S2 command used the existing `v10/dplus/step_020000` checkpoint, v0.8 window-5 candidate cache, and v0.8 reference checkpoint. No training was repeated.

## Previously measured v1.0 values

| Stage / metric | Recorded value |
|---|---:|
| S1 chunk violation; changed / unchanged | 0.178; 0.192 / 0.176 |
| S1 `P(d≤5)` | 0.1995 |
| S1 P2 violation | 0.0785 |
| S1 `λ_chunk` / `λ_match` | 4.22 / 1.41 |
| S1 matched-pair mean distance; reference | 16.18; 19.15 |
| L⁺ accepted / retained bridges | 76,056 / 24,704 |
| L⁺ reachability / impossible-bridge fraction | 0.9784 / 0.792 |
| L⁺ impossible bridge `n_p=0 / 1 / 2 / 3 / ≥4` | 5,066 / 76 / 54 / 178 / 19,330 |
| L⁺ discounted target mean / capped fraction | 66.4 / 2.2% |

## S2 — candidate distance audit

The cache contains 6,385,024 cross-episode pairs. The exact C₀ optimistic rule from `build_judged_graph.py` selected 224,910 pairs (`C₀(s) − C₀(s′) ≥ 10` and finite destination). Press counts use the exact GF(2) minimum-weight solution. `d_θ` is the v0.8 reference checkpoint distance. Fractions are the portion of each row with `d⁺≤20`.

### All cached candidates

| `n_p` | Candidate count | d⁺ p10 | d⁺ p50 | d⁺ p90 | dθ p50 | Fraction `d⁺≤20` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 944,245 | 0.468 | 5.788 | 40.124 | 7.099 | 0.7187 |
| 1 | 43,397 | 3.032 | 35.443 | 89.291 | 26.558 | 0.2811 |
| 2 | 63,105 | 31.570 | 56.595 | 105.035 | 53.201 | 0.0389 |
| 3 | 139,226 | 23.163 | 65.910 | 106.374 | 58.465 | 0.0892 |
| ≥4 | 5,195,051 | 1.012 | 34.488 | 72.564 | 28.985 | 0.3104 |
| ∞ | 0 | — | — | — | — | — |

For combined `n_p≥2`, including `∞`, the `d⁺≤20` fraction is **0.3015**.

### C₀ optimistic candidates only

| `n_p` | Candidate count | d⁺ p10 | d⁺ p50 | d⁺ p90 | dθ p50 | Fraction `d⁺≤20` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24,497 | 0.478 | 6.222 | 40.125 | 7.978 | 0.7129 |
| 1 | 1,604 | 4.684 | 37.113 | 102.648 | 29.167 | 0.2219 |
| 2 | 2,571 | 31.156 | 54.489 | 236.894 | 48.732 | 0.0432 |
| 3 | 5,647 | 27.893 | 68.888 | 366.423 | 58.183 | 0.0784 |
| ≥4 | 190,591 | 1.081 | 34.788 | 74.856 | 29.232 | 0.3026 |
| ∞ | 0 | — | — | — | — | — |

For combined `n_p≥2`, including `∞`, the `d⁺≤20` fraction is **0.2929**.

### S2 gates

| Gate | Measured value | Threshold | Result |
|---|---:|---:|---|
| Held-out chunk violation `d⁺>6.25` | 0.1783 | ≤0.05 | Fail |
| Held-out matched P2 violation `d⁺>dθ+1.25`, both directions | 0.0785 | ≤0.05 | Fail |
| Median order `n_p=0 < n_p=1 < n_p≥2` | 5.788 < 35.443 < 34.823 | Strictly increasing | Fail |
| Combined `n_p≥2` candidates with `d⁺≤20` | 0.3015 | ≤0.01 | Fail |
| S2 overall | 0 / 4 gates passed | All four | **Fail** |

Audit JSON: `outputs/puzzle/standard-3x3-task3/v10/dplus_audit.json`.

## Execution boundary

- The audit stopped at S2 because all four S2 gates failed.
- S3 graph building and board check were not run. S4, S5, and S6 were not run.
- No training, QC, simulator run, or other task was executed.
