# Graph Critic v1.0 — matched-object constraint results

Date: 2026-11-11  
Branch: `quasi_develop`

This file is the v1.0 S1–S6 result record. The implementation and runbook have been prepared and the S0 code checks passed. No v1.0 training, graph build, offline gate, or QC benchmark run has been performed in this work.

## Run provenance and checks

| Item | Result |
|---|---|
| Root implementation commit | Not recorded; implementation is uncommitted |
| QC submodule commit | Not recorded; QC changes are uncommitted |
| Root `git_dirty` | `true` (implementation changes present) |
| QC submodule `git_dirty` | `true` (implementation changes present) |
| Root focused tests | 34 passed, 1 upstream deprecation warning |
| QC tests | 5 passed, 3 upstream deprecation warnings |

| Stage | Elapsed time | Status / stop value |
|---|---:|---|
| S0 tests | Root 48.29 s; QC 7.48 s | Pass |
| S1 d⁺ training, 20k | — | Not run |
| S2 chunk/P2/P3 audit | — | Not run |
| S3 L⁺ build and board check | — | Not run |
| S4 four action heads and offline gates | — | Not run |
| S5 direct-Q QC and same-checkpoint BC evaluation | — | Not run |
| S6 discounted-cost critic QC | — | Not run |

## S1 — d⁺ training

| Metric | Result |
|---|---:|
| Train matched / unmatched candidate pairs | Not measured |
| Held-out matched candidate pairs | Not measured |
| Final chunk constraint violation | Not measured |
| Final held-out P2 violation | Not measured |
| `λ_chunk` / `λ_match` | Not measured |
| F3 collapse alarm | Not measured |
| Dead-unit fraction by layer | Not measured |
| Checkpoint path | Not created |

## S2 — d⁺ audit and stop gates

| Model | Held-out chunk violation | Held-out P2 violation | Median d⁺, `n_p=0 / 1 / ≥2` | `n_p≥2`, fraction `d⁺≤20` | `n_p=0`, fraction `d⁺≤20` | Gate |
|---|---:|---:|---|---:|---:|---|
| v0.8 `d_θ` reference | 2.9% (historical) | — | To be measured by same audit | To be measured | To be measured | Reference |
| v1.0 `d⁺` | Not measured | Not measured | Not measured | Not measured | Not measured | Not evaluated |

| `n_p` group | Candidate count | d⁺ p10 | d⁺ p50 | d⁺ p90 | Fraction `d⁺≤20` |
|---|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — |
| 1 | — | — | — | — | — |
| ≥2 | — | — | — | — | — |
| ∞ | — | — | — | — | — |

| S2 gate | Threshold | Result |
|---|---:|---|
| Held-out chunk violation | ≤5% | Not measured |
| Held-out P2 violation | ≤5% | Not measured |
| Median order `n_p=0 < 1 < ≥2` | Strictly increasing | Not measured |
| `n_p≥2` candidates with `d⁺≤20` | ≤1% | Not measured |
| S2 overall | All four pass | Not evaluated |

## S3 — learned L⁺ graph

| Metric | L⁺ result | v0.8.4 L reference | v0.8.4 O reference | v0.6 G-R reference |
|---|---:|---:|---:|---:|
| Retained bridge edges | Not measured | 30,209 | 7,707 | 1,401,747 |
| Reachability | Not measured | 98.2% | 86.1% | 99.7% |
| Impossible bridge fraction | Not measured | 85.8% | 1.8% | 0 |
| Shortest-path bridge count 0 / 1 / 2 | Not measured | 1 / 52 / 47% | 2 / 62 / 36% | — |
| Median cost C | Not measured | 95.5 | 352 | — |
| L⁺ graph hash | Not created | — | — | — |
| Board-check artifact | Not created | — | — | — |

## S4 — action heads and offline gates

| Head seed | C1 violation | C2 violation | F3 alarm | Dead-unit fraction | `λ₁` / `λ₂` | Gate |
|---:|---:|---:|---|---:|---|---|
| 0 | — | — | — | — | — | Not measured |
| 1 | — | — | — | — | — | Not measured |
| 2 | — | — | — | — | — | Not measured |
| 3 | — | — | — | — | — | Not measured |

| Offline metric | Result |
|---|---:|
| G1 ensemble d̄ median | Not measured |
| G1 fraction d̄ > 6.25 | Not measured |
| G2 median `|Q − (5 + C(s_{t+5}))|` | Not measured |
| G3 alternative-action distance increases | Not measured |
| G3 median distance increase | Not measured |
| Offline gate result | Not evaluated |

## S5/S6 — QC

Success and wrong-button counts use 50 evaluation episodes per checkpoint. Wrong buttons are the mean sign-reversed `episode.final_reward`.

| Method | 50k | 100k | 150k | 200k | 250k | 300k | Final graph-target Spearman | Final critic loss | Final wrong buttons | Run directory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L⁺ direct-Q, with same-checkpoint BC comparison | — | — | — | — | — | — | N/A | N/A | — | Not run |
| L⁺ discounted-cost critic | — | — | — | — | — | — | — | — | — | Not run |
| O_np1 regret critic, v0.8.4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.222 | N/A | 4.16 | Existing reference |
| O_np1 cost critic, v0.8.4b | 0.04 | 0.24 | 0.26 | 0.16 | 0.28 | 0.30 | 0.980 | 2,978.6704 | 2.98 | Existing reference |
| L cost critic, v0.8.4b | 0.00 | 0.06 | 0.02 | 0.04 | 0.02 | 0.06 | 0.890 | 610.5286 | 2.96 | Existing reference |
| TD QC, v0.6 | 0.10 | 1.00 | 0.98 | 1.00 | — | — | N/A | N/A | — | Existing reference |

## Stop point

No benchmark experiments were run. The next action is to commit the implementation and QC submodule changes, confirm a clean checkout, then follow [RUN_v1.0.md](RUN_v1.0.md) in order.
