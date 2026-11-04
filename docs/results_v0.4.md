# Graph Critic v0.4 implementation and results

Date: 2026-11-04

This report follows the v0.4 run and passive-coordinate bridge specification. Simulation outputs and model artifacts live under ignored `outputs/`; this document records the measured gates and implementation boundary.

## Implementation

- **P1, physical state schema:** `StateSchema` exposes `r`, `o`, `e`, `q_p`, `mode`, and `q_a`. The Scene adapter reads robot/object coordinates from the aligned qpos sidecar, computes EE coordinates in metres/radians, handles quarter-turn cube yaw symmetry, and switches a contacted cube to object-relative passive coordinates while grasped. A low-dimensional robomimic adapter is also present. `OGBenchVerifier.sim_readout()` exposes physical simulator coordinates for the controller. `envs/boost_export.py` validates the same field contract on the BOOST side; BOOST runtime integration is not in this checkout.
- **P2, pose servo:** `ServoTrackingPolicy` follows temporal path vertices by feedback, treats bridge edges as explicit landing steps, and records active tracking, passive drift, mode mismatch, and timeout outcomes. Replay and feed-forward-plus-servo modes remain available as comparisons. The evaluator reports zero-bridge success, bridge retention, landing by bridge kind/passive distance/τ, path success by bridge count, cost/time rank, and the graph/simulator gates.
- **P3, R bridges:** Scene passive tolerances are estimated from contact-free, object-distant transitions and floored at 5 mm / 2 degrees. Candidate search uses the epsilon-normalized L∞ metric and componentwise validation. R costs use `τ_kin + settle`; six Scene configs cover epsilon multipliers `{1,3,10}` and settle margins `{2,4}`. Diagnostics include cross-episode nearest passive distances and bridge endpoint differences.
- **P4, M bridge preparation:** `passive_rm` adds `(r,o,e)` frame-gap input and applies `τ_M=max(τ_learned, τ_kin+settle, τ_pass)`. Kinematic/passive violation metrics are recorded. Frame-gap validation now uses episode-cluster bootstrap intervals for coverage. No new `(r,o,e)` checkpoint or M-bridge Scene graph has been trained: the guide makes this conditional on the R-only stage and its reachability results.
- **P5, build/evaluation gates:** Graph-target building is still offline and simulator-free; simulator verification is a separate evaluation step. The builder enforces the M-bridge kinematic-violation gate, while the simulator evaluator reports graph and relative-simulation gate outcomes.

## D0: existing graph audit

For the Scene task1 dataset, 1,001,000 state rows had a same-mode nearest neighbor from another episode. With the estimated passive tolerances (`[5,5,5] mm`, `2°`, and about `[33,57] mm` for drawer/window), the nearest-neighbor distance had median 1.81 ε, p95 3.54 ε, and only 16.10% were within ε. This predicts sparse R-only stitching at the strict default tolerance.

| Existing graph | Bridge edges | Bridge passive difference p50 / p95 (ε) | `τ_learned < τ_kin` | `τ_learned < τ_pass` | Historical πG success, 0 bridges / 1–3 bridges | Retention |
|---|---:|---:|---:|---:|---:|---:|
| Geometric, Λ₀=0 | 724,969 | 0.67 / 5.26 | 0.00% | 0.13% | 0.630 / 0.149 | 0.236 |
| Geometric, Λ₀=10 | 724,969 | 0.67 / 5.26 | 0.00% | 0.13% | 0.614 / 0.083 | 0.136 |
| Learned, q=.9 | 3,182,404 | 3.75 / 16.07 | 1.67% | 1.84% | 0.643 / 0.037 | 0.058 |
| Learned, r_H, q=.8 | 7,436,179 | 6.46 / 22.13 | 7.05% | 4.74% | 0.750 / 0.000 | 0.000 |

The learned q=.8 graph exceeds the proposed 5% kinematic-violation limit. Its very large edge count and high passive-coordinate deltas match the v0.3 optimism concern. Historical per-bridge landing attribution is approximate because those older evaluation files did not retain bridge-level records. The four D0 JSON reports retain this caveat.

For the existing r_H, q=.8 observation-input checkpoint, the stored test coverage is 0.8017. A fresh fixed-seed re-evaluation produced 0.7913 over 10,000 near pairs; the 1,000-replicate episode-cluster 95% interval was `[0.7778, 0.8048]` (width 0.0270 across 100 test episodes), which contains the nominal 0.8. This re-evaluation uses the legacy checkpoint and does not validate the new `(r,o,e)` input model.

The stored historical attempts show a strong passive-distance pattern, with that same attribution caveat: for geometric Λ₀=0, inferred landings were `9/503` at ≤1 ε, `2/507` at 1–3 ε, and `0/650` above 3 ε; for Λ₀=10 they were `4/232`, `0/363`, and `1/478`. The learned q=.9 and q=.8 runs inferred `3/187` and `1/171` landings at ≤3 ε, and zero landings above 3 ε. The tiny overall landing counts mean these are suggestive evidence for C2, not a precise causal estimate.

## D1: controller check

The 500-start replay evaluation on the geometric Λ₀=10 graph reproduced the baseline: zero-bridge success `43/70 = 0.614`, 1–3 bridge success `12/144 = 0.083`, and relative retention `0.136`.

The v0.4 pose servo did not satisfy its prerequisite controller gate. At unit feedback gain, zero-bridge success was `3/70 = 0.043`; bridge-path success was `0/144`, so retention was zero. Failures were active tracking `326/500`, passive drift `170/500`, and timeout `2/500`. The graph-only checks passed (`reachable=0.595`, failed-episode reachable `0.582`, valid non-success cap fraction `0.423`), but the simulator gate failed because zero-bridge success was below 0.8 and bridge retention was below 0.5. Thus the current bridge path outcomes cannot be used to rank the R configurations.

The feed-forward-plus-servo ablation at gain 1 also failed the controller gate: zero-bridge success was `8/70 = 0.114`, bridge-path success was `0/144`, with active tracking `276/500`, passive drift `151/500`, mode mismatch `7/500`, and timeout `58/500`. Adding recorded actions did not fix bridge execution.

The free-space servo diagnostic also missed its 0.95 gate: at the specified unit gain it reached εa in `2/200 = 0.01` trials by `τ_kin+2`; gains 2 and 3 reached it in `52/200 = 0.26` and `66/200 = 0.33`. Larger gains help but do not validate the prescribed controller. The direct pose action is not realized as the same per-step motion by the simulator's IK/control stack. This is an outstanding controller calibration issue; no gain change has been promoted as a fix.

Because D1 failed, the six-configuration D2 simulation sweep and conditional D3 M-bridge sweep have not been run. The configs and implementation are present, but their simulation gates remain unmeasured.

## D4a: QC pilot

The prescribed seed-0, 300k-step QC graph-target pilot used W&B offline and the geometric Λ₀=10 v0.3 targets. Its success was `0.02 / 0.00 / 0.00` at 100k / 200k / 300k (50 evaluation episodes per checkpoint), versus `0.96 / 0.98 / 1.00` for the existing TD run. The graph-target pilot declines and is far below TD, so the D4a improvement criterion fails. The run and three parameter checkpoints are under `third_party/qc/exp/qc/e1-scene-v04-d4a-geometric/scene-play-singletask-task1-v0/sd00020261104_194245/`; this pilot is descriptive and does not select a graph configuration.

## Verification and remaining work

- Core suite: `94 passed, 1 skipped`.
- `compileall` and `git diff --check`: passed.
- A full-size R-only graph/target build for `ε multiplier=1, settle=2` completed in 258 seconds (peak RSS about 3.52 GB): 1,001,000 state nodes, 5,527,828 R bridges, `τ` p50/p95 of 5/8 steps, and no M bridges. Graph gates passed: reachability `0.791`, failed-episode reachability `0.783`, bridge-use `0.995`, valid non-success cap fraction `0.211`. The h=5 target artifact has 996,000 valid rows (99.6%). Round-trip loading validated graph schema `0.4`, 6,527,828 total edges, bridge types `{-1,0}` (temporal/R), and the targets against the graph hash. This is a build smoke check, not a selected setting, because the servo simulator gate failed.
- Graph and target generation used saved transitions and the aligned sidecar only; it did not construct or step a simulator.
- The graph core accepts the BOOST field names through a validated adapter, but it has not been run inside the separate BOOST/G1 repository.
- The current Scene relative-pose representation covers cube EE poses; drawer/window coordinates are represented as passive world/articulation values and do not yet provide handle-relative EE pose features.
- Before v0.4 QC/FQL comparison (D4), resolve the servo tracking mismatch, build the remaining five R configurations, and evaluate D2 only after the zero-bridge controller gate is met. Train a new `(r,o,e)` frame-gap model and evaluate M bridges only if R-only simulation/reachability results require it. E2 multi-task/multi-seed work is not started.

