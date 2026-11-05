# Graph Critic v0.5 implementation and results

Date: 2026-11-05

This report follows the v0.5 cause-confirmation and conservative-bridge specification. Transition data and all experiment artifacts are under ignored `outputs/`; this document records the code boundary, measured results, and gates.

## Implementation

- `BridgeConfig.enabled=false` builds a temporal-only graph (G0).
- Scene passive bridges can use dataset-estimated τ_real, strict free-space endpoints, 1 cm drawer/window tolerance floors, and per-edge ρ costs. The strict R search uses the epsilon-normalized L∞ metric and validates componentwise distances and τ_real ≥ τ_kin.
- `experiments/calibrate_real_time.py` estimates free-space actuator response, realized speed, same-episode time checks, static drawer/window tolerances, and strict-R source exclusion from saved transitions and the aligned sidecar. It does not step a simulator or load a TD checkpoint.
- `ServoTrackingPolicy` implements the v3 free/contact controller, current-path-vertex contact correction, recorded gripper control, physical tracking tubes, and stall/passive-drift detection. The controller is a post-build witness only.
- `experiments/target_agreement.py` evaluates the TD critic ensemble mean on the exact five-action chunks and compares its Q ordering to graph targets. It is diagnostic only and does not select a graph.
- `BridgeConfig.include_m_bridges` extends strict R graphs with separately searched M candidates. R endpoints remain free-only; M candidates can cross contact states. M travel time is lower-bounded by learned frame-gap time, τ_real, and passive-coordinate motion time, then capped by `tau_max_steps`. A new physical `(r,o,e)` model is being trained because the strict-R reachability result triggered the guide's conditional M branch.
- Graph schema 0.5 remains able to load 0.3/0.4 graphs and artifacts written during the v0.5 config-field addition.

Graph construction and target generation use saved transitions and the aligned sidecar only. Simulator stepping is limited to the separate witness evaluations; TD is used only by target-agreement diagnostics.

## A1: QC cause control

All runs use Scene task1, QC graph critic, best-of-n=32, h=5, 300k offline steps, seed 0, and 50 evaluation episodes per checkpoint. W&B was offline.

| Targets | 100k | 200k | 300k | Interpretation |
|---|---:|---:|---:|---|
| G0 temporal-only | 0.58 | 0.62 | 0.60 | Does not collapse; near the expected behavior-cloning level |
| G3 v0.4 R-only, τ_kin | 0.32 | 0.18 | 0.24 | Below G0 at all checkpoints |
| D4a v0.3 geometric Λ₀=10 | 0.02 | 0.00 | 0.00 | Existing run |
| TD reference | 0.96 | 0.98 | 1.00 | Existing run |

G0 clears the ≥ 0.4 branch threshold. G3 and D4a both underperform G0, supporting the guide's diagnosis that erroneous bridge targets harm best-of-n selection. These are single-seed results with 50 evaluation episodes, so the table is descriptive rather than a significance test. The G0 checkpoint files and G3 run are retained under their QC experiment directories inside `third_party/qc/exp/`.

## A2: TD target agreement

TD checkpoint: `third_party/qc/exp/qc/e1-scene/scene-play-singletask-task1-v0/sd00020261103_153119/params_300000.pkl`. There are 996,000 valid h=5 Q rows. The graph target conversion is `Q_G = -(1-γ^(y-1))/(1-γ)`, with γ=0.99. A negative median `Q_G-Q_TD` means the graph target is more conservative; a positive value means it is optimistic.

| Graph | b rows p50 / p95 / max | g1 finite, b=0: n / Spearman / median gap | g2 finite, b≥1: n / Spearman / median gap | g3 cap: n / Spearman / median gap |
|---|---:|---:|---:|---:|
| G0 | 0 / 0 / 0 | 31,423 / 0.613 / -22.61 | 0 / n/a / n/a | 961,656 / constant-Q / -25.83 |
| G3 v0.4 R | 16 / 32 / 58 | 1,396 / 0.965 / -0.24 | 781,965 / -0.176 / -20.41 | 209,718 / constant-Q / -26.92 |
| v0.3 geometric Λ₀=10 | 3 / 10 / 21 | 12,822 / 0.873 / -7.62 | 559,715 / 0.238 / -25.42 | 420,542 / constant-Q / -20.07 |
| v0.3 learned q=.8 | 4 / 8 / 15 | 4,265 / 0.986 / -2.15 | 987,336 / 0.397 / +11.01 | 1,478 / constant-Q / -21.44 |

The finite bridge-free groups have strong rank agreement; the bridge-using groups lose rank agreement. The learned q=.8 graph is optimistic in its bridge group (median gap +11.01). Cap-group Spearman is undefined because graph Q is constant there. This analysis is development-only and is not used to choose ρ or report a best graph.

## B/C: witness and τ_real calibration

### Response and speed from free data

The calibration used 463,596 free transitions and 462,568 moving transitions. Each vector is ordered `(x,y,z,yaw,gripper)`.

| Dimension | α immediate | R² immediate | α one-step lag | R² lag | v95 | v75 | v95 / action limit |
|---|---:|---:|---:|---:|---:|---:|---:|
| x (m) | 0.428 | 0.948 | 0.392 | 0.813 | 0.02245 | 0.01511 | 0.449 |
| y (m) | 0.436 | 0.984 | 0.404 | 0.853 | 0.02283 | 0.01699 | 0.457 |
| z (m) | 0.386 | 0.972 | 0.360 | 0.855 | 0.01889 | 0.01350 | 0.378 |
| yaw (rad) | 0.210 | 0.842 | 0.196 | 0.733 | 0.09000 | 0.06172 | 0.300 |
| gripper | 0.300 | 0.821 | 0.233 | 0.495 | 0.11844 | 0.05774 | 0.118 |

The immediate model has higher R² on all five coordinates. XYZ and yaw speed/action-limit ratios are close to the fitted immediate response; the gripper ratio is lower. The applied feedback gain is clipped `1/α` by dimension: `[2.338, 2.296, 2.591, 4.0, 3.332]`.

The 50,000-pair same-episode check found `P(elapsed ≤ τ_real)=0.0231` and zero sampled τ_real<τ_kin violations. The low coverage is expected for meandering data paths. In distance bins up to 0.4 m, the minimum observed elapsed time was at least the minimum predicted τ_real; above 0.4 m, the sample minimum was 19 steps versus 15 predicted. This tail is a caution on the rate estimator. Both drawer and window static tolerances used the 1 cm floor: q99 adjacent deltas were 5.40×10⁻⁶ m and 7.90×10⁻⁶ m, based on about 459k static-free steps each, with no fallback due to insufficient data.

### Witness checks

| Check | Result | Gate |
|---|---:|---|
| Free-space arrival within τ_real+2, v95 | 197/200 = 0.985 | Pass (≥ 0.9) |
| Free-space arrival within τ_real+2, v75 | 197/200 = 0.985 | Pass (≥ 0.9) |
| Zero-bridge path success, Kc=0.3, geometric Λ₀=10 graph | 40/70 = 0.571 | Fail (replay reference 0.614) |
| Zero-bridge path success, Kc=0.5, same graph | 40/70 = 0.571 | Fail (replay reference 0.614) |
| Bridge-path retention for 1–3 bridges, Kc=0.3 / 0.5 | 0.620 / 0.644 | Both pass numeric threshold, but not interpreted because the witness-normal gate failed |

At Kc=0.3, failures among all 500 starts were active tracking 171, passive drift 144, mode mismatch 12, and timeout 54. The response-calibrated free-space controller now passes its isolated arrival test, but the end-to-end zero-bridge witness remains below replay. Connection-accuracy results must not be treated as a passing v0.5 gate unless the specific graph's witness-normal stage also passes.

## D: conservative graph sweep

### G1: strict R, ρ=1

| Metric | Result | Gate |
|---|---:|---|
| R bridges | 1,390,445 | — |
| τ_real p50 / p95 | 7 / 10 steps | τ_max=10 |
| Retained R edges with τ_real<τ_kin | 0 | Pass |
| Reachable states | 0.0395 | — |
| Failed-episode reachability | 0.0051 | Fail (>0.1) |
| Valid non-success cap fraction | 0.9634 | Fail (≤0.8) |
| Cross-episode nearest passive distance p50 / p95 | 2.697 / 4.825 ε | 9.19% within ε |

The strict free-endpoint rule removes 51.42% of non-success source states from R consideration. The candidate graph is physically conservative but far too disconnected for QC.

The separate direct bridge landing check sampled 200 strict-R edges and landed on 97 (`0.485`), below the required `0.8`. In the later G1+M graph policy run, strict-R path landings were 77/166 (`0.464`); this confirms the direct landing failure is not just an artifact of the standalone probe.

The q75 speed ablation produced 901,082 R bridges with τ_real p50/p95 of 7/10. Reachability was `0.0388`, failed-episode reachability `0.0045`, and valid non-success cap fraction `0.9639`; it fails the same graph gates and reduces bridge use relative to q95 without improving connectivity.

### G2: ρ sensitivity on the same R edges

| ρ | Λ(ρ) steps | Failed-episode reachability | Valid non-success cap fraction | Graph gate |
|---:|---:|---:|---:|---|
| 0.9 | 10.483 | 0.0051 | 0.9635 | Fail |
| 0.7 | 35.489 | 0.0051 | 0.9639 | Fail |
| 0.5 | 68.968 | 0.0051 | 0.9643 | Fail |

The bridge penalty changes target values and bridge-use frequency but cannot repair the strict graph's low reachability. No G1/G2 QC run is allowed by the guide's graph gate.

### G3′: v0.4 R candidates with τ_real

The graph build completed with 5,527,828 R bridges, τ_real p50/p95 of 9/17 steps (maximum 37), reachability 0.7925, failed-episode reachability 0.7846, cap fraction 0.2120, and zero τ_real<τ_kin violations. Its graph-only necessary gates pass. Unlike G1, G3′ does not require both R endpoints to be free, so it isolates the effect of realized time from the strict endpoint filter.

The 500-start v3 witness passed its normal stage: zero-bridge success was 11/11 (`1.0`) and cost/time Spearman was `0.817`. Only 11 starts had zero-bridge paths, so this estimate is imprecise. Connection accuracy failed: 1–3 bridge retention was `0.0667`, and R landing was 219/684 (`0.320`). Overall πG success was `0.032`; 444/500 starts required four or more bridges and only 2 of those succeeded. Since the graph-normal gate passes, these connection results are interpretable; they do not meet the required retention threshold.

## Conditional M extension

G1 failed reachability and cap gates, triggering the guide's M branch. A new 48-step frame-gap model was trained on physical `(r,o,e)` inputs from the same saved Scene transitions, with episode-level calibration/test splits and the `r_H` far-label exclusion. The q=.9 test coverage was `0.9029`, with a 1,000-replicate episode-cluster 95% interval `[0.8929, 0.9140]`; the calibrated bootstrap gate passed. The q=.8 test coverage was `0.7953`, interval `[0.7811, 0.8095]`, also passing its calibration gate. Test far-class AUROC was `0.9909`, median τ-versus-distance Spearman was `0.8630`, and 13,579 of 20,000 proposed r-to-r_H far pairs were excluded to avoid label conflicts.

The initial G1+M build used the common 48-step M horizon as the R cap too, which admitted R edges beyond G1's specified 10-step limit. That graph and its witness are invalid for this comparison. `bridge.r_tau_max_steps` now keeps R at 10 while M uses the learned model's 48-step cap.

The corrected G1+M graph retained exactly 1,390,445 R bridges (τ p50/p95 7/10) and 6,374,434 M bridges (25/47), with zero kinematic violations. Reachability was `0.9980`, failed-episode reachability `0.9979`, valid non-success cap fraction `0.00208`, and bridge use `0.9915`; graph necessary gates pass. Build time was 23.1 minutes and peak RSS about 4.42 GB. Graph hash: `8fd6ffe48705de11df715a6b0c3824dd0a2fa02ce9157e57be2f16bc5cd520b8`. The graph records the R/M caps separately, the M checkpoint SHA256, calibrated quantile `.9`, horizon 48, split seed 0, and `r_H` far-label exclusion.

The corrected 500-start witness passed its normal stage: free-space arrival `0.985` and zero-bridge path success 23/25 (`0.92`) against replay `0.614`. The small zero-bridge subgroup makes that estimate imprecise. Cost/time Spearman was `0.964`. Connection accuracy failed: 1–3 bridge retention was `0.1012`; strict-R single-bridge landing remained `0.485` (200 direct trials; in-policy R landing 76/154 = `0.494`); M landing was 348/575 = `0.605`. Overall πG success was `0.092`; 228/500 starts had four or more bridges and none succeeded. M therefore repairs the graph reachability/cap conditions, but bridge paths still fail the QC connection gate.

No G1, G2, G3′, or G1+M QC pilot was launched: every candidate failed at least one required graph or connection-accuracy gate. The completed eligible-control runs remain G0 and the historical TD comparison. The graph outputs, M checkpoint, and simulator witness JSONs remain under ignored `outputs/`.

## Verification

- An initial GPU-contended test run had two obsolete servo fixture failures and one JAX CUDA out-of-memory failure; the fixtures were corrected. The final full suite passed with CPU-only JAX: `102 passed` (one upstream AnyIO/pytest deprecation warning).
- Final `compileall` and `git diff --check` passed.

