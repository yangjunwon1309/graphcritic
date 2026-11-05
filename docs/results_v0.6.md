# Graph Critic v0.6 implementation and results

Date: 2026-11-05

This report records the v0.6 regret-target and OGBench Puzzle work. Graph and target generation reads saved transitions plus aligned sidecars; simulator stepping is confined to the separate witness runs. TD checkpoints are not used by graph construction or target generation.

## Implementation

- Graph targets now use schema 0.6, default to non-negative chunk regret, and store a `loss_mask` independently of QC chunk validity. Source code 4 identifies rows where both graph costs are capped; these remain valid QC chunks but have placeholder target 0 and are excluded from critic loss. The cost target remains available for controlled comparison. Graph artifacts remain schema 0.5.
- The QC fork, based on submodule commit `ce81e33`, checks target kind, dataset provenance, horizon, row count, QC chunk validity, and explicit loss-mask presence. It attaches both masks at the chunk start, fits non-negative `softplus(Q)` to eligible targets, logs regret means and batch Spearman, and keeps actor selection on negative ensemble-maximum cost/regret. FQL's distillation Q term uses the same non-negative ensemble maximum.
- Puzzle support includes a physical `StateSchema`, exact integer button bitfields, calibrated press/free modes, GF(2) solving, exact-board free-state R bridges, realized-time edge costs, and ρ penalties. Bridge lookup uses sorted exact-board buckets followed by EE kNN. Diagnostics now include bucket sizes, press-height fit, τ distribution, speed quantile, and bridge parameters.
- Puzzle data loading uses QC's loader and aligned qpos/qvel/button sidecars, while task success is derived from the static OGBench task definition. It does not instantiate or step a simulator.
- The P13 generator uses OGBench's noisy `ButtonPlanOracle` for low-level control, seeded GF(2) solutions, paired detours, optional post-goal random play, and optional 50/50 standard-play mixing. It checkpoints completed episodes, NumPy/Python/environment RNG state, and transition counts so a host restart can resume without duplicating rows. Dataset IDs include the generator script hash, parameters, seed, and archive SHA. Archives include terminal-only final states so the QC loader receives the requested number of usable transitions. Fixed dataset paths are supported in both graph building and QC training.
- Puzzle witness evaluation now compares against a matched replay run. The baseline must have the same dataset ID, graph hash, start count, and exact start-selection hash; the old Scene threshold is not reused for Puzzle.

## P10: regret and QC checks

The toy checks cover non-negativity, zero regret on shortest-path chunks, same-state ordering preservation, masking of both-capped rows, target-schema round trips, and rejection of legacy targets without an explicit mask. A one-update QC integration smoke using a generated Puzzle archive verified fixed-path loading and the row-aligned mask path. The logged critic keys included `regret_pred_mean`, `regret_target_mean`, `regret_batch_spearman`, and `graph_target_valid_frac`.

The required 10k graph-critic Spearman pilot completed on the 3×3 task-3 G-R ρ=1 regret targets. At step 10,000, the logged batch Spearman was `0.6102`, below the required `0.7`; the P10 training-quality gate therefore fails. The run and CSV are under `third_party/qc/exp/qc/v06-p10-regret-10k/`. The same run logged critic prediction and target regret means and the valid-row fraction. No 300k graph-target pilot has started; the low Spearman is retained as a failed gate, not treated as evidence for a successful QC comparison.

## P11: standard Puzzle data

All archives opened through QC's `load_dataset`; the actual OGBench singletask relabeler agreed with exact button-board equality for all 15 task/size checks. The schema's button bitfield matched the sidecar on the first 1,000 rows for each size.

| Standard data | QC transitions | Episodes × length | Obs / button width | Button changes | Visited configurations | Task results |
|---|---:|---|---:|---:|---:|---|
| 3×3 | 1,000,000 | 1,000 × 1,000 | 55 / 9 | 30,897 (3.09%); 32.35 steps between changes | 512 / 512 | Tasks 1–5 connected; `presses_data / presses_opt = 1` for every task |
| 4×4 | 1,000,000 | 1,000 × 1,000 | 83 / 16 | 30,031 (3.00%); 33.26 steps between changes | 23,400 / 65,536 | Tasks 1–5 disconnected |
| 4×5 | 3,000,000 | 3,000 × 1,000 | 99 / 20 | 88,765 (2.96%); 33.77 steps between changes | 83,442 / 1,048,576 | Tasks 1–5 disconnected |

Episode starts were varied: 437/1,000, 991/1,000, and 2,995/3,000 distinct button boards for 3×3, 4×4, and 4×5. Arm qpos rounded to 1e-3 was unique at every episode start. Every observed board change matched one Lights Out button press.

The exact minimum press counts for 4×4 tasks 1–5 are 4, 6, 6, 6, and 7; for 4×5 they are 4, 10, 14, 16, and 20. No standard-play episode solved any 4×4 or 4×5 task. The data branch therefore stays standard for 3×3 and triggers P13 for 4×4. The selected comparisons are 3×3 tasks 3/4 and 4×4 tasks 2/5. The 4×5 generator branch is implemented but outside the current graph/QC run set.

## P12: Puzzle calibration and graphs

The selected 3×3 standard transition set has 354,324 free states (35.4%). The fitted press-height threshold is 0.14172 m from 30,897 observed board toggles. Immediate actuator response α is `[0.440, 0.430, 0.384, 0.217, 0.294]` with R² `[0.973, 0.995, 0.960, 0.921, 0.921]` for `(x,y,z,yaw,gripper)`. The q95 realized speed is `[0.01974, 0.02037, 0.01227, 0.08120, 0.04808]` in `(m/s,m/s,m/s,rad/s,gripper/s)`.

On 50,000 sampled same-episode pairs, 98.548% had predicted τ_real no greater than observed elapsed time, with zero τ_real < τ_kin violations. The small fraction where τ_real exceeded observed elapsed time shows that this empirical rate estimate is not a strict statistical lower bound; the graph builder separately enforces the kinematic lower bound and the configured 20-step cap.

| 3×3 task | Graph | R bridges | τ_real p50 / p95; max | Reachable / failed-episode reachable | Regret loss mask | Bridge use | Regret p90 / Rmax fraction |
|---|---|---:|---|---|---:|---:|---|
| 3 | G0 | 0 | — | 0.0299 / 0 | 0.0300 | 0 | 0 / 0 |
| 3 | G-R, ρ=1 | 1,401,747 | 4 / 5; 20 | 0.9973 / 0.9973 | 0.9989 | 0.9959 | 16 / 0.00357 |
| 3 | G-R, ρ=0.9 | 1,401,747 | 4 / 5; 20 | 0.9973 / 0.9973 | 0.9989 | 0.9906 | 19.48 / 0.00769 |
| 4 | G0 | 0 | — | 0.0283 / 0 | 0.0284 | 0 | 0 / 0 |
| 4 | G-R, ρ=1 | 1,401,539 | 4 / 5; 20 | 0.9973 / 0.9973 | 0.9989 | 0.9961 | 16 / 0.00369 |
| 4 | G-R, ρ=0.9 | 1,401,539 | 4 / 5; 20 | 0.9973 / 0.9973 | 0.9989 | 0.9912 | 19.48 / 0.00793 |

All four G-R graphs have zero regret property-1 violations, zero τ_real < τ_kin violations, and about 0.22% valid non-success capped targets. Their graph build gates pass. G0 has very low reachability and only about 3% loss-eligible rows because most episodes never encounter that task's goal. The ρ=0.9 penalty changes regret magnitudes and bridge-use frequency but not bridge endpoints or reachability. The exact-board bucket sizes for the shared 3×3 data have p10/p50/p90 of 573/691/822 free states, maximum 1,130, over 512 visited boards; each source queries at most 64 EE neighbors.

The required task-3 G0-cost comparison artifact is also built. Its 97.0% valid non-success target cap fraction produces a target median of 1,006 steps and mean of 984 steps (loss mask 1.0); G0-regret excludes both-capped rows and has a 3.0% loss mask. No QC comparison has yet been run on these targets.

The standard 4×4 task-2 G-R build confirms that exact-board stitching cannot repair missing button-transition connectivity. It has 1,001,000 state nodes and 741,535 R bridges (1,741,535 total edges); R τ_real is p50/p95/max `8/15/20` steps. The 23,248 observed free-board buckets have sizes p10/p50/p90 `7/13/27` and max `96`. Despite those bridges, reachability is only `0.000046`, failed-episode reachability is zero, bridge use is zero, and 995,954 rows are both-capped and excluded from regret loss. The graph has 16 components and zero regret property-1 violations. Standard 4×4 task-5 G0 contains no goal nodes, has zero reachable rows, and a zero regret loss mask. Its G-R graph adds 741,535 R bridges with the same τ and bucket distributions, but still has no goal nodes, zero reachability, zero bridge use, and no loss-eligible rows. These results support the conditional P13 branch: repeated robot states help only within board configurations already connected by observed presses.

### 3×3 task 3 witness

- Free-space arrival within τ_real+2: 197/200 = 0.985 (gate ≥0.9).
- Direct R landing: 198/200 = 0.990; the two failures were active-tracking outcomes, with no passive drift or timeout.
- Paired replay and servo each succeeded on 5/6 sampled zero-bridge paths (0.833). This clears the comparison gate, but the subgroup is small.
- Servo path success for 1–3 bridges was 36/64 = 0.563; relative retention against the zero-bridge success was 0.675. Including the zero-bridge cases, the ≤3-bridge rate was 41/70 = 0.586. In-policy R landing was 1,795/1,879 = 0.955; cost/time Spearman was 0.993. The combined witness gates pass for this task/configuration.
- Across all 500 servo starts, 120 succeeded; failures were passive drift 351, timeout 21, and mode mismatch 8. Most selected paths required four or more bridges, where success was 79/430 = 0.184. Thus the connection gates pass, but overall long-path success remains low.

### 3×3 task 4 witness

- Free-space arrival within τ_real+2: 197/200 = 0.985; direct R landing: 199/200 = 0.995. Both pass their gates.
- The paired 500-start evaluation matched the replay run's graph, dataset, and start selection. Zero-bridge servo success was 5/5 = 1.0 (the small subgroup is imprecise); 1–3 bridge retention was 0.640, above the 0.5 gate. In-policy R landing was 1,749/1,860 = 0.940, above the 0.8 gate; cost/time Spearman was 0.990. The task-4 witness gates pass.
- Overall success was 110/500 = 0.220. Failures were passive drift 349, timeout 30, and mode mismatch 14. For paths with 4+ bridges, success was 73/445 = 0.164, so long-path success remains limited despite passing the prescribed short-path gates.

## P13: task-directed data and graph targets

The 4×4 task-2 generator smoke produced 1,000 QC transitions from one goal-reaching episode. The QC loader read exactly 1,000 rows, the board bitfield matched all 1,000 checked sidecars, and the official relabel agreed with goal equality. Its observed data path used 16 presses versus the six-press optimum (2.67×), including paired detours.

A separate 4,000-transition 50/50 smoke produced exactly 2,000 generated plus 2,000 standard QC transitions. QC read all 4,000 rows; two generated episodes reached the goal, and the combined task graph used 8 presses versus the six-press optimum (1.33×). A 10k resume smoke was interrupted after 803 transitions, resumed from its episode-boundary checkpoint, and completed with 10,000 QC transitions. The QC loader confirmed the official relabel, 18 accepted episodes, 17 single-episode solutions, and `presses_data / presses_opt = 1.0`.

The full P13 archive contains exactly 1,000,000 QC transitions from 1,321 accepted episodes (20 failed attempts); metadata records 1,321 goal-reaching episodes and 646 that continued after goal. Its SHA256 is `6f73eb1d6bfe8d950e567055edc3f4a55e88f115885945efac21640924817de6`. QC's loader opened the archive and the official relabel matched goal equality. Puzzle inspection matched the button bitfield against the sidecar for 1,000 rows, found 30,074 board-change transitions (each a single Lights Out press), and reported 4,023/65,536 configurations visited. For task 2, the observed configuration graph was connected, `presses_data / presses_opt = 1.0` (6/6), and all 1,321 episodes individually reached the goal.

The full 4×4 task-2 G0, G-R ρ=1, and G-R ρ=0.9 h=5 regret graphs and targets are built from this generated dataset. Each has 1,001,321 state occurrences; both G-R variants have 1,362,215 R edges. G0 reaches 71.36% of rows and has 71.47% regret-loss coverage; its valid regrets are all zero. G-R reaches 99.80%, uses an R edge on 92.10% of reachable states at ρ=1 and 80.34% at ρ=0.9, and has 99.90% regret-loss coverage. For both G-R targets, property-1 violations are zero, regret p90 is 13 steps, and the non-success cap fraction is 0.15%; zero-regret fractions are 59.58% (ρ=1) and 71.93% (ρ=0.9). The generic failed-episode reachability metric is inapplicable because all 1,321 generated episodes solve the task; the build gate now records this explicitly. G0 passes the loss-mask threshold, though its all-zero regret provides no action-ordering signal. Artifacts are under ignored `outputs/puzzle/p13-4x4-task2-100-generated/{g0,gr-rho1,gr-rho09}/`.

P13 witness checks pass free-space arrival (192/200 = 0.96) and direct R landing (199/200 = 0.995). The matched 500-start replay witness for G-R ρ=1 achieved 3% overall success, zero-bridge success 4/12 = 0.333, and 1–3 bridge retention 0.275. The matched servo achieved 24.6% overall success, 10/12 zero-bridge success (0.833), 1–3 bridge retention 0.69, 87.1% in-policy R landing, and cost/time Spearman 0.995. The v0.5 graph, normal-witness, and connection-accuracy gates pass. Paths with four or more bridges remain difficult (44/368 succeeded). The small 12-start zero-bridge subgroup makes its rate imprecise.

The graph-target QC runs use the generated task-2 archive, seed 0, best-of-32, h=5, `online_steps=0`, 300k offline updates, and 50-episode evaluations every 50k. G0-regret completed with success `0.00 / 0.00 / 0.02 / 0.00 / 0.00 / 0.00` at 50k through 300k; its final batch regret Spearman was 0.0463. G-R ρ=1 completed with `0.04 / 0.12 / 0.08 / 0.14 / 0.08 / 0.10` success and final batch Spearman 0.7675. Thus G-R ρ=1 exceeded G0 at each matching checkpoint, but its absolute final success remained 10% in this one-seed, 50-episode evaluation. The CSVs also contain a duplicate 299,999-step final evaluation from the old evaluation condition; it was 0% for G0 and 10% for G-R ρ=1 and is not a scheduled 50k checkpoint. The QC evaluation condition has since been corrected to avoid that duplicate.

The matching G-R ρ=0.9 QC run completed 300k, with evaluations `0.04 / 0.08 / 0.14 / 0.12 / 0.16 / 0.14` at 50k through 300k and final batch regret Spearman 0.6762. Compared with ρ=1, ρ=0.9 finished at 14% versus 10% success; outcomes vary across checkpoints. Both are single-seed, 50-episode evaluations. Its CSV has one evaluation per scheduled checkpoint.

### P13 task 5 interim data and G0 graph

The task-5 P13 generator completed its 1,000,000-transition budget: 1,444 accepted episodes, 14 failed attempts, all 1,444 episodes reached the goal, and 745 continued after goal. QC loaded the full archive; inspection found 30,032 board-change transitions, all matching a single button press, and sidecar/bitfield agreement on the first 1,000 rows. It visited 3,924/65,536 board configurations. The observed task-5 configuration graph is connected; shortest data and optimal press counts are both 7, and every generated episode individually solved the task. Dataset SHA256: `4bb36a8c75c9338a3e57c41b5c598ba7a60d682bf596b90e6a7799961b4e5628`.

The h=5 G0-regret graph and targets have built successfully from this archive (graph hash `f23ecb7921eb531690f52a13c36e75a6edd6d9a7c834a29e2f39dbb39ba73528`). It contains 1,001,444 state occurrences, 1,000,000 temporal edges, 47,256 goal nodes, and no bridges. Reachability is 55.25%; regret-loss coverage is 55.29%, and all eligible regrets are zero. All 1,444 episodes solve, so failed-episode reachability is inapplicable. This G0 build passes its configured training gate but, as for task 2, provides no regret ranking signal.

The task-5 G-R ρ=1 and ρ=0.9 graphs and targets have built successfully. They share 1,351,970 R bridges (2,351,970 total edges), τ_real p50/p95/max `5/9/20` steps, zero τ_real < τ_kin violations, 99.69% reachability, 99.81% regret-loss coverage, and 0.185% valid non-success cap fraction. For ρ=1, bridge use is 91.59%, regret p90 12, and zero-regret fraction 60.43% (graph hash `ed7645a95111240861187bc72ac2c82baf1d0445eba8909da489cc80155c5bfb`). For ρ=0.9, bridge use is 78.57%, regret p90 11.52, and zero-regret fraction 73.80% (graph hash `5672bd2ad835943606c3e50fd11884f14042492eabd48c82ee448aa0b403c6cd`). Both build gates pass; failed-episode reachability is inapplicable because every episode solves. Artifacts are under `outputs/puzzle/p13-4x4-task5-100-generated/{gr-rho1,gr-rho09}/`. The task-5 free-space controller reached its target in 195/200 trials (0.975; gate ≥0.9), and direct R-bridge landing succeeded on 195/200 sampled edges (0.975; gate ≥0.8). The matched 500-start replay completed at 4.2% overall success; zero-bridge success was 11/17 = 0.647, 1–3 bridge success 0.093, retention 0.143, and cost/time Spearman 0.970. Most paths had 4+ bridges (375/500), with no successes in that group. The matched servo achieved 23.2% overall success, 13/17 = 0.765 zero-bridge success, 0.666 retention on 1–3 bridge paths, 0.878 in-policy R landing, and cost/time Spearman 0.994. The 4+ bridge group remained difficult; failures were mainly passive drift. All v0.5 graph, witness-normal, and connection-accuracy gates pass. Task-5 G0-regret reached 85k after its 50k evaluation scored 0/50; its 100k evaluation is not due yet. G-R ρ=1 is at 50k with its first evaluation in progress. G-R ρ=0.9 QC is queued.

## P0 and remaining experiments

The prescribed 1M-step TD QC baseline was attempted on standard 3×3 task 3 with best-of-32, h=5, sparse reward, and seed 0. The first attempt reached 185k updates; its 100k evaluation checkpoint reports success 1.0, then a CUDA launch failure stopped it before 1M. A retry with 100k save intervals reached 30k updates, but a second unclean host reboot at 15:47 stopped it before the first model checkpoint. A later retry reached 25k with a 100k save interval; it was stopped before its first checkpoint and restarted from step 0 with both saves and evaluations every 50k. The replacement run group is `v06-p0-qc-task3-save50k`. It was intentionally stopped at 217,582 updates when the user redirected the work to graph-target QC; the latest saved checkpoint is 200k. Checkpoint evaluations at 50k / 100k / 150k / 200k were 0.10 / 1.00 / 0.98 / 1.00 over 50 episodes. These checkpoints remain under `third_party/qc/exp/qc/v06-p0-qc-task3-save50k/`. Public plot data reports 1M QC success 1.0 for both selected 3×3 tasks (task 3: CI [1,1]; task 4: CI [1,1]); corresponding FQL values are 0.9748 and 0.9847. No reference-only TD or FQL run is being continued.

The 3×3 task-3 G-R graph passes graph and witness gates, but its 10k regret-learning Spearman is 0.6102, below the 0.7 fit gate. The 300k graph-target follow-up, task-4 graph-target QC, and task-3 G0-cost graph-target QC remain outstanding; standard-data G0-regret is excluded because its loss mask is only about 3%. Standard 4×4 task-2 graphs reach only 46 of 1,000,000 rows. P13 task-2 G0 and G-R graph-build, loss-mask, and simulator-witness gates pass. Task-2 G0-regret QC completed 300k with 2% at its best checkpoint (150k) and 0% at 300k; G-R ρ=1 completed at 10% at 300k; G-R ρ=0.9 completed at 14% at 300k. P13 task-5 data, graph builds, free-space, direct landing, and matched replay/servo witnesses are complete and pass the v0.5 gates. Task-5 G0-regret has saved 50k; G-R ρ=1 has begun. Task-5 G-R ρ=0.9 QC is queued. The user requested graph-target QC runs and no further reference-only TD/FQL runs. Save and evaluation intervals are 50k steps.

## Verification

- Core suite: `112 passed, 2 skipped`; one expected unreachable-goal warning from a synthetic regret-mask fixture.
- QC graph-target suite: `2 passed`; upstream AnyIO and TensorFlow Probability/JAX deprecation warnings only.
- `compileall` and `git diff --check` passed before the latest all-success-episode gate/evaluator adjustment. The updated P13 graph build and 500-start servo witness completed; the full suite has not been rerun after that adjustment.
- The one-update QC fixed-dataset smoke completed with graph mode and logged the new regret/mask metrics.
