# Graph Critic v0.7 implementation and validation

Date: 2026-11-06

This report records the v0.7 implementation and measured model/graph checks. Training artifacts, graphs, targets, and witness outputs are under ignored `outputs/`. QC remains stopped and has not been launched for v0.7.

## Implementation, step by step

1. **State and episode split.** `prepare_puzzle_model_data` builds the learning vector from the physical Puzzle `StateSchema` robot readout (`r`, 19 continuous values) and the 16 exact binary button states. It splits episodes 90/10 with a fixed seed, fits continuous state mean/std, five-action delta scale, and one-step q90 endpoint tolerances using training episodes only. The P13 task-2 data produced 1,001,321 aligned states, 1,189 training episodes, and 132 held-out episodes.
2. **Policy and transition models.** `experiments/train_models.py` implements the prescribed four-layer, width-512 GELU MLP with LayerNorm, Adam at `3e-4`, batch 256, and 300k steps by default. The policy uses 10-step goal-conditioned flow matching over five-action chunks. The transition model has separate continuous-delta and button-logit heads; its delta scale is fit from training-split five-action targets. JAX/Flax execution lives in `experiments/v07_runtime.py`, keeping `graphcritic` importable without the ML stack. Both checkpoints and held-out reports are saved every 50k steps. Reports include one-chunk coordinate errors in ε units, all-button and changed-button accuracy, 1–4 chunk cumulative rollout error, and policy endpoint rates by `j=1..20`.
3. **Candidate and arrival checks.** `graphcritic.bridges.check.make_and_evaluate` draws 64 policy samples per candidate, rolls each through the single transition model for up to four chunks, and accepts an arrival only when every continuous coordinate is within ε and all button bits match exactly. It retains the selected action chunks for later simulation checks. Candidates use exactly k=32 cross-episode nearest neighbors in the normalized state vector, widening the KD-tree query until enough eligible neighbors are found. Per M1, successful dataset states are valid destinations and ordinary zero-cost goal vertices.
4. **Sequential insertion.** `IterativeBridgeGraph` initializes G0 reverse-Dijkstra costs in steps. It selects up to 2,048 optimistic candidates per round, rechecks each before model evaluation after preceding insertions, requires at least five steps of improvement, and applies the approved `B_max=2` path limit. A decrease-only reverse predecessor update propagates both costs and bridge counts. Unreachable sources can connect to finite-cost targets, matching the pseudocode's `∞ - finite` improvement case. The loop stops after a round adds no edge.
5. **Comparison and artifacts.** `experiments/eval_bridges.py` supports the sequential method and bulk-insertion ablation, saves a schema-valid graph plus h=5 regret targets and selected action chunks, reports reachability against an optional G-R reference graph, and directly executes sampled accepted chunks for Q1. `experiments/eval_graph_policy.py` accepts matched-start witnesses and reports the prescribed one-sided binomial test against the measured V0 success rate, plus a paired McNemar comparison and a bootstrap cost/time-rank comparison. Starts unreachable in the candidate graph count as failures. Simulator checks use saved sidecars; no simulator is accessed during graph building or model training.
6. **Dataset separation (M2).** P13 and standard 4×4 each have their own episode split, normalization, ε, delta scale, and π/f checkpoints. The graph evaluator reads both model metadata files and rejects mismatched π/f IDs, mismatched model dimensions, or a checkpoint dataset ID different from its transition dataset.

## Checks completed

- The v0.7 graph/model tests pass, including M1 goal-destination cost semantics and M2 dataset-ID rejection.
- Full repository suite after target-cap, unreachable-start, cache, and Q3-statistic fixes: **133 passed** (three upstream/runtime warnings).
- `compileall` and `git diff --check` pass for the new code.
- P13 and standard task-2 π/f training each completed 300k updates with checkpoints and held-out reports every 50k. Summary and graph experiments are recorded below as they complete.
- A prior small smoke run used 10 candidate source states and found all 32 cross-episode neighbors for each (320 pairs); its untrained model accepted 0 edges. It is only a code-path smoke check.

## v0.6 G-R ρ=0.9 reference (V0)

The prescribed 500-start task-2 witness is fixed to start-selection hash `fa3dff0344f471d7228a1f16c4e3a61d578c8a23ddfeef59e5f1d978080dd7dd` across the paired replay and servo runs. Direct R landing succeeded on 199/200 edges (`0.995`). Replay success was 44/500 (`0.088`); servo success was 192/500 (`0.384`). Their zero-bridge rates were `0.486` and `0.700`, respectively. These are the v0.6 reference measurements needed by v0.7; they do not measure Graph Critic v0.7.

## Stopped previous QC run

At the user's direction, the unrelated v0.6 task-5 G-R ρ=0.9 QC run was stopped. It had saved model parameters through 250k updates; the last completed evaluation in `eval.csv` was at 200k. It was not resumed and is not evidence for v0.7.

## v0.7 experiments

### R1/R2: model training

Both prescribed 300k runs completed with checkpoint and held-out evaluation every 50k, seed 0, batch 256. P13 used 1,189 train / 132 held-out episodes; standard used 900 / 100. Their dataset IDs and fitted tolerances are distinct:

| Dataset | Dataset ID suffix | ε raw, first four continuous fields | Checkpoint |
|---|---|---|---|
| P13 task 2 | `...script9ab803500c6b:6f73eb1d…` | `[0.04528, 0.04649, 0.03976, 0.04213]` | `outputs/puzzle/p13-4x4-task2-100-generated/v07/p13-task2/step_300000` |
| Standard 4×4 task 2 | `...:396a029c…` | `[0.04556, 0.04639, 0.03971, 0.04218]` | `outputs/puzzle/standard-4x4-task2/v07/standard-task2/step_300000` |

The policy endpoint diagnostic uses 64 held-out source/goal pairs per horizon and 64 sampled chunks per pair; its per-checkpoint j rates have sampling variation. Transition-model summaries:

| Dataset | Step | Changed-button accuracy | Exact board, 1 chunk | Exact board, 4 chunks | π endpoint rate, j=5 |
|---|---:|---:|---:|---:|---:|
| P13 | 50k | 0.9730 | 0.9840 | 0.9320 | 0.4277 |
| P13 | 100k | 0.9784 | 0.9881 | 0.9579 | 0.5513 |
| P13 | 150k | 0.9893 | 0.9906 | 0.9626 | 0.5247 |
| P13 | 200k | 0.9872 | 0.9917 | 0.9694 | 0.5669 |
| P13 | 250k | 0.9890 | 0.9917 | 0.9695 | 0.5923 |
| P13 | 300k | 0.9916 | 0.9915 | 0.9723 | 0.6384 |
| Standard | 50k | 0.9745 | 0.9829 | 0.9335 | 0.4092 |
| Standard | 100k | 0.9833 | 0.9883 | 0.9547 | 0.4651 |
| Standard | 150k | 0.9871 | 0.9908 | 0.9630 | 0.5339 |
| Standard | 200k | 0.9903 | 0.9920 | 0.9681 | 0.6262 |
| Standard | 250k | 0.9871 | 0.9916 | 0.9713 | 0.5693 |
| Standard | 300k | 0.9852 | 0.9921 | 0.9725 | 0.5981 |

All coordinate-wise mean/p50/p90 ε errors, 1–4 chunk coordinate rollouts, changed-button metrics, and j=1…20 endpoint tables are retained in each ignored `training.jsonl`. Correct P13 and standard checkpoint loads were verified, and loading P13 weights for the standard data was rejected by the M2 dataset-ID check.

### R3–R6: graph and simulator checks

#### P13 sequential graph (R3)

The full P13 sequential build completed after about 6h15m and inserted **479** learned bridges into the 1,000,000 temporal-edge G0 graph. The saved graph hash is `be9017075958dc16018b58178f28c3887e02961e22b0a3ba710e3a8a5e7908ed`; reachability is **74.893%**. The initial run saved the graph, then stopped while making regret targets: the target code treated any finite cost above `c_max_steps=1001` as unreachable. These states had finite costs around 1002–1005 steps, while their five-step endpoints were finite around 997–1000, yielding valid zero regret. Target construction and diagnostics now distinguish finite-over-cap costs from truly unreachable states. The recovered h=5 targets have 99.472% valid rows, zero property-1 violations, 0.00169% valid non-success capped targets, and 0.7496 regret loss-mask coverage. Regret p90 is zero.

The original action-plan array was not saved because target finalization stopped first. A fresh π/f sampling pass regenerated a valid plan for **452/479** graph edges; its direct simulator Q1 check landed on **175/200 = 0.875**, passing the ≥0.8 gate. The run and regenerated Q1 sample are under `outputs/puzzle/p13-4x4-task2-100-generated/v07/graphcritic-sequential/`.

#### P13 bulk graph, Q1, and Q4

The bulk ablation used the same 32,042,272 cross-episode kNN pairs and selected the top 2,048 G0-optimistic candidates. The model predicted an acceptable arrival for **424/2,048 = 20.70%**; all 424 also met the cost-improvement threshold and were inserted together in round 0. None of these accepted edges ended at a success node. The graph has 424 learned bridges and 73.541% node reachability (736,386/1,001,321), versus the P13 G-R reference's 0.0046%. Its maximum shortest-path bridge count is 1. The deterministic recovery re-evaluation matched all 424 saved edges and action chunks exactly after strict JSON output rejected the initial infinite-improvement record from an unreachable source.

The bulk Q1 witness landed on **168/200 = 0.840**, passing the ≥0.8 gate. The graph, regret targets, candidate cache, action chunks, Q1 record, and recovered insertion report are under `outputs/puzzle/p13-4x4-task2-100-generated/v07/graphcritic-bulk/`.

#### P13 matched witnesses (R5) and insertion comparison (Q4)

Both controllers used the V0 start hash `fa3dff0344f471d7228a1f16c4e3a61d578c8a23ddfeef59e5f1d978080dd7dd`.

| Controller | Graph Critic | V0 G-R ρ=0.9 | One-sided binomial p vs V0 | Paired McNemar p | Cost/time Spearman (common successes) |
|---|---:|---:|---:|---:|---:|
| Servo | 169/500 = 0.338 | 192/500 = 0.384 | 0.9852 | 0.9653 | 0.9976 (V0: 0.9686) |
| Replay | 74/500 = 0.148 | 44/500 = 0.088 | 8.31e-6 | 6.91e-7 | 0.9969 (V0: 0.9786) |

The prescribed Q3 servo success gate **fails** for both graphs: sequential is 169/500 = 0.338 (p=0.9852 vs V0's 0.384), and bulk is 167/500 = 0.334 (p=0.9909). Servo success on zero-bridge paths was 164/356 = 0.461 for both; on 1–3 bridge paths it was 5/22 = 0.227 sequential and 3/12 = 0.250 bulk. Unreachable starts were 122 sequential and 132 bulk. Sequential failures were active tracking 120, passive drift 90, and graph-unreachable 122; bulk failures were active tracking 116, passive drift 86, and graph-unreachable 132. Cost/time Spearman on common successes was 0.9976 sequential and 0.9976 bulk.

Q4 does not support sequential insertion over bulk: servo success was 33.8% vs 33.4%, with 3 sequential-only successes and 1 bulk-only success (one-sided paired McNemar p=0.3125). Replay success was 14.8% for each graph, with 0/22 vs 0/12 successes on 1–3 bridge paths. Both graphs pass Q1 but fail the servo Q3 improvement criterion. Standard-task Q2/Q1 is in progress. QC has not been started.

#### Standard 4×4 G-R ρ=0.9 reference (R6 baseline)

The matched standard task-2 reference graph was built from the standard dataset with `configs/graph/puzzle_gr_rho09.yaml`. It has 1,001,000 nodes, 1,000,000 temporal edges, and 741,535 R bridges. Only 46 nodes are reachable (`0.0046%`); no state from a failed episode is reachable. Its training gate fails on failed-episode reachability, as expected from the weak reference connectivity. This is the required G-R ρ=0.9 comparator for the pending standard-task Graph Critic Q2 result, not a candidate selected for training. Artifacts are under `outputs/puzzle/standard-4x4-task2/gr-rho09/`; graph hash `7e5c0668b6cbf0cc216d854362f8cd8dc330d447eb5b1201b9baea31a5bd9125`.
