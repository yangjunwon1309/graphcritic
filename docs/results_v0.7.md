# Graph Critic v0.7 implementation and validation

Date: 2026-11-06

This report records the v0.7 implementation and code checks. No full v0.7 model training, graph sweep, or QC run is included here. The one-step model smoke below checks the execution path only; its random checkpoint is not a benchmark model.

## Implementation, step by step

1. **State and episode split.** `prepare_puzzle_model_data` builds the learning vector from the physical Puzzle `StateSchema` robot readout (`r`, 19 continuous values) and the 16 exact binary button states. It splits episodes 90/10 with a fixed seed, fits continuous state mean/std, five-action delta scale, and one-step q90 endpoint tolerances using training episodes only. The P13 task-2 data produced 1,001,321 aligned states, 1,189 training episodes, and 132 held-out episodes.
2. **Policy and transition models.** `experiments/train_models.py` implements the prescribed four-layer, width-512 GELU MLP with LayerNorm, Adam at `3e-4`, batch 256, and 300k steps by default. The policy uses 10-step goal-conditioned flow matching over five-action chunks. The transition model has separate continuous-delta and button-logit heads; its delta scale is fit from training-split five-action targets. JAX/Flax execution lives in `experiments/v07_runtime.py`, keeping `graphcritic` importable without the ML stack. Both checkpoints and held-out reports are saved every 50k steps. Reports include one-chunk coordinate errors in ε units, all-button and changed-button accuracy, 1–4 chunk cumulative rollout error, and policy endpoint rates by `j=1..20`.
3. **Candidate and arrival checks.** `graphcritic.bridges.check.make_and_evaluate` draws 64 policy samples per candidate, rolls each through the single transition model for up to four chunks, and accepts an arrival only when every continuous coordinate is within ε and all button bits match exactly. It retains the selected action chunks for later simulation checks. Candidates use exactly k=32 cross-episode nearest neighbors in the normalized state vector, widening the KD-tree query until enough eligible neighbors are found; success nodes are never candidate destinations.
4. **Sequential insertion.** `IterativeBridgeGraph` initializes G0 reverse-Dijkstra costs in steps. It selects up to 2,048 optimistic candidates per round, rechecks each before model evaluation after preceding insertions, requires at least five steps of improvement, and applies the approved `B_max=2` path limit. A decrease-only reverse predecessor update propagates both costs and bridge counts. Unreachable sources can connect to finite-cost targets, matching the pseudocode's `∞ - finite` improvement case. The loop stops after a round adds no edge.
5. **Comparison and artifacts.** `experiments/eval_bridges.py` supports the sequential method and bulk-insertion ablation, saves a schema-valid graph plus h=5 regret targets and selected action chunks, reports reachability against an optional G-R reference graph, directly executes sampled accepted chunks for Q1, and can run a matched 500-start Q3 path witness using a supplied G-R ρ=0.9 baseline. The simulator checks use saved sidecars and the trained action chunks; no simulator is accessed during graph building or model training.

## Checks completed

- The v0.7 graph/model tests pass: **8 passed**. They cover incremental decreases against full recomputation, `B_max`, disconnected-source insertion, sequential rechecks before model evaluation, exact cross-episode kNN degree, and the model-check arrival path.
- Full repository suite: **123 passed** (three pre-existing upstream/runtime warnings).
- `compileall` and `git diff --check` pass for the new code.
- A one-update P13 task-2 training smoke completed and wrote checkpoints that the evaluator loaded. It exercised one optimizer update and held-out metric generation; the random initialization's metrics are not reported as learning results.
- A small end-to-end graph/target smoke used 10 candidate source states and found all 32 cross-episode neighbors for each (320 candidate pairs). The untrained smoke model accepted **0 edges**. The 1,001,321-node / 1,000,000-edge graph and 994,716 valid h=5 regret targets loaded and validated against their hashes. No simulator witness was run. This is not a full graph build or a Q1 measurement.

## v0.6 G-R ρ=0.9 reference (V0)

The prescribed 500-start task-2 witness is fixed to start-selection hash `fa3dff0344f471d7228a1f16c4e3a61d578c8a23ddfeef59e5f1d978080dd7dd` across the paired replay and servo runs. Direct R landing succeeded on 199/200 edges (`0.995`). Replay success was 44/500 (`0.088`); servo success was 192/500 (`0.384`). Their zero-bridge rates were `0.486` and `0.700`, respectively. These are the v0.6 reference measurements needed by v0.7; they do not measure Graph Critic v0.7.

## Stopped previous QC run

At the user's direction, the unrelated v0.6 task-5 G-R ρ=0.9 QC run was stopped. It had saved model parameters through 250k updates; the last completed evaluation in `eval.csv` was at 200k. It was not resumed and is not evidence for v0.7.

## Not measured yet

- Full 300k policy and transition training on P13 task 2 (and standard task 2 if selected).
- Full k=32 candidate generation and sequential/bulk graph builds.
- v0.7 Q1 accepted-edge landing rate, Q2 standard 4×4 reachability comparison, Q3 matched path result and significance tests, and Q4 sequential-versus-bulk comparison.
- QC. The v0.7 guide allows QC only after Q1 and Q3 pass; no QC was launched.

The implementation therefore makes the specified experiments runnable, but it has not yet established that the new method resolves the earlier graph failures.
