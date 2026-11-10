# Graph Critic v0.8 window-5 runbook

**Implementation code commit:** `72c6dca8b40e62fff60c1505f475c5e57b1e3e0d`

This runbook contains the v0.8 **five-frame window** experiment and its downstream checks only. The quasimetric training and graph judging use window length 5; the constraint horizon is 5 and ε is 1.25. There are no state-input quasimetric runs or comparisons in this procedure.

The v0.8.1 focused checks and full test suite were run before commit. Model training, benchmark graph builds, and the board report remain user-run. The v0.8.2 rule that oracle graphs do not produce QC targets has the explicitly approved v0.8.3 upper-bound exception in §9. Simulator witnesses remain outside this runbook.

## 1. Prepare the pinned worktree

The CLI error `unrecognized arguments: --window --constraint-horizon --epsilon` occurs when running an older checkout. The implementation at the code commit above defines all three options. Update an existing run worktree as well as creating one when it is absent:

```bash
cd /home/url/graphcritic
# Run this only if /home/url/gc-run-v08 is not already a worktree:
git worktree add --detach /home/url/gc-run-v08 quasi_develop
git fetch origin quasi_develop
git -C /home/url/gc-run-v08 checkout --detach origin/quasi_develop
git -C /home/url/gc-run-v08 rev-parse HEAD                    # record the pushed run commit
git -C /home/url/gc-run-v08 status --short
```

`status --short` should print nothing. The detached worktree uses the latest commit pushed to `quasi_develop`, including the runbook update. Then set the run paths:

```bash
cd /home/url/gc-run-v08
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D=$ROOT/outputs/puzzle/p13-4x4-task2-100-generated
COMMON="--transitions $D/g0/transitions.npz --sim-state $D/g0/sim_state.npz"
```

All generated data, checkpoints, embedding arrays, and candidate caches below stay under ignored `outputs/`.

## 2. Run the code checks

```bash
$PY -m pytest tests/test_quasimetric_v08.py -q
```

The suite checks chunk-horizon loss, window padding and boundary-node selection, batched cached-node judging, IQE embedding/head equivalence (maximum absolute error `≤1e-5`), and—when a GPU is available—float32 cache reuse and cached/direct pair agreement (`≤1e-4`). Stop and report the failure output if a check fails.

## 3. Train the five-frame quasimetric

This is a fresh 20k-step run with F1–F3, a five-frame input window, a five-step constraint horizon, and ε=1.25:

```bash
$PY experiments/train_quasimetric.py $COMMON \
  --window 5 --constraint-horizon 5 --epsilon 1.25 \
  --activation gelu --layernorm --grad-clip 1.0 --dual adam \
  --steps 20000 --log-every 1000 --save-every 10000 \
  --out $D/v08/check/window5_chunk
```

**Expected time:** approximately 12–20 minutes on the configured GPU. Outputs are under `$D/v08/check/window5_chunk/`. Exit code `3` means F3 saved a checkpoint and stopped after a collapse gate fired.

**Check:** continue only if the run completed without an F3 alarm and wrote `step_020000`. Review `train_metrics.jsonl` for random-pair distance, λ, constraint gap, pre-clip gradient norm, per-layer dead-unit fraction, and projector variance.

Evaluate the completed checkpoint on the common five-step chunk metrics:

```bash
$PY experiments/eval_quasimetric_chunk.py \
  --ckpt $D/v08/check/window5_chunk/step_020000 $COMMON
```

The report includes chunk violation `d(x_t,x_{t+5}) > 6.25`, split by chunks with and without internal button changes; `P(d≤j)` for `j=5,10,15,20,40,80`; projector variance; F3 indicators; and triangle violations. Review the output before starting a longer run.

## 4. Build the shared window-5 candidate cache

Build this cache after the five-frame checkpoint has passed the checks above. Candidate pairs use k=32, cross-episode 175D window features, boundary sources, and success states as eligible destinations.

```bash
$PY experiments/build_candidate_cache.py \
  --g0-graph $D/g0/graph.npz \
  --transitions $D/g0/transitions.npz --sim-state $D/g0/sim_state.npz \
  --out $D/v08/candidates_window5.npz \
  --window 5 --knn 32 --parity-size 50000 --seed 0
```

The compressed cache and JSON report are `$D/v08/candidates_window5.npz` and `$D/v08/candidates_window5.json`. Candidate generation may take several minutes to tens of minutes depending on CPU/GPU throughput. Check that `knn_parity.mismatched_rows` is `0`; record actual source, destination, and success-destination counts. Stop if parity fails.

## 5. Check embedding-cache equivalence

Run this on the completed five-frame checkpoint. Add `--candidate-cache` to check the actual kNN pairs from section 4 instead of random cross-episode pairs.

```bash
$PY experiments/check_embedding_cache.py \
  --ckpt $D/v08/check/window5_chunk/step_020000 $COMMON --pairs 10000 \
  --candidate-cache $D/v08/candidates_window5.npz
```

The first cache build takes several minutes on GPU. The float32 embedding table uses `4 × node_count × 2048` bytes: about 1.6 GB for 200k nodes. This dataset produced 235,437 cache nodes, or about 1.93 GB. The default cache path is beside the checkpoint under `embedding_cache/`; use `--embedding-cache PATH` to choose another ignored output path.

**Pass criteria:** `max_absolute_distance_difference ≤1e-4`, `bridge_decision_mismatches_at_K20 = 0`, and `pass = true`. Record `embedding_seconds`, `cached_pair_seconds`, and direct pair time. Do not build a quasimetric graph if this check fails.

## 6. Build and compare judged graphs

After the five-frame checkpoint and cache checks pass, build the quasimetric graph using the shared five-frame candidate cache and the prescribed G-R reference graph:

```bash
$PY experiments/build_judged_graph.py --judge quasimetric \
  --g0-graph $D/g0/graph.npz --reference-graph $D/gr-rho09/graph.npz \
  --transitions $D/g0/transitions.npz --sim-state $D/g0/sim_state.npz \
  --checkpoint $D/v08/check/window5_chunk/step_020000 \
  --candidate-cache $D/v08/candidates_window5.npz \
  --embedding-pair-batch-size 4096 \
  --out $D/v08/quasimetric
```

The 4,096-pair batch bounds the GPU gathers to about 32 MiB per endpoint for 2,048D float32 embeddings. Reduce it further if the GPU still runs out of memory; the existing embedding cache is reused.

If the compatible v0.7 flow checkpoint is available, evaluate it on the same candidate cache for the judge comparison:

```bash
$PY experiments/build_judged_graph.py --judge flow --candidate-window 5 \
  --g0-graph $D/g0/graph.npz --reference-graph $D/gr-rho09/graph.npz \
  --transitions $D/g0/transitions.npz --sim-state $D/g0/sim_state.npz \
  --checkpoint <V07_FLOW_CHECKPOINT> \
  --candidate-cache $D/v08/candidates_window5.npz \
  --out $D/v08/flow

$PY experiments/compare_judges.py \
  --flow-candidates $D/v08/flow/judge_candidates.npz \
  --quasimetric-candidates $D/v08/quasimetric/judge_candidates.npz \
  --out $D/v08/judge_decision_comparison.json
```

Judging millions of pairs is GPU- and checkpoint-dependent. Each graph build writes graph/target artifacts and `result.json` under its output directory. The report includes candidate and bridge counts, reachability against G0/reference, bridge-count distribution, cost improvements, judge/embedding timing, and target validity. `compare_judges.py` reports agreement cells split by whether endpoint board states differ.

Before interpreting agreement, confirm both `optimistic_candidate_sha256` values match. Q1 must be at least 0.8 for both judges before comparing Q3. These commands do not step a simulator; simulator witnesses and QC require a separate approved procedure.

## 7. v0.8.1: bridge Λ and temporary board check

The per-bridge penalty is applied to accepted bridge costs before layered reverse Dijkstra. It does not affect the quasimetric acceptance rule (`dθ ≤ 20`), optimistic candidate filtering, or the stored bridge distance `dist`, which remains `dθ`. Λ is recorded as `bridge_lambda_steps` in graph metadata. The legacy integer `kappa` stores `ceil(dθ + Λ)`, so the graph's `tau_max_steps` validation cap is `ceil(20 + Λ)` while the judge's distance threshold stays 20. The zero setting preserves the v0.8 graph hash; unit tests check this and the positive-Λ cost/cap behavior.

Run the focused implementation checks:

```bash
$PY -m pytest tests/test_v08_1.py tests/test_puzzle.py tests/test_quasimetric_v08.py -q
```

Implementation verification after the Λ cap fix: v0.8.1 regression tests **10 passed**; the full repository suite **164 passed** with three existing upstream/runtime warnings. The benchmark graph builds and board audit below were not run by Codex.

Then build both requested graphs. Each run should take about 1–2 minutes. The 4,096-pair embedding batch limits GPU memory use; reduce it to 1,024 if needed.

```bash
for LAM in 0 10.5; do
  $PY experiments/build_judged_graph.py --judge quasimetric \
    --g0-graph $D/g0/graph.npz --reference-graph $D/gr-rho09/graph.npz \
    --transitions $D/g0/transitions.npz --sim-state $D/g0/sim_state.npz \
    --checkpoint $D/v08/check/window5_chunk/step_020000 \
    --candidate-cache $D/v08/candidates_window5.npz \
    --embedding-pair-batch-size 4096 \
    --bridge-lambda-steps "$LAM" \
    --out "$D/v08/quasimetric_lam$LAM"
done
```

The board check is puzzle-specific, offline, and does not step a simulator. It reads each graph's retained `kind == 1` edges and raw `dist`, maps node representatives through `rep_ref`, and uses the existing GF(2) minimum-press solver. The data-derived lower bound is `L(0)=0` and `L(n_p)=(n_p−1)m+1` for `n_p≥1`, where `m` is the p1 interval between consecutive within-episode board changes. A bridge is flagged when `dθ < L(n_p)` or the board difference has no GF(2) solution. A pass is not evidence that a bridge is executable; the check is only a lower-bound rejection test.

```bash
$PY experiments/tmp_board_check.py \
  --graph $D/v08/quasimetric_lam0/graph.npz \
          $D/v08/quasimetric_lam10.5/graph.npz \
  --transitions $D/g0/transitions.npz --sim-state $D/g0/sim_state.npz \
  --out $D/v08/board_check.json
```

Report these values side by side. The previous Λ=0 result is included for comparison; fill the rerun and Λ=10.5 columns from each graph's `result.json` and `board_check.json`.

| Measure | Λ=0, previous | Λ=0, rerun | Λ=10.5 |
|---|---:|---:|---:|
| Shortest-path bridge count distribution (0 / 1 / 2) | 10.3% / 16.5% / 73.1% | — | — |
| Retained bridge count | 124,147 | — | — |
| Reachable-state fraction | 99.9% | — | — |
| Originally reachable states with lower `C` | 85.6% | — | — |
| Retained bridges to success states | 2,669 | — | — |
| Board effort `n_p` distribution (0 / 1 / 2 / 3 / ≥4 / ∞) | — | — | — |
| Impossible bridge count / fraction | — | — | — |

Also report the board-change interval sample count and its minimum, p1, and p5, plus impossible rates by `n_p`. Expected direction: Λ=10.5 reduces the two-bridge path share and the fraction of originally reachable states with lower `C`, while keeping reachability close to Λ=0. A high board-impossibility rate indicates that the quasimetric judge itself needs further review, regardless of Λ.

## 8. v0.8.2: separate object event and motion judges

This extension keeps the v0.8 candidate cache, optimistic `C0` filter, layered Dijkstra, path-pruned bridges, regret targets, and per-bridge `dθ + Λ` cost. It adds two object-only judges. Puzzle object frames have 64 values: 16 two-way button one-hots, 16 press depths, and 16 press velocities. The continuous 32 coordinates use training-split statistics; five-frame windows are 320D.

Movement labels use raw press-depth displacement only: `δ_k(t)=|p_k(t+5)-p_k(t)|`, `ε_k=0.05×q99.9(δ_k)`, and `moved(t)=1` if any `δ_k>ε_k` or a button bit changes. Velocity remains in the network input but does not set labels. Moved chunk starts are grouped into contact intervals; gaps containing fewer than five frames merge. Each event constraint is the pair of stationary object windows ending at `a` and `b+9`, covering frames `a−4…a` and `b+5…b+9` around contact interval `[a,b]`. `T_I` is the p1 interval between contact starts.

The trainer applies the same QRL objective, random independent-window maximization term, Adam dual, GELU/LayerNorm, gradient clipping, and collapse checks as v0.8. Event constraints use one pair per valid contact interval with cost 1 and ε=0.25; movement constraints use all five-frame chunks with cost `5×moved(t)` and ε=1.25. The model is encoder `(256,256,128)`, projector `(256,512)`, IQE `16×32`, Adam `5e-4`, batch 4,096, and 20k updates. A D0 label audit runs before training; the trainer repeats the gate and stops if it fails. New checkpoints use label version `contact-segment-position-q999-v2`; older object checkpoints are rejected by the graph builder.

Run the implementation checks first:

```bash
$PY -m pytest tests/test_object_quasimetric_v082.py tests/test_v08_1.py \
  tests/test_puzzle.py tests/test_quasimetric_v08.py -q
```

Implementation checks and measured D0/D2/D2.5 results are recorded in [results_v0.8.2b.md](results_v0.8.2b.md). D2.5 failed, so the gated D3/D4 experiments were not run.

D1 builds the puzzle-only exact GF(2) upper-bound diagnostic. This graph is marked `qc_eligible: false`, stores `oracle_filter` metadata, and intentionally writes no QC target artifact:

```bash
BUILD="--judge quasimetric --g0-graph $D/g0/graph.npz \
  --reference-graph $D/gr-rho09/graph.npz $COMMON \
  --checkpoint $D/v08/check/window5_chunk/step_020000 \
  --candidate-cache $D/v08/candidates_window5.npz \
  --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5"

$PY experiments/build_judged_graph.py $BUILD \
  --diagnostic-oracle-np-max 1 --out $D/v08/obj_oracle_np1
```

D0 audits the contact labels before learning. Continue only if `gate_passed` is true. It writes contact/board event counts, one-toggle purity, uncovered board changes, p1/p5 `T_I`, moved fraction, contact lengths, and thresholds.

```bash
$PY experiments/check_object_labels.py $COMMON --window 5 \
  --out $D/v08/object_labels_check_v082b.json
```

D2 fits new event and motion models from random initialization; do not resume the v0.8.2 checkpoints. The trainer repeats the D0 gate before the first optimizer update. The event model uses contact-boundary window pairs, and the motion model uses the corrected position-based moved labels.

```bash
for COST in events motion; do
  $PY experiments/train_object_quasimetric.py $COMMON \
    --cost "$COST" --window 5 --steps 20000 \
    --log-every 1000 --save-every 10000 \
    --out "$D/v08/object2_$COST"
done
```

Check both directories for `step_020000`, `heldout.jsonl`, finite collapse metrics, and the D0 report embedded in their metadata. The motion report compares `mψ` with moved-chunk count × 5; the event report checks held-out contact constraints.

D2.5 evaluates the event model on the exact shared candidate pairs before any D3 graph build. Proceed only if the `n_p=0,1,≥2` score medians increase in that order and the false-positive rate among `n_p>1` (including unreachable boards) is at most 1%.

```bash
$PY experiments/tmp_board_check.py --candidates-only \
  --candidate-cache $D/v08/candidates_window5.npz \
  --object-event-checkpoint $D/v08/object2_events/step_020000 \
  $COMMON --out $D/v08/e_phi_check_v082b.json
```

D3 applies `dθ≤20`, `mψ≤20`, and `eφ≤E_max+0.5` as an AND. With `--event-max auto`, the builder reads the corrected contact-start `T_I=p1` and sets `E_max=ceil(20/T_I)`. Distances `mψ` and `eφ` affect acceptance only; stored bridge distance and cost remain `dθ` and `dθ+Λ`.

```bash
$PY experiments/build_judged_graph.py $BUILD \
  --object-event-checkpoint $D/v08/object2_events/step_020000 \
  --object-motion-checkpoint $D/v08/object2_motion/step_020000 \
  --event-max auto \
  --oracle-reference-candidates $D/v08/obj_oracle_np1/judge_candidates.npz \
  --out $D/v08/obj2_learned
```

When supplied, the builder verifies that D1 and D3 use the same optimistic candidate pairs in the same order and that their candidate hash matches. It reports D3 recall on D1's exact `n_p≤1, dθ≤20` subset in `result.json`.

D4 adds `eφ` diagnostics to the existing offline board audit. It recomputes event scores for retained bridges and reports their distribution by exact `n_p`, plus the `eφ≤1.5` versus `n_p≤1` table. It does not step a simulator.

```bash
$PY experiments/tmp_board_check.py \
  --graph $D/v08/quasimetric_lam10.5/graph.npz \
          $D/v08/obj_oracle_np1/graph.npz \
          $D/v08/obj2_learned/graph.npz \
  --object-event-checkpoint $D/v08/object2_events/step_020000 \
  $COMMON --out $D/v08/board_check_v082b.json
```

Report the v0.8.1 Λ=10.5, D1, and D3 columns together. Include retained bridges, impossible-bridge fraction, `n_p` distribution, reachability, shortest-path bridge-count distribution, reduced-`C` fraction, and D3's `dθ`/`mψ`/`eφ` individual and joint rejection counts. The D3 guide gates are: impossible fraction ≤1%; at least 90% of D1's `n_p≤1, dθ≤20` candidates retained by D3; and reachability within one percentage point of D1. The oracle graph is a diagnostic upper bound only and must not be used for QC.

## 9. v0.8.3: D1 oracle QC upper bound

This approved experiment is an exception to the v0.8.2 oracle-target prohibition. The oracle target must be described as an upper-bound ablation, never as the learned Graph Critic method. The output graph and target metadata explicitly label it `oracle_upper_bound`; QC also requires `oracle` in its run group. R3b (quasimetric-only QC) and R3c (TD QC) remain approval-gated and are not part of this run.

Run only the two prescribed checks and the R2/R3 commands below. R3 uses P13 task 2, seed 0, sparse rewards, best-of-32, horizon 5, 300k offline updates, zero online updates, 50k evaluation/save intervals, and 50 evaluation episodes.

### R1. Approved code checks

```bash
$PY -m pytest tests/test_quasimetric_v08.py -q
JAX_PLATFORM_NAME=cpu pixi run -e bench pytest -q third_party/qc/tests/test_graph_targets.py
```

### R2. D1 graph and explicit oracle regret targets

Use a new output directory. Without `--write-oracle-targets`, the diagnostic mode continues to omit QC targets.

```bash
BUILD="--judge quasimetric --g0-graph $D/g0/graph.npz \
  --reference-graph $D/gr-rho09/graph.npz $COMMON \
  --checkpoint $D/v08/check/window5_chunk/step_020000 \
  --candidate-cache $D/v08/candidates_window5.npz \
  --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5"

$PY experiments/build_judged_graph.py $BUILD \
  --diagnostic-oracle-np-max 1 --write-oracle-targets \
  --out $D/v08/obj_oracle_np1_qc
```

Check `result.json` for graph reachability, `h5_regret_valid_fraction`, path bridge-count distribution (0/1/2), h=5 regret p90, and zero-regret fraction. Compare reachability and loss coverage with the v0.6 G-R reference values of 99.80% and 99.90%. The target's `config.dataset_id` is passed to QC below; the target JSON also records `target_provenance`.

### R3. D1 oracle target QC run

```bash
RUN_ROOT="$(git rev-parse --show-toplevel)"
cd "$RUN_ROOT/third_party/qc"
GRAPH_DATASET_ID="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' \
  "$D/v08/obj_oracle_np1_qc/targets.json")"

MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$RUN_ROOT:$RUN_ROOT/third_party/qc" \
  "$PY" main.py \
  --run_group=v083-p13-d1-oracle \
  --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --agent.critic_mode=graph --agent.graph_target_kind=regret \
  --env_name=puzzle-4x4-play-singletask-task2-v0 \
  --ogbench_dataset_path="$D/train.npz" \
  --sparse=True --horizon_length=5 \
  --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 \
  --graph_targets="$D/v08/obj_oracle_np1_qc/targets.npz" \
  --graph_dataset_id="$GRAPH_DATASET_ID" --seed=0 \
  --save_dir="$D/v08/qc"
```

Before R3 starts, QC prints the resolved provenance and rejects an oracle target if `run_group` omits `oracle`. `flags.json` and W&B config retain provenance; `offline_agent.csv` and `eval.csv` include `graph_target_provenance`. Compare the six success rates and final batch regret Spearman with v0.6 G-R ρ=0.9 (14% final success, 0.676 final Spearman). A single-seed difference below 10 percentage points is descriptive only. Record the main and QC commit hashes, `git_dirty: false`, target hash, and provenance. Do not run R3b/R3c without approval.
