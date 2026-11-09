# Graph Critic v0.8 window-5 runbook

**Implementation code commit:** `808576c7e4670c08ad4e7401a6d99198c16dfd40`

This runbook contains the v0.8 **five-frame window** experiment and its downstream checks only. The quasimetric training and graph judging use window length 5; the constraint horizon is 5 and ε is 1.25. There are no state-input quasimetric runs or comparisons in this procedure.

The v0.8.1 focused checks and full test suite were run before commit. Model training, benchmark graph builds, and the board report remain user-run. Simulator witnesses and QC are outside this runbook.

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

The per-bridge penalty is applied to accepted bridge costs before layered reverse Dijkstra. It does not affect the quasimetric acceptance rule (`dθ ≤ 20`), optimistic candidate filtering, or the stored bridge distance `dist`, which remains `dθ`. Λ is recorded as `bridge_lambda_steps` in graph metadata. The zero setting preserves the v0.8 graph hash; the focused unit test checks the invariant and checks that a positive Λ changes edge cost while retaining raw distance.

Run the focused implementation checks:

```bash
$PY -m pytest tests/test_v08_1.py tests/test_puzzle.py tests/test_quasimetric_v08.py -q
```

Implementation verification: focused tests **32 passed**; the full repository suite **163 passed** with three existing upstream/runtime warnings. The benchmark graph builds and board audit below were not run by Codex.

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
