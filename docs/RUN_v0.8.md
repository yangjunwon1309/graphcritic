# Graph Critic v0.8 recovery and windowed-bridge runs

**Implementation code commit:** `58423b2faf85a38bab1ca0bb8d9e5b97476cb35c`

This runbook covers v0.8 §2.1–2.3, §3.1, and the simulator-free part of §4. It contains user-run commands only. Per §5.3, Codex does not run pytest, training, experiments, or cache creation. Q1/Q3 simulator witnesses and QC are not included: the guide leaves the common diagnostic executor pending approval.

## Prepare a clean, pinned worktree

Use the implementation commit so saved metadata records a clean, reproducible source tree. The commands assume the repository is `/home/url/graphcritic`, the benchmark environment is installed there, and `/home/url/gc-run-v08` is reserved for the run.

```bash
cd /home/url/graphcritic
git worktree add --detach /home/url/gc-run-v08 quasi_develop   # only if this worktree does not already exist
git -C /home/url/gc-run-v08 checkout --detach 58423b2faf85a38bab1ca0bb8d9e5b97476cb35c
git -C /home/url/gc-run-v08 status --short                    # should print nothing
cd /home/url/gc-run-v08

ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D=$ROOT/outputs/puzzle/p13-4x4-task2-100-generated
CK=$D/v08/quasimetric
COMMON="--transitions $D/g0/transitions.npz --sim-state $D/g0/sim_state.npz"
```

All generated data, checkpoints, embedding arrays, and candidate caches below stay under ignored `outputs/`. The interrupted run has P13 state-input checkpoints under `$CK/`; it has no standard-task checkpoint.

## 0. Unit tests

```bash
$PY -m pytest tests/test_quasimetric_v08.py -q
```

**Expected time:** a few minutes, depending on GPU availability for the optional integration checks.

**Check:** all tests pass. The suite checks chunk-horizon loss, window padding and boundary-node selection, batched cached-node judging, IQE embedding/head equivalence (maximum absolute error `≤1e-5`), and—when a GPU is available—float32 cache reuse and cached/direct pair agreement (`≤1e-4`). Stop and send the failure output if any test fails. This command is for the user to run.

## A0. Inspect the existing P13 checkpoints

```bash
$PY experiments/analyze_quasimetric_ckpt.py \
  --ckpt $CK/step_050000 $CK/step_100000 $CK/step_150000 $COMMON
```

**Expected time:** a few minutes. This reads checkpoints and prints held-out diagnostics; it does not train or simulate.

**Check:** inspect `projector_1` dead-unit fraction at step 50k. If it is **greater than 0.95**, run A2/A3 from scratch without `--init-from $CK/step_050000`. Otherwise use the warm-start form below. Record per-layer dead fractions, encoder/projector variances, λ, held-out constraint rates, `P(d≤5 | j=5)`, and triangle violations.

## A1. Reproduce the old setup with collapse monitoring

```bash
$PY experiments/train_quasimetric.py $COMMON --resume-from $CK/step_100000 \
  --activation relu --no-layernorm --grad-clip 0 --dual adam \
  --steps 20000 --log-every 1000 --save-every 10000 --out $D/v08/check/a1_reproduce
```

**Expected time:** about 12 minutes. Training metrics go to `$D/v08/check/a1_reproduce/train_metrics.jsonl`; checkpoints and held-out reports go to the same directory. Exit code `3` means F3 saved an alarm checkpoint and stopped because a collapse gate fired.

**Check:** summarize the 5k rows and alarm step with the command in “Compare A1–A3 logs.” This checks that F3 makes the collapse visible during training.

## A2. GELU, LayerNorm, clipping, Adam dual

If A0 found the step-50k projector dead fraction at or below 0.95:

```bash
$PY experiments/train_quasimetric.py $COMMON --init-from $CK/step_050000 \
  --activation gelu --layernorm --grad-clip 1.0 --dual adam \
  --steps 20000 --log-every 1000 --save-every 10000 --out $D/v08/check/a2_fix123
```

If it was above 0.95, use the same command without `--init-from $CK/step_050000`.

**Expected time:** about 12 minutes. Warm start copies Dense weights, IQE α, and λ; LayerNorm starts as identity and optimizer state is fresh. Exit code `3` means F3 saved an alarm checkpoint and stopped.

## A3. Add projected-SGD dual update

Use the same warm-start decision as A2:

```bash
$PY experiments/train_quasimetric.py $COMMON --init-from $CK/step_050000 \
  --activation gelu --layernorm --grad-clip 1.0 --dual sgd --dual-lr 1.0 \
  --steps 20000 --log-every 1000 --save-every 10000 --out $D/v08/check/a3_fix1234
```

If A0 found the projector dead fraction above 0.95, omit `--init-from $CK/step_050000`.

**Expected time:** about 12 minutes. This adds `λ ← max(0, λ + gap)` with η=1. Exit code `3` has the same F3 meaning.

## Compare A1–A3 logs

```bash
$PY experiments/summarize_qrl_log.py --log $D/v08/check/a2_fix123/train_metrics.jsonl \
  --compare $D/v08/check/a3_fix1234/train_metrics.jsonl $D/v08/check/a1_reproduce/train_metrics.jsonl
```

The command prints 5k-step rows and alarm steps. Send this output, the three `train_metrics.jsonl` files, and the `training.jsonl` held-out reports for interpretation.

### A2/A3 gate

At step 20k a setting passes only if all conditions hold:

- No F3 alarm occurred.
- Random-pair mean `d(s,g)` is not trending downward in the 5k summary.
- Held-out one-step violation `d(s,s′) > 1.25` is at most **0.223**.
- Held-out `P(d≤5 | same-episode j=5)` is at least **0.623**.

If neither A2 nor A3 passes, stop and return the logs. If either passes, report the metrics; the full-run setting is selected after review. A2/A3 results are one-frame measurements and must not be treated as the window/chunk result.

## A5. Five-frame window with a five-step constraint

This is a fresh 20k-step run with F1–F3, a five-frame input window, and the chunk constraint from v0.8 §2.3. It deliberately does not warm-start from the 35D state-input checkpoints.

```bash
$PY experiments/train_quasimetric.py $COMMON \
  --window 5 --constraint-horizon 5 --epsilon 1.25 \
  --activation gelu --layernorm --grad-clip 1.0 --dual adam \
  --steps 20000 --log-every 1000 --save-every 10000 --out $D/v08/check/a5_window_chunk
```

**Expected time:** approximately 12–20 minutes on the configured GPU; actual time depends on GPU throughput. Outputs are under `$D/v08/check/a5_window_chunk/`. Exit code `3` means F3 saved a checkpoint and stopped.

**Check:** proceed to the shared-metric comparison only if the A5 run completed without an F3 alarm and wrote `step_020000`. Review `train_metrics.jsonl` for random-pair distance, λ, constraint gap, pre-clip gradient norm, per-layer dead-unit fraction, and projector variance.

### Compare A2 and A5 with the same chunk metrics

Run only if A2 and A5 both have `step_020000` checkpoints:

```bash
$PY experiments/eval_quasimetric_chunk.py \
  --ckpt $D/v08/check/a2_fix123/step_020000 \
        $D/v08/check/a5_window_chunk/step_020000 $COMMON
```

**Expected time:** several minutes, depending on GPU throughput. The report is printed to stdout; no cache or simulator is used.

**Check:** compare chunk violation `d(x_t,x_{t+5}) > 6.25`, separately for chunks with any internal button change and chunks without one; compare `P(d≤j)` for `j=5,10,15,20,40,80`, projector variance, F3 indicators, and triangle violations. This is a diagnostic comparison; the guide does not assign a numeric A5 selection threshold beyond the F3 stability gates. Send the output before starting a full 200k run.

## Window-5 candidate cache

Build the shared boundary-window candidate cache after the windowed features are selected for evaluation. Candidate pairs use k=32, cross-episode 175D window features, boundary sources, and success states as eligible destinations.

```bash
$PY experiments/build_candidate_cache.py \
  --g0-graph $D/g0/graph.npz \
  --transitions $D/g0/transitions.npz --sim-state $D/g0/sim_state.npz \
  --out $D/v08/candidates_window5.npz \
  --window 5 --knn 32 --parity-size 50000 --seed 0
```

**Expected time:** several minutes to tens of minutes, depending on GPU and CPU kNN parity throughput. The compressed cache and JSON report are `$D/v08/candidates_window5.npz` and `$D/v08/candidates_window5.json`.

**Check:** `knn_parity.mismatched_rows` must be `0`; candidate nodes should be around 200k and pair count around 6.4m for this dataset. Record the actual source, destination, and success-destination counts. Stop if the parity check fails.

## §3.1 Embedding-cache equivalence check

Run this on a completed window-5 checkpoint. The command below is the guide’s random cross-episode pair check. To check actual kNN candidates from the cache above, add `--candidate-cache $D/v08/candidates_window5.npz`.

```bash
$PY experiments/check_embedding_cache.py \
  --ckpt $D/v08/check/a5_window_chunk/step_020000 $COMMON --pairs 10000
```

**Expected time:** several minutes on GPU for a first cache build; a repeated run reuses the on-disk cache. The embedding table is float32 and is approximately 1.6 GB for 200k nodes × 2048 values. The default cache path is beside the checkpoint under `embedding_cache/`; pass `--embedding-cache PATH` to choose another ignored output path.

**Check:** `max_absolute_distance_difference ≤ 1e-4`, `bridge_decision_mismatches_at_K20 = 0`, and `pass = true`. Record `embedding_seconds`, `cached_pair_seconds`, and direct pair time. The test also records cache hit status and bytes. Do not build a quasimetric graph if this check fails.

## Later simulator-free graph comparison

The full P13 and standard-task training commands are added after A2/A3/A5 outputs are reviewed, as required by v0.8 §6. Once the selected window-5 checkpoint and corresponding G0/reference graphs are available, the builder accepts the same window-5 cache for both judges. Flow consumes the original 35D frame endpoints; quasimetric consumes the corresponding 175D windows. `--candidate-window 5` explicitly makes the flow run use the same cached boundary pairs.

Example quasimetric build:

```bash
$PY experiments/build_judged_graph.py --judge quasimetric \
  --g0-graph $D/g0/graph.npz --reference-graph $D/gr-rho09/graph.npz \
  --transitions $D/g0/transitions.npz --sim-state $D/g0/sim_state.npz \
  --checkpoint <SELECTED_WINDOW5_CHECKPOINT> \
  --candidate-cache $D/v08/candidates_window5.npz \
  --out $D/v08/quasimetric
```

If a compatible v0.7 flow checkpoint is available, build it from the exact same candidate cache:

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

**Expected time:** not established; judging millions of pairs is GPU- and checkpoint-dependent. These commands write graph/target artifacts and `result.json` under the two judge-specific directories. The result reports candidate and bridge counts, reachability against G0/reference, bridge-count distribution, cost improvements, judge/embedding timing, and target validity. `compare_judges.py` reports the four agreement cells split by whether endpoint board states differ.

**Gate:** confirm both `optimistic_candidate_sha256` values match before interpreting decision agreement. Q1 must be at least 0.8 for both judges before comparing Q3; the v0.8 guide does not authorize the simulator witness yet, so neither command above steps a simulator. QC is not part of this run.
