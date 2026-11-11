# Graph Critic v1.0 — matched-object constraint runbook

This runbook implements the v1.0 P2/P3 experiment on standard Puzzle 3×3 task 3. It reuses the v0.8.4 candidate cache, G0, G-R reference graph, and window-5 checkpoint. The GF(2) press-count solution is used only for S2 reporting. Training, graph construction, action selection, and QC do not receive press-count labels.

The expected flow is S0 → S1 → S2 → S3 → S4 → S5 → S6. Stop at a failed stop gate. S3 is report-only. The commands below do not launch TD/FQL, a separate BC run, a different data size/task, or a simulator diagnosis. S5 includes same-checkpoint BC evaluation as the prescribed comparison.

## 0. Pin the implementation

Commit the implementation and QC submodule changes first, then use a clean checkout. Save both commit hashes and verify `git_dirty=false` at the root and `third_party/qc` before running.

```bash
set -euo pipefail
cd /home/url/graphcritic
git switch quasi_develop
git status --short
git rev-parse HEAD
git -C third_party/qc status --short
git -C third_party/qc rev-parse HEAD

ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
COMMON3="--transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz"
OLD=$D3/v08/check/window5_chunk/step_020000
CAND=$D3/v08/candidates_window5.npz
V=$D3/v10
```

Use the exact task-3 data and existing artifacts above. Do not regenerate `G0`, `G-R`, `CAND`, or `OLD` for this run.

## S0. Code checks

```bash
set -euo pipefail
$PY -m pytest tests/test_v10.py tests/test_quasimetric_v08.py -q
```

Stop if either command fails. The root tests cover object equality/caching, P2/P3 signs and direction sampling, the 50/50 maximization batch, discounted targets and cap, and the v0.9 direct-Q contracts. The QC suite covers the graph-direct update path, including keeping both critic parameter sets fixed.

## S1. Train d⁺ from scratch

```bash
set -euo pipefail
$PY experiments/train_quasimetric.py $COMMON3 \
  --window 5 --constraint-horizon 5 --epsilon 1.25 \
  --activation gelu --layernorm --grad-clip 1.0 --dual adam \
  --match-constraint object --candidate-cache $CAND --reference-checkpoint $OLD \
  --epsilon-match 1.25 --unmatched-max-frac 0.5 \
  --steps 20000 --log-every 1000 --save-every 10000 --out $V/dplus
```

This trains with 4,096 chunk constraints, 4,096 matched-object P2 pairs, and 4,096 maximization pairs split equally between random and unmatched-object pairs. The v0.8 checkpoint is only the fixed `d_R` reference for P2; it does not initialize d⁺.

Stop if training returns a collapse alarm (exit 3), or the final checkpoint has any dead-unit fraction above zero (exit 4). Continue with `$V/dplus/step_020000` only if S1 passes.

## S2. Audit d⁺ (stop gate)

```bash
set -euo pipefail
$PY experiments/eval_quasimetric_chunk.py --ckpt $V/dplus/step_020000 $COMMON3
$PY experiments/tmp_board_check.py --candidates-only --candidate-cache $CAND \
  --distance-checkpoint $V/dplus/step_020000 --reference-checkpoint $OLD \
  $COMMON3 --out $V/dplus_audit.json
```

The first command reports held-out chunk metrics. The JSON audit reports full-cache and C₀-optimistic-subset d⁺ distributions by exact GF(2) minimum press count, reference dθ medians, P2 violation on held-out matched pairs in both directions, and the same chunk threshold in machine-readable form. If any S2 gate fails, stop: do not run S3, S4, S5, or S6. Continue only if all four gates pass:

1. Held-out chunk violation `d⁺ > 6.25` is at most 5%.
2. Held-out matched-pair P2 violation `d⁺ > d_R + 1.25` is at most 5%.
3. Median d⁺ strictly increases from `n_p=0` to `n_p=1` to `n_p≥2`.
4. At most 1% of `n_p≥2` candidates have `d⁺≤20`.

Record p10/p50/p90 by `n_p`, the `n_p=0` fraction with `d⁺≤20`, and the `n_p=1` fraction with `d⁺≤20`. `dplus_audit.json` contains `gates.s2_all_passed`.

## S3. Build L⁺ and run the board lower-bound check

Run only after S2 passes.

```bash
set -euo pipefail
$PY experiments/build_judged_graph.py --judge quasimetric \
  --g0-graph $D3/g0/graph.npz --reference-graph $D3/gr-rho09/graph.npz $COMMON3 \
  --checkpoint $V/dplus/step_020000 --candidate-cache $CAND \
  --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5 \
  --target-kind discounted --discount-gamma 0.99 --out $V/Lplus
$PY experiments/tmp_board_check.py --graph $V/Lplus/graph.npz $COMMON3 \
  --out $V/Lplus_board_check.json
```

Record retained bridges, reachability, bridge-count distribution, median graph cost, and impossible-bridge fraction with the nₚ breakdown. S3 is report-only; do not apply a new stop rule here.

## S4. Train four action heads and run offline gates

```bash
set -euo pipefail
$PY experiments/train_action_quasimetric.py $COMMON3 \
  --state-checkpoint $V/dplus/step_020000 \
  --window 5 --horizon 5 --epsilon-c1 1.25 --epsilon-c2 0.25 \
  --steps 20000 --log-every 1000 --save-every 10000 \
  --seeds 0 1 2 3 --out $V/action_heads

$PY experiments/write_direct_q_config.py \
  --graph $V/Lplus/graph.npz --candidate-cache $CAND \
  --state-checkpoint $V/dplus/step_020000 \
  --action-head-root $V/action_heads $COMMON3 \
  --anchor-mode dplus --out $V/direct_q_Lplus.json

$PY experiments/eval_direct_q.py --config $V/direct_q_Lplus.json $COMMON3 \
  --n 20000 --seed 0 --out $V/offline_gate.json
```

The config writer discovers the matching d⁺ embedding cache written by the L⁺ build; it validates the graph/checkpoint/cache hashes and all four heads. Stop before QC if any head has held-out C1 violation above 5%, if any head triggers F3, or if any head has nonzero dead units. Record G1 (ensemble distance median and fraction above 6.25), G2 (median absolute error against `5 + C(s_{t+5})`), and G3 (fraction and median increase when a different recorded action chunk is substituted).

## S5. QC with direct graph Q and same-checkpoint BC evaluation

Run only if all four action heads pass S4. Run from the QC fork. This is one 300k flow-BC actor run; direct graph Q selects among 32 proposals during `eval`, and `eval_bc_also=True` evaluates one unscored proposal from the same actor checkpoint.

```bash
set -euo pipefail
cd $ROOT/third_party/qc
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" $PY main.py \
  --run_group=v10-3x3t3-Lplus-direct \
  --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --agent.critic_mode=graph_direct --direct_q_config=$V/direct_q_Lplus.json --eval_bc_also=True \
  --agent.action_chunking=True --agent.alpha=100 --agent.batch_size=256 --agent.lr=0.0003 \
  --agent.num_qs=2 --agent.layer_norm=True --agent.q_agg=mean \
  --env_name=$ENV --sparse=True --horizon_length=5 --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 --seed=0 \
  --save_dir=$V/qc/Lplus_direct
```

The `v10-...` run group is learned-graph provenance; it does not need the v0.9 `oracle` label. Record direct-Q and `eval_bc/*` success and average wrong-button count at each checkpoint, along with direct-Q fallback and timing diagnostics.

## S6. QC with discounted graph-cost critic

Run after S5 completes. `graph_target_gamma` must match the target metadata (`0.99`).

```bash
set -euo pipefail
GRAPH_DATASET_ID=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' $V/Lplus/targets.json)
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" $PY main.py \
  --run_group=v10-3x3t3-Lplus-discounted \
  --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --agent.critic_mode=graph --agent.graph_target_kind=discounted --agent.graph_target_unit=steps \
  --agent.graph_target_gamma=0.99 --agent.action_chunking=True --agent.alpha=100 \
  --agent.batch_size=256 --agent.lr=0.0003 --agent.num_qs=2 \
  --agent.layer_norm=True --agent.q_agg=mean \
  --env_name=$ENV --sparse=True --horizon_length=5 --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 \
  --graph_targets=$V/Lplus/targets.npz --graph_dataset_id="$GRAPH_DATASET_ID" --seed=0 \
  --save_dir=$V/qc/Lplus_discounted
```

## Reporting

Record completed steps and wall time in [results_v1.0.md](results_v1.0.md). Include both root and QC commits, dirty state, each gate, and exact artifact paths. If a stop gate fails, stop at that step and record the measured value; do not run later stages.
