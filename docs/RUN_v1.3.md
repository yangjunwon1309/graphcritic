# Graph Critic v1.3 — FQL versus FQL with graph critic

Task: standard `puzzle-3x3-play-singletask-task3-v0`, branch `quasi_develop`; QC fork branch `graph-critic`.

The run reuses the v1.2b `Lsame` graph construction, its `d_theta` checkpoint, and the v1.1 candidate cache. It creates an h=1 discounted target artifact for the same graph, then compares an FQL actor with the TD critic against an identical FQL actor with the graph critic. No graph, distance model, candidate cache, or QC training implementation is changed for the experiment.

For each transition row, the h=1 discounted label is

\[
y_t = \frac{1-\gamma^{1+C(s_{t+1})}}{1-\gamma},\qquad \gamma=0.99.
\]

## Allowed tests

Run only the h=1 graph-target/FQL update test and `tests/test_v12.py`:

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
cd "$ROOT"
"$PY" -m pytest third_party/qc/tests/test_graph_discounted_fql.py tests/test_v12.py -q
```

Commit implementation and runbook changes before running experiments. The execution commit must have `git_dirty=false`.

## S1. Build and verify the h=1 target artifact

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v13
COMMON3=(--transitions "$D3/g0/transitions.npz" --sim-state "$D3/g0/sim_state.npz")
OLD=$D3/v08/check/window5_chunk/step_020000
CAND=$D3/v11/candidates_obj.npz

"$PY" experiments/build_judged_graph.py --judge quasimetric \
  --g0-graph "$D3/g0/graph.npz" \
  --reference-graph "$D3/gr-rho09/graph.npz" "${COMMON3[@]}" \
  --checkpoint "$OLD" --candidate-cache "$CAND" \
  --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5 \
  --bridge-object-change same --b-max 1 \
  --target-kind discounted --discount-gamma 0.99 --target-h 1 \
  --out "$V/Lsame_h1"

EXPECTED=3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa
ACTUAL=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["graph_hash"])' "$V/Lsame_h1/result.json")
test "$ACTUAL" = "$EXPECTED"
"$PY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["config"]["h"] == 1' "$V/Lsame_h1/targets.json"
```

Stop before either QC run if the graph hash differs from the expected hash above or target metadata does not report `h=1`.

## S2. FQL with the TD critic

Run from the QC fork. This is the TD baseline; the FQL actor and training options match S3.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v13
export MUJOCO_GL=egl WANDB_MODE=offline
export PYTHONPATH="$ROOT:$ROOT/third_party/qc"
cd "$ROOT/third_party/qc"

"$PY" main.py --run_group=v13-3x3t3-fql-td \
  --agent.actor_type=distill-ddpg --agent.critic_mode=td \
  --agent.action_chunking=False --horizon_length=1 \
  --agent.alpha=100 --agent.batch_size=256 --agent.lr=0.0003 \
  --agent.num_qs=2 --agent.layer_norm=True --agent.q_agg=mean \
  --env_name="$ENV" --sparse=True --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 --seed=0 \
  --save_dir="$V/qc/fql_td"
```

## S3. FQL with the discounted graph critic

Run after S2 completes. The target dataset ID comes from the newly generated h=1 artifact.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v13
export MUJOCO_GL=egl WANDB_MODE=offline
export PYTHONPATH="$ROOT:$ROOT/third_party/qc"
GRAPH_DATASET_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' "$V/Lsame_h1/targets.json")
cd "$ROOT/third_party/qc"

"$PY" main.py --run_group=v13-3x3t3-fql-graph \
  --agent.actor_type=distill-ddpg --agent.critic_mode=graph \
  --agent.graph_target_kind=discounted --agent.graph_target_unit=steps \
  --agent.action_chunking=False --horizon_length=1 \
  --agent.alpha=100 --agent.batch_size=256 --agent.lr=0.0003 \
  --agent.num_qs=2 --agent.layer_norm=True --agent.q_agg=mean \
  --env_name="$ENV" --sparse=True --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 \
  --graph_targets="$V/Lsame_h1/targets.npz" --graph_dataset_id="$GRAPH_DATASET_ID" \
  --seed=0 --save_dir="$V/qc/fql_graph"
```

## Report

Write numeric results to `docs/results_v1.3.md`: implementation and execution commits, `git_dirty`, allowed test result, stage durations, graph hash and target statistics, both QC evaluation series at 50k–300k, final wrong-button counts, critic loss, graph-target Spearman, and final actor loss terms (`bc_flow_loss`, `distill_loss`, and Q loss). Include the supplied v1.2b, v0.6 TD, and BC reference rows. If a stage stops, record its last completed checkpoint and measured reason; do not start later stages after the S1 hash gate fails.
