# Graph Critic v1.4b — P13 4×4 FQL run

This is a generated P13 task-2 dataset (1,321 episodes, 1,000,000 transitions; all generated episodes reach the goal), not a standard benchmark. The graph, candidate cache, and state-distance checkpoint are reused from v1.4. Only the h=1 target is generated here.

## Setup

Commit this runbook before starting the runs. Confirm the root, QC, and FQL worktrees are clean. Do not push from this workflow.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D=$ROOT/outputs/puzzle/p13-4x4-task2-100-generated
V=$D/v14b
ENV=puzzle-4x4-play-singletask-task2-v0
OLD=$D/v08/check/window5_chunk/step_020000
CAND=$D/v14/candidates_obj.npz
COMMON="--transitions $D/g0/transitions.npz --sim-state $D/g0/sim_state.npz"
BUILD="--judge quasimetric --g0-graph $D/g0/graph.npz --reference-graph $D/gr-rho09/graph.npz $COMMON --checkpoint $OLD --candidate-cache $CAND --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5 --bridge-object-change same --b-max 1 --target-kind discounted --discount-gamma 0.99"
```

## S1 — h=1 discounted target

```bash
set -euo pipefail
cd "$ROOT"
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" \
  $PY experiments/build_judged_graph.py $BUILD --target-h 1 --out "$V/Lsame_h1"
GRAPH_HASH=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["graph_hash"])' "$V/Lsame_h1/result.json")
test "$GRAPH_HASH" = "6bd8f696e65e362770facddea65caca03f57abdb78927f054a730f32afbc01aa"
```

Stop if the graph-hash check fails.

## Shared QC flags

`alpha=100` is used for this P13 comparison. The P13 dataset is at `$D/train.npz`; the environment is not the OGBench standard-data task-2 split.

```bash
set -euo pipefail
COMMON_QC="--agent.actor_type=distill-ddpg --agent.action_chunking=False --horizon_length=1 --agent.alpha=100 --agent.batch_size=256 --agent.lr=0.0003 --agent.num_qs=2 --agent.layer_norm=True --agent.q_agg=mean --env_name=$ENV --ogbench_dataset_path=$D/train.npz --sparse=True --offline_steps=300000 --online_steps=0 --eval_interval=50000 --eval_episodes=50 --save_interval=50000 --seed=0"
```

## S2 — FQL with TD critic

```bash
set -euo pipefail
cd "$ROOT/third_party/qc"
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" \
  $PY main.py --run_group=v14b-4x4p13-fql-td $COMMON_QC --save_dir="$V/qc/fql_td"
```

## S3 — FQL with graph critic

```bash
set -euo pipefail
GRAPH_DATASET_ID=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' "$V/Lsame_h1/targets.json")
cd "$ROOT/third_party/qc"
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" \
  $PY main.py --run_group=v14b-4x4p13-fql-graph $COMMON_QC \
  --agent.critic_mode=graph --agent.graph_target_kind=discounted --agent.graph_target_unit=steps \
  --graph_targets="$V/Lsame_h1/targets.npz" --graph_dataset_id="$GRAPH_DATASET_ID" \
  --save_dir="$V/qc/fql_graph"
```

## Report fields

Record the execution commits and clean-worktree state, elapsed time for S1–S3, h=1 target valid/loss-mask/capped fractions and cost statistics, graph hash gate, and both QC runs' 50k–300k success, final wrong-button count, critic loss, graph-target Spearman, and final BC-flow/distill/Q actor-loss terms. Keep the push commands section at the end of the results report; do not push during the run.
