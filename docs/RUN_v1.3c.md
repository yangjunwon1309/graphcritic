# Graph Critic v1.3c — FQL family with alpha=1000

Task: standard `puzzle-3x3-play-singletask-task3-v0`. Root branch: `quasi_develop`; QC fork branch: `graph-critic`.

Run four experiments sequentially, in the order F1, F2, F3, F4. Reuse the existing h=1 and h=5 discounted targets. Do not rebuild or modify graphs, targets, or code for these runs.

| Experiment | Actor | Action chunking | Horizon | Critic | Target |
|---|---|---:|---:|---|---|
| F1 | FQL | false | 1 | TD | — |
| F2 | FQL | false | 1 | discounted graph | `v13/Lsame_h1` |
| F3 | QC-FQL | true | 5 | TD | — |
| F4 | QC-FQL | true | 5 | discounted graph | `v12/Lsame` |

The graph hash is `3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa`. The h=1 and h=5 target artifacts both use `kind=discounted`, `gamma=0.99`; their horizons are 1 and 5, respectively.

Before running, commit this runbook and confirm `git_dirty=false` in the root, QC fork, and FQL fork. FQL is read-only for this experiment. Do not start the next run until the previous command exits successfully. All runs use W&B offline.

## F1 — FQL, TD critic

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v13c
export MUJOCO_GL=egl WANDB_MODE=offline
export PYTHONPATH="$ROOT:$ROOT/third_party/qc"
cd "$ROOT/third_party/qc"

"$PY" main.py --run_group=v13c-fql-td \
  --agent.actor_type=distill-ddpg --agent.action_chunking=False --horizon_length=1 \
  --agent.alpha=1000 --agent.batch_size=256 --agent.lr=0.0003 \
  --agent.num_qs=2 --agent.layer_norm=True --agent.q_agg=mean \
  --env_name="$ENV" --sparse=True --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 --seed=0 \
  --save_dir="$V/qc/fql_td"
```

## F2 — FQL, discounted graph critic

Run after F1 completes. This uses the existing h=1 target.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v13c
export MUJOCO_GL=egl WANDB_MODE=offline
export PYTHONPATH="$ROOT:$ROOT/third_party/qc"
GRAPH_DATASET_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' "$D3/v13/Lsame_h1/targets.json")
cd "$ROOT/third_party/qc"

"$PY" main.py --run_group=v13c-fql-graph \
  --agent.actor_type=distill-ddpg --agent.action_chunking=False --horizon_length=1 \
  --agent.alpha=1000 --agent.batch_size=256 --agent.lr=0.0003 \
  --agent.num_qs=2 --agent.layer_norm=True --agent.q_agg=mean \
  --env_name="$ENV" --sparse=True --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 --seed=0 \
  --agent.critic_mode=graph --agent.graph_target_kind=discounted --agent.graph_target_unit=steps \
  --graph_targets="$D3/v13/Lsame_h1/targets.npz" --graph_dataset_id="$GRAPH_DATASET_ID" \
  --save_dir="$V/qc/fql_graph"
```

## F3 — QC-FQL, TD critic

Run after F2 completes.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v13c
export MUJOCO_GL=egl WANDB_MODE=offline
export PYTHONPATH="$ROOT:$ROOT/third_party/qc"
cd "$ROOT/third_party/qc"

"$PY" main.py --run_group=v13c-qcfql-td \
  --agent.actor_type=distill-ddpg --agent.action_chunking=True --horizon_length=5 \
  --agent.alpha=1000 --agent.batch_size=256 --agent.lr=0.0003 \
  --agent.num_qs=2 --agent.layer_norm=True --agent.q_agg=mean \
  --env_name="$ENV" --sparse=True --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 --seed=0 \
  --save_dir="$V/qc/qcfql_td"
```

## F4 — QC-FQL, discounted graph critic

Run after F3 completes. This uses the existing h=5 target.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v13c
export MUJOCO_GL=egl WANDB_MODE=offline
export PYTHONPATH="$ROOT:$ROOT/third_party/qc"
GRAPH_DATASET_ID=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' "$D3/v12/Lsame/targets.json")
cd "$ROOT/third_party/qc"

"$PY" main.py --run_group=v13c-qcfql-graph \
  --agent.actor_type=distill-ddpg --agent.action_chunking=True --horizon_length=5 \
  --agent.alpha=1000 --agent.batch_size=256 --agent.lr=0.0003 \
  --agent.num_qs=2 --agent.layer_norm=True --agent.q_agg=mean \
  --env_name="$ENV" --sparse=True --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 --seed=0 \
  --agent.critic_mode=graph --agent.graph_target_kind=discounted --agent.graph_target_unit=steps \
  --graph_targets="$D3/v12/Lsame/targets.npz" --graph_dataset_id="$GRAPH_DATASET_ID" \
  --save_dir="$V/qc/qcfql_graph"
```

## Report

Record root/QC commits and dirty status, elapsed time for each run, all six evaluation success rates, final mean wrong-button count, critic loss, graph-target Spearman, and final `bc_flow`, `distill`, and Q loss terms in `docs/results_v1.3c.md`. Include the supplied alpha=100 reference rows. Do not push; include the required copyable push commands and commit lists at the end of the results report.
