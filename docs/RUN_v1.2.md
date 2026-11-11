# Graph Critic v1.2 — same-object bridges and minimum bridge budget

Run this workflow on branch `quasi_develop` with the QC fork on branch
`graph-critic`. The task is `puzzle-3x3-play-singletask-task3-v0`. Commit the
implementation and runbook before starting so every generated report records
`git_dirty=false`. Run each section in order. A failed gate is a hard stop at
the section named below; do not continue to later sections.

## S1 — sweep bridge budgets

Each sweep judges its candidate set once and performs one layered Dijkstra
with `max_b=32`. It writes the reachability `R(b)` and finite-cost p50/p90 for
each budget. The same-object sweep is the Gate A input. If its `R_sat` is
below `0.90`, stop after S1.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v12
COMMON3="--transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz"
GRAPH_IN="--judge quasimetric --g0-graph $D3/g0/graph.npz --reference-graph $D3/gr-rho09/graph.npz $COMMON3 \
  --checkpoint $D3/v08/check/window5_chunk/step_020000 --candidate-cache $D3/v11/candidates_obj.npz \
  --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5"

$PY experiments/sweep_b_max.py $GRAPH_IN --bridge-object-change same --max-b 32 --out $V/sweep_same.json
$PY experiments/sweep_b_max.py $GRAPH_IN --bridge-object-change any --max-b 32 --out $V/sweep_any.json
$PY -c 'import json,sys; d=json.load(open(sys.argv[1])); print("Gate A R_sat=",d["r_sat"]); assert d["r_sat"] >= 0.90' $V/sweep_same.json
```

## S2 — build the same-object graph and audit retained bridges

Continue only if Gate A passes. `B*` is the first budget with
`R(b) >= R_sat - 0.005`. Stop after the board audit if any retained bridge
has exact minimum press count `n_p >= 1`.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v12
COMMON3="--transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz"
GRAPH_IN="--judge quasimetric --g0-graph $D3/g0/graph.npz --reference-graph $D3/gr-rho09/graph.npz $COMMON3 \
  --checkpoint $D3/v08/check/window5_chunk/step_020000 --candidate-cache $D3/v11/candidates_obj.npz \
  --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5"
BSTAR=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["b_star"])' $V/sweep_same.json)

$PY experiments/build_judged_graph.py $GRAPH_IN --bridge-object-change same --b-max "$BSTAR" \
  --target-kind discounted --discount-gamma 0.99 --out $V/Lsame
$PY experiments/tmp_board_check.py --graph $V/Lsame/graph.npz $COMMON3 --out $V/Lsame_board_check.json
$PY -c 'import json,sys; d=json.load(open(sys.argv[1]))["graphs"][0]["same_object_gate"]; print("Gate B n_p>=1:",d["n_p_ge_1_count"]); assert d["n_p_ge_1_count"] == 0' $V/Lsame_board_check.json
```

## S3 — train action heads and run the offline direct-Q gate

Continue only if Gate B passes. Four action heads are trained from the frozen
state checkpoint. Stop if any head reports collapse/F3 or held-out C1
violation above `0.05`. G1, G2, and G3 from `offline_gate.json` are report-only.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v12
COMMON3="--transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz"
OLD=$D3/v08/check/window5_chunk/step_020000
CAND=$D3/v11/candidates_obj.npz

$PY experiments/train_action_quasimetric.py $COMMON3 --state-checkpoint $OLD \
  --window 5 --horizon 5 --epsilon-c1 1.25 --epsilon-c2 0.25 \
  --steps 20000 --log-every 1000 --save-every 10000 --seeds 0 1 2 3 --out $V/action_heads
$PY -c 'import json,sys; d=json.load(open(sys.argv[1])); h=d["heads"]; assert not d.get("stopped_on_collapse",False) and not d.get("stopped_on_gate",False); assert set(h)=={"0","1","2","3"}; assert all(not x["collapse_alarm"] and x["c1_violation_rate_d_gt_h_plus_epsilon"] <= .05 for x in h.values())' $V/action_heads/summary.json
$PY experiments/write_direct_q_config.py --graph $V/Lsame/graph.npz --candidate-cache $CAND \
  --state-checkpoint $OLD --action-head-root $V/action_heads $COMMON3 \
  --anchor-mode same_object --out $V/direct_q_Lsame.json
$PY experiments/eval_direct_q.py --config $V/direct_q_Lsame.json $COMMON3 \
  --n 20000 --seed 0 --out $V/offline_gate.json
```

## S4 — direct graph Q with BC evaluation

Run after the four-head gate in S3 passes.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v12
cd $ROOT/third_party/qc
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" $PY main.py \
  --run_group=v12-3x3t3-Lsame-direct \
  --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --agent.critic_mode=graph_direct --direct_q_config=$V/direct_q_Lsame.json --eval_bc_also=True \
  --agent.action_chunking=True --agent.alpha=100 --agent.batch_size=256 --agent.lr=0.0003 \
  --env_name=$ENV --sparse=True --horizon_length=5 --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 --seed=0 \
  --save_dir=$V/qc/Lsame_direct
```

## S5 — discounted graph-cost critic

Run after S4 completes.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v12
GRAPH_DATASET_ID=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' $V/Lsame/targets.json)
cd $ROOT/third_party/qc
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" $PY main.py \
  --run_group=v12-3x3t3-Lsame-discounted \
  --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --agent.critic_mode=graph --agent.graph_target_kind=discounted --agent.graph_target_unit=steps \
  --agent.action_chunking=True --agent.alpha=100 --agent.batch_size=256 --agent.lr=0.0003 \
  --agent.num_qs=2 --agent.layer_norm=True --agent.q_agg=mean \
  --env_name=$ENV --sparse=True --horizon_length=5 --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 \
  --graph_targets=$V/Lsame/targets.npz --graph_dataset_id="$GRAPH_DATASET_ID" --seed=0 \
  --save_dir=$V/qc/Lsame_discounted
```

## Reporting

Record commits and dirty state, tests, per-stage elapsed time, the stop stage
and gate values, and the following in [results_v1.2.md](results_v1.2.md):

- For same and any: `R(b)` and reachable-state cost p50/p90 at budgets
  `0,1,2,3,4,6,8,12,16,32`; for same also `R_sat` and `B*`.
- For `Lsame`: candidate/accepted/retained bridge counts, reachability,
  loss-mask fraction, bridge-count distribution `0..B*`, exact `n_p` audit,
  lower-bound rejects, discounted-target statistics and capped fraction.
- For four action heads: C1/C2 violation rates and `d` medians at offsets
  `5,10,15,20`; offline G1/G2/G3, fallback rate and causes.
- For QC: success at `50k..300k`, mean wrong buttons at the last evaluation,
  and for direct Q fallback rate/causes, mean `|N(s)|`, candidate-Q standard
  deviation and decision time. Include the reference rows given in the v1.2
  development specification.

Commit the numerical report. Push `quasi_develop` and the QC fork's
`graph-critic` branch when credentials are available; record authentication
failure if push is unavailable.
