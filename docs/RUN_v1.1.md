# Graph Critic v1.1 — observed object-change bridges

Branch: `quasi_develop`  
Task: standard `puzzle-3x3-play-singletask-task3-v0`

This run reuses the v0.8 frozen distance checkpoint as `d_theta`. It does not train `d+`. The only training in this run is the four action-conditioned heads. GF(2) minimum press counts are computed for audit tables only; they are not inputs to candidate generation, bridge acceptance, graph building, or action selection.

Run each numbered stage only after the previous stop gate passes. If a gate fails, stop there. S6 and S7 run sequentially only after S5 passes its head gates and the direct-Q offline report is saved. The implementation and QC submodule must be committed before data runs; both worktrees must report `git_dirty=false`.

## 0. Pin the implementation

```bash
set -euo pipefail
cd /home/url/graphcritic
git switch quasi_develop
git status --short --branch
git rev-parse HEAD
git -C third_party/qc status --short --branch
git -C third_party/qc rev-parse HEAD

ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
COMMON3="--transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz"
OLD=$D3/v08/check/window5_chunk/step_020000
V=$D3/v11
```

Do not regenerate G0, G-R, or `OLD`. Do not run TD/FQL, another data size or task, another lambda, `d_theta` training, or simulator diagnostics.

## S0. Specified tests

```bash
set -euo pipefail
cd /home/url/graphcritic
$PY -m pytest tests/test_v11.py tests/test_v10.py tests/test_quasimetric_v08.py -q
```

Proceed only if all three requested files pass.

## S1. Build observed object-change set

```bash
set -euo pipefail
cd /home/url/graphcritic
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v11
COMMON3="--transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz"
$PY experiments/build_object_change_set.py $COMMON3 \
  --max-horizon 20 --min-count 10 --out $V/change_set.json
```

Record each retained XOR pattern as changed button indices, its count and minimum frame offset. Also record omitted-pattern count and omitted occurrence total.

## S2. Build object-conditioned candidate cache

```bash
set -euo pipefail
cd /home/url/graphcritic
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v11
COMMON3="--transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz"
$PY experiments/build_candidate_cache.py --g0-graph $D3/g0/graph.npz $COMMON3 \
  --object-change-set $V/change_set.json --feature-mode robot --window 5 \
  --knn 32 --knn-changed 32 --parity-size 50000 --seed 0 \
  --out $V/candidates_obj.npz
```

The source and destination node universe follows the v0.8.4 boundary/goal rule. Each eligible source receives up to 32 closest `Δ=0` destinations and 32 closest nonzero observed-change destinations, using robot-only window features and cross-episode destinations.

## S3. Candidate audit (Gate A)

```bash
set -euo pipefail
cd /home/url/graphcritic
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v11
COMMON3="--transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz"
OLD=$D3/v08/check/window5_chunk/step_020000
$PY experiments/tmp_board_check.py --candidates-only --candidate-cache $V/candidates_obj.npz \
  --distance-checkpoint $OLD $COMMON3 --out $V/candidates_audit.json
```

The audit reports `d_theta` p10/p50/p90 and `d_theta≤20` by candidate bundle (`Δ=0` / `Δ≠0`) and minimum GF(2) press-count class. Continue only if the total fraction with `n_p≥2` (including unsolvable `∞`) is at most 1%. If Gate A fails, do not run S4–S7.

## S4. Build Lobj and audit retained bridges (Gate B)

Run only after Gate A passes.

```bash
set -euo pipefail
cd /home/url/graphcritic
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v11
COMMON3="--transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz"
OLD=$D3/v08/check/window5_chunk/step_020000
$PY experiments/build_judged_graph.py --judge quasimetric \
  --g0-graph $D3/g0/graph.npz --reference-graph $D3/gr-rho09/graph.npz $COMMON3 \
  --checkpoint $OLD --candidate-cache $V/candidates_obj.npz \
  --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5 \
  --target-kind discounted --discount-gamma 0.99 --out $V/Lobj
$PY experiments/tmp_board_check.py --graph $V/Lobj/graph.npz $COMMON3 \
  --out $V/Lobj_board_check.json
```

Record graph and target summaries, including accepted/retained bridges, reachability, bridge-count distribution, loss mask, discounted target mean/cap fraction, impossible-bridge fraction and its `n_p` distribution. Continue only if the impossible retained-bridge fraction is at most 5%. If Gate B fails, do not run S5–S7.

## S5. Train four action heads and run offline G1/G2/G3

Run only after Gate B passes.

```bash
set -euo pipefail
cd /home/url/graphcritic
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v11
COMMON3="--transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz"
OLD=$D3/v08/check/window5_chunk/step_020000
$PY experiments/train_action_quasimetric.py $COMMON3 --state-checkpoint $OLD \
  --window 5 --horizon 5 --epsilon-c1 1.25 --epsilon-c2 0.25 \
  --steps 20000 --log-every 1000 --save-every 10000 \
  --seeds 0 1 2 3 --out $V/action_heads
$PY experiments/write_direct_q_config.py --graph $V/Lobj/graph.npz \
  --candidate-cache $V/candidates_obj.npz --state-checkpoint $OLD \
  --action-head-root $V/action_heads $COMMON3 --anchor-mode same_object \
  --out $V/direct_q_Lobj.json
$PY experiments/eval_direct_q.py --config $V/direct_q_Lobj.json $COMMON3 \
  --n 20000 --seed 0 --out $V/offline_gate.json
```

Continue only if all four heads have no F3/collapse alarm and held-out `d>6.25` C1 violation at most 5%. G1/G2/G3 are report-only. Record head C1/C2 violation, `d` medians at offsets 5/10/15/20, offline G1/G2/G3, fallback rate/reasons, mean `|N(s)|`, candidate-Q standard deviation and decision latency. If a head gate fails, do not run S6 or S7.

## S6. QC direct graph Q and same-checkpoint BC

Run after S5 passes. This is one 300k actor run; `eval_bc_also=True` evaluates the same actor checkpoint with an unscored proposal.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v11
cd $ROOT/third_party/qc
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" $PY main.py \
  --run_group=v11-3x3t3-Lobj-direct \
  --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --agent.critic_mode=graph_direct --direct_q_config=$V/direct_q_Lobj.json --eval_bc_also=True \
  --agent.action_chunking=True --agent.alpha=100 --agent.batch_size=256 --agent.lr=0.0003 \
  --agent.num_qs=2 --agent.layer_norm=True --agent.q_agg=mean \
  --env_name=$ENV --sparse=True --horizon_length=5 --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 --seed=0 \
  --save_dir=$V/qc/Lobj_direct
```

## S7. QC discounted graph-cost critic

Run after S6 completes. Target gamma is fixed at 0.99.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
ENV=puzzle-3x3-play-singletask-task3-v0
D3=$ROOT/outputs/puzzle/standard-3x3-task3
V=$D3/v11
cd $ROOT/third_party/qc
GRAPH_DATASET_ID=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' $V/Lobj/targets.json)
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" $PY main.py \
  --run_group=v11-3x3t3-Lobj-discounted \
  --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --agent.critic_mode=graph --agent.graph_target_kind=discounted --agent.graph_target_unit=steps \
  --agent.graph_target_gamma=0.99 --agent.action_chunking=True --agent.alpha=100 \
  --agent.batch_size=256 --agent.lr=0.0003 --agent.num_qs=2 \
  --agent.layer_norm=True --agent.q_agg=mean \
  --env_name=$ENV --sparse=True --horizon_length=5 --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 \
  --graph_targets=$V/Lobj/targets.npz --graph_dataset_id="$GRAPH_DATASET_ID" --seed=0 \
  --save_dir=$V/qc/Lobj_discounted
```

## Report

Write numerical values and artifact paths to [results_v1.1.md](results_v1.1.md). Record root/QC commits and dirty states, test result, stage durations, the stop stage/value, observed-change set counts, candidate audit, graph/target statistics, action-head/offline checks and QC metrics. Compare QC with the v0.8.4 L/O regret, v0.8.4b L/O cost, and v0.6 TD rows specified in the development guide. Do not add a result interpretation. Commit the report. Push `quasi_develop` if credentials are available; otherwise record the authentication failure.
