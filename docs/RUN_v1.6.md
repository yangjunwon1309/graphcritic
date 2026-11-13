# Graph Critic v1.6 — DQC TD versus graph critic

This run compares DQC with its original TD chunk critic against DQC with
`gamma_pow` graph targets on standard 3×3 task 3 and generated P13 4×4 task 2.
The P13 archive contains 1,000,000 generated transitions and is not a standard
benchmark. DQC keeps its 25-step chunk critic, 5-step flow policy, expectile
partial-critic distillation, and best-of-32 action selection.

## 0. Clean commits and fixed inputs

Commit the root and DQC changes first. QC and FQL are unmodified. Each
experiment must start with all four repositories clean; the DQC fork is on
`graph-critic`.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D3=$ROOT/outputs/puzzle/standard-3x3-task3
D4=$ROOT/outputs/puzzle/p13-4x4-task2-100-generated
V3=$D3/v16
V4=$D4/v16
ENV3=puzzle-3x3-play-singletask-task3-v0
ENV4=puzzle-4x4-play-singletask-task2-v0
COMMON3="--transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz"
COMMON4="--transitions $D4/g0/transitions.npz --sim-state $D4/g0/sim_state.npz"
OLD3=$D3/v08/check/window5_chunk/step_020000
OLD4=$D4/v08/check/window5_chunk/step_020000
CAND3=$D3/v11/candidates_obj.npz
CAND4=$D4/v14/candidates_obj.npz

cd "$ROOT"
git status -sb
git -C third_party/qc status -sb
git -C third_party/fql status -sb
git -C third_party/dqc status -sb
git rev-parse HEAD
git -C third_party/qc rev-parse HEAD
git -C third_party/fql rev-parse HEAD
git -C third_party/dqc rev-parse HEAD
```

## 1. Build h=25 `gamma_pow` targets

The only target option changes are `kind=gamma_pow`, `h=25`, and `gamma=0.999`.
The graph inputs and bridge settings match the existing Lsame builds. Stop if
either hash differs from its recorded Lsame graph.

```bash
set -euo pipefail
cd "$ROOT"
GRAPH3="--judge quasimetric --g0-graph $D3/g0/graph.npz --reference-graph $D3/gr-rho09/graph.npz $COMMON3 --checkpoint $OLD3 --candidate-cache $CAND3 --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5"
GRAPH4="--judge quasimetric --g0-graph $D4/g0/graph.npz --reference-graph $D4/gr-rho09/graph.npz $COMMON4 --checkpoint $OLD4 --candidate-cache $CAND4 --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5"

$PY experiments/build_judged_graph.py $GRAPH3 --bridge-object-change same --b-max 1 \
  --target-kind gamma_pow --target-h 25 --discount-gamma 0.999 --out $V3/Lsame_h25_gp
$PY experiments/build_judged_graph.py $GRAPH4 --bridge-object-change same --b-max 1 \
  --target-kind gamma_pow --target-h 25 --discount-gamma 0.999 --out $V4/Lsame_h25_gp

$PY -c 'import json,sys; expected=sys.argv[2]; actual=json.load(open(sys.argv[1]))["graph_hash"]; print(actual); assert actual==expected' \
  $V3/Lsame_h25_gp/result.json 3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa
$PY -c 'import json,sys; expected=sys.argv[2]; actual=json.load(open(sys.argv[1]))["graph_hash"]; print(actual); assert actual==expected' \
  $V4/Lsame_h25_gp/result.json 6bd8f696e65e362770facddea65caca03f57abdb78927f054a730f32afbc01aa
```

Also confirm each `targets.json` has `kind=gamma_pow`, `h=25`, `gamma=0.999`,
and the dataset ID for its task before running DQC.

## 2. Measure DQC update speed and select one batch size

Run 1,000 updates on standard 3×3 task 3 with the DQC default batch size 4096.
`offline_steps=999` makes the upstream inclusive loop execute exactly 1,000
updates. No evaluation runs in this measurement. Use the logged elapsed training
time to estimate `300000 / update_rate`; if it exceeds three hours, repeat at
1024, then at 256 if 1024 also exceeds three hours. Use the selected batch for
all four benchmark runs.

```bash
set -euo pipefail
cd "$ROOT/third_party/dqc"
BATCH=4096
GROUP=v16-speed-b$BATCH
SPEED_DIR=$V3/speed/b$BATCH
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/dqc" $PY main.py \
  --run_group=$GROUP --task_mode=single --critic_mode=td \
  --agent.backup_horizon=25 --agent.policy_chunk_size=5 --agent.use_chunk_critic=True \
  --agent.kappa_b=0.9 --agent.kappa_d=0.5 --agent.discount=0.999 \
  --agent.distill_method=expectile --agent.implicit_backup_type=quantile \
  --agent.batch_size=$BATCH --agent.num_qs=2 --agent.q_agg=mean \
  --env_name=$ENV3 --offline_steps=999 --eval_interval=0 --eval_episodes=0 \
  --video_episodes=0 --log_interval=999 --save_interval=1000000 --seed=0 \
  --save_dir=$SPEED_DIR

TRAIN_CSV=$(find "$SPEED_DIR" -name train.csv -print -quit)
$PY -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1]))); row=max(rows,key=lambda r:int(r["step"])); seconds=float(row["training/time/total_time"]); rate=1000/seconds; print(f"seconds={seconds:.3f} updates_per_second={rate:.3f} estimated_300k_hours={300000/rate/3600:.3f}")' "$TRAIN_CSV"
```

If the estimate exceeds 3 hours, repeat the same block with `BATCH=1024` and a
new `GROUP`/`SPEED_DIR`. If that estimate also exceeds 3 hours, repeat with
`BATCH=256`. Record each speed measurement and use the first batch at or below
the three-hour estimate; if all exceed it, use 256.

## 3. DQC runs, sequentially

Run R1 through R4 in order, one process at a time. Set `BATCH` to the choice
from section 2. The TD runs omit all graph-target flags. Graph runs validate
target kind, horizon, gamma, dataset archive hash, row alignment, and chunk
validity before the first update. The single-task sampler keeps complete
25-action chunks within each episode; graph chunks may end on the episode's
last transition, with the compact terminal-state observation as endpoint.

### R1 — 3×3 DQC TD

```bash
set -euo pipefail
cd "$ROOT/third_party/dqc"
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/dqc" $PY main.py \
  --run_group=v16-3x3t3-dqc-td --task_mode=single --critic_mode=td \
  --agent.backup_horizon=25 --agent.policy_chunk_size=5 --agent.use_chunk_critic=True \
  --agent.kappa_b=0.9 --agent.kappa_d=0.5 --agent.discount=0.999 \
  --agent.distill_method=expectile --agent.implicit_backup_type=quantile \
  --agent.batch_size=$BATCH --agent.num_qs=2 --agent.q_agg=mean \
  --env_name=$ENV3 --offline_steps=300000 --eval_interval=50000 \
  --eval_episodes=50 --video_episodes=0 --save_interval=50000 --seed=0 \
  --save_dir=$V3/qc/td
```

### R2 — 3×3 DQC graph critic

```bash
set -euo pipefail
GRAPH_DATASET_ID3=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' $V3/Lsame_h25_gp/targets.json)
cd "$ROOT/third_party/dqc"
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/dqc" $PY main.py \
  --run_group=v16-3x3t3-dqc-graph --task_mode=single --critic_mode=graph \
  --graph_targets=$V3/Lsame_h25_gp/targets.npz --graph_dataset_id="$GRAPH_DATASET_ID3" \
  --agent.backup_horizon=25 --agent.policy_chunk_size=5 --agent.use_chunk_critic=True \
  --agent.kappa_b=0.9 --agent.kappa_d=0.5 --agent.discount=0.999 \
  --agent.distill_method=expectile --agent.implicit_backup_type=quantile \
  --agent.batch_size=$BATCH --agent.num_qs=2 --agent.q_agg=mean \
  --env_name=$ENV3 --offline_steps=300000 --eval_interval=50000 \
  --eval_episodes=50 --video_episodes=0 --save_interval=50000 --seed=0 \
  --save_dir=$V3/qc/graph
```

### R3 — 4×4 P13 DQC TD

```bash
set -euo pipefail
cd "$ROOT/third_party/dqc"
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/dqc" $PY main.py \
  --run_group=v16-4x4p13-dqc-td --task_mode=single --critic_mode=td \
  --ogbench_dataset_path=$D4/train.npz \
  --agent.backup_horizon=25 --agent.policy_chunk_size=5 --agent.use_chunk_critic=True \
  --agent.kappa_b=0.9 --agent.kappa_d=0.5 --agent.discount=0.999 \
  --agent.distill_method=expectile --agent.implicit_backup_type=quantile \
  --agent.batch_size=$BATCH --agent.num_qs=2 --agent.q_agg=mean \
  --env_name=$ENV4 --offline_steps=300000 --eval_interval=50000 \
  --eval_episodes=50 --video_episodes=0 --save_interval=50000 --seed=0 \
  --save_dir=$V4/qc/td
```

### R4 — 4×4 P13 DQC graph critic

```bash
set -euo pipefail
GRAPH_DATASET_ID4=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' $V4/Lsame_h25_gp/targets.json)
cd "$ROOT/third_party/dqc"
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/dqc" $PY main.py \
  --run_group=v16-4x4p13-dqc-graph --task_mode=single --critic_mode=graph \
  --ogbench_dataset_path=$D4/train.npz \
  --graph_targets=$V4/Lsame_h25_gp/targets.npz --graph_dataset_id="$GRAPH_DATASET_ID4" \
  --agent.backup_horizon=25 --agent.policy_chunk_size=5 --agent.use_chunk_critic=True \
  --agent.kappa_b=0.9 --agent.kappa_d=0.5 --agent.discount=0.999 \
  --agent.distill_method=expectile --agent.implicit_backup_type=quantile \
  --agent.batch_size=$BATCH --agent.num_qs=2 --agent.q_agg=mean \
  --env_name=$ENV4 --offline_steps=300000 --eval_interval=50000 \
  --eval_episodes=50 --video_episodes=0 --save_interval=50000 --seed=0 \
  --save_dir=$V4/qc/graph
```

## 4. Checks and report

Run only the v1.6 tests:

```bash
set -euo pipefail
cd "$ROOT"
JAX_PLATFORMS=cpu $PY -m pytest tests/test_v16.py -q
cd "$ROOT/third_party/dqc"
PYTHONPATH="$ROOT:$ROOT/third_party/dqc" JAX_PLATFORMS=cpu \
  $PY -m pytest tests/test_v16_graph_critic.py -q
```

Record repository commits/clean state, test results, target statistics, speed
and selected batch size, run durations, and each task's success/wrong-button
curves, final chunk-critic and partial-critic losses, and first steps above 0.5
and 0.9 success in [results_v1.6.md](results_v1.6.md). Keep the required
historical reference rows and a push-command section at the end. Do not push.
