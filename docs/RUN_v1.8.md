# Graph Critic v1.8 — cube-double object-threshold bridges

Use the standard low-dimensional OGBench `cube-double-play-v0` archive. One
shared dθ model and five shared candidate/score sets feed task-specific graphs
and h=5 targets for singletask tasks 1–5. The graph is built from saved data;
only the QC/DQC evaluations step simulators.

## 0. Clean repositories and fixed data

The root, QC, FQL, and DQC repositories must be clean before training or
benchmark runs. No push is performed by this workflow.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
DATA=/home/url/.ogbench/data/cube-double-play-v0.npz
V=$ROOT/outputs/cube-double-play/v18
STATE20=$V/dtheta_robot_window5
STATE=$V/dtheta_robot_window5_r40k

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

## 1. S1 — data and row alignment

This verifies the 37D observation layout, transition row order, next-observation
alignment, terminal rows, and h=5 validity against the regular QC and compact
DQC loaders. It also counts goal states under each ordered task goal.

```bash
set -euo pipefail
cd "$ROOT"
$PY experiments/check_cube_bridges.py --cube-double-dataset "$DATA" \
  --alignment-only --out "$V/data_alignment.json"
```

Stop if `alignment.gate_passed` is false or any task has zero goal states.

## 2. S2 — train the shared robot-only dθ model

The 19D robot observation alone is normalized on a seed-0, episode-level 90/10
split and flattened into a five-frame (95D) input. The model is shared across
all five tasks. The command follows the v0.8 QRL/IQE training defaults with the
approved v1.8 window, horizon, and epsilon.

```bash
set -euo pipefail
cd "$ROOT"
JAX_PLATFORMS=cuda $PY experiments/train_quasimetric.py \
  --cube-double-dataset "$DATA" --input-part robot \
  --window 5 --constraint-horizon 5 --epsilon 1.25 \
  --activation gelu --layernorm --grad-clip 1.0 --dual adam \
  --steps 20000 --log-every 1000 --save-every 10000 --seed 0 \
  --out "$STATE20"
```

The command exits nonzero on a collapse alarm. Enforce both the collapse and
F3 chunk-violation gates with:

```bash
set -euo pipefail
cd "$ROOT"
$PY experiments/check_v18_model.py --run "$STATE20" --step 20000
```

This requires `collapse_alarm=false` in the final `train_metrics.jsonl` row and
`heldout_chunk_constraint.overall_violation_rate_d_gt_limit <= 0.05` in the
final `training.jsonl` row. The chunk threshold is `d > 6.25`.

### Resume from the saved 20k checkpoint to 40k

For the v1.8 continuation, resume the full serialized state, including primal
optimizer moments, dual λ and its optimizer moments, and the PRNG key. The new
directory preserves the original 10k/20k run and saves 30k and 40k checkpoints.

```bash
set -euo pipefail
cd "$ROOT"
JAX_PLATFORMS=cuda $PY experiments/train_quasimetric.py \
  --cube-double-dataset "$DATA" --input-part robot \
  --window 5 --constraint-horizon 5 --epsilon 1.25 \
  --activation gelu --layernorm --grad-clip 1.0 --dual adam \
  --steps 20000 --log-every 1000 --save-every 10000 --seed 0 \
  --resume-from "$STATE20/step_020000" --out "$STATE"
```

The output metadata records the restored step, optimizer/dual restoration flags,
source training-state SHA256, and starting λ. Confirm those fields against the
source checkpoint before using the 40k result. Evaluate step 40k with stop mode
first. If F3 alone fails, report mode permits continuing while recording the
F3 violation; collapse, dead units, and 19D/95D input checks remain hard gates.

```bash
set -euo pipefail
cd "$ROOT"
$PY experiments/check_v18_model.py --run "$STATE" --step 40000 --violation-gate stop
```

If stop mode fails only because `F3_passed=false`, run the report-mode check:

```bash
set -euo pipefail
cd "$ROOT"
$PY experiments/check_v18_model.py --run "$STATE" --step 40000 --violation-gate report
```

The following offline diagnostic divides held-out 5-step pairs into joint-speed
deciles and compares cube-moving with cube-stationary windows. It does not train
or use a simulator.

```bash
set -euo pipefail
cd "$ROOT"
$PY experiments/diag_v18_dtheta_violation.py --cube-double-dataset "$DATA" \
  --checkpoint "$STATE/step_040000" --out "$STATE/violation_diagnostic.json"
```

## 3. S3–S5 — δ sweep, G2 audit, task graphs and targets

This creates candidate and dθ score caches for δp = 0.5, 1, 2, 3, and 4 cm,
with δR = 15°, k = 32, K = 20, and bridge cost `dθ + 10.5`. It selects the
smallest δp whose five-task mean reachability at b=32 is within 0.5 percentage
points of the 4 cm result. A failed G2 audit stops before task graph creation.
Each task graph directory contains both QC `discounted` targets
(`targets.npz`, γ=0.99) and DQC `gamma_pow` targets (`targets_gamma_pow.npz`,
γ=0.99).

```bash
set -euo pipefail
cd "$ROOT"
JAX_PLATFORMS=cuda $PY experiments/sweep_object_delta.py \
  --cube-double-dataset "$DATA" --checkpoint "$STATE/step_040000" \
  --max-budget 32 --embedding-pair-batch-size 4096 --out "$V/bridge_sweep"
```

The script stops if more than 5% of accepted bridges disagree on contact or
elevated-cube state, or if any accepted bridge violates δp. G2 is also
independently recomputed for every task graph:

```bash
set -euo pipefail
cd "$ROOT"
SELECTED=$($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); print(format(d["selected_delta_pos_m"], ".3f"))' \
  "$V/bridge_sweep/delta_sweep.json")
$PY experiments/check_cube_bridges.py --cube-double-dataset "$DATA" \
  --candidate-cache "$V/bridge_sweep/candidates/delta_${SELECTED}.npz" \
  --graph "$V/bridge_sweep/graphs/task1" "$V/bridge_sweep/graphs/task2" \
         "$V/bridge_sweep/graphs/task3" "$V/bridge_sweep/graphs/task4" \
         "$V/bridge_sweep/graphs/task5" \
  --out "$V/bridge_sweep/g2_recheck.json"
```

Require `alignment.gate_passed=true`, all five graph hashes to recompute, zero
accepted position/rotation violations, and grasp mismatch at most 0.05.

## 4. S6/S8 — QC, one task/seed at a time

The QC README's cube-domain command uses `sparse=False`; use that for both TD
and graph runs. The graph run changes only the critic target. Settings follow
the v1.2b QC graph comparison: best-of-32, five-step action chunks, batch 256,
α=100, two Qs, layer norm, mean Q aggregation, and learning rate 3e-4. Run
seed 0 first, then seeds 1 and 2 after the complete seed-0 QC and DQC blocks.

```bash
set -euo pipefail
cd "$ROOT/third_party/qc"
for SEED in 0; do
  for TASK in 1 2 3 4 5; do
    ENV=cube-double-play-singletask-task${TASK}-v0
    TARGET="$V/bridge_sweep/graphs/task${TASK}"
    DATASET_ID=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' \
      "$TARGET/targets.json")
    MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" $PY main.py \
      --run_group="v18-cube-t${TASK}-qc-td-s${SEED}" \
      --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
      --agent.action_chunking=True --agent.critic_mode=td --agent.alpha=100 \
      --agent.batch_size=256 --agent.lr=0.0003 --agent.num_qs=2 \
      --agent.layer_norm=True --agent.q_agg=mean --env_name="$ENV" \
      --ogbench_dataset_path="$DATA" --sparse=False --horizon_length=5 \
      --offline_steps=300000 --online_steps=0 --eval_interval=25000 \
      --eval_episodes=50 --save_interval=25000 --seed="$SEED" \
      --save_dir="$V/runs/QC_td/task${TASK}/seed${SEED}"
    MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" $PY main.py \
      --run_group="v18-cube-t${TASK}-qc-graph-s${SEED}" \
      --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
      --agent.action_chunking=True --agent.critic_mode=graph \
      --agent.graph_target_kind=discounted --agent.graph_target_unit=steps \
      --agent.graph_target_gamma=0.99 --agent.alpha=100 \
      --agent.batch_size=256 --agent.lr=0.0003 --agent.num_qs=2 \
      --agent.layer_norm=True --agent.q_agg=mean --env_name="$ENV" \
      --ogbench_dataset_path="$DATA" --sparse=False --horizon_length=5 \
      --offline_steps=300000 --online_steps=0 --eval_interval=25000 \
      --eval_episodes=50 --save_interval=25000 --seed="$SEED" \
      --graph_targets="$TARGET/targets.npz" --graph_dataset_id="$DATASET_ID" \
      --save_dir="$V/runs/QC_graph/task${TASK}/seed${SEED}"
  done
done
```

For seed 0, retain the outer value `SEED=0` above. Repeat the same block for
`SEED=1` and `SEED=2` after the DQC seed-0 block. Never run training processes
concurrently.

## 5. S7/S8 — DQC, one task/seed at a time

The TD baseline uses the approved DQC cube-triple coefficients with γ=0.99:
`h=25`, `h_a=5`, `κ_b=0.93`, `κ_d=0.8`. The graph run follows v1.7 R6:
direct action-level graph targets, `h=h_a=5`, no chunk critic, and γ=0.99.
Both use best-of-32, batch 256, two Qs, mean aggregation, and 300k updates.

```bash
set -euo pipefail
cd "$ROOT/third_party/dqc"
for SEED in 0; do
  for TASK in 1 2 3 4 5; do
    ENV=cube-double-play-singletask-task${TASK}-v0
    TARGET="$V/bridge_sweep/graphs/task${TASK}"
    DATASET_ID=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' \
      "$TARGET/targets_gamma_pow.json")
    MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/dqc" $PY main.py \
      --run_group="v18-cube-t${TASK}-dqc-td-s${SEED}" --task_mode=single --critic_mode=td \
      --agent.backup_horizon=25 --agent.policy_chunk_size=5 --agent.use_chunk_critic=True \
      --agent.kappa_b=0.93 --agent.kappa_d=0.8 --agent.discount=0.99 \
      --agent.distill_method=expectile --agent.implicit_backup_type=quantile \
      --agent.batch_size=256 --agent.num_qs=2 --agent.q_agg=mean \
      --agent.best_of_n=32 --agent.flow_steps=10 --agent.lr=0.0003 \
      --env_name="$ENV" --ogbench_dataset_path="$DATA" \
      --offline_steps=300000 --eval_interval=25000 --eval_episodes=50 \
      --video_episodes=0 --save_interval=25000 --seed="$SEED" \
      --save_dir="$V/runs/DQC_td/task${TASK}/seed${SEED}"
    MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/dqc" $PY main.py \
      --run_group="v18-cube-t${TASK}-dqc-graph-s${SEED}" --task_mode=single --critic_mode=graph \
      --graph_targets="$TARGET/targets_gamma_pow.npz" --graph_dataset_id="$DATASET_ID" \
      --agent.backup_horizon=5 --agent.policy_chunk_size=5 --agent.use_chunk_critic=False \
      --agent.graph_target_level=action --agent.kappa_b=0.9 --agent.kappa_d=0.5 \
      --agent.discount=0.99 --agent.distill_method=expectile \
      --agent.implicit_backup_type=quantile --agent.batch_size=256 \
      --agent.num_qs=2 --agent.q_agg=mean --agent.best_of_n=32 \
      --agent.flow_steps=10 --agent.lr=0.0003 --env_name="$ENV" \
      --ogbench_dataset_path="$DATA" --offline_steps=300000 \
      --eval_interval=25000 --eval_episodes=50 --video_episodes=0 \
      --save_interval=25000 --seed="$SEED" \
      --save_dir="$V/runs/DQC_graph/task${TASK}/seed${SEED}"
  done
done
```

For the remaining runs, repeat each block with `SEED=1` and then `SEED=2`.
Record the 25k–300k success curve, last-four-evaluation mean, first steps above
0.5 and 0.9, and area under the 25k–300k curve.
