# Graph Critic v1.7 — direct action-chunk graph targets in DQC

This run compares the 3×3 task-3 v1.6 DQC graph-distillation result with DQC
whose 5-step action critic directly regresses the h=5 `gamma_pow` targets.
The standard OGBench puzzle dataset is used; no new dataset is generated.

## 0. Repositories and relocated inputs

Commit the root and DQC changes before building or training. QC and FQL are
unchanged. All four worktrees must be clean at each experiment start.

```bash
set -euo pipefail
ROOT=/home/url/graphcritic
PY=$ROOT/.pixi/envs/bench/bin/python
D3=$ROOT/outputs/puzzle-3x3-task3
V=$D3/v17
ENV=puzzle-3x3-play-singletask-task3-v0
G0=$D3/artifacts/reference_graphs/g0
GR=$D3/artifacts/reference_graphs/gr-rho09
OLD=$D3/artifacts/d_theta/window5_chunk/step_020000
CAND=$D3/artifacts/candidates/v11/candidates_obj.npz
Lsame=$D3/artifacts/graph_Lsame
R2=$D3/runs/A4_dqc_graph_distill
TARGET=$V/Lsame_h5_gp

cd "$ROOT"
git status -sb
git -C third_party/qc status -sb
git -C third_party/fql status -sb
git -C third_party/dqc status -sb
```

## 1. Build h=5 `gamma_pow` targets

This uses the v1.6 graph inputs/settings and writes a new artifact directory.
The generated graph hash must match the existing Lsame graph before continuing.

```bash
set -euo pipefail
cd "$ROOT"
$PY experiments/build_judged_graph.py \
  --judge quasimetric --g0-graph "$G0/graph.npz" \
  --reference-graph "$GR/graph.npz" \
  --transitions "$G0/transitions.npz" --sim-state "$G0/sim_state.npz" \
  --checkpoint "$OLD" --candidate-cache "$CAND" \
  --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5 \
  --bridge-object-change same --b-max 1 \
  --target-kind gamma_pow --target-h 5 --discount-gamma 0.999 \
  --out "$TARGET"

$PY -c 'import json,sys; actual=json.load(open(sys.argv[1]))["graph_hash"]; expected=sys.argv[2]; print(actual); assert actual == expected' \
  "$TARGET/result.json" 3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa
```

The target must record `kind=gamma_pow`, `h=5`, `gamma=0.999`, the standard
task-3 dataset ID, and the matching Lsame graph hash. Stop if any check fails.

## 2. R5 — DQC graph-action

Only these settings differ from v1.6 R2: `backup_horizon=5`,
`use_chunk_critic=False`, `graph_target_level=action`, and the h=5 target path.
The v1.6 R2 settings are `q_agg=mean`, `num_qs=2`, batch size 256,
`discount=0.999`, best-of-N 32, flow steps 10, seed 0, and 300k updates.

```bash
set -euo pipefail
GRAPH_DATASET_ID=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' "$TARGET/targets.json")
cd "$ROOT/third_party/dqc"
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/dqc" $PY main.py \
  --run_group=v17-3x3t3-dqc-graph-action --task_mode=single --critic_mode=graph \
  --graph_targets="$TARGET/targets.npz" --graph_dataset_id="$GRAPH_DATASET_ID" \
  --agent.backup_horizon=5 --agent.policy_chunk_size=5 --agent.use_chunk_critic=False \
  --agent.graph_target_level=action \
  --agent.kappa_b=0.9 --agent.kappa_d=0.5 --agent.discount=0.999 \
  --agent.distill_method=expectile --agent.implicit_backup_type=quantile \
  --agent.batch_size=256 --agent.num_qs=2 --agent.q_agg=mean --agent.best_of_n=32 \
  --agent.flow_steps=10 --env_name="$ENV" \
  --offline_steps=300000 --eval_interval=50000 --eval_episodes=50 \
  --video_episodes=0 --save_interval=50000 --seed=0 --save_dir="$V/runs/R5_dqc_graph_action"
```

### Conditional R6

Run only if R5's 300k success rate is `≤0.90`. Rebuild the same graph with
`gamma=0.99`, then use `discount=0.99`; all other R5 settings stay fixed.

```bash
set -euo pipefail
TARGET99=$V/Lsame_h5_gp_g099
cd "$ROOT"
$PY experiments/build_judged_graph.py \
  --judge quasimetric --g0-graph "$G0/graph.npz" \
  --reference-graph "$GR/graph.npz" \
  --transitions "$G0/transitions.npz" --sim-state "$G0/sim_state.npz" \
  --checkpoint "$OLD" --candidate-cache "$CAND" \
  --embedding-pair-batch-size 4096 --bridge-lambda-steps 10.5 \
  --bridge-object-change same --b-max 1 \
  --target-kind gamma_pow --target-h 5 --discount-gamma 0.99 \
  --out "$TARGET99"
$PY -c 'import json,sys; actual=json.load(open(sys.argv[1]))["graph_hash"]; expected=sys.argv[2]; print(actual); assert actual == expected' \
  "$TARGET99/result.json" 3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa

GRAPH_DATASET_ID99=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["config"]["dataset_id"])' "$TARGET99/targets.json")
cd "$ROOT/third_party/dqc"
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/dqc" $PY main.py \
  --run_group=v17-3x3t3-dqc-graph-action-g099 --task_mode=single --critic_mode=graph \
  --graph_targets="$TARGET99/targets.npz" --graph_dataset_id="$GRAPH_DATASET_ID99" \
  --agent.backup_horizon=5 --agent.policy_chunk_size=5 --agent.use_chunk_critic=False \
  --agent.graph_target_level=action \
  --agent.kappa_b=0.9 --agent.kappa_d=0.5 --agent.discount=0.99 \
  --agent.distill_method=expectile --agent.implicit_backup_type=quantile \
  --agent.batch_size=256 --agent.num_qs=2 --agent.q_agg=mean --agent.best_of_n=32 \
  --agent.flow_steps=10 --env_name="$ENV" \
  --offline_steps=300000 --eval_interval=50000 --eval_episodes=50 \
  --video_episodes=0 --save_interval=50000 --seed=0 --save_dir="$V/runs/R6_dqc_graph_action_g099"
```

## 3. Fixed-row critic-target diagnostic

After R5 (and R6 if run), compare the v1.6 R2 distilled action critic and the
new direct action critic on the same 10,000 h=5 loss-mask rows. This only loads
data/checkpoints and evaluates critic outputs; it does not train or step an
environment.

```bash
set -euo pipefail
cd "$ROOT/third_party/dqc"
PYTHONPATH="$ROOT:$ROOT/third_party/dqc" $PY experiments/diagnose_action_graph_targets.py \
  --target-h5 "$TARGET/targets.npz" \
  --r2-run "$R2" --r5-run "$V/runs/R5_dqc_graph_action" \
  --n-pairs 10000 --seed 0 --out "$V/action_target_diagnostic.json"
```

The diagnostic resolves the unique completed 300k checkpoint under each run
base directory. If R6 is run, add
`--r6-run "$V/runs/R6_dqc_graph_action_g099" --r6-target-h5 "$TARGET99/targets.npz"`;
R6 is compared against its gamma=0.99 h=5 target while R2/R5 use gamma=0.999.

## 4. Tests

```bash
set -euo pipefail
cd "$ROOT/third_party/dqc"
PYTHONPATH="$ROOT:$ROOT/third_party/dqc" JAX_PLATFORMS=cpu \
  $PY -m pytest tests/test_v17_action_graph.py -q
```

Record target statistics, clean commits, the R5/R6 run curves and losses, and
the paired action-critic Spearman/MAE in `docs/results_v1.7.md`. Do not push.
