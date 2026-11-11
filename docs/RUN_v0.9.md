# Graph Critic v0.9 — direct graph Q runbook

This runbook evaluates the approved critic replacement on standard Puzzle 3×3 task 3. It uses the existing v0.8.4b `O_np1` graph, fixed window-5 state quasimetric checkpoint, candidate cache, and embedding cache. It trains four action-conditioned quasimetric heads, checks them offline, then trains only the QC flow-BC actor while selecting evaluation actions with direct graph Q. It does not rebuild the graph or train a critic.

The implementation preserves the existing graph, Dijkstra, bridge-judge, and target paths. Direct Q reads the graph's stored cost and converts it back to steps. Since the policy observation does not contain every joint position used by the graph features, simulator evaluation reads aligned `qpos` and `button_states` from `env.get_state()` when available, with a MuJoCo readout fallback; those values are combined with observations using the same state-feature transform as the graph builder.

## 0. Pin the run checkout

Run from a clean `quasi_develop` checkout after committing the implementation. Record the commit and make sure both the root checkout and QC submodule match the run version.

```bash
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
STATE_CKPT=$D3/v08/check/window5_chunk/step_020000
EMBEDDINGS=$D3/v08/embeddings/embeddings-9adaf65cb96691db-d8e00c609a016926.npy
```

The v0.9 scorer is pinned to O graph hash `bb5b26bf1776785d7196745d97ee1b98e177feeb7112c69ef6344c9c3b3548f9`, state checkpoint SHA256 `9adaf65cb96691dbece0448ebe32535edfe5fecf3c2125c6cddada77557965cd`, and the existing 201,876-row embedding cache. Do not regenerate these artifacts for this run.

## R0. Run the focused tests

```bash
$PY -m pytest tests/test_direct_q_v09.py tests/test_quasimetric_v08.py -q
JAX_PLATFORM_NAME=cpu pixi run -e bench pytest -q \
  third_party/qc/tests/test_graph_direct.py \
  third_party/qc/tests/test_graph_targets.py
```

Stop if either command fails. The tests cover QRL constraint signs and ε slack, observation plus live-readout feature parity, the `h + C(x)` min-plus identity, infinity/fallback behavior, an unchanged critic and target critic after an update, and existing v0.8 embedding behavior.

## R1. Train the four action heads

```bash
$PY experiments/train_action_quasimetric.py $COMMON3 \
  --state-checkpoint $STATE_CKPT \
  --window 5 --horizon 5 --epsilon-c1 1.25 --epsilon-c2 0.25 \
  --steps 20000 --log-every 1000 --save-every 10000 \
  --seeds 0 1 2 3 --out $D3/v09/action_heads
```

Each seed trains independently with its own dual variables. The trainer writes `seed_N/step_010000` and `seed_N/step_020000`, `heldout.json`, `metadata.json`, `training.jsonl`, and a combined `summary.json`.

**Gate A, each head:** continue only if the run reaches 20k for all four seeds, no F3 collapse alarm fires, the dead-unit fraction is zero, and held-out `d_C1 > 6.25` is at most 5%. Record C2 violation, the median distance to offsets 5/10/15/20, and final λ₁/λ₂. Exit code 3 indicates an F3 collapse alarm; exit code 4 indicates a failed per-head gate. Either code stops the v0.9 run before R2.

## R2. Write the scorer config and run the offline gates

The writer checks the exact O graph hash and provenance, dataset identity, candidate-node order, fixed state-checkpoint SHA, embedding-cache hashes, and all four completed action heads.

```bash
$PY experiments/write_direct_q_config.py \
  --graph $D3/v08/O_np1/graph.npz \
  --candidate-cache $D3/v08/candidates_window5.npz \
  --embedding-cache $EMBEDDINGS \
  --state-checkpoint $STATE_CKPT \
  --action-head-root $D3/v09/action_heads \
  --transitions $D3/g0/transitions.npz --sim-state $D3/g0/sim_state.npz \
  --out $D3/v09/direct_q_O.json

$PY experiments/eval_direct_q.py --config $D3/v09/direct_q_O.json $COMMON3 \
  --n 20000 --seed 0 --out $D3/v09/offline_gate.json
```

The evaluator samples held-out episode boundary rows, excludes each row's episode from its neighborhood, and compares `Q(s_t,a_t)` with `5 + C(s_{t+5})`. Continue to R3 only if both gates pass:

- **G1:** median ensemble distance to `x_{t+5}` is in `[4, 6.25]`, and the fraction above `6.25` is at most 5%.
- **G2:** median absolute Q error against `5 + C(s_{t+5})` is at most 5 steps.

The JSON also reports overestimation (`Q > y + 5`), lower-than-reference values (`Q < y − 5`), Spearman correlation, fallback rate and reasons, mean neighborhood size, decision latency, and sample counts. If either gate fails, stop before QC.

## R3. Train QC with direct Q replacing the critic

Run from the QC fork. W&B remains offline. The flow-BC actor trains for 300k updates; direct Q is used only to select among 32 action chunks during evaluation. `eval_bc` evaluates one unscored flow proposal from the same actor checkpoint.

```bash
cd $ROOT/third_party/qc
MUJOCO_GL=egl WANDB_MODE=offline PYTHONPATH="$ROOT:$ROOT/third_party/qc" $PY main.py \
  --run_group=v09-3x3t3-oracle-direct \
  --direct_q_config=$D3/v09/direct_q_O.json --eval_bc_also=True \
  --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --agent.critic_mode=graph_direct --agent.action_chunking=True \
  --agent.alpha=100 --agent.batch_size=256 --agent.lr=0.0003 \
  --agent.num_qs=2 --agent.layer_norm=True --agent.q_agg=mean \
  --env_name=$ENV --sparse=True --horizon_length=5 \
  --offline_steps=300000 --online_steps=0 \
  --eval_interval=50000 --eval_episodes=50 --save_interval=50000 --seed=0 \
  --save_dir=$D3/v09/qc/oracle_direct
```

The run group must contain `oracle`, because the graph's exact-board neighborhoods are an explicit oracle upper bound. The evaluator logs success and final wrong-button count alongside direct-Q fallback rate and reason counts, mean `|N(s)|`, selected Q, finite candidate-Q standard deviation, and mean decision time. BC metrics are written under `eval_bc/*`; direct-Q metrics are under `eval/*`.

## Results to record

Write measured results to `docs/results_v0.9.md` after the run. Record the clean implementation and execution commits, root/submodule dirty state, focused test outcomes, R1 head gates and λ values, R2 G1/G2 and diagnostics, and QC success at 50k intervals from 50k to 300k. Compare direct-Q and same-checkpoint BC at every evaluation. Include the v0.8.4b O cost-critic, O regret-critic, and v0.6 TD reference rows from the development specification. This runbook does not authorize new graph builds, other datasets/tasks, TD/BC training runs, or simulator diagnostics beyond the specified QC evaluation.
