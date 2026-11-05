# Benchmark environment and fork notes

Status date: 2026-11-05. The `bench` Pixi environment is Linux x86_64 only. It keeps the benchmark stack separate from the project's NumPy 1/2 core environments.

## v0.6 regret targets and OGBench Puzzle (2026-11-05)

- Graph targets use schema 0.6, default to non-negative chunk regret, and store a separate `loss_mask`; rows with both endpoint costs capped remain QC-valid but do not train the critic. Graph artifacts remain schema 0.5 and retain their previous format.
- The QC worktree is based on submodule commit `ce81e33`. Its local v0.6 changes validate target kind and explicit loss-mask provenance, attach both graph masks to sampled chunk starts, regress `softplus(Q)` on masked regret targets, log masked batch Spearman, and leave actor selection on the negative ensemble maximum. The edits are currently uncommitted in the submodule.
- `envs/ogbench.py` has a Puzzle-specific data path that reads the task's static button goal and QC-aligned arrays without constructing or stepping a MuJoCo simulator. The simulator remains available only to the separate servo/witness evaluations.
- `experiments/inspect_puzzle.py` opens standard play archives through QC's `load_dataset`, verifies sidecar/observation alignment, and measures button-configuration connectivity and Lights Out press counts for all five task definitions.
- `graphcritic.puzzle` contains the simulator-free GF(2) solver, minimum-press enumerator, and button-state transition helpers. `graphcritic.puzzle_bridges` searches R edges inside exact discrete-board buckets and records realized-time and bucket diagnostics.
- `experiments/gen_puzzle_task_data.py` implements the conditional P13 task-directed data branch with the official noisy button oracle, seeded GF(2) solutions, paired detours, optional post-success play, standard-data mixing, and script/config/seed provenance. Its transition budget counts QC-usable actions and writes a separate terminal-only final state for each episode.
- Generated immutable archives can be passed to `experiments/build_graph.py --ogbench-dataset ... --dataset-path ...`; the adjacent JSON tag and SHA are verified and included in the graph dataset ID. QC accepts the same path through `--ogbench_dataset_path` for graph-target training without dataset replacement.
- Puzzle actuator calibration and simulator witnesses use `--schema ogbench_puzzle`. The replay baseline is matched by graph, data, seed, and exact start-selection hash before puzzle witness-normal gates are assessed.
- Bench-only QC plumbing checks live in `third_party/qc/tests/test_graph_targets.py`; run with `JAX_PLATFORM_NAME=cpu pixi run -e bench pytest -q third_party/qc/tests/test_graph_targets.py`.

## v0.3 Scene observation and replay findings (2026-11-04)

The v0.3 `SceneObservationLayout` is checked against the installed OGBench `SceneEnv.compute_observation()` layout. On the first 1,000 Scene train rows, qpos-derived cube, drawer, and window positions agree with the named observation fields to a maximum absolute error of `2.1e-7` (acceptance tolerance `1e-5`). The `ogbench_scene_pos` representation uses 13 dimensions; its within-mode `d_step` is `0.1428919`, and all per-feature `rho_j` values are at most `0.1110`.

The 1,000-row feature-space replay check passes: `q99/d_step=0.0363` and `max/d_step=0.3862`, under the limits `0.1` and `0.5`. Raw observation replay has a max error of `1.8944`, concentrated in the raw cube yaw encoding. At row `525285`, the replay and sidecar quaternion-derived yaw differ by `0` radians while the raw observation yaw pair differs by `1.4763`; this supports the yaw-discontinuity hypothesis and justifies using `cos(4θ), sin(4θ)` for graph distance.

The first 100 deterministic-reset trials initially matched bitwise in only 93%. The sidecar omits MuJoCo's `qacc_warmstart`; explicitly clearing that solver warm-start vector after restoring qpos/qvel makes all 100 repeated 10-step rollouts bitwise identical. This reset behavior is part of the replay verifier, not the graph-building algorithm.

## Sources and pinned versions

| Component | Source / version | Use |
| --- | --- | --- |
| QC | `third_party/qc`, v0.6 working tree based on `ce81e33` | Main benchmark runner, dataset loaders, QC and QC-FQL agents |
| FQL | `third_party/fql`, commit `e8cd16eb490332924dfa2492097219f181765933` | Pinned upstream reference; the Kitchen environment is excluded for the dependency conflict below |
| Python | Pixi `3.11.*` (resolved in `pixi.lock`) | Bench interpreter |
| QC requirements | Exact versions from `third_party/qc/requirements.txt` | JAX, CUDA 12 wheels, NumPy, SciPy, OGBench, Gymnasium, MuJoCo and QC runtime |
| robomimic | v0.4.0, Git commit `7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9` | Reads v1.5 demonstration datasets; v0.4.0 is installed from its official Git tag because it is not published as a PyPI release |
| robosuite | `1.5.1` | Matches the v1.5 robomimic dataset generation/runtime |
| h5py | `3.13.0` | HDF5 demonstration loading |
| OpenCV | `opencv-python==4.11.0.86` | robosuite runtime dependency |
| PyTorch | `2.8.0` CPU wheel, with `torchvision==0.23.0` CPU wheel | Imported by robomimic utilities; QC training itself uses JAX/CUDA 12 |
| graphcritic | Local editable package | Allows core project imports and checks inside `bench` |

The full resolved dependency graph and wheel hashes are in `pixi.lock`. To create the environment from a clean checkout:

```bash
git submodule update --init --recursive
pixi install -e bench
```

## Dataset locations

QC expects robomimic datasets at:

```text
~/.robomimic/lift/mh/low_dim_v15.hdf5
~/.robomimic/can/mh/low_dim_v15.hdf5
~/.robomimic/square/mh/low_dim_v15.hdf5
```

The official robomimic v0.4 dataset guide provides these v1.5 MH files. Its download script can be run from the robomimic source checkout; the low-dimension files do not require post-processing. OGBench downloads its datasets automatically into `~/.ogbench/data` when QC creates the environment.

The Scene smoke uses QC dataset ID `scene-play-singletask-task1-v0`, backed by the official files:

```text
~/.ogbench/data/scene-play-v0.npz
~/.ogbench/data/scene-play-v0-val.npz
```

The E1 Cube comparison uses `cube-double-play-singletask-task2-v0`. The triple-cube task2 result is zero for all compared methods at 1M and is not a useful comparison.

## Data and reward conventions

### robomimic

- `third_party/qc/envs/robomimic_utils.py` sorts demonstrations by numeric ID and concatenates `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, and `object` from each episode's `obs` and `next_obs` with its actions, rewards, and `dones`.
- On the v1.5 MH files, the success reward and `dones` become 1 at the first successful transition and stay 1 through the remaining saved rows. For example, Lift `demo_0` has success and terminal flags at rows 42–47. QC preserves every such flag, so `terminals` and `masks = 1 - dones` can have multiple terminal rows inside one demonstration. Its online wrapper stops at the first positive reward.
- A Lift `demo_0` replay from the stored simulator state reached environment success at row 42, matching the first positive offline reward. The success label therefore belongs to `s_{t+1}`. The replay reset observation matched the dataset within `1.1e-7`; the maximum next-observation difference over the episode was `7.6e-3` from simulator drift.
- The robomimic TransitionSet adapter preserves these terminal flags and assigns a new sampler segment after every terminal row. It stores current-state success separately so sticky success rows remain goal states after a segment boundary.
- `third_party/qc/main.py::process_train_dataset` subtracts one from robomimic rewards. For the binary success reward this makes failures `-1` and successes `0`. With `--sparse=True`, the subsequent nonzero-to-`-1` mapping keeps failures penalized and successes at zero.
- `RobomimicLowdimWrapper.step` treats a positive environment reward as success, sets `info["success"] = 1`, and terminates. Otherwise it sets success to zero; the task-specific maximum episode length produces a truncation.

### OGBench

- `third_party/qc/envs/ogbench_utils.py::load_dataset` creates `next_observations` by shifting states while removing episode-final rows, so a transition does not join the end of one episode to the start of another. It shifts terminal markers and marks the final row terminal.
- For singletask tasks, QC calls `ogbench.relabel_utils.relabel_dataset` to add task rewards. QC's `--sparse=True` option then maps nonzero dataset rewards to `-1` and zero rewards to `0`.
- The current QC relabel path evaluates `qpos` and `button_states` aligned with `observations` (`s_t`). The Graph Critic `TransitionSet` stores `success_cur` and `success` separately for `s_t` and `s_{t+1}`. On the full 1M-transition Scene train split, 146 binary success labels differ between these endpoints (74 current-only, 72 next-only); dense relabel rewards differ on 14,763 rows. `envs/ogbench.py` obtains both labels by running OGBench's relabeler on the QC-aligned current and shifted next-state arrays.
- On evaluation, success is read from the environment's returned `info`; the benchmark success metric is not inferred from the transformed training reward.

### Sequence validity and episode boundaries

`third_party/qc/utils/datasets.py::Dataset.sample_sequence` samples contiguous slices from the flattened dataset. A slice can include rows after a terminal, but it marks those subsequent rows invalid (`valid=0`). It accumulates masks with `min` and terminal flags with `max`. QC's critic loss uses the final validity value, and its action-chunk behavior masks invalid actions. This is how sequences that reach an episode boundary are prevented from contributing cross-episode targets.

## T5 commands

Run the QC runner from its fork directory because its default agent config path is relative to that directory. These commands use offline W&B logging.

```bash
cd third_party/qc

# FQL baseline, horizon 1
MUJOCO_GL=egl WANDB_MODE=offline pixi run -e bench --manifest-path ../../pixi.toml -- python main.py \
  --run_group=reproduce --agent.alpha=100 \
  --env_name=lift-mh-low_dim --sparse=False --horizon_length=1 \
  --offline_steps=1000000 --online_steps=0

# QC, horizon 5 with best-of-32 actor sampling
MUJOCO_GL=egl WANDB_MODE=offline pixi run -e bench --manifest-path ../../pixi.toml -- python main.py \
  --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --env_name=lift-mh-low_dim --sparse=False --horizon_length=5 \
  --offline_steps=1000000 --online_steps=0

# Short OGBench loader/training/evaluation smoke
MUJOCO_GL=egl WANDB_MODE=offline pixi run -e bench --manifest-path ../../pixi.toml -- python main.py \
  --run_group=t5-smoke --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --env_name=scene-play-singletask-task1-v0 --sparse=True --horizon_length=5 \
  --offline_steps=1 --online_steps=0 --eval_interval=1 --eval_episodes=1 --video_episodes=0 --seed=0
```

The following Scene commands pair FQL (`h=1`) and QC (`h=5`) with their graph-cost variants. Run them from `third_party/qc`; Scene is downloaded automatically by QC. Graph targets use schema v0.3 and must match the dataset, horizon, unit, and flattened row order.

```bash
cd third_party/qc

# FQL baseline, h=1
MUJOCO_GL=egl WANDB_MODE=offline pixi run -e bench --manifest-path ../../pixi.toml -- python main.py \
  --run_group=e1-scene --agent.alpha=300 --env_name=scene-play-singletask-task1-v0 \
  --sparse=True --horizon_length=1 --offline_steps=1000000 --online_steps=0 --seed=0

# The baseline run above downloads the official Scene archive on first use.
DATASET_ID="ogbench:scene-play-singletask-task1-v0:$(sha256sum ~/.ogbench/data/scene-play-v0.npz | cut -d ' ' -f1)"

# FQL with graph costs, h=1
MUJOCO_GL=egl WANDB_MODE=offline pixi run -e bench --manifest-path ../../pixi.toml -- python main.py \
  --run_group=e1-scene-graph --agent.alpha=300 --agent.critic_mode=graph \
  --env_name=scene-play-singletask-task1-v0 --sparse=True --horizon_length=1 \
  --offline_steps=1000000 --online_steps=0 --graph_targets=../../outputs/scene-h1/targets.npz \
  --graph_dataset_id="$DATASET_ID" --seed=0

# QC baseline, h=5
MUJOCO_GL=egl WANDB_MODE=offline pixi run -e bench --manifest-path ../../pixi.toml -- python main.py \
  --run_group=e1-scene --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --env_name=scene-play-singletask-task1-v0 --sparse=True --horizon_length=5 \
  --offline_steps=1000000 --online_steps=0 --seed=0

# QC with graph costs, h=5
MUJOCO_GL=egl WANDB_MODE=offline pixi run -e bench --manifest-path ../../pixi.toml -- python main.py \
  --run_group=e1-scene-graph --agent.actor_type=best-of-n --agent.actor_num_samples=32 \
  --agent.critic_mode=graph --env_name=scene-play-singletask-task1-v0 --sparse=True --horizon_length=5 \
  --offline_steps=1000000 --online_steps=0 --graph_targets=../../outputs/scene-h5/targets.npz \
  --graph_dataset_id="$DATASET_ID" --seed=0
```

Build the v0.3 geometric reference from the OGBench dataset directly. This uses the raw simulator-state sidecar for position features and modes, but graph construction does not step or roll out the simulator:

```bash
pixi run -e bench -- python experiments/build_graph.py \
  --ogbench-dataset scene-play-singletask-task1-v0 \
  --config configs/graph/scene_v03_geometric.yaml --h 5 --unit steps \
  --out outputs/scene-v03/geometric-r15-rho1-l0 --force
```

For the learned bridge model, train the frame-gap classifier first, then build with `configs/graph/scene.yaml`:

```bash
pixi run -e bench -- python experiments/train_tau.py \
  --ogbench-dataset scene-play-singletask-task1-v0 \
  --config configs/graph/scene.yaml --out outputs/tau/scene --seed 0

pixi run -e bench -- python experiments/build_graph.py \
  --ogbench-dataset scene-play-singletask-task1-v0 \
  --config configs/graph/scene.yaml --h 5 --unit steps \
  --out outputs/scene-v03/learned-q90-r2-l5 --force

MUJOCO_GL=egl pixi run -e bench -- python experiments/eval_graph_policy.py \
  --graph outputs/scene-v03/learned-q90-r2-l5/graph.npz \
  --ogbench-dataset scene-play-singletask-task1-v0 \
  --env scene-singletask-task1-v0 \
  --out outputs/scene-v03/learned-q90-r2-l5/pi_g.json
```

The `r_H`, q=0.8 calibration ablation uses a separate checkpoint and config:

```bash
pixi run -e bench -- python experiments/train_tau.py \
  --ogbench-dataset scene-play-singletask-task1-v0 \
  --config configs/graph/scene.yaml --out outputs/tau/scene-rH --seed 0 \
  --far-exclusion-radius r_H

pixi run -e bench -- python experiments/build_graph.py \
  --ogbench-dataset scene-play-singletask-task1-v0 \
  --config configs/graph/scene_v03_learned_rh_q80.yaml --h 5 --unit steps \
  --out outputs/scene-v03/learned-rh-q80-r2-l5 --force

MUJOCO_GL=egl pixi run -e bench -- python experiments/eval_graph_policy.py \
  --graph outputs/scene-v03/learned-rh-q80-r2-l5/graph.npz \
  --ogbench-dataset scene-play-singletask-task1-v0 --starts 500 \
  --out outputs/scene-v03/learned-rh-q80-r2-l5/pi_g_500.json
```

The command reproduces the R6-passing q=0.8 calibration model and graph build. The current Scene πG success gate does not pass for this configuration; see [v0.3 results](results_v0.3.md).

The graph build writes `graph.npz`/`.json`, targets, `sim_state.npz`, transitions, and diagnostics. The sidecar is required for named Scene position features and contact/task modes; no simulator action is executed during time-policy graph construction. Run `eval_graph_policy.py` only after the graph has been saved. The legacy `--path-exec` open-loop replay applies only to the legacy bridge policies; time-policy paths use the graph tracker.

For E1, use the same task commands for `lift-mh-low_dim`, `can-mh-low_dim`, and `square-mh-low_dim`; then use `scene-play-singletask-task1-v0` with `--sparse=True` and `cube-double-play-singletask-task2-v0` with `--sparse=False`. The public plot data combines offline steps from 0–1M and online steps from 1–2M. An offline-only run should therefore use `--offline_steps=1000000` and compare evaluation index 10 (1.0M), not the 2.0M endpoint.

`main.py` defaults to seed 0. The public plot files aggregate the 0–1M offline segment and 1–2M online segment and provide 95% bootstrap confidence intervals (5,000 resamples), but do not specify the number or IDs of seeds behind the published curves; the pickle files contain aggregate means and confidence bounds, not raw seed runs. Do not treat a single local seed as a full statistical reproduction.

## Kitchen limitation

The FQL requirements pin `gym==0.23.1` and request `shimmy[gym-v21,gym-v26]`. The available Shimmy version that provides `gym-v26` requires `gym>=0.26.2`, which conflicts with FQL's exact Gym pin. Per the T5 guide's fallback, Kitchen is excluded from the bench environment; QC/FQL OGBench and robomimic tasks remain available through the QC fork.

## Verification record

The T5 environment and smoke checks completed on 2026-11-01:

- Pixi `bench` installed successfully; imports pass for `jax`, `mujoco`, `ogbench`, `robomimic`, `robosuite`, and `graphcritic`. All 86 PyPI pins listed by QC match the fork requirements exactly. The resolved lock is in `pixi.lock`.
- JAX reports the RTX 3080 as its GPU backend/device. PyTorch is intentionally CPU-only. MuJoCo EGL rendering was exercised and reported `NVIDIA GeForce RTX 3080/PCIe/SSE2` as the renderer.
- Official OGBench Scene train/validation files downloaded to the paths above and passed HTTP range, size, and NPZ CRC checks. Train is 267,055,508 bytes (SHA256 `66625e0cd9f2fcf92f5988f4d1bedeb2aa8b1e748316e507e89f4a9afb87c341`); validation is 26,795,398 bytes (SHA256 `6700c42a1602fa4a9457414395613d9d8ad6350181662a79751ca02dc0423cb2`). QC loading produced 1,000,000 transitions with observations `(1000000, 40)`, actions `(1000000, 5)`, 1,000 episode terminals, and matching next-observation lengths.
- The OGBench QC smoke command exited 0 after one offline update and one evaluation episode. Evaluation completed 750 environment steps, returned `-1500`, and had success `0`; this only checks execution, not policy quality.
- The `scene-singletask-task1-v0` environment reset and accepted a zero-action step with the expected 40 observation features and 5 action dimensions. No Cube dataset-backed training run was part of this smoke.
- Official robomimic v1.5 MH HDF5 files passed QC loader checks: Lift (300 demos/31,127 transitions, SHA256 `d8bc0de2ca48d0b834ab36561182182d101f4e84bfcec365291ec7acf12124dd`), Can (300/62,756, SHA256 `c4a34c837913446d295c6ca38f2b31888d69812198478af470ecb84a99e9f7d7`), and Square (300/80,731, SHA256 `da7dee0b6da49feae81dbde8fc6508660788e2131098f72ca44f60528405172d`). Lift reset/step replay matched the initial dataset observation within `1.1e-7`; first environment success matched the first positive reward at row 42. These files retain sticky post-success reward/done flags, so the adapter must represent success at its first transition and normalize episode boundaries to the schema.
- A QC-FQL Lift smoke completed one offline GPU update and one 300-step evaluation episode; evaluation success and return were both zero, as expected for an untrained one-step smoke.
- No full E1 runs were performed. Matching the offline portions requires the five tasks, 1M offline updates per run, multiple seeds, and comparison at evaluation index 10; the source does not publish the seed IDs/count. Smoke runs are not reproduction results, and `repro-v0.1` must not be tagged from these checks.

## Upstream references

- [robomimic v0.4.0 release notes](https://github.com/ARISE-Initiative/robomimic/releases/tag/v0.4.0)
- [robomimic v0.4 dataset guide](https://robomimic.github.io/docs/v0.4/datasets/robomimic_v0.1.html)
- [robosuite 1.5.1 release on PyPI](https://pypi.org/project/robosuite/1.5.1/)
- [PyTorch CPU wheel installation matrix](https://pytorch.org/get-started/previous-versions/)
