"""Generate task-directed OGBench Puzzle data with the official low-level oracle.

This is a controlled P13 dataset, not a replacement for OGBench's standard
play benchmark. It uses the official temporally correlated ButtonPlanOracle
for each press and changes only the high-level press sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphcritic.puzzle import apply_presses, solve_gf2, toggle_matrix


def _episode_arrays(rows: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.asarray(values) for name, values in rows.items()}


def _make_press_events(
    solution: np.ndarray,
    rng: np.random.Generator,
    *,
    p_detour: float,
) -> list[int]:
    main = [int(x) for x in rng.permutation(np.flatnonzero(solution))]
    outside = [button for button in range(solution.size) if not solution[button]]
    detours = [int(rng.choice(outside)) for _ in main if outside and rng.random() < p_detour]
    events = list(main)
    for button in detours:
        first = int(rng.integers(0, len(events) + 1))
        events.insert(first, button)
        second = int(rng.integers(first + 1, len(events) + 2))
        events.insert(second, button)
    return events


def _append_press(
    env,
    oracle,
    observation: np.ndarray,
    button: int,
    rows: dict[str, list[np.ndarray]],
    *,
    transition_budget: int,
    episode_budget: int,
) -> tuple[np.ndarray, int, bool]:
    """Run one official noisy press plan; return observation, steps, complete."""
    unwrapped = env.unwrapped
    before = np.asarray(unwrapped._cur_button_states, dtype=np.int64).copy()
    desired = before.copy()
    desired[button] ^= 1
    unwrapped._target_button = int(button)
    unwrapped._target_button_states = desired.copy()
    info = unwrapped.get_reset_info()
    oracle.reset(observation, info)
    used = 0
    ended = False
    while not oracle.done and used < min(transition_budget, episode_budget):
        action = np.clip(np.asarray(oracle.select_action(observation, info)), -1.0, 1.0)
        next_observation, _, terminated, truncated, next_info = env.step(action)
        rows["observations"].append(np.asarray(observation, dtype=np.float32).copy())
        rows["actions"].append(np.asarray(action, dtype=np.float32).copy())
        rows["qpos"].append(np.asarray(next_info["prev_qpos"], dtype=np.float32).copy())
        rows["qvel"].append(np.asarray(next_info["prev_qvel"], dtype=np.float32).copy())
        rows["button_states"].append(np.asarray(next_info["prev_button_states"], dtype=np.int64).copy())
        rows["terminals"].append(np.asarray(False))
        used += 1
        observation, info = next_observation, next_info
        if terminated or truncated:
            ended = True
            break
    complete = bool(oracle.done and not ended)
    expected = before.copy()
    expected ^= toggle_matrix(env.unwrapped._num_rows, env.unwrapped._num_cols)[:, button]
    if complete and not np.array_equal(np.asarray(unwrapped._cur_button_states), expected):
        complete = False
    return observation, used, complete


def _pack_numpy_random_state() -> dict[str, Any]:
    name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": keys.tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_random_state(state: dict[str, Any]) -> None:
    np.random.set_state((
        str(state["name"]),
        np.asarray(state["keys"], dtype=np.uint32),
        int(state["pos"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    ))


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_tuple_tree(item) for item in value]
    if isinstance(value, list):
        return [_tuple_tree(item) for item in value]
    return value


def _restore_tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_restore_tuple_tree(item) for item in value)
    return value


def _write_progress(path: Path, state: dict[str, Any], arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _collect_generated(
    *,
    size: str,
    task: int,
    transitions: int,
    seed: int,
    p_detour: float,
    p_continue: float,
    noise: float,
    noise_smoothing: float,
    max_episode_steps: int,
    max_attempts: int,
    checkpoint_dir: Path,
    checkpoint_every: int,
    resume: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import gymnasium as gym
    import ogbench  # noqa: F401, register official environments
    from ogbench.manipspace.envs.puzzle_env import PuzzleEnv
    from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle

    n_rows, n_cols = (int(part) for part in size.split("x"))
    static = PuzzleEnv.__new__(PuzzleEnv)
    static._num_rows, static._num_cols = n_rows, n_cols
    static._num_buttons = n_rows * n_cols
    static._reward_task_id = 0
    static.set_tasks()
    if task < 1 or task > len(static.task_infos):
        raise ValueError(f"task must be in 1..{len(static.task_infos)} for puzzle-{size}")
    task_info = static.task_infos[task - 1]
    start_board = np.asarray(task_info["init_button_states"], dtype=np.uint8)
    goal_board = np.asarray(task_info["goal_button_states"], dtype=np.uint8)
    matrix = toggle_matrix(n_rows, n_cols)
    solution = solve_gf2(
        matrix,
        start_board ^ goal_board,
        rng=np.random.default_rng(seed),
    )
    if not np.array_equal(apply_presses(start_board, solution, matrix), goal_board):
        raise RuntimeError("GF(2) solution failed the simulator-free task check")

    config = {
        "size": size,
        "task": task,
        "transitions": transitions,
        "seed": seed,
        "p_detour": p_detour,
        "p_continue": p_continue,
        "noise": noise,
        "noise_smoothing": noise_smoothing,
        "max_episode_steps": max_episode_steps,
        "max_attempts": max_attempts,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    progress_path = checkpoint_dir / "progress.json"
    if resume:
        if not progress_path.is_file():
            raise FileNotFoundError(f"no resumable progress file: {progress_path}")
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("config") != config:
            raise ValueError("checkpoint parameters or generator source do not match this run")
        specs = state["array_specs"]
        capacity = transitions + max_attempts + 1
        arrays: dict[str, np.memmap] = {
            name: np.load(checkpoint_dir / f"{name}.npy", mmap_mode="r+")
            for name in specs
        }
        expected_capacity = capacity
        if any(array.shape[0] != expected_capacity for array in arrays.values()):
            raise ValueError("checkpoint array capacity does not match the requested run")
        total = int(state["total_transitions"])
        raw_cursor = int(state["raw_rows"])
        episodes = int(state["episodes"])
        attempts = int(state["attempts"])
        successful_episodes = int(state["episodes_reaching_goal"])
        continued_episodes = int(state["episodes_continued_after_goal"])
        failed_attempts = int(state["failed_attempts"])
        collected_transitions = int(state["collected_before_final_truncation"])
    else:
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise FileExistsError(
                f"checkpoint exists at {checkpoint_dir}; pass --resume or remove it explicitly"
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {}
        arrays = {}
        specs: dict[str, dict[str, Any]] = {}
        capacity = transitions + max_attempts + 1
        total = raw_cursor = episodes = attempts = 0
        successful_episodes = continued_episodes = failed_attempts = 0
        collected_transitions = 0

    def save_checkpoint(*, complete: bool) -> None:
        state.update({
            "config": config,
            "array_specs": specs,
            "total_transitions": total,
            "raw_rows": raw_cursor,
            "episodes": episodes,
            "attempts": attempts,
            "episodes_reaching_goal": successful_episodes,
            "episodes_continued_after_goal": continued_episodes,
            "failed_attempts": failed_attempts,
            "collected_before_final_truncation": collected_transitions,
            "complete": complete,
            "rng_state": rng.bit_generator.state,
            "numpy_random_state": _pack_numpy_random_state(),
            "python_random_state": _tuple_tree(random.getstate()),
            "env_random_state": env.unwrapped.np_random.bit_generator.state,
        })
        _write_progress(progress_path, state, arrays)

    def metadata() -> dict[str, Any]:
        return {
            "generated_transitions": int(transitions),
            "generated_raw_rows_including_terminal_states": int(raw_cursor),
            "collected_before_final_truncation": int(collected_transitions),
            "generated_episodes": int(episodes),
            "attempts": int(attempts),
            "failed_attempts": int(failed_attempts),
            "episodes_reaching_goal": int(successful_episodes),
            "episodes_continued_after_goal": int(continued_episodes),
            "solution_press_count": int(solution.sum()),
            "solution": np.flatnonzero(solution).astype(int).tolist(),
            "task_start": start_board.astype(int).tolist(),
            "task_goal": goal_board.astype(int).tolist(),
        }

    if resume and state.get("complete"):
        generated = {name: np.asarray(array[:raw_cursor]).copy() for name, array in arrays.items()}
        return generated, metadata()

    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(
        f"puzzle-{size}-v0",
        terminate_at_goal=False,
        mode="data_collection",
        max_episode_steps=max_episode_steps + 1,
    )
    oracle = ButtonPlanOracle(
        env=env,
        noise=noise,
        noise_smoothing=noise_smoothing,
        gripper_always_closed=True,
    )
    if resume:
        # Initialize Gym's per-env RNG, then restore all streams to the latest
        # episode-boundary checkpoint before taking another reset.
        env.reset(seed=seed)
        rng.bit_generator.state = state["rng_state"]
        _restore_numpy_random_state(state["numpy_random_state"])
        random.setstate(_restore_tuple_tree(state["python_random_state"]))
        env.unwrapped.np_random.bit_generator.state = state["env_random_state"]

    try:
        while total < transitions:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"generated {total}/{transitions} transitions after {max_attempts} episode attempts"
                )
            observation, _ = env.reset(seed=seed if attempts == 1 else None)
            unwrapped = env.unwrapped
            unwrapped._cur_button_states = start_board.copy()
            unwrapped._target_button_states = goal_board.copy()
            unwrapped._apply_button_states()
            observation = unwrapped.compute_observation()
            rows: dict[str, list[np.ndarray]] = {
                name: [] for name in ("observations", "actions", "qpos", "qvel", "button_states", "terminals")
            }
            remaining = max_episode_steps
            complete = True
            press_count = 0
            for button in _make_press_events(solution, rng, p_detour=p_detour):
                observation, used, complete = _append_press(
                    env, oracle, observation, button, rows,
                    transition_budget=remaining - len(rows["observations"]),
                    episode_budget=max_episode_steps - len(rows["observations"]),
                )
                press_count += 1
                if not complete or len(rows["observations"]) >= remaining:
                    break
            goal_reached = np.array_equal(unwrapped._cur_button_states, goal_board)
            if complete and not goal_reached and len(rows["observations"]) < remaining:
                complete = False
            main_plan_succeeded = bool(complete and goal_reached)
            continue_random = bool(rng.random() < p_continue)
            if main_plan_succeeded:
                successful_episodes += 1
                if continue_random:
                    continued_episodes += 1
                    while len(rows["observations"]) < remaining:
                        button = int(rng.integers(n_rows * n_cols))
                        observation, _, complete = _append_press(
                            env, oracle, observation, button, rows,
                            transition_budget=remaining - len(rows["observations"]),
                            episode_budget=max_episode_steps - len(rows["observations"]),
                        )
                        press_count += 1
                        if not complete:
                            break
            if not rows["observations"]:
                failed_attempts += 1
                continue
            # A failed high-level plan is discarded so generated task data has
            # known press semantics. Failures never enter the saved dataset.
            if not main_plan_succeeded:
                failed_attempts += 1
                continue
            terminal_info = unwrapped.get_reset_info()
            rows["observations"].append(np.asarray(observation, dtype=np.float32).copy())
            rows["actions"].append(np.zeros_like(rows["actions"][-1], dtype=np.float32))
            rows["qpos"].append(np.asarray(terminal_info["qpos"], dtype=np.float32).copy())
            rows["qvel"].append(np.asarray(terminal_info["qvel"], dtype=np.float32).copy())
            rows["button_states"].append(
                np.asarray(terminal_info["button_states"], dtype=np.int64).copy()
            )
            rows["terminals"].append(np.asarray(True))
            episode_arrays = _episode_arrays(rows)
            n = int(episode_arrays["observations"].shape[0] - 1)
            collected_transitions += n
            take = min(n, transitions - total)
            episode_arrays = {
                name: values[: take + 1].copy()
                for name, values in episode_arrays.items()
            }
            episode_arrays["terminals"][:] = False
            episode_arrays["terminals"][-1] = True
            episode_arrays["actions"][-1] = 0.0
            if not arrays:
                specs = {
                    name: {"tail_shape": list(values.shape[1:]), "dtype": values.dtype.str}
                    for name, values in episode_arrays.items()
                }
                arrays = {
                    name: np.lib.format.open_memmap(
                        checkpoint_dir / f"{name}.npy",
                        mode="w+",
                        dtype=np.dtype(spec["dtype"]),
                        shape=(capacity, *spec["tail_shape"]),
                    )
                    for name, spec in specs.items()
                }
            end = raw_cursor + take + 1
            if end > capacity:
                raise RuntimeError("generated data exceeded the checkpoint memmap capacity")
            for name, values in episode_arrays.items():
                arrays[name][raw_cursor:end] = values
            raw_cursor = end
            total += take
            episodes += 1
            if episodes == 1 or episodes % checkpoint_every == 0 or total >= transitions:
                save_checkpoint(complete=total >= transitions)
                print(
                    f"generated puzzle-{size} task{task}: {total}/{transitions} transitions, "
                    f"{episodes} accepted episodes, {failed_attempts} retries",
                    file=sys.stderr, flush=True,
                )
    finally:
        env.close()

    if total != transitions:
        raise RuntimeError(f"generated only {total}/{transitions} transitions")
    save_checkpoint(complete=True)
    generated = {name: np.asarray(array[:raw_cursor]).copy() for name, array in arrays.items()}
    return generated, metadata()


def _standard_rows(path: Path, transitions: int, seed: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = ("observations", "actions", "terminals", "qpos", "qvel", "button_states")
        raw = {key: np.asarray(archive[key]) for key in required}
    terminal_rows = np.flatnonzero(raw["terminals"])
    starts = np.r_[0, terminal_rows[:-1] + 1]
    ends = terminal_rows + 1
    order = np.random.default_rng(seed).permutation(starts.size)
    picked: dict[str, list[np.ndarray]] = {key: [] for key in raw}
    count = 0
    for ep in order:
        start, end = int(starts[ep]), int(ends[ep])
        episode_transitions = max(0, end - start - 1)
        take = min(episode_transitions, transitions - count)
        if take <= 0:
            continue
        for key in raw:
            picked[key].append(np.array(raw[key][start : start + take + 1], copy=True))
        picked["terminals"][-1][:] = False
        picked["terminals"][-1][-1] = True
        picked["actions"][-1][-1] = 0.0
        count += take
        if count >= transitions:
            break
    if count != transitions:
        raise ValueError(f"standard dataset supplied only {count}/{transitions} requested QC transitions")
    return {key: np.concatenate(values, axis=0) for key, values in picked.items()}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("3x3", "4x4", "4x5"), required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--standard", type=Path, help="Standard puzzle play training archive for the mix")
    parser.add_argument("--transitions", type=int, help="Default: standard archive transition count")
    parser.add_argument("--mix", type=float, choices=(0.5, 1.0), default=1.0,
                        help="Generated fraction; 0.5 creates a 50/50 generated/standard dataset")
    parser.add_argument("--p-detour", type=float, default=0.3)
    parser.add_argument("--p-continue", type=float, default=0.5)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--noise-smoothing", type=float, default=0.5)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Persist a resumable episode-boundary checkpoint every N accepted episodes",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the output's progress directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.p_detour <= 1 or not 0 <= args.p_continue <= 1:
        parser.error("p-detour and p-continue must lie in [0, 1]")
    if args.noise < 0 or args.noise_smoothing < 0 or args.max_episode_steps < 1 or args.max_attempts < 1:
        parser.error("noise must be non-negative and episode/attempt limits must be positive")
    if args.checkpoint_every < 1:
        parser.error("checkpoint-every must be positive")

    standard_path = args.standard or (Path("~/.ogbench/data").expanduser() / f"puzzle-{args.size}-play-v0.npz")
    if not standard_path.is_file():
        raise FileNotFoundError(standard_path)
    with np.load(standard_path, allow_pickle=False) as archive:
        standard_count = int(np.count_nonzero(~np.asarray(archive["terminals"], dtype=np.bool_)))
    target_count = args.transitions or standard_count
    generated_count = int(round(target_count * args.mix))
    output = args.out
    if output.suffix != ".npz":
        output = output.with_suffix(".npz")
    checkpoint_dir = output.with_name(f"{output.stem}.progress")
    generated, generation_meta = _collect_generated(
        size=args.size,
        task=args.task,
        transitions=generated_count,
        seed=args.seed,
        p_detour=args.p_detour,
        p_continue=args.p_continue,
        noise=args.noise,
        noise_smoothing=args.noise_smoothing,
        max_episode_steps=args.max_episode_steps,
        max_attempts=args.max_attempts,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )
    arrays = [generated]
    standard_count_used = target_count - generated_count
    if standard_count_used:
        arrays.append(_standard_rows(standard_path, standard_count_used, args.seed + 1))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(arrays)
    merged = {
        key: np.concatenate([part[key] for part in arrays], axis=0)
        for key in ("observations", "actions", "terminals", "qpos", "qvel", "button_states")
    }
    qc_transition_count = int(merged["observations"].shape[0] - np.count_nonzero(merged["terminals"]))
    if qc_transition_count != target_count or not bool(merged["terminals"][-1]):
        raise RuntimeError(
            "final task-directed archive has an invalid QC transition count or episode ending: "
            f"{qc_transition_count}/{target_count}"
        )

    source_hash = _hash_file(Path(__file__).resolve())
    tag = (
        f"p13-{args.size}-task{args.task}-mix{args.mix:g}-d{args.p_detour:g}-"
        f"c{args.p_continue:g}-seed{args.seed}-script{source_hash[:12]}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    validation_source = standard_path.with_name(f"{standard_path.stem}-val.npz")
    validation_path = output.with_name(f"{output.stem}-val.npz")
    if not validation_source.is_file():
        raise FileNotFoundError(validation_source)
    validation_path.write_bytes(validation_source.read_bytes())
    metadata = {
        "dataset_tag": tag,
        "dataset_sha256": _hash_file(output),
        "source_script_sha256": source_hash,
        "standard_dataset": str(standard_path),
        "standard_dataset_sha256": _hash_file(standard_path),
        "validation_source": str(validation_source),
        "validation_is_standard_play": True,
        "size": args.size,
        "task": args.task,
        "seed": args.seed,
        "mix_generated_fraction": args.mix,
        "p_detour": args.p_detour,
        "p_continue": args.p_continue,
        "noise": args.noise,
        "noise_smoothing": args.noise_smoothing,
        "total_transitions": target_count,
        "raw_rows_including_terminal_states": int(merged["observations"].shape[0]),
        "qc_transition_count_verified": qc_transition_count,
        "generated_transition_budget": generated_count,
        "standard_transitions": standard_count_used,
        **generation_meta,
        "train_path": str(output),
        "validation_path": str(validation_path),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(checkpoint_dir)
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
