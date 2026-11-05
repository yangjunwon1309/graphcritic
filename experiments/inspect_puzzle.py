"""Inspect OGBench Puzzle play data and exact button-configuration overlap."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphcritic.puzzle import button_code, minimum_press_solution, toggle_matrix


def _task_infos(n_rows: int, n_cols: int):
    from ogbench.manipspace.envs.puzzle_env import PuzzleEnv

    env = PuzzleEnv.__new__(PuzzleEnv)
    env._num_rows = n_rows
    env._num_cols = n_cols
    env._num_buttons = n_rows * n_cols
    env._reward_task_id = 0
    env.set_tasks()
    return env.task_infos


def _shortest_press_path(adjacency: dict[int, set[int]], start: int, goal: int) -> int | None:
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        state, distance = queue.popleft()
        if state == goal:
            return distance
        for nxt in adjacency.get(state, ()):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def _episode_fields(terminals: np.ndarray):
    episode = np.cumsum(np.r_[0, terminals[:-1].astype(np.int64)], dtype=np.int64)
    time = np.empty(terminals.size, dtype=np.int64)
    prior = -1
    step = 0
    for i, ep in enumerate(episode):
        if int(ep) != prior:
            prior, step = int(ep), 0
        time[i] = step
        step += 1
    return episode, time


def inspect_size(
    size: str,
    dataset_dir: Path,
    *,
    dataset_path: Path | None = None,
    task_filter: int | None = None,
) -> dict:
    import ogbench
    from ogbench.relabel_utils import relabel_dataset
    from third_party.qc.envs.ogbench_utils import load_dataset
    from graphcritic.state_schema import OGBenchPuzzleStateSchema, estimate_puzzle_press_height
    from graphcritic.puzzle import button_code

    n_rows, n_cols = tuple(int(x) for x in size.split("x"))
    stem = f"puzzle-{size}-play-v0"
    path = dataset_path.expanduser().resolve() if dataset_path is not None else dataset_dir / f"{stem}.npz"
    if not path.is_file():
        ogbench.download_datasets([stem], str(dataset_dir))
    # Open the source through QC's own OGBench row loader. It removes the
    # terminal-only observation, aligns qpos/button sidecars to current rows,
    # and builds next_observations with the same shift used for training.
    qc_data = load_dataset(path, compact_dataset=False, add_info=True)
    with np.load(path, allow_pickle=False) as archive:
        required = {"observations", "actions", "terminals", "qpos", "qvel", "button_states"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        raw_terminals = np.asarray(archive["terminals"], dtype=np.bool_)
        raw_terminal_rows = np.flatnonzero(raw_terminals)
        terminal_buttons = np.asarray(archive["button_states"][raw_terminal_rows], dtype=np.int64)

    # This is the same transition-row mask and shifted terminal convention
    # used by OGBench/QC's regular-dataset loader; assert the implementation
    # above still agrees with this explicit source-array interpretation.
    current_mask = ~raw_terminals
    terminal_next = np.r_[raw_terminals[1:], True][current_mask]
    observations = np.asarray(qc_data["observations"], dtype=np.float32)
    qpos = np.asarray(qc_data["qpos"], dtype=np.float32)
    buttons = np.asarray(qc_data["button_states"], dtype=np.int64)
    actions = np.asarray(qc_data["actions"], dtype=np.float32)
    if not (len(observations) == len(qpos) == len(buttons) == len(terminal_next) == len(actions)):
        raise ValueError("QC transition masking did not align puzzle observations and sidecars")
    if not np.array_equal(np.asarray(qc_data["terminals"], dtype=np.bool_), terminal_next):
        raise ValueError("QC loader terminal shift differs from the expected next-state convention")
    if np.asarray(qc_data["next_observations"]).shape != observations.shape:
        raise ValueError("QC loader next observations do not align with current observations")
    terminal_rows = np.flatnonzero(terminal_next)
    if terminal_rows.size != terminal_buttons.shape[0]:
        raise ValueError("raw terminal button states do not align with QC episode ends")
    next_buttons = np.empty_like(buttons)
    nonterminal_rows = np.flatnonzero(~terminal_next)
    next_buttons[nonterminal_rows] = buttons[nonterminal_rows + 1]
    next_buttons[terminal_rows] = terminal_buttons
    if observations.shape[1] != 19 + 4 * (n_rows * n_cols):
        raise ValueError(f"unexpected {size} observation width {observations.shape[1]}")

    schema_fit = estimate_puzzle_press_height(
        observations,
        buttons,
        next_observations=np.asarray(qc_data["next_observations"], dtype=np.float32),
        next_button_states=next_buttons,
    )
    schema = OGBenchPuzzleStateSchema(press_height=float(schema_fit["press_height_m"]))
    schema_check_n = min(1000, len(observations))
    schema_coordinates = schema.coordinates(
        observations[:schema_check_n],
        {"qpos": qpos[:schema_check_n], "button_states": buttons[:schema_check_n]},
    )
    expected_codes = np.asarray([button_code(row) for row in buttons[:schema_check_n]], dtype=np.int64)
    schema_codes = schema_coordinates.q_p[:, 0].astype(np.int64)
    if not np.array_equal(schema_codes, expected_codes):
        raise ValueError("puzzle StateSchema button bitfield does not match button_states sidecar")
    mode_values, mode_counts = np.unique(schema_coordinates.mode, return_counts=True)

    episode, times = _episode_fields(terminal_next)
    starts = np.r_[0, np.flatnonzero(episode[1:] != episode[:-1]) + 1]
    ends = np.r_[starts[1:], len(episode)]
    lengths = ends - starts
    current_code = np.asarray([button_code(row) for row in buttons], dtype=np.int64)
    next_code = np.asarray([button_code(row) for row in next_buttons], dtype=np.int64)
    initial_codes = current_code[starts]
    initial_qpos = qpos[starts, :14]
    changed_rows = np.flatnonzero(np.any(buttons != next_buttons, axis=1))

    # Verify the observation's one-hot board signal against the sidecar.
    n_buttons = n_rows * n_cols
    check_n = min(1000, len(buttons))
    inferred = np.stack([
        np.argmax(observations[:check_n, 19 + 4 * i : 21 + 4 * i], axis=1)
        for i in range(n_buttons)
    ], axis=1)
    observation_match = int(np.count_nonzero(np.all(inferred == buttons[:check_n], axis=1)))

    adjacency: dict[int, set[int]] = {}
    for row in changed_rows:
        a, b = int(current_code[row]), int(next_code[row])
        adjacency.setdefault(a, set()).add(b)
    visited = set(current_code.tolist()) | set(next_code.tolist())
    config_episodes: dict[int, set[int]] = {}
    for code, ep in zip(np.r_[current_code, next_code], np.r_[episode, episode], strict=True):
        config_episodes.setdefault(int(code), set()).add(int(ep))
    episode_counts = np.asarray([len(v) for v in config_episodes.values()], dtype=np.int64)
    press_interval = []
    for start, end in zip(starts, ends, strict=True):
        local = changed_rows[(changed_rows >= start) & (changed_rows < end)]
        if local.size >= 2:
            press_interval.extend(np.diff(times[local]).tolist())

    matrix = toggle_matrix(n_rows, n_cols)
    toggle_match = 0
    if changed_rows.size:
        for row in changed_rows:
            difference = buttons[row] ^ next_buttons[row]
            toggle_match += int(np.any(np.all(matrix.T == difference[None, :], axis=1)))

    infos = _task_infos(n_rows, n_cols)
    tasks = []
    class _RelabelEnv:
        unwrapped = None
        _reward_task_id = 1
        _target_button_states = None

        def __init__(self, target):
            self.unwrapped = self
            self._target_button_states = np.asarray(target, dtype=np.int64)
            self._num_buttons = int(self._target_button_states.size)

        def reset(self):
            return None

    for task_idx, info in enumerate(infos, start=1):
        if task_filter is not None and task_idx != task_filter:
            continue
        start_board = np.asarray(info["init_button_states"], dtype=np.int64)
        goal_board = np.asarray(info["goal_button_states"], dtype=np.int64)
        start_code, goal_code = button_code(start_board), button_code(goal_board)
        presses_opt = int(minimum_press_solution(matrix, start_board ^ goal_board).sum())
        presses_data = _shortest_press_path(adjacency, start_code, goal_code)
        ratio = (
            float(presses_data / presses_opt) if presses_data is not None and presses_opt > 0
            else (1.0 if presses_data == 0 else None)
        )
        episode_solves = 0
        for ep_start, ep_end in zip(starts, ends, strict=True):
            if int(current_code[ep_start]) != start_code:
                continue
            if np.any(current_code[ep_start:ep_end] == goal_code) or int(next_code[ep_end - 1]) == goal_code:
                episode_solves += 1

        # Invoke OGBench's actual relabel function with a tiny data-only stub;
        # no MuJoCo environment is constructed or stepped here.
        relabel_data = {"button_states": buttons.copy()}
        relabel_dataset(f"puzzle-{size}-v0", _RelabelEnv(goal_board), relabel_data)
        relabel_ok = bool(np.array_equal(
            relabel_data["masks"] == 0,
            np.all(buttons == goal_board[None, :], axis=1),
        ))
        tasks.append({
            "task": task_idx,
            "start_code": start_code,
            "goal_code": goal_code,
            "start_button_states": start_board.tolist(),
            "goal_button_states": goal_board.tolist(),
            "task_connected": presses_data is not None,
            "presses_data": presses_data,
            "presses_opt": presses_opt,
            "presses_data_over_opt": ratio,
            "single_episode_solves": episode_solves,
            "official_relabel_matches_goal_equality": relabel_ok,
        })

    return {
        "size": size,
        "dataset": stem if dataset_path is None else str(path),
        "source_path": str(path),
        "transition_count": int(len(observations)),
        "episode_count": int(starts.size),
        "episode_length": {
            "p10": float(np.percentile(lengths, 10)),
            "p50": float(np.percentile(lengths, 50)),
            "p90": float(np.percentile(lengths, 90)),
            "max": int(np.max(lengths)),
        },
        "observation_dim": int(observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "button_sidecar_width": int(buttons.shape[1]),
        "button_observation_matches_first_1000": observation_match,
        "state_schema": {
            "button_bitfield_matches_sidecar_first_1000": int(np.count_nonzero(schema_codes == expected_codes)),
            "checked_rows": int(schema_check_n),
            "press_height_m": float(schema_fit["press_height_m"]),
            "toggle_pairs_for_press_height": int(schema_fit["toggle_pairs"]),
            "mode_counts_first_1000": {str(int(mode)): int(count) for mode, count in zip(mode_values, mode_counts, strict=True)},
        },
        "initial_button_configs": int(np.unique(initial_codes).size),
        "initial_arm_poses_rounded_1e-3": int(np.unique(np.round(initial_qpos, 3), axis=0).shape[0]),
        "n_configs_visited": int(len(visited)),
        "n_configs_total": int(1 << n_buttons),
        "visited_fraction": float(len(visited) / (1 << n_buttons)),
        "episodes_per_config": {
            "p10": float(np.percentile(episode_counts, 10)),
            "p50": float(np.percentile(episode_counts, 50)),
            "p90": float(np.percentile(episode_counts, 90)),
            "max": int(np.max(episode_counts)),
        },
        "button_change_transition_count": int(changed_rows.size),
        "button_change_transition_fraction": float(changed_rows.size / max(1, len(buttons))),
        "mean_steps_between_button_changes": float(np.mean(press_interval)) if press_interval else None,
        "observed_changes_matching_one_lights_out_press": int(toggle_match),
        "tasks": tasks,
        "qc_loader_row_contract": "QC load_dataset(compact_dataset=False, add_info=True) opened the archive; official relabel_dataset checked with a data-only stub",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=["3x3", "4x4", "4x5"])
    parser.add_argument("--dataset-dir", type=Path, default=Path("~/.ogbench/data").expanduser())
    parser.add_argument("--dataset-path", type=Path, help="Inspect a generated/custom NPZ instead of standard play")
    parser.add_argument("--task", type=int, help="Inspect only this task's connectivity and press ratio")
    parser.add_argument("--out", type=Path, default=Path("outputs/puzzle/inspect_puzzle.json"))
    args = parser.parse_args()
    if args.dataset_path is not None and (len(args.sizes) != 1 or args.task is None):
        parser.error("--dataset-path requires exactly one --sizes entry and a --task")
    reports = [
        inspect_size(
            size,
            args.dataset_dir,
            dataset_path=args.dataset_path,
            task_filter=args.task,
        )
        for size in args.sizes
    ]
    payload = {"schema_version": "0.6", "reports": reports}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
