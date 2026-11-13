"""State and episode-split preparation for Graph Critic v0.7 models.

The module deliberately contains no simulator access.  Training and prediction
consume only aligned offline state/action arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from graphcritic.types import SchemaError, SimStateSet, TransitionSet


@dataclass(frozen=True)
class PuzzleModelData:
    """Episode-aligned state/action arrays and leakage-safe split metadata."""

    state_raw: np.ndarray
    state_cont: np.ndarray
    state_bits: np.ndarray
    actions: np.ndarray
    episode_id: np.ndarray
    t: np.ndarray
    train_rows: np.ndarray
    heldout_rows: np.ndarray
    train_episodes: np.ndarray
    heldout_episodes: np.ndarray
    episode_end_rows: np.ndarray
    terminal_state_index: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    delta_std: np.ndarray
    epsilon_raw: np.ndarray
    epsilon_z: np.ndarray
    action_limits: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_cont(self) -> int:
        return int(self.state_cont.shape[1])

    @property
    def n_bits(self) -> int:
        return int(self.state_bits.shape[1])


@dataclass(frozen=True)
class CubeRobotModelData:
    """Robot-only 19D model inputs prepared from standard cube-double data."""

    state_raw: np.ndarray
    state_cont: np.ndarray
    state_bits: np.ndarray
    actions: np.ndarray
    episode_id: np.ndarray
    t: np.ndarray
    train_rows: np.ndarray
    heldout_rows: np.ndarray
    train_episodes: np.ndarray
    heldout_episodes: np.ndarray
    episode_end_rows: np.ndarray
    terminal_state_index: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    delta_std: np.ndarray
    epsilon_raw: np.ndarray
    epsilon_z: np.ndarray
    metadata: dict[str, Any]

    @property
    def n_cont(self) -> int:
        return int(self.state_cont.shape[1])

    @property
    def n_bits(self) -> int:
        return 0


def prepare_cube_robot_model_data(
    dataset_path: str,
    *,
    seed: int = 0,
    train_fraction: float = 0.9,
) -> tuple[TransitionSet, dict[str, str], CubeRobotModelData]:
    """Load aligned standard OGBench cube-double transition rows.

    The raw OGBench archive stores one extra terminal observation per episode.
    This adapter reproduces the row order used by the QC regular loader and
    DQC compact loader while retaining the terminal endpoint side table.
    """
    from pathlib import Path

    from graphcritic.envs.ogbench_cube import (
        OBS_DIM,
        archive_dataset_id,
        cube_double_goals,
        make_cube_transition_set,
        validate_observations,
    )

    path = Path(dataset_path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        required = {"observations", "actions", "terminals"}
        if not required.issubset(archive.files):
            raise SchemaError(f"cube-double archive is missing {sorted(required - set(archive.files))}")
        all_obs = validate_observations(archive["observations"])
        all_actions = np.asarray(archive["actions"], dtype=np.float32)
        raw_terminal = np.asarray(archive["terminals"], dtype=bool)
    if all_actions.ndim != 2 or all_actions.shape != (len(all_obs), 5):
        raise SchemaError("cube-double archive actions must align with observations and have width five")
    dataset_id = archive_dataset_id(path, "cube-double-play-v0")
    # Task labels are intentionally fixed to task 1 in the training adapter;
    # they do not enter the robot-only objective. Graph building relabels all
    # five tasks independently.
    trans = make_cube_transition_set(
        all_obs, all_actions, raw_terminal, goals=cube_double_goals()[1]
    )
    episode_id = trans.episode_id
    terminal = trans.terminal
    obs, next_obs = trans.feat, trans.next_feat
    n_rows = trans.n_rows
    current_raw_rows = trans.row_idx
    times = trans.t

    episodes = np.unique(episode_id)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)
    n_train = min(max(1, int(round(len(episodes) * train_fraction))), len(episodes) - 1)
    train_episodes = np.sort(shuffled[:n_train])
    heldout_episodes = np.sort(shuffled[n_train:])
    train_rows = np.flatnonzero(np.isin(episode_id, train_episodes)).astype(np.int64)
    heldout_rows = np.flatnonzero(np.isin(episode_id, heldout_episodes)).astype(np.int64)

    # State order mirrors state_table: all current transition occurrences,
    # followed by one terminal next-state occurrence for each terminal row.
    terminal_rows = np.flatnonzero(terminal)
    terminal_obs = next_obs[terminal_rows]
    robot_raw = np.concatenate((obs[:, :19], terminal_obs[:, :19]), axis=0).astype(np.float32)
    train_terminal = terminal_rows[np.isin(episode_id[terminal_rows], train_episodes)]
    terminal_ordinals = np.searchsorted(terminal_rows, train_terminal)
    train_state_rows = np.concatenate((train_rows, n_rows + terminal_ordinals))
    mean = np.mean(robot_raw[train_state_rows], axis=0, dtype=np.float64)
    std = np.maximum(np.std(robot_raw[train_state_rows], axis=0, dtype=np.float64), 1e-6)
    normalized = ((robot_raw.astype(np.float64) - mean) / std).astype(np.float32)

    ends = np.empty(n_rows, dtype=np.int64)
    terminal_state_index = np.full(n_rows, -1, dtype=np.int64)
    for ordinal, row in enumerate(terminal_rows):
        terminal_state_index[row] = n_rows + ordinal
    for episode in episodes:
        rows = np.flatnonzero(episode_id == episode)
        ends[rows] = int(rows[-1])
    nonterminal = np.flatnonzero(~terminal)
    next_ids = np.empty(n_rows, dtype=np.int64)
    next_ids[nonterminal] = nonterminal + 1
    next_ids[terminal_rows] = terminal_state_index[terminal_rows]
    delta = normalized[next_ids] - normalized[:n_rows]
    train_mask = np.isin(episode_id, train_episodes)
    epsilon_z = np.maximum(np.quantile(np.abs(delta[train_mask]), 0.90, axis=0), 0.05)
    epsilon_raw = epsilon_z * std
    chunk_rows = train_rows[train_rows + 5 <= ends[train_rows] + 1]
    chunk_end = chunk_rows + 5
    terminal_chunk = chunk_end == ends[chunk_rows] + 1
    chunk_end[terminal_chunk] = terminal_state_index[chunk_rows[terminal_chunk]]
    if not len(chunk_rows):
        raise SchemaError("cube-double training split has no valid five-step chunks")
    delta_std = np.maximum(np.std(normalized[chunk_end] - normalized[chunk_rows], axis=0), 1e-6)
    state_bits = np.empty((len(robot_raw), 0), dtype=np.float32)
    meta = {
        "schema": "v1.8-cube-double-robot",
        "continuous_fields": ["cube-double low-dimensional observation r[0:19]"],
        "continuous_dim": 19,
        "state_dim": 19,
        "button_width": 0,
        "train_fraction": float(train_fraction),
        "split_seed": int(seed),
        "n_episodes": int(len(episodes)),
        "n_train_episodes": int(len(train_episodes)),
        "n_heldout_episodes": int(len(heldout_episodes)),
        "train_episode_ids": train_episodes.tolist(),
        "heldout_episode_ids": heldout_episodes.tolist(),
        "state_mean": mean.tolist(),
        "state_std": std.tolist(),
        "delta_std": delta_std.tolist(),
        "epsilon_raw": epsilon_raw.tolist(),
        "epsilon_z": epsilon_z.tolist(),
        "input_part": "robot",
        "observation_dim": OBS_DIM,
        "source_dataset": str(path),
        "dataset_hash": dataset_id.rsplit(":", 1)[-1],
    }
    data = CubeRobotModelData(
        state_raw=robot_raw,
        state_cont=normalized,
        state_bits=state_bits,
        actions=trans.action,
        episode_id=episode_id,
        t=times,
        train_rows=train_rows,
        heldout_rows=heldout_rows,
        train_episodes=train_episodes,
        heldout_episodes=heldout_episodes,
        episode_end_rows=ends,
        terminal_state_index=terminal_state_index,
        state_mean=mean.astype(np.float32),
        state_std=std.astype(np.float32),
        delta_std=delta_std.astype(np.float32),
        epsilon_raw=epsilon_raw.astype(np.float32),
        epsilon_z=epsilon_z.astype(np.float32),
        metadata=meta,
    )
    return trans, {"dataset_id": dataset_id, "source": str(path)}, data


def prepare_puzzle_model_data(
    trans: TransitionSet,
    sim_state: SimStateSet,
    *,
    seed: int = 0,
    train_fraction: float = 0.9,
) -> PuzzleModelData:
    """Create the v0.7 state vector ``[physical robot r, button bits]``.

    ``r`` is the 19D physical robot readout from ``OGBenchPuzzleStateSchema``
    (14 joint/gripper qpos values plus the five action-coordinate values).
    The puzzle button values remain exact binary inputs and targets. Statistics,
    one-step tolerances, and delta scales are fit on training episodes only.
    """
    trans.validate()
    if trans.action is None or trans.action.shape[1] != 5:
        raise SchemaError("v0.7 puzzle models require recorded five-dimensional actions")
    sim_state.validate(trans.n_rows + int(np.count_nonzero(trans.terminal)))
    if sim_state.button_states is None:
        raise SchemaError("v0.7 puzzle models require aligned button_states")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")

    from graphcritic.featurize import state_table
    from graphcritic.state_schema import OGBenchPuzzleStateSchema

    states, _, _, _ = state_table(trans)
    schema = OGBenchPuzzleStateSchema()
    aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel,
           "button_states": sim_state.button_states}
    robot = schema.robot(states, aux).astype(np.float32)
    buttons = np.asarray(sim_state.button_states, dtype=np.float32)
    if buttons.shape[1] not in {9, 16, 20}:
        raise SchemaError("puzzle models require a supported 3x3, 4x4, or 4x5 button layout")
    state_raw = np.concatenate((robot, buttons), axis=1)
    n = trans.n_rows
    if state_raw.shape[0] != n + int(np.count_nonzero(trans.terminal)):
        raise SchemaError("state vectors do not match transition and terminal-state ordering")

    episodes = np.unique(trans.episode_id)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episodes)
    n_train_eps = min(max(1, int(round(len(episodes) * train_fraction))), len(episodes) - 1)
    train_eps = np.sort(shuffled[:n_train_eps])
    heldout_eps = np.sort(shuffled[n_train_eps:])
    train_rows = np.flatnonzero(np.isin(trans.episode_id, train_eps)).astype(np.int64)
    heldout_rows = np.flatnonzero(np.isin(trans.episode_id, heldout_eps)).astype(np.int64)

    train_state_rows = train_rows
    terminal_rows = np.flatnonzero(trans.terminal)
    train_terminal = terminal_rows[np.isin(trans.episode_id[terminal_rows], train_eps)]
    train_state_ids = np.concatenate((train_state_rows, n + np.searchsorted(terminal_rows, train_terminal)))
    cont = state_raw[:, :robot.shape[1]].astype(np.float64)
    mean = np.mean(state_raw[train_state_ids, :robot.shape[1]], axis=0, dtype=np.float64)
    std = np.maximum(np.std(state_raw[train_state_ids, :robot.shape[1]], axis=0, dtype=np.float64), 1e-6)
    z_cont = ((cont - mean) / std).astype(np.float32)
    bits = state_raw[:, robot.shape[1]:].astype(np.float32)

    # Build one-step endpoint indices for the entire occurrence table.
    state_next = np.empty(n, dtype=np.int64)
    nonterminal = np.flatnonzero(~trans.terminal)
    state_next[nonterminal] = nonterminal + 1
    state_next[terminal_rows] = n + np.arange(terminal_rows.size, dtype=np.int64)
    one_step_delta = cont[state_next] - cont[:n]
    train_transition_mask = np.isin(trans.episode_id, train_eps)
    one_step_abs_z = np.abs(one_step_delta[train_transition_mask] / std)
    epsilon_z = np.maximum(np.quantile(one_step_abs_z, 0.90, axis=0), 0.05)
    epsilon_raw = epsilon_z * std

    ends = np.empty(n, dtype=np.int64)
    terminal_lookup: dict[int, int] = {}
    for ordinal, row in enumerate(terminal_rows):
        terminal_lookup[int(trans.episode_id[row])] = n + ordinal
    terminal_state_index = np.asarray(
        [terminal_lookup.get(int(ep), -1) for ep in trans.episode_id], dtype=np.int64
    )
    for ep in episodes:
        rows = np.flatnonzero(trans.episode_id == ep)
        ends[rows] = int(rows[-1])

    # f predicts a complete h=5 chunk, so its output scale must be fitted on
    # the same five-action displacement targets used by the training loss.
    train_chunk_starts = train_rows[
        train_rows + 4 <= ends[train_rows]
    ]
    chunk_endpoint = train_chunk_starts + 5
    crosses_episode_end = chunk_endpoint > ends[train_chunk_starts]
    if np.any(crosses_episode_end):
        terminal_endpoint = terminal_state_index[train_chunk_starts[crosses_episode_end]]
        chunk_endpoint[crosses_episode_end] = terminal_endpoint
    valid_endpoint = chunk_endpoint >= 0
    if not np.any(valid_endpoint):
        raise SchemaError("training split has no valid five-action transition chunks")
    five_step_delta = cont[chunk_endpoint[valid_endpoint]] - cont[train_chunk_starts[valid_endpoint]]
    delta_std = np.maximum(np.std(five_step_delta, axis=0), 1e-6)
    metadata = {
        "schema": "v0.7-puzzle-r-plus-bits",
        "continuous_fields": ["robot_qpos_0_13", "eef_xyz_yaw_gripper"],
        "button_width": int(buttons.shape[1]),
        "continuous_dim": int(robot.shape[1]),
        "state_dim": int(state_raw.shape[1]),
        "train_fraction": float(train_fraction),
        "split_seed": int(seed),
        "n_episodes": int(len(episodes)),
        "n_train_episodes": int(len(train_eps)),
        "n_heldout_episodes": int(len(heldout_eps)),
        "train_episode_ids": train_eps.tolist(),
        "heldout_episode_ids": heldout_eps.tolist(),
        "state_mean": mean.tolist(),
        "state_std": std.tolist(),
        "delta_std": delta_std.tolist(),
        "epsilon_raw": epsilon_raw.tolist(),
        "epsilon_z": epsilon_z.tolist(),
        "action_limits": OGBenchPuzzleStateSchema().action_limits.tolist(),
    }
    return PuzzleModelData(
        state_raw=state_raw.astype(np.float32), state_cont=z_cont, state_bits=bits,
        actions=np.asarray(trans.action, dtype=np.float32), episode_id=trans.episode_id.copy(),
        t=trans.t.copy(), train_rows=train_rows, heldout_rows=heldout_rows,
        train_episodes=train_eps, heldout_episodes=heldout_eps, episode_end_rows=ends,
        terminal_state_index=terminal_state_index,
        state_mean=mean.astype(np.float32), state_std=std.astype(np.float32),
        delta_std=delta_std.astype(np.float32), epsilon_raw=epsilon_raw.astype(np.float32),
        epsilon_z=epsilon_z.astype(np.float32),
        action_limits=np.asarray(OGBenchPuzzleStateSchema().action_limits, dtype=np.float32),
        metadata=metadata,
    )


