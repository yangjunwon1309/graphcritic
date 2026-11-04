"""Adapter for the QC fork's low-dimensional robomimic demonstrations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from graphcritic.types import SchemaError, TransitionSet


LOW_DIM_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _qc_episode_ids(terminals: np.ndarray, source_episode_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split at every QC terminal marker, including sticky robomimic markers."""
    n_rows = terminals.size
    episode_id = np.empty(n_rows, dtype=np.int64)
    t = np.empty(n_rows, dtype=np.int64)
    segment = 0
    step = 0
    for row in range(n_rows):
        if row and source_episode_ids[row] != source_episode_ids[row - 1]:
            if not terminals[row - 1]:
                raise SchemaError("each robomimic demo must end on a QC terminal row")
            step = 0
        episode_id[row] = segment
        t[row] = step
        if terminals[row]:
            segment += 1
            step = 0
        else:
            step += 1
    if n_rows and not terminals[-1]:
        raise SchemaError("the final robomimic row must be terminal")
    return episode_id, t


def load_robomimic_transitions(
    env_name: str,
    dataset_path: str | Path | None = None,
) -> tuple[TransitionSet, str]:
    """Load a QC-compatible MH dataset and return rows plus a content-based ID.

    ``success`` always labels the next state. ``success_cur`` labels the current
    state from the previous transition's raw reward within the same source demo.
    Sticky QC ``dones`` are retained row for row; each terminal marker closes a
    sampler segment so the strict TransitionSet episode invariant still holds.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - only the bench environment has h5py
        raise RuntimeError("robomimic loading requires the bench environment (h5py)") from exc

    parts = env_name.split("-")
    if len(parts) != 3 or parts[1] not in {"mh", "ph", "mg"} or parts[2] != "low_dim":
        raise ValueError("env_name must look like 'lift-mh-low_dim'")
    task, dataset_type, _ = parts
    if dataset_path is None:
        filename = "low_dim_sparse_v15.hdf5" if dataset_type == "mg" else "low_dim_v15.hdf5"
        dataset_path = Path.home() / ".robomimic" / task / dataset_type / filename
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    obs_parts: list[np.ndarray] = []
    next_obs_parts: list[np.ndarray] = []
    reward_parts: list[np.ndarray] = []
    terminal_parts: list[np.ndarray] = []
    cur_success_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    source_episode_parts: list[np.ndarray] = []

    with h5py.File(path, "r") as source:
        demos = list(source["data"].keys())
        demos.sort(key=lambda name: int(name.removeprefix("demo_")))
        for source_episode, demo_name in enumerate(demos):
            group = source[f"data/{demo_name}"]
            actions = np.asarray(group["actions"], dtype=np.float32)
            rewards = np.asarray(group["rewards"], dtype=np.float32).reshape(-1)
            terminals = np.asarray(group["dones"], dtype=np.bool_).reshape(-1)
            observations = np.concatenate(
                [np.asarray(group[f"obs/{key}"]) for key in LOW_DIM_KEYS], axis=-1
            ).astype(np.float32, copy=False)
            next_observations = np.concatenate(
                [np.asarray(group[f"next_obs/{key}"]) for key in LOW_DIM_KEYS], axis=-1
            ).astype(np.float32, copy=False)
            length = actions.shape[0]
            if any(array.shape[0] != length for array in (rewards, terminals, observations, next_observations)):
                raise SchemaError(f"{demo_name} has misaligned QC arrays")
            if length == 0 or not terminals[-1]:
                raise SchemaError(f"{demo_name} does not end with a QC terminal marker")

            current_success = np.zeros(length, dtype=np.bool_)
            if length > 1:
                current_success[1:] = rewards[:-1] > 0.0

            obs_parts.append(observations)
            next_obs_parts.append(next_observations)
            reward_parts.append(rewards)
            terminal_parts.append(terminals)
            cur_success_parts.append(current_success)
            action_parts.append(actions)
            source_episode_parts.append(np.full(length, source_episode, dtype=np.int64))

    if not obs_parts:
        raise SchemaError("robomimic file contains no demonstrations")
    observations = np.concatenate(obs_parts, axis=0)
    next_observations = np.concatenate(next_obs_parts, axis=0)
    rewards = np.concatenate(reward_parts)
    terminals = np.concatenate(terminal_parts)
    success_cur = np.concatenate(cur_success_parts)
    source_episode_ids = np.concatenate(source_episode_parts)
    episode_id, t = _qc_episode_ids(terminals, source_episode_ids)

    transitions = TransitionSet(
        feat=observations,
        next_feat=next_observations,
        episode_id=episode_id,
        t=t,
        success=(rewards > 0.0),
        success_cur=success_cur,
        terminal=terminals,
        row_idx=np.arange(observations.shape[0], dtype=np.int64),
        action=np.concatenate(action_parts, axis=0),
    )
    dataset_id = f"robomimic:{env_name}:{_sha256(path)}"
    return transitions, dataset_id
