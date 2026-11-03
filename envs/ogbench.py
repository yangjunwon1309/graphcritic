"""Adapter for QC's state-based OGBench dataset rows."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from graphcritic.types import SchemaError, TransitionSet


def _episode_fields(terminals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign one contiguous sampler episode per QC terminal boundary."""
    terminal = np.asarray(terminals, dtype=np.bool_)
    n_rows = terminal.size
    episode_id = np.cumsum(np.r_[0, terminal[:-1]], dtype=np.int64)
    t = np.empty(n_rows, dtype=np.int64)
    last_episode = -1
    step = 0
    for row, episode in enumerate(episode_id):
        if episode != last_episode:
            step = 0
            last_episode = int(episode)
        t[row] = step
        step += 1
    if n_rows and not terminal[-1]:
        raise SchemaError("QC OGBench rows must end on a terminal boundary")
    return episode_id, t


def _dataset_parts(dataset_name: str) -> tuple[str, str]:
    parts = dataset_name.split("-")
    if "singletask" not in parts:
        raise ValueError("dataset_name must be a QC OGBench singletask ID")
    pos = parts.index("singletask")
    if pos == 0 or len(parts) <= pos + 1:
        raise ValueError(f"invalid OGBench dataset ID {dataset_name!r}")
    env_name = "-".join(parts[: pos - 1] + parts[pos:])
    file_stem = "-".join(parts[:pos] + parts[-1:])
    return env_name, file_stem


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ogbench_transitions(
    dataset_name: str,
    *,
    dataset_dir: str | Path = "~/.ogbench/data",
    dataset_path: str | Path | None = None,
    split: str = "train",
) -> tuple[TransitionSet, str]:
    """Load QC's flattened OGBench rows and relabel both state endpoints.

    QC removes terminal-only observation rows and shifts the remaining state
    array to form `next_observations`. This adapter uses the same loader and
    masks, then applies OGBench's official relabeler separately to current and
    next `qpos`/button-state arrays. `success` is always the label for
    `s_{t+1}`; `success_cur` is the label for `s_t`.
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    try:
        from third_party.qc.envs.ogbench_utils import make_ogbench_env_and_datasets
        import ogbench  # noqa: F401  # registers the official Gym environments
        from ogbench.relabel_utils import relabel_dataset
    except ImportError as exc:  # pragma: no cover - only the bench environment has these deps
        raise RuntimeError("OGBench loading requires the bench environment and initialized QC submodule") from exc

    env_name, file_stem = _dataset_parts(dataset_name)
    expanded_dir = Path(os.path.expanduser(str(dataset_dir)))
    train_path = (
        Path(dataset_path).expanduser().resolve()
        if dataset_path is not None
        else expanded_dir / f"{file_stem}.npz"
    )
    selected_path = train_path if split == "train" else train_path.with_name(f"{train_path.stem}-val.npz")

    env = eval_env = None
    try:
        env, eval_env, train_data, val_data = make_ogbench_env_and_datasets(
            dataset_name,
            dataset_dir=str(expanded_dir),
            dataset_path=str(train_path) if dataset_path is not None else None,
            add_info=True,
        )
        data: Mapping[str, Any] = train_data if split == "train" else val_data
        if not selected_path.is_file():
            raise FileNotFoundError(selected_path)

        with np.load(selected_path, allow_pickle=False) as raw:
            raw_terminal = np.asarray(raw["terminals"], dtype=np.bool_)
            current_mask = ~raw_terminal
            next_mask = np.r_[False, current_mask[:-1]]
            raw_row_idx = np.flatnonzero(current_mask).astype(np.int64)
            if "qpos" not in raw.files:
                raise SchemaError("OGBench singletask archive is missing qpos")
            next_states: dict[str, np.ndarray] = {
                "qpos": np.asarray(raw["qpos"][next_mask])
            }
            if "button_states" in raw.files:
                next_states["button_states"] = np.asarray(raw["button_states"][next_mask])

        n_rows = len(data["observations"])
        if raw_row_idx.shape != (n_rows,) or next_states["qpos"].shape[0] != n_rows:
            raise SchemaError("QC rows do not align with OGBench source state rows")
        if "masks" in data:
            current_success = np.asarray(data["masks"]) == 0
        else:
            current_states = {"qpos": np.asarray(data["qpos"])}
            if "button_states" in data:
                current_states["button_states"] = np.asarray(data["button_states"])
            relabel_dataset(env_name, env, current_states)
            current_success = np.asarray(current_states["masks"]) == 0

        relabel_dataset(env_name, env, next_states)
        success_next = np.asarray(next_states["masks"]) == 0
        terminals = np.asarray(data["terminals"], dtype=np.bool_)
        episode_id, t = _episode_fields(terminals)

        transitions = TransitionSet(
            feat=np.asarray(data["observations"], dtype=np.float32),
            next_feat=np.asarray(data["next_observations"], dtype=np.float32),
            episode_id=episode_id,
            t=t,
            success=success_next,
            success_cur=current_success,
            terminal=terminals,
            row_idx=raw_row_idx,
        )
        dataset_id = f"ogbench:{dataset_name}:{_sha256(selected_path)}"
        return transitions, dataset_id
    finally:
        if env is not None:
            env.close()
        if eval_env is not None and eval_env is not env:
            eval_env.close()
