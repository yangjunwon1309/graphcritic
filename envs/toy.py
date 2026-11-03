"""Small deterministic graph-learning datasets for CPU tests and demos."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from graphcritic.types import SchemaError, TransitionSet

Coord = tuple[int, ...]


def _pack_episodes(episodes: list[list[Coord]], goal: Coord) -> TransitionSet:
    feat_parts: list[np.ndarray] = []
    next_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    success_parts: list[np.ndarray] = []
    success_cur_parts: list[np.ndarray] = []
    terminal_parts: list[np.ndarray] = []
    row_offset = 0
    for episode_id, states in enumerate(episodes):
        if len(states) < 2:
            continue
        current = np.asarray(states[:-1], dtype=np.float32)
        nxt = np.asarray(states[1:], dtype=np.float32)
        n = current.shape[0]
        feat_parts.append(current)
        next_parts.append(nxt)
        episode_parts.append(np.full(n, episode_id, dtype=np.int64))
        time_parts.append(np.arange(n, dtype=np.int64))
        success_parts.append(np.all(nxt == np.asarray(goal, dtype=np.float32), axis=1))
        success_cur_parts.append(np.all(current == np.asarray(goal, dtype=np.float32), axis=1))
        terminal = np.zeros(n, dtype=np.bool_)
        terminal[-1] = True
        terminal_parts.append(terminal)
        row_offset += n

    if not feat_parts:
        raise SchemaError("toy generator produced no transitions")
    feat = np.concatenate(feat_parts, axis=0)
    next_feat = np.concatenate(next_parts, axis=0)
    return TransitionSet(
        feat=feat,
        next_feat=next_feat,
        episode_id=np.concatenate(episode_parts),
        t=np.concatenate(time_parts),
        success=np.concatenate(success_parts),
        success_cur=np.concatenate(success_cur_parts),
        terminal=np.concatenate(terminal_parts),
        row_idx=np.arange(row_offset, dtype=np.int64),
    )


def _episode_rollout(
    start: Coord,
    goal: Coord,
    step,
    *,
    max_len: int,
) -> list[Coord]:
    states = [start]
    for _ in range(max_len):
        states.append(step(states[-1]))
        if states[-1] == goal:
            break
    return states


def make_chain(
    L: int = 20,
    n_episodes: int = 50,
    p_forward: float = 0.7,
    max_len: int = 200,
    seed: int = 0,
) -> tuple[TransitionSet, dict]:
    """Generate random walks on ``0..L-1`` with ``L-1`` as the goal."""
    if isinstance(L, bool) or L < 2 or n_episodes < 1 or max_len < 1:
        raise ValueError("L must be >=2 and n_episodes/max_len must be positive")
    if not np.isfinite(p_forward) or not 0.0 <= p_forward <= 1.0:
        raise ValueError("p_forward must be between zero and one")
    rng = np.random.default_rng(seed)
    goal = (L - 1,)
    episodes: list[list[Coord]] = []
    for _ in range(n_episodes):
        def step(state: Coord) -> Coord:
            direction = 1 if rng.random() < p_forward else -1
            return (max(0, min(L - 1, state[0] + direction)),)

        episodes.append(_episode_rollout((0,), goal, step, max_len=max_len))
    coords = [(x,) for x in range(L)]
    d_star = {coord: float(L - 1 - coord[0]) for coord in coords}
    return _pack_episodes(episodes, goal), {"d_star": d_star, "coords": coords}


def _normalize_walls(
    W: int,
    H: int,
    walls: Iterable[tuple[int, int]] | np.ndarray | None,
) -> set[tuple[int, int]]:
    if walls is None:
        # Deliberate bottleneck: a vertical wall with one opening at y=0.
        wall_set = {(W // 2, y) for y in range(H) if y != 0}
    elif isinstance(walls, np.ndarray):
        if walls.shape != (W, H):
            raise ValueError("boolean wall arrays must have shape (W, H)")
        wall_set = {tuple(map(int, pair)) for pair in np.argwhere(walls)}
    else:
        wall_set = {tuple(map(int, cell)) for cell in walls}
    if any(len(cell) != 2 or not (0 <= cell[0] < W and 0 <= cell[1] < H) for cell in wall_set):
        raise ValueError("wall coordinates must lie inside the grid")
    return wall_set


def _grid_distances(
    W: int,
    H: int,
    walls: set[tuple[int, int]],
    goal: tuple[int, int],
) -> dict[tuple[int, int], float]:
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    distances: dict[tuple[int, int], float] = {goal: 0.0}
    queue = deque([goal])
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < W and 0 <= nxt[1] < H) or nxt in walls or nxt in distances:
                continue
            distances[nxt] = distances[(x, y)] + 1.0
            queue.append(nxt)
    return { (x, y): distances.get((x, y), float("inf"))
             for x in range(W) for y in range(H) if (x, y) not in walls }


def make_grid(
    W: int = 10,
    H: int = 10,
    walls: Iterable[tuple[int, int]] | np.ndarray | None = None,
    n_episodes: int = 100,
    eps: float = 0.3,
    max_len: int = 200,
    seed: int = 0,
) -> tuple[TransitionSet, dict]:
    """Generate grid walks with epsilon-random deviations from BFS actions."""
    if isinstance(W, bool) or isinstance(H, bool) or W < 2 or H < 2:
        raise ValueError("W and H must both be >=2")
    if n_episodes < 1 or max_len < 1:
        raise ValueError("n_episodes and max_len must be positive")
    if not np.isfinite(eps) or not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be between zero and one")
    wall_set = _normalize_walls(W, H, walls)
    goal = (W - 1, H - 1)
    if goal in wall_set:
        raise ValueError("the grid goal cannot be a wall")
    d_star = _grid_distances(W, H, wall_set, goal)
    starts = [coord for coord, distance in d_star.items() if np.isfinite(distance) and coord != goal]
    if not starts:
        raise ValueError("grid has no reachable non-goal start state")

    rng = np.random.default_rng(seed)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def neighbors(state: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = state
        result = []
        for dx, dy in directions:
            nxt = (x + dx, y + dy)
            if 0 <= nxt[0] < W and 0 <= nxt[1] < H and nxt not in wall_set:
                result.append(nxt)
            else:
                result.append(state)
        return result

    episodes: list[list[Coord]] = []
    for _ in range(n_episodes):
        start = starts[int(rng.integers(len(starts)))]

        def step(state: tuple[int, int]) -> tuple[int, int]:
            options = neighbors(state)
            if rng.random() < 1.0 - eps:
                best_distance = d_star[state] - 1.0
                optimal = [candidate for candidate in options if d_star.get(candidate, float("inf")) == best_distance]
                if optimal:
                    return optimal[int(rng.integers(len(optimal)))]
            return options[int(rng.integers(len(options)))]

        episodes.append(_episode_rollout(start, goal, step, max_len=max_len))
    coords = [(x, y) for x in range(W) for y in range(H) if (x, y) not in wall_set]
    return _pack_episodes(episodes, goal), {"d_star": d_star, "coords": coords}


def make_stitch(L: int = 20, seed: int = 0) -> tuple[TransitionSet, dict]:
    """Generate two chain fragments that connect only when episodes are stitched."""
    if isinstance(L, bool) or L < 4:
        raise ValueError("L must be >=4")
    midpoint = L // 2
    goal = (L - 1,)
    episode_a = [(x,) for x in range(midpoint + 1)]
    episode_b = [(x,) for x in range(midpoint, L)]
    coords = [(x,) for x in range(L)]
    d_star = {coord: float(L - 1 - coord[0]) for coord in coords}
    trans = _pack_episodes([episode_a, episode_b], goal)
    return trans, {"d_star": d_star, "coords": coords, "seed": int(seed)}
