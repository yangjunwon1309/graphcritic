"""Direct graph-Q scoring for v0.9 and v1.0 Puzzle experiments.

This module keeps the learned local action distance separate from graph
construction.  It only reads an already-built graph, its state embedding
cache, and four action-conditioned distance checkpoints.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import time
from typing import Any

import numpy as np


def puzzle_observation_to_frame_features(
    observations: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    *,
    qpos: np.ndarray,
    button_states: np.ndarray,
    n_buttons: int = 9,
) -> np.ndarray:
    """Convert OGBench Puzzle observations to the v0.8 robot+button frame.

    Some robot joint coordinates used by the graph feature are not present in
    the policy observation. The aligned live qpos/button readout is therefore
    required, just as it is when building the graph from the dataset sidecar.
    """
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.ndim != 2 or n_buttons not in (9, 16, 20):
        raise ValueError("Puzzle observations must be a matrix for a supported board size")
    if obs.shape[1] != 19 + 4 * n_buttons:
        raise ValueError(
            f"expected observation width {19 + 4 * n_buttons}, found {obs.shape[1]}"
        )
    mean = np.asarray(state_mean, dtype=np.float64)
    std = np.asarray(state_std, dtype=np.float64)
    if mean.shape != (19,) or std.shape != (19,) or np.any(std <= 0):
        raise ValueError("state normalization must contain 19 positive-scale robot coordinates")

    qpos = np.asarray(qpos, dtype=np.float32)
    bits = np.asarray(button_states, dtype=np.float32)
    if qpos.shape != (len(obs), 14 + n_buttons) or bits.shape != (len(obs), n_buttons):
        raise ValueError("live qpos and button states must align with observations")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("button states must be binary")
    robot = np.concatenate((qpos[:, :14], _action_coordinates(obs)), axis=1)
    normalized = (robot.astype(np.float64) - mean[None, :]) / std[None, :]
    return np.ascontiguousarray(np.concatenate((normalized, bits), axis=1), dtype=np.float32)


def _action_coordinates(obs: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(obs[:, 16], obs[:, 15])
    grip = np.clip(obs[:, 17] / 3.0, 0.0, 1.0)
    xyz = obs[:, 12:15] / 10.0 + np.asarray([0.425, 0.0, 0.0], dtype=np.float32)
    return np.asarray(np.column_stack((xyz, yaw, grip)), dtype=np.float32)


def action_qrl_loss_components(
    c1,
    c2,
    independent_goal_distance,
    lambda1,
    lambda2,
    *,
    horizon: float = 5.0,
    epsilon_c1: float = 1.25,
    epsilon_c2: float = 0.25,
):
    """Return v0.9 primal objective and the two constraint gaps."""
    import jax
    import jax.numpy as jnp

    c1 = jnp.asarray(c1)
    c2 = jnp.asarray(c2)
    d_goal = jnp.asarray(independent_goal_distance)
    gap1 = jnp.mean(jnp.square(jax.nn.relu(c1 - float(horizon)))) - float(epsilon_c1) ** 2
    gap2 = jnp.mean(jnp.square(jax.nn.relu(c2))) - float(epsilon_c2) ** 2
    maximize = jnp.mean(jax.nn.softplus((500.0 - d_goal) * 0.01) / 0.01)
    total = maximize + jnp.asarray(lambda1) * gap1 + jnp.asarray(lambda2) * gap2
    return total, maximize, gap1, gap2


def min_plus_direct_q(distance_by_head: np.ndarray, cost_steps: np.ndarray, *, k_limit: float = 20.0):
    """Apply ensemble-max local distance and min-plus graph cost.

    ``distance_by_head`` is ``(M, n_actions, n_nodes)``. Returns action
    values and the ensemble-conservative distance matrix.
    """
    distances = np.asarray(distance_by_head, dtype=np.float32)
    costs = np.asarray(cost_steps, dtype=np.float32)
    if distances.ndim != 3 or costs.shape != (distances.shape[-1],):
        raise ValueError("distances must be (heads,actions,nodes) with aligned node costs")
    if not np.isfinite(k_limit) or k_limit <= 0:
        raise ValueError("k_limit must be finite and positive")
    conservative = np.max(distances, axis=0)
    route = np.where(conservative <= k_limit, conservative + costs[None, :], np.inf)
    return np.min(route, axis=1), conservative


def select_direct_q_action(q_values: np.ndarray) -> tuple[int, bool]:
    """Stable argmin; if every route is infinite, choose the first BC proposal."""
    values = np.asarray(q_values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("q_values must be a non-empty vector")
    if not np.any(np.isfinite(values)):
        return 0, True
    return int(np.argmin(values)), False


def puzzle_observation_window(
    observation_history: list[np.ndarray] | np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    *,
    qpos_history: list[np.ndarray] | np.ndarray,
    button_history: list[np.ndarray] | np.ndarray,
    n_buttons: int = 9,
    window: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one trailing normalized window and its exact final board bits."""
    history = np.asarray(observation_history, dtype=np.float32)
    qpos_values = np.asarray(qpos_history, dtype=np.float32)
    button_values = np.asarray(button_history, dtype=np.uint8)
    if history.ndim != 2 or not len(history):
        raise ValueError("observation_history must be a non-empty (T,obs_dim) array")
    if qpos_values.shape[0] != len(history) or button_values.shape[0] != len(history):
        raise ValueError("live simulator readout history must align with observations")
    offsets = np.arange(window, dtype=np.int64) - (window - 1)
    indices = np.clip(len(history) - 1 + offsets, 0, len(history) - 1)
    frames = puzzle_observation_to_frame_features(
        history[indices], state_mean, state_std,
        qpos=qpos_values[indices], button_states=button_values[indices],
        n_buttons=n_buttons,
    )
    board = np.rint(frames[-1, -n_buttons:]).astype(np.uint8)
    return np.ascontiguousarray(frames.reshape(-1)), board


def _make_action_head():
    import jax
    from flax import linen as nn

    class ActionHead(nn.Module):
        widths: tuple[int, ...] = (1024, 1024, 256, 1024, 1024, 2048)

        @nn.compact
        def __call__(self, inputs, *, return_diagnostics=False):
            value = inputs
            activations = {}
            for index, width in enumerate(self.widths):
                value = nn.Dense(
                    width, kernel_init=nn.initializers.xavier_uniform(),
                    name=f"dense_{index}",
                )(value)
                if index < len(self.widths) - 1:
                    value = nn.LayerNorm(name=f"norm_{index}")(value)
                    if return_diagnostics:
                        activations[f"dense_{index}"] = value
                    value = jax.nn.gelu(value)
            if return_diagnostics:
                return {"embedding": value, "preactivations": activations}
            return value

    return ActionHead()


class DirectQScorer:
    """Min-plus scorer using a frozen state embedding and an action-head ensemble."""

    def __init__(
        self,
        *,
        candidate_ids: np.ndarray,
        candidate_features: np.ndarray,
        candidate_boards: np.ndarray,
        candidate_embeddings: np.ndarray,
        episode_ids: np.ndarray,
        times: np.ndarray,
        cost_steps: np.ndarray,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        n_buttons: int,
        action_dim: int,
        horizon: int,
        action_heads: list[tuple[Any, Any]],
        raw_alpha: Any,
        anchor_mode: str = "same_object",
        state_embedder: Any | None = None,
        state_params: Any | None = None,
        k: int = 32,
        successors: int = 4,
        max_local_distance: float = 20.0,
        device: Any | None = None,
    ):
        import jax
        import jax.numpy as jnp
        from experiments.v08_runtime import iqe_distance_from_embeddings

        ids = np.asarray(candidate_ids, dtype=np.int64)
        features = np.asarray(candidate_features, dtype=np.float32)
        boards = np.asarray(candidate_boards, dtype=np.uint8)
        embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
        eps = np.asarray(episode_ids, dtype=np.int64)
        ts = np.asarray(times, dtype=np.int64)
        costs = np.asarray(cost_steps, dtype=np.float32)
        if not (len(ids) == len(features) == len(boards) == len(embeddings)):
            raise ValueError("candidate IDs, features, boards, and embeddings must align")
        if len(action_heads) != 4:
            raise ValueError("direct Q requires exactly four action heads")
        if anchor_mode not in {"same_object", "dplus"}:
            raise ValueError("anchor_mode must be 'same_object' or 'dplus'")
        if anchor_mode == "dplus" and (state_embedder is None or state_params is None):
            raise ValueError("dplus anchoring requires the fixed state embedding model and parameters")
        if embeddings.ndim != 2 or embeddings.shape[1] != 2048:
            raise ValueError("frozen state embeddings must have width 2048")
        if costs.shape != eps.shape or costs.shape != ts.shape:
            raise ValueError("node costs, episode IDs, and times must align")
        if np.any(ids < 0) or (len(ids) > 1 and np.any(ids[1:] <= ids[:-1])):
            raise ValueError("candidate_ids must be sorted, unique, and non-negative")

        self.candidate_ids = ids
        self.features = features
        self.boards = boards
        self.embeddings_np = embeddings
        self.episode_ids = eps
        self.times = ts
        self.cost_steps = costs
        self.state_mean = np.asarray(state_mean, dtype=np.float64)
        self.state_std = np.asarray(state_std, dtype=np.float64)
        self.n_buttons = int(n_buttons)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.k = int(k)
        self.successors = int(successors)
        self.max_local_distance = float(max_local_distance)
        self.action_heads = action_heads
        self.raw_alpha = raw_alpha
        self.anchor_mode = anchor_mode
        self._jax, self._jnp = jax, jnp
        self._device = device or jax.devices()[0]
        # Keep the 1.65 GB v0.8 cache memory-mapped on host. Each decision
        # transfers only its ≤160 selected embeddings to the accelerator.
        self._embeddings = embeddings
        self._head_predictors = []
        for head, params in action_heads:
            apply = jax.jit(lambda batch, p=params, m=head: m.apply({"params": p}, batch))
            self._head_predictors.append(apply)
        self._board_to_candidates: dict[bytes, np.ndarray] = {}
        bucket: dict[bytes, list[int]] = defaultdict(list)
        for local, board in enumerate(boards):
            bucket[np.ascontiguousarray(board).tobytes()].append(local)
        self._board_to_candidates = {
            key: np.asarray(value, dtype=np.int64) for key, value in bucket.items()
        }
        self._node_to_local = {int(node): i for i, node in enumerate(ids)}
        self._time_to_node = {
            (int(ep), int(t)): int(node)
            for node, ep, t in zip(ids, eps[ids], ts[ids], strict=True)
        }
        self._pair_distance = jax.jit(lambda source, target: iqe_distance_from_embeddings(
            source, target, raw_alpha, n_components=64, component_dim=32, jnp=jnp
        ))
        if anchor_mode == "dplus":
            self._query_embedding = jax.jit(
                lambda values: state_embedder.apply({"params": state_params}, values)
            )
            self._anchor_batch_size = 8192
        self._input_dim = int(features.shape[1])

    def _action_embeddings(self, window: np.ndarray, chunks: np.ndarray):
        jax, jnp = self._jax, self._jnp
        action = np.asarray(chunks, dtype=np.float32)
        if action.ndim == 3:
            action = action.reshape(len(action), -1)
        inputs = np.concatenate((np.broadcast_to(window, (len(action), len(window))), action), axis=1)
        return [
            predict(jax.device_put(jnp.asarray(inputs), self._device)).reshape((len(action), 64, 32))
            for predict in self._head_predictors
        ]

    def distance_to_node(self, window_obs: np.ndarray, chunk: np.ndarray, node_id: int) -> float:
        """Return the ensemble-maximum local distance to a cached graph node."""
        jax, jnp = self._jax, self._jnp
        window = np.asarray(window_obs, dtype=np.float32).reshape(-1)
        local = self._node_to_local.get(int(node_id))
        if local is None:
            raise ValueError(f"state node {node_id} is absent from the embedding cache")
        phi = self._action_embeddings(window, np.asarray(chunk).reshape(1, -1))
        target = jax.device_put(jnp.asarray(self._embeddings[[local]]), self._device)
        values = []
        for source in phi:
            values.append(self._pair_distance(source, target.reshape((1, 64, 32)))[0])
        return float(jax.device_get(jnp.max(jnp.stack(values))))

    def _neighborhood(
        self,
        window: np.ndarray,
        board: np.ndarray,
        *,
        exclude_episode: int | None = None,
    ) -> tuple[np.ndarray, str | None]:
        if self.anchor_mode == "dplus":
            query = self._query_embedding(
                self._jnp.asarray(np.asarray(window, dtype=np.float32)[None, :])
            ).reshape((1, 64, 32))
            eligible = np.ones(len(self.candidate_ids), dtype=np.bool_)
            if exclude_episode is not None:
                eligible &= self.episode_ids[self.candidate_ids] != int(exclude_episode)
            eligible_local = np.flatnonzero(eligible)
            if not len(eligible_local):
                return np.empty(0, dtype=np.int64), "no_candidate_nodes"
            distances = np.full(len(self.candidate_ids), np.inf, dtype=np.float32)
            batch_size = self._anchor_batch_size
            jax, jnp = self._jax, self._jnp
            for start in range(0, len(eligible_local), batch_size):
                local = eligible_local[start : start + batch_size]
                count = len(local)
                target_host = np.zeros((batch_size, 2048), dtype=np.float32)
                target_host[:count] = self._embeddings[local]
                target = jax.device_put(jnp.asarray(target_host), self._device).reshape(
                    (batch_size, 64, 32)
                )
                source = jnp.broadcast_to(query, (batch_size, 64, 32))
                batch_distances = np.asarray(
                    jax.device_get(self._pair_distance(source, target)), dtype=np.float32
                )
                distances[local] = batch_distances[:count]
            anchor_order = np.lexsort((
                self.candidate_ids[eligible_local], distances[eligible_local]
            ))[: self.k]
            base = eligible_local[anchor_order].astype(np.int64, copy=False)
            if not len(base):
                return np.empty(0, dtype=np.int64), "no_candidate_nodes"
        else:
            key = np.ascontiguousarray(board, dtype=np.uint8).tobytes()
            base = self._board_to_candidates.get(key, np.empty(0, dtype=np.int64))
            if exclude_episode is not None and len(base):
                base = base[self.episode_ids[self.candidate_ids[base]] != int(exclude_episode)]
            if not len(base):
                return np.empty(0, dtype=np.int64), "no_board_nodes"
            delta = self.features[base] - np.asarray(window, dtype=np.float32)[None, :]
            distance = np.einsum("ij,ij->i", delta, delta, optimize=True)
            order = np.lexsort((self.candidate_ids[base], distance))[: self.k]
            base = base[order]
        anchors = base
        node_ids = [int(self.candidate_ids[local]) for local in anchors]
        for node in tuple(node_ids):
            ep = int(self.episode_ids[node])
            t = int(self.times[node])
            for step in range(1, self.successors + 1):
                candidate = self._time_to_node.get((ep, t + step * self.horizon))
                if candidate is not None:
                    if exclude_episode is None or ep != int(exclude_episode):
                        node_ids.append(candidate)
        unique_nodes = np.asarray(sorted(set(node_ids)), dtype=np.int64)
        local_ids = np.asarray([self._node_to_local.get(int(n), -1) for n in unique_nodes])
        local_ids = local_ids[local_ids >= 0]
        finite = np.isfinite(self.cost_steps[self.candidate_ids[local_ids]])
        local_ids = local_ids[finite]
        if not len(local_ids):
            return local_ids, "no_finite_cost_nodes"
        return local_ids, None

    def score(
        self,
        window_obs: np.ndarray,
        chunks: np.ndarray,
        *,
        board: np.ndarray | None = None,
        exclude_episode: int | None = None,
        use_fallback_value: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Score proposed action chunks and return per-decision diagnostics."""
        jax, jnp = self._jax, self._jnp
        started = time.perf_counter()
        window = np.asarray(window_obs, dtype=np.float32).reshape(-1)
        action = np.asarray(chunks, dtype=np.float32)
        if action.ndim == 3:
            action = action.reshape(len(action), -1)
        if action.ndim != 2 or action.shape[1] != self.horizon * self.action_dim:
            raise ValueError("chunks must have shape (N,horizon,action_dim) or flattened equivalent")
        if window.shape != (self._input_dim,):
            raise ValueError("window feature width does not match state checkpoint")
        if board is None:
            board = np.rint(window[-self.n_buttons:]).astype(np.uint8)
        else:
            board = np.asarray(board, dtype=np.uint8)
        q_values = np.full(len(action), np.inf, dtype=np.float32)
        node_locals, fallback_reason = self._neighborhood(
            window, board, exclude_episode=exclude_episode
        )
        diag: dict[str, Any] = {
            "fallback": False,
            "fallback_reason": fallback_reason,
            "neighborhood_size": int(len(node_locals)),
            "selected_q": None,
            "candidate_q_std": None,
            "decision_seconds": None,
        }
        if len(node_locals):
            node_ids = self.candidate_ids[node_locals]
            candidate_embeddings = jax.device_put(
                jnp.asarray(np.asarray(self._embeddings[node_locals], dtype=np.float32)), self._device
            )
            candidate_costs = np.asarray(self.cost_steps[node_ids], dtype=np.float32)
            ensemble = []
            for phi in self._action_embeddings(window, action):
                source = jnp.broadcast_to(
                    phi[:, None, :, :], (len(action), len(node_locals), 64, 32)
                ).reshape((-1, 64, 32))
                target = jnp.broadcast_to(
                    candidate_embeddings[None, :, :], (len(action), len(node_locals), 2048)
                ).reshape((-1, 64, 32))
                distances = self._pair_distance(source, target).reshape((len(action), len(node_locals)))
                ensemble.append(distances)
            distance_values = np.stack([np.asarray(jax.device_get(x)) for x in ensemble], axis=0)
            q_values, conservative = min_plus_direct_q(
                distance_values, candidate_costs,
                k_limit=self.max_local_distance,
            )
            if np.all(~np.isfinite(q_values)):
                fallback_reason = "no_route_within_K"
                diag["fallback_reason"] = fallback_reason
        selected, did_fallback = select_direct_q_action(q_values)
        if did_fallback and use_fallback_value:
            q_values[0] = 0.0
            diag["fallback"] = True
        elif did_fallback:
            diag["fallback"] = True
        selected, _ = select_direct_q_action(q_values)
        finite_q = q_values[np.isfinite(q_values)]
        diag.update({
            "fallback_reason": fallback_reason if diag["fallback"] else None,
            "selected_index": selected,
            "selected_q": float(q_values[selected]) if np.isfinite(q_values[selected]) and not diag["fallback"] else None,
            "candidate_q_std": float(np.std(finite_q)) if len(finite_q) else None,
            "decision_seconds": float(time.perf_counter() - started),
        })
        return q_values, diag

    def score_batch(
        self,
        windows: np.ndarray,
        chunks: np.ndarray,
        boards: np.ndarray,
        *,
        exclude_episodes: np.ndarray | None = None,
        target_node_ids: np.ndarray | None = None,
        use_fallback_value: bool = False,
    ) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray | None]:
        """Vectorized offline scoring for one recorded chunk per held-out row."""
        jax, jnp = self._jax, self._jnp
        started = time.perf_counter()
        window_values = np.asarray(windows, dtype=np.float32)
        action_values = np.asarray(chunks, dtype=np.float32)
        board_values = np.asarray(boards, dtype=np.uint8)
        if action_values.ndim == 3:
            action_values = action_values.reshape(len(action_values), -1)
        count = len(window_values)
        if count < 1:
            raise ValueError("direct-Q batch must contain at least one row")
        if (window_values.shape != (count, self._input_dim)
                or action_values.shape != (count, self.horizon * self.action_dim)
                or board_values.shape != (count, self.n_buttons)):
            raise ValueError("batched windows, chunks, and boards must align")
        if exclude_episodes is None:
            exclude_episodes = np.full(count, -1, dtype=np.int64)
        else:
            exclude_episodes = np.asarray(exclude_episodes, dtype=np.int64)
            if exclude_episodes.shape != (count,):
                raise ValueError("exclude_episodes must have one entry per sample")
        neighborhoods = []
        reasons = []
        for i in range(count):
            exclude = None if exclude_episodes[i] < 0 else int(exclude_episodes[i])
            local, reason = self._neighborhood(window_values[i], board_values[i], exclude_episode=exclude)
            neighborhoods.append(local)
            reasons.append(reason)
        max_nodes = max(1, max(map(len, neighborhoods), default=0))
        padded_local = np.zeros((count, max_nodes), dtype=np.int64)
        padded_cost = np.full((count, max_nodes), np.inf, dtype=np.float32)
        mask = np.zeros((count, max_nodes), dtype=np.bool_)
        sizes = np.zeros(count, dtype=np.int64)
        for i, local in enumerate(neighborhoods):
            sizes[i] = len(local)
            if len(local):
                nodes = self.candidate_ids[local]
                padded_local[i, :len(local)] = local
                padded_cost[i, :len(local)] = self.cost_steps[nodes]
                mask[i, :len(local)] = np.isfinite(padded_cost[i, :len(local)])
        inputs = jax.device_put(
            jnp.asarray(np.concatenate((window_values, action_values), axis=1)), self._device
        )
        phi_heads = self._action_embeddings_batched(inputs)
        target_host = np.asarray(self._embeddings[padded_local], dtype=np.float32)
        targets = jax.device_put(jnp.asarray(target_host), self._device).reshape((count, max_nodes, 64, 32))
        distances_by_head = []
        target_distances_by_head = []
        target_local = None
        target_mask = None
        target_embedding = None
        if target_node_ids is not None:
            target_node_ids = np.asarray(target_node_ids, dtype=np.int64)
            if target_node_ids.shape != (count,):
                raise ValueError("target_node_ids must align with the batch")
            target_local = np.asarray([self._node_to_local.get(int(n), -1) for n in target_node_ids])
            target_mask = target_local >= 0
            safe = np.maximum(target_local, 0)
            target_embedding = jax.device_put(
                jnp.asarray(np.asarray(self._embeddings[safe], dtype=np.float32)), self._device
            ).reshape((count, 64, 32))
        for phi in phi_heads:
            source = jnp.broadcast_to(phi[:, None, :, :], (count, max_nodes, 64, 32)).reshape((-1, 64, 32))
            target = targets.reshape((-1, 64, 32))
            distances = self._pair_distance(source, target).reshape((count, max_nodes))
            distances_by_head.append(np.asarray(jax.device_get(distances), dtype=np.float32))
            if target_embedding is not None:
                td = self._pair_distance(phi, target_embedding)
                target_distances_by_head.append(np.asarray(jax.device_get(td), dtype=np.float32))
        q_values = np.full(count, np.inf, dtype=np.float32)
        q_std = np.full(count, np.nan, dtype=np.float32)
        dbar = np.max(np.stack(distances_by_head), axis=0)
        route = np.where(mask & (dbar <= self.max_local_distance), dbar + padded_cost, np.inf)
        q_values = np.min(route, axis=1)
        if use_fallback_value:
            q_values[~np.isfinite(q_values)] = 0.0
        diagnostics: list[dict[str, Any]] = []
        elapsed_per_row = float(time.perf_counter() - started) / max(count, 1)
        for i in range(count):
            fallback = not bool(np.isfinite(np.min(route[i])))
            reason = reasons[i]
            if fallback and reason is None:
                reason = "no_route_within_K"
            finite_routes = route[i][np.isfinite(route[i])]
            q_std[i] = float(np.std(finite_routes)) if len(finite_routes) else np.nan
            diagnostics.append({
                "fallback": fallback,
                "fallback_reason": reason if fallback else None,
                "neighborhood_size": int(sizes[i]),
                "selected_q": None if fallback else float(np.min(route[i])),
                "candidate_q_std": None if not len(finite_routes) else float(q_std[i]),
                "decision_seconds": elapsed_per_row,
            })
        target_distances = None
        if target_embedding is not None:
            target_distances = np.max(np.stack(target_distances_by_head), axis=0)
            target_distances[~target_mask] = np.nan
        return q_values, diagnostics, target_distances

    def _action_embeddings_batched(self, inputs):
        return [
            predict(inputs).reshape((len(inputs), 64, 32))
            for predict in self._head_predictors
        ]


def load_direct_q_scorer(config_path: str | Path):
    """Load a v0.9/v1.0 scorer from its validated JSON config."""
    from graphcritic.io import load_graph, load_sim_state, load_transitions
    from graphcritic.bridges.models import prepare_puzzle_model_data
    from graphcritic.bridges.windows import puzzle_state_windows, puzzle_window_candidate_nodes
    from graphcritic.state_graph import state_episode_time
    from graphcritic.weights import step_cost
    from experiments.v08_runtime import (
        load_quasimetric_checkpoint, make_quasimetric_embedder, _jax_modules,
    )

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    trans, transition_meta = load_transitions(cfg["transitions"])
    sim = load_sim_state(cfg["sim_state"])
    if transition_meta["dataset_id"] != cfg["dataset_id"]:
        raise ValueError("direct Q config dataset ID does not match transitions")
    data = prepare_puzzle_model_data(trans, sim, seed=int(cfg.get("split_seed", 0)))
    frame = np.concatenate((data.state_cont, data.state_bits), axis=1).astype(np.float32)
    windows = puzzle_state_windows(frame, trans, data.terminal_state_index, window=5)
    graph = load_graph(cfg["graph"], trans=trans)
    if graph.content_hash != cfg["graph_hash"]:
        raise ValueError("direct Q config graph hash does not match graph artifact")
    candidate_ids, _, _ = puzzle_window_candidate_nodes(trans, graph.nodes.is_goal, window=5)
    candidate_features = windows[candidate_ids]
    candidate_boards = data.state_bits[candidate_ids].astype(np.uint8)
    emb_file = Path(cfg["embedding_cache"])
    emb_meta = json.loads(emb_file.with_suffix(emb_file.suffix + ".json").read_text())
    if emb_meta.get("params_sha256") != cfg["state_checkpoint_sha256"]:
        raise ValueError("embedding cache was produced by a different state checkpoint")
    if emb_meta.get("dataset_id") != cfg["dataset_id"] or emb_meta.get("shape", [None])[0] != len(candidate_ids):
        raise ValueError("embedding cache dataset or node count mismatch")
    embeddings = np.load(emb_file, mmap_mode="r", allow_pickle=False)
    if embeddings.dtype != np.float32 or embeddings.shape != (len(candidate_ids), 2048):
        raise ValueError("embedding cache has an invalid shape or dtype")
    from experiments.v08_runtime import _array_sha256
    if _array_sha256(candidate_ids) != emb_meta.get("node_ids_sha256"):
        raise ValueError("embedding cache node IDs do not match the candidate-node order")
    if _array_sha256(windows[candidate_ids]) != emb_meta.get("node_features_sha256"):
        raise ValueError("embedding cache node features do not match the d-theta data preparation")
    episode, times = state_episode_time(trans)
    costs = np.asarray(graph.cost / step_cost(graph.config.gamma), dtype=np.float32)
    model, state_params, state_meta = load_quasimetric_checkpoint(
        cfg["state_checkpoint"], input_dim=windows.shape[1], expected_dataset_id=cfg["dataset_id"]
    )
    if state_meta.get("params_sha256") != cfg["state_checkpoint_sha256"]:
        raise ValueError("state checkpoint SHA256 changed after config creation")
    anchor_mode = str(cfg.get("anchor_mode", "same_object"))
    if cfg.get("spec_version") == "v1.0" and anchor_mode != "dplus":
        raise ValueError("v1.0 direct-Q config must use dplus anchors")
    if cfg.get("spec_version") == "v1.0":
        if state_meta.get("spec_version") != "v1.0" or state_meta.get("match_constraint") != "object":
            raise ValueError("v1.0 direct-Q anchors require the trained dplus checkpoint")
        bridge_meta = graph.meta.get("bridge_model_metadata", {})
        if bridge_meta.get("params_sha256") != cfg["state_checkpoint_sha256"]:
            raise ValueError("L+ graph bridge judge and direct-Q anchor checkpoints differ")
    jax, jnp, _, serialization, _ = _jax_modules()
    raw_alpha = state_params["raw_alpha"]
    heads = []
    for checkpoint in cfg["action_head_checkpoints"]:
        ckpt = Path(checkpoint)
        meta = json.loads((ckpt / "metadata.json").read_text())
        if meta.get("model_kind") not in {
            "action-conditioned-quasimetric-v0.9",
            "action-conditioned-quasimetric-v1.0",
        } or int(meta.get("step", -1)) != 20_000:
            raise ValueError(f"action head {ckpt} is not a completed v0.9 20k checkpoint")
        if cfg.get("spec_version") == "v1.0" and meta.get("model_kind") != "action-conditioned-quasimetric-v1.0":
            raise ValueError(f"action head {ckpt} was not trained for v1.0")
        if meta.get("state_checkpoint_sha256") != cfg["state_checkpoint_sha256"]:
            raise ValueError(f"action head {ckpt} uses a different frozen state checkpoint")
        if meta.get("dataset_id") != cfg["dataset_id"]:
            raise ValueError(f"action head {ckpt} dataset ID mismatch")
        if int(meta.get("horizon", -1)) != 5 or int(meta.get("action_dim", -1)) != int(cfg["action_dim"]):
            raise ValueError(f"action head {ckpt} action layout mismatch")
        head = _make_action_head()
        in_dim = windows.shape[1] + int(meta["horizon"]) * int(meta["action_dim"])
        if int(meta.get("input_dim", -1)) != in_dim:
            raise ValueError(f"action head {ckpt} input dimension mismatch")
        template = head.init(jax.random.PRNGKey(0), jnp.zeros((1, in_dim), jnp.float32))["params"]
        raw = (ckpt / "params.msgpack").read_bytes()
        if __import__("hashlib").sha256(raw).hexdigest() != meta.get("params_sha256"):
            raise ValueError(f"action head params hash mismatch: {ckpt}")
        heads.append((head, serialization.from_bytes(template, raw)))
    return DirectQScorer(
        candidate_ids=candidate_ids, candidate_features=candidate_features,
        candidate_boards=candidate_boards, candidate_embeddings=embeddings,
        episode_ids=episode, times=times, cost_steps=costs,
        state_mean=np.asarray(data.metadata["state_mean"], dtype=np.float64),
        state_std=np.asarray(data.metadata["state_std"], dtype=np.float64),
        n_buttons=data.n_bits, action_dim=int(cfg["action_dim"]), horizon=5,
        action_heads=heads, raw_alpha=raw_alpha,
        anchor_mode=anchor_mode,
        state_embedder=(make_quasimetric_embedder(model) if anchor_mode == "dplus" else None),
        state_params=({k: v for k, v in state_params.items() if k != "raw_alpha"}
                      if anchor_mode == "dplus" else None),
        k=int(cfg.get("k", 32)), successors=int(cfg.get("successors", 4)),
        max_local_distance=float(cfg.get("max_local_distance", 20.0)),
    )

