"""Shortest-path extraction and closed-loop graph tracking diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from graphcritic.featurize import Featurizer, state_table
from graphcritic.modes import MODE_FNS, resolve_state_modes, smooth_mode_changes, state_episode_time
from graphcritic.types import GraphArtifact, SchemaError, SimStateSet, TransitionSet


@dataclass(frozen=True)
class PathStep:
    """One path vertex and its outgoing edge type and duration.

    The final goal vertex has ``kind=None`` and ``tau=0``. Otherwise kind is
    0 for a temporal edge and 1 for a bridge edge.
    """

    vertex: int
    kind: int | None
    tau: int


def _successor_edge_table(artifact: GraphArtifact) -> tuple[NDArray, NDArray]:
    """Index only the selected shortest-path edge for each vertex."""
    n_nodes = artifact.nodes.n_nodes
    src = artifact.edges.src
    candidates = np.flatnonzero(artifact.succ[src] == artifact.edges.dst)
    kind_by_src = np.full(n_nodes, -1, dtype=np.int8)
    tau_by_src = np.zeros(n_nodes, dtype=np.int64)
    if candidates.size == 0:
        return kind_by_src, tau_by_src
    order = np.lexsort((candidates, artifact.weights[candidates], src[candidates]))
    ordered = candidates[order]
    ordered_src = src[ordered]
    first = np.r_[True, ordered_src[1:] != ordered_src[:-1]]
    chosen = ordered[first]
    kind_by_src[src[chosen]] = artifact.edges.kind[chosen]
    tau_by_src[src[chosen]] = artifact.edges.kappa[chosen]
    return kind_by_src, tau_by_src


def extract_path(artifact: GraphArtifact, start_vertex: int) -> list[PathStep]:
    """Follow the saved shortest-path policy from a vertex to a goal."""
    artifact.validate()
    kind_by_src, tau_by_src = _successor_edge_table(artifact)
    return _extract_path_with_table(artifact, start_vertex, kind_by_src, tau_by_src)


def _extract_path_with_table(
    artifact: GraphArtifact,
    start_vertex: int,
    kind_by_src: NDArray,
    tau_by_src: NDArray,
) -> list[PathStep]:
    if isinstance(start_vertex, bool) or not isinstance(start_vertex, (int, np.integer)):
        raise SchemaError("start_vertex must be an integer graph vertex")
    vertex = int(start_vertex)
    if vertex < 0 or vertex >= artifact.nodes.n_nodes:
        raise SchemaError("start_vertex is outside the graph")
    if not np.isfinite(artifact.cost[vertex]):
        raise SchemaError("cannot extract a path from an unreachable vertex")
    path: list[PathStep] = []
    seen: set[int] = set()
    while True:
        if vertex in seen:
            raise SchemaError("shortest-path successor chain contains a cycle")
        seen.add(vertex)
        if artifact.nodes.is_goal[vertex]:
            path.append(PathStep(vertex, None, 0))
            return path
        nxt = int(artifact.succ[vertex])
        kind = int(kind_by_src[vertex])
        if kind < 0:
            raise SchemaError("successor edge is missing from the artifact")
        tau = int(tau_by_src[vertex])
        path.append(PathStep(vertex, kind, tau))
        vertex = nxt
        if len(path) > artifact.nodes.n_nodes:
            raise SchemaError("shortest-path successor chain exceeded the graph size")


class GraphTrackingPolicy:
    """Track an extracted graph path with nearest-state recorded actions."""

    def __init__(
        self,
        artifact: GraphArtifact,
        trans: TransitionSet,
        featurizer: Featurizer,
        mode_fn: str | Callable[[NDArray, Mapping[str, NDArray]], NDArray] = "none",
        *,
        sim_state: SimStateSet | None = None,
        state_features: NDArray | None = None,
        window: int = 10,
        fail_radius_mult: float = 2.0,
    ) -> None:
        artifact.validate(trans)
        if artifact.config.graph_kind != "state":
            raise SchemaError("GraphTrackingPolicy requires an occurrence-level state graph")
        if trans.action is None:
            raise SchemaError("GraphTrackingPolicy requires recorded actions")
        if window < 0 or fail_radius_mult <= 0:
            raise SchemaError("window must be non-negative and fail_radius_mult positive")
        states, _, _, _ = state_table(trans)
        if state_features is None:
            aux = None
            if sim_state is not None:
                sim_state.validate(states.shape[0])
                aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
                if sim_state.button_states is not None:
                    aux["button_states"] = sim_state.button_states
            state_features = featurizer.transform(states, aux=aux)
        features = np.asarray(state_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != artifact.nodes.n_nodes:
            raise SchemaError("state_features must contain one feature vector per graph vertex")
        self.artifact = artifact
        self.trans = trans
        self.featurizer = featurizer
        self.features = features
        self.window = int(window)
        self.fail_radius_mult = float(fail_radius_mult)
        self.radius = float(artifact.config.bridge.radius)
        if artifact.config.bridge.radius_unit == "step":
            self.radius *= float(artifact.meta.get("d_step", 1.0))
        self.fail_radius = self.radius * self.fail_radius_mult
        if isinstance(mode_fn, str):
            if mode_fn not in MODE_FNS:
                raise SchemaError(f"unknown mode function {mode_fn!r}")
            self.mode_fn: Callable = MODE_FNS[mode_fn]
            self.state_modes = resolve_state_modes(
                trans, mode_fn, sim_state=sim_state, min_dwell=artifact.config.min_dwell
            )
        else:
            self.mode_fn = mode_fn
            feature_aux: dict[str, NDArray] = {}
            if sim_state is not None:
                sim_state.validate(states.shape[0])
                feature_aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
                if sim_state.button_states is not None:
                    feature_aux["button_states"] = sim_state.button_states
            raw_modes = np.asarray(mode_fn(states, feature_aux), dtype=np.int64)
            if raw_modes.shape != (states.shape[0],):
                raise SchemaError("mode function must return one value per state occurrence")
            episode, time = state_episode_time(trans)
            self.state_modes = smooth_mode_changes(
                raw_modes, episode, time, artifact.config.min_dwell
            )
        # Every state occurrence is represented by a transition row. For a
        # terminal-next node, the associated action is the recorded action in
        # that row that produced the terminal state.
        self._action_row = np.asarray(artifact.nodes.rep_ref[:, 0], dtype=np.int64).copy()
        # The graph hash and edge arrays are large. Validate and index them
        # once at construction, not once per reset/start in an evaluation.
        self._successor_kind, self._successor_tau = _successor_edge_table(artifact)
        self.path: list[PathStep] = []
        self.pointer = 0
        self.selected_vertex = -1
        self.follow_distance = float("inf")
        self.bridge_use_count = 0
        self._crossed_positions: set[int] = set()
        self._last_crossed: list[tuple[int, int]] = []

    def reset(self, start_vertex: int) -> None:
        self.path = _extract_path_with_table(
            self.artifact, start_vertex, self._successor_kind, self._successor_tau
        )
        self.pointer = 0
        self.selected_vertex = int(start_vertex)
        self.follow_distance = float("inf")
        self.bridge_use_count = 0
        self._crossed_positions.clear()
        self._last_crossed = []

    def _current_features(self, obs: NDArray, aux: Mapping[str, NDArray]) -> tuple[NDArray, int]:
        value = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        feature_aux = {
            key: np.asarray(item).reshape(1, -1)
            for key, item in aux.items()
        }
        phi = self.featurizer.transform(value, aux=feature_aux)[0]
        mode_values = np.asarray(self.mode_fn(value, feature_aux), dtype=np.int64).reshape(-1)
        if mode_values.size != 1:
            raise SchemaError("mode function must return one mode for a single observation")
        return phi, int(mode_values[0])

    def act(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> NDArray:
        if not self.path:
            raise SchemaError("call reset(start_vertex) before requesting an action")
        if aux is None:
            aux = {}
        phi, mode = self._current_features(obs, aux)
        end = min(len(self.path), self.pointer + self.window + 1)
        positions = np.arange(self.pointer, end, dtype=np.int64)
        candidates: list[int] = []
        for pos in positions:
            step = self.path[int(pos)]
            if step.kind == 1 or self._action_row[step.vertex] < 0:
                continue
            candidates.append(int(pos))
        if not candidates:
            # A chain of consecutive bridges can fill the configured lookahead.
            # In that case, continue at the first later recorded temporal
            # action rather than failing before the graph path can be tracked.
            for pos in range(end, len(self.path)):
                step = self.path[pos]
                if step.kind != 1 and self._action_row[step.vertex] >= 0:
                    candidates.append(pos)
                    break
            if (
                not candidates
                and self.path
                and self.path[-1].kind is None
                and self._action_row[self.path[-1].vertex] >= 0
            ):
                # Keep applying the recorded action that reaches a terminal
                # goal while the simulator finishes its success transition.
                candidates.append(len(self.path) - 1)
            if not candidates:
                raise SchemaError("graph path has no actionable temporal vertex before its goal")
        candidate_positions = np.asarray(candidates, dtype=np.int64)
        candidate_vertices = np.asarray([self.path[pos].vertex for pos in candidates], dtype=np.int64)
        same_mode = self.state_modes[candidate_vertices] == mode
        if np.any(same_mode):
            candidate_positions = candidate_positions[same_mode]
            candidate_vertices = candidate_vertices[same_mode]
        distances = np.linalg.norm(self.features[candidate_vertices] - phi[None, :], axis=1)
        selected = int(np.argmin(distances))
        pos = int(candidate_positions[selected])
        vertex = int(candidate_vertices[selected])
        self.selected_vertex = vertex
        self.follow_distance = float(distances[selected])
        action_row = int(self._action_row[vertex])
        if action_row < 0:
            raise SchemaError("selected path vertex has no recorded action")
        old_pointer = self.pointer
        self.pointer = max(self.pointer, pos + 1)
        crossed: list[tuple[int, int]] = []
        for path_pos in range(old_pointer, self.pointer):
            step = self.path[path_pos]
            if step.kind == 1 and path_pos not in self._crossed_positions:
                self._crossed_positions.add(path_pos)
                crossed.append((path_pos, step.tau))
        self._last_crossed = crossed
        self.bridge_use_count += len(crossed)
        return np.asarray(self.trans.action[action_row]).copy()

    def distance_to_selected(self, obs: NDArray, aux: Mapping[str, NDArray] | None = None) -> float:
        if self.selected_vertex < 0:
            return float("inf")
        phi, _ = self._current_features(obs, aux or {})
        return float(np.linalg.norm(phi - self.features[self.selected_vertex]))

    def distance_to_vertex(
        self,
        obs: NDArray,
        aux: Mapping[str, NDArray] | None,
        vertex: int,
    ) -> float:
        """Return feature distance to a vertex, or infinity across modes."""
        if isinstance(vertex, bool) or not isinstance(vertex, (int, np.integer)):
            raise SchemaError("vertex must be an integer graph vertex")
        vertex = int(vertex)
        if vertex < 0 or vertex >= self.artifact.nodes.n_nodes:
            raise SchemaError("vertex is outside the graph")
        phi, mode = self._current_features(obs, aux or {})
        if mode != int(self.state_modes[vertex]):
            return float("inf")
        return float(np.linalg.norm(phi - self.features[vertex]))

    def pending_bridge(self) -> tuple[int, int] | None:
        if not self.path or self.pointer >= len(self.path):
            return None
        step = self.path[self.pointer]
        if step.kind != 1 or self.pointer + 1 >= len(self.path):
            return None
        return int(self.path[self.pointer + 1].vertex), int(step.tau)

    def status(self) -> dict[str, Any]:
        return {
            "pointer": int(self.pointer),
            "selected_vertex": int(self.selected_vertex),
            "follow_distance": float(self.follow_distance),
            "bridge_use_count": int(self.bridge_use_count),
            "pending_bridge": self.pending_bridge(),
            "crossed_bridges": list(self._last_crossed),
        }
