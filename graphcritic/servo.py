"""Hybrid replay/pose witness controller for graph paths (v0.5)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from graphcritic.path import PathStep, _extract_path_with_table, _successor_edge_table
from graphcritic.physical_time import free_space_mask
from graphcritic.state_schema import StateCoordinates, StateSchema, passive_difference
from graphcritic.types import GraphArtifact, SchemaError, TransitionSet
from graphcritic.weights import step_cost


def contact_replay_correction(
    recorded_action: NDArray,
    current_to_vertex_error: NDArray,
    action_limits: NDArray,
    gain: float,
) -> NDArray[np.float32]:
    """Replay contact actions with pose correction to the current path vertex."""
    action = np.asarray(recorded_action, dtype=np.float64).copy()
    error = np.asarray(current_to_vertex_error, dtype=np.float64)
    limits = np.asarray(action_limits, dtype=np.float64)
    if action.ndim != 1 or error.shape != action.shape or limits.shape != action.shape:
        raise SchemaError("contact correction action, error, and limits must be aligned vectors")
    if action.size < 4 or np.any(limits <= 0) or not np.isfinite(gain) or gain < 0:
        raise SchemaError("contact correction requires pose dimensions and a non-negative finite gain")
    action[:4] += float(gain) * error[:4] / limits[:4]
    return np.clip(action, -1.0, 1.0).astype(np.float32)


class ServoTrackingPolicy:
    """Follow temporal graph segments by pose servo and bridge edges by landing."""

    def __init__(
        self,
        artifact: GraphArtifact,
        trans: TransitionSet,
        schema: StateSchema,
        coordinates: StateCoordinates,
        *,
        mode: str = "servo",
        window: int = 10,
        settle: int = 2,
        epsilon_a: NDArray | None = None,
        epsilon_p: NDArray | None = None,
        feedback_gain: float = 1.0,
        response_gain: NDArray | None = None,
        contact_gain: float = 0.3,
        free_distance: float = 0.05,
        tracking_tube: NDArray | None = None,
    ) -> None:
        artifact.validate(trans)
        if artifact.config.graph_kind != "state":
            raise SchemaError("ServoTrackingPolicy requires an occurrence-level state graph")
        if mode not in {"servo", "ff+servo", "replay"}:
            raise SchemaError("servo mode must be servo, ff+servo, or replay")
        if trans.action is None:
            raise SchemaError("the v0.5 witness requires recorded actions for gripper/contact control")
        if (window < 0 or settle < 0 or not np.isfinite(feedback_gain) or feedback_gain < 0
                or not np.isfinite(contact_gain) or contact_gain < 0
                or not np.isfinite(free_distance) or free_distance <= 0):
            raise SchemaError("servo window and settle must be non-negative")
        n = artifact.nodes.n_nodes
        for name in ("r", "o", "e", "q_p", "mode", "q_a"):
            value = np.asarray(getattr(coordinates, name))
            if value.shape[0] != n:
                raise SchemaError(f"StateCoordinates.{name} must align with graph vertices")
        qa = np.asarray(coordinates.q_a, dtype=np.float64)
        if qa.ndim != 2 or qa.shape[1] != len(schema.action_limits):
            raise SchemaError("q_a width must equal StateSchema.action_limits")
        self.artifact = artifact
        self.trans = trans
        self.schema = schema
        self.coordinates = coordinates
        self.mode = mode
        self.feedback_gain = float(feedback_gain)
        self.window = int(window)
        self.settle = int(settle)
        self.q_a = qa
        self.q_p = np.asarray(coordinates.q_p, dtype=np.float64)
        self.modes = np.asarray(coordinates.mode, dtype=np.int64)
        self.action_limits = np.asarray(schema.action_limits, dtype=np.float64)
        self.free_distance = float(free_distance)
        self.contact_gain = float(contact_gain)
        if response_gain is None:
            response_gain = np.ones(qa.shape[1], dtype=np.float64)
        self.response_gain = np.asarray(response_gain, dtype=np.float64)
        if self.response_gain.shape != (qa.shape[1],) or np.any(~np.isfinite(self.response_gain)) or np.any(self.response_gain <= 0):
            raise SchemaError("response_gain must be a positive finite vector matching q_a")
        if epsilon_a is None:
            rows = np.flatnonzero(~trans.terminal)
            rows = rows[trans.episode_id[rows] == trans.episode_id[rows + 1]]
            delta_a = qa[rows + 1] - qa[rows] if rows.size else np.empty((0, qa.shape[1]))
            for dim in getattr(schema, "periodic_action", ()):
                delta_a[:, dim] = (delta_a[:, dim] + np.pi) % (2.0 * np.pi) - np.pi
            steps = np.abs(delta_a)
            eps_a = np.zeros(qa.shape[1], dtype=np.float64)
            for dim in range(qa.shape[1]):
                positive = steps[:, dim][steps[:, dim] > 1e-7] if steps.size else np.empty(0)
                eps_a[dim] = np.median(positive) if positive.size else self.action_limits[dim]
            epsilon_a = eps_a
        self.epsilon_a = np.asarray(epsilon_a, dtype=np.float64)
        if self.epsilon_a.shape != (qa.shape[1],) or np.any(~np.isfinite(self.epsilon_a)) or np.any(self.epsilon_a <= 0):
            raise SchemaError("epsilon_a must be a positive finite vector matching q_a")
        if epsilon_p is None:
            epsilon_p = np.full(self.q_p.shape[1], 0.005, dtype=np.float64)
            for dim in schema.periodic_passive:
                epsilon_p[dim] = np.deg2rad(2.0)
        self.epsilon_p = np.asarray(epsilon_p, dtype=np.float64)
        if self.epsilon_p.shape != (self.q_p.shape[1],) or np.any(self.epsilon_p <= 0):
            raise SchemaError("epsilon_p must be a positive vector matching q_p")
        if tracking_tube is None:
            if qa.shape[1] == 5:
                tracking_tube = np.asarray([0.01, 0.01, 0.01, np.deg2rad(5.0), 0.1])
            else:
                tracking_tube = self.epsilon_a.copy()
        self.tracking_tube = np.asarray(tracking_tube, dtype=np.float64)
        if self.tracking_tube.shape != (qa.shape[1],) or np.any(~np.isfinite(self.tracking_tube)) or np.any(self.tracking_tube <= 0):
            raise SchemaError("tracking_tube must be a positive finite vector matching q_a")
        self.action_row = np.asarray(artifact.nodes.rep_ref[:, 0], dtype=np.int64)
        self._successor_kind, self._successor_tau = _successor_edge_table(artifact)
        self._successor_edge = np.full(n, -1, dtype=np.int64)
        candidate_edges = np.flatnonzero(artifact.succ[artifact.edges.src] == artifact.edges.dst)
        if candidate_edges.size:
            order = np.lexsort((candidate_edges, artifact.weights[candidate_edges], artifact.edges.src[candidate_edges]))
            ordered = candidate_edges[order]
            ordered_src = artifact.edges.src[ordered]
            first = np.r_[True, ordered_src[1:] != ordered_src[:-1]]
            chosen = ordered[first]
            self._successor_edge[ordered_src[first]] = chosen
        self.path: list[PathStep] = []
        self.pointer = 0
        self.start_vertex = -1
        self.stage = "idle"
        self.selected_vertex = -1
        self.follow_error = float("inf")
        self.target_vertex = -1
        self.bridge_started = -1
        self.bridge_records: list[dict[str, Any]] = []
        self.failure_reason: str | None = None
        self.failure_detail: str | None = None
        self.step_count = 0
        self._active_bad = 0
        self._active_best_tube_error = float("inf")
        self._active_stale_steps = 0
        self._passive_bad = 0
        self._mode_bad = 0
        self._live_mode = 0
        self._candidate_mode: int | None = None
        self._candidate_mode_count = 0
        self.min_dwell = int(artifact.config.min_dwell)
        self.max_steps = 0
        self.epsilon_a_by_dim = self.epsilon_a.copy()

    def reset(self, start_vertex: int) -> None:
        self.path = _extract_path_with_table(
            self.artifact, int(start_vertex), self._successor_kind, self._successor_tau
        )
        self.pointer = 0
        self.start_vertex = int(start_vertex)
        self.stage = "temporal"
        self.selected_vertex = int(start_vertex)
        self.target_vertex = int(start_vertex)
        self.bridge_started = -1
        self.bridge_records = []
        self.failure_reason = None
        self.failure_detail = None
        self.step_count = 0
        self._active_bad = self._mode_bad = 0
        self._active_best_tube_error = float("inf")
        self._active_stale_steps = 0
        self._passive_bad = 0
        self._live_mode = int(self.modes[start_vertex])
        self._candidate_mode = None
        self._candidate_mode_count = 0
        graph_steps = float(self.artifact.cost[start_vertex] / step_cost(self.artifact.config.gamma))
        self.max_steps = int(np.ceil(1.5 * graph_steps) + 20)

    def _action_delta(self, current: NDArray, target: NDArray) -> NDArray[np.float64]:
        delta = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
        for dim in getattr(self.schema, "periodic_action", ()):
            delta[dim] = (delta[dim] + np.pi) % (2.0 * np.pi) - np.pi
        return delta

    def _set_failure(self, reason: str, detail: str) -> None:
        if self.failure_reason is None:
            self.failure_reason = reason
            self.failure_detail = detail

    def _mode_path_vertex(self) -> int:
        return int(self.path[min(self.pointer, len(self.path) - 1)].vertex)

    def _begin_bridge_if_needed(self) -> bool:
        if self.pointer >= len(self.path) or self.path[self.pointer].kind != 1:
            return False
        source = self.path[self.pointer]
        self.stage = "bridge"
        if self.bridge_started < 0:
            self.bridge_started = self.step_count
            self.bridge_records.append({
                "source": source.vertex,
                "target": self.path[self.pointer + 1].vertex,
                "tau": source.tau,
                "bridge_type": int(self.artifact.edges.bridge_type[self._edge_index(source.vertex)]),
                "passive_diff_epsilon": float(np.max(
                    np.abs(passive_difference(
                        self.schema,
                        self.q_p[source.vertex],
                        self.q_p[self.path[self.pointer + 1].vertex],
                    )) / self.epsilon_p
                )),
                "start_step": self.step_count,
                "landed": False,
                "landed_step": None,
            })
        return True

    def _edge_index(self, source: int) -> int:
        edge_index = int(self._successor_edge[source])
        if edge_index < 0:
            raise SchemaError("path successor edge is missing")
        return edge_index

    def _readout(self, sim_readout: Mapping[str, NDArray]) -> tuple[NDArray, NDArray, int, bool, bool]:
        required = {"q_a", "q_p", "mode"}
        missing = required - sim_readout.keys()
        if missing:
            raise SchemaError(f"sim_readout is missing required coordinates: {sorted(missing)}")
        qa = np.asarray(sim_readout["q_a"], dtype=np.float64).reshape(-1)
        qp = np.asarray(sim_readout["q_p"], dtype=np.float64).reshape(-1)
        observed_mode = int(np.asarray(sim_readout["mode"]).item())
        contact = bool(np.any(np.asarray(sim_readout.get("gripper_contact", 0.0)) > 0.0))
        if observed_mode == self._live_mode:
            self._candidate_mode = None
            self._candidate_mode_count = 0
        elif observed_mode == self._candidate_mode:
            self._candidate_mode_count += 1
        else:
            self._candidate_mode = observed_mode
            self._candidate_mode_count = 1
        if self._candidate_mode_count >= self.min_dwell:
            self._live_mode = observed_mode
            self._candidate_mode = None
            self._candidate_mode_count = 0
        mode = self._live_mode
        if "observation" in sim_readout:
            obs = np.asarray(sim_readout["observation"], dtype=np.float32).reshape(1, -1)
            aux = {
                key: np.asarray(sim_readout[key], dtype=np.float32).reshape(1, -1)
                for key in (
                    "qpos", "qvel", "button_states", "eef_position", "eef_yaw",
                    "gripper_opening", "gripper_contact",
                ) if key in sim_readout
            }
            # q_p is a physical coordinate frame: switch to object-relative
            # coordinates on the measured contact state, even while the
            # graph's discrete mode label is still within its dwell window.
            qp = np.asarray(
                self.schema.passive(obs, aux, np.asarray([observed_mode])),
                dtype=np.float64,
            ).reshape(-1)
            if "free" in sim_readout:
                is_free = bool(np.asarray(sim_readout["free"]).item())
            elif "qpos" in aux:
                current_coordinates = self.schema.coordinates(obs, aux)
                is_free = bool(free_space_mask(
                    obs, aux["qpos"], current_coordinates, distance=self.free_distance
                )[0])
            else:
                is_free = False
        else:
            is_free = bool(np.asarray(sim_readout.get("free", False)).item())
        is_free = is_free and not contact
        if qa.shape != self.action_limits.shape or qp.shape != self.epsilon_p.shape:
            raise SchemaError("sim_readout q_a/q_p has an incompatible shape")
        if not np.all(np.isfinite(qa)) or not np.all(np.isfinite(qp)):
            raise SchemaError("sim_readout coordinates must be finite")
        return qa, qp, mode, is_free, contact

    def _choose_temporal_target(
        self, current: NDArray, current_passive: NDArray, current_mode: int
    ) -> int:
        end = min(len(self.path), self.pointer + self.window + 1)
        # A temporal lookahead must stop at the first bridge source. Otherwise
        # nearest-neighbor selection could jump over a bridge and bypass its
        # landing protocol entirely.
        bridge_positions = [
            pos for pos in range(self.pointer, end) if self.path[pos].kind == 1
        ]
        if bridge_positions:
            end = bridge_positions[0]
        candidates = [
            pos for pos in range(self.pointer, end)
            if self.path[pos].kind != 1
        ]
        if not candidates:
            candidates = [
                pos for pos in range(end, len(self.path))
                if self.path[pos].kind != 1
            ][:1]
        vertices = np.asarray([self.path[pos].vertex for pos in candidates], dtype=np.int64)
        same_mode = self.modes[vertices] == current_mode if vertices.size else np.empty(0, dtype=np.bool_)
        if np.any(same_mode):
            candidates = list(np.asarray(candidates)[same_mode])
            vertices = vertices[same_mode]
            self._mode_bad = 0
        else:
            self._mode_bad += 1
            if self._mode_bad >= 10:
                self._set_failure("mode_mismatch", "simulator mode differs from every temporal lookahead vertex for 10 steps")
        if vertices.size == 0:
            raise SchemaError("graph path has no temporal target in the lookahead window")
        # Repeated end-effector poses can occur before and after an object has
        # moved. Keep the active-pose lookahead on the same passive-world
        # segment when such a candidate exists; otherwise the controller can
        # jump to a later contact state and mistake that path jump for physical
        # passive drift.
        passive_delta = passive_difference(
            self.schema, self.q_p[vertices], current_passive[None, :]
        )
        passive_error = np.max(
            np.abs(passive_delta) / self.epsilon_p[None, :], axis=1
        )
        passive_compatible = passive_error <= 3.0
        if np.any(passive_compatible):
            candidates = list(np.asarray(candidates)[passive_compatible])
            vertices = vertices[passive_compatible]
        errors = np.stack([np.max(np.abs(self._action_delta(current, self.q_a[v])) / self.epsilon_a) for v in vertices])
        selected = int(np.argmin(errors))
        path_pos = int(candidates[selected])
        self.pointer = max(self.pointer, min(path_pos + 1, len(self.path) - 1))
        self.selected_vertex = int(vertices[selected])
        next_pos = min(path_pos + 1, len(self.path) - 1)
        self.target_vertex = int(self.path[next_pos].vertex)
        self.follow_error = float(errors[selected])
        return self.target_vertex

    def act(self, sim_readout: Mapping[str, NDArray]) -> NDArray[np.float32]:
        if not self.path:
            raise SchemaError("call reset(start_vertex) before requesting an action")
        if self.failure_reason is not None:
            return np.zeros(self.action_limits.shape, dtype=np.float32)
        current, passive, mode, is_free, contact = self._readout(sim_readout)
        self.step_count += 1
        if self.step_count > self.max_steps:
            self._set_failure("timeout", "graph-cost-derived episode timeout")
            return np.zeros(self.action_limits.shape, dtype=np.float32)

        is_bridge = self._begin_bridge_if_needed()
        if is_bridge:
            target = int(self.path[self.pointer + 1].vertex)
            self.target_vertex = target
            self.selected_vertex = int(self.path[self.pointer].vertex)
            self.stage = "bridge"
            expected_mode = int(self.modes[target])
            source_type = int(self.artifact.edges.bridge_type[self._edge_index(self.path[self.pointer].vertex)])
            elapsed = self.step_count - self.bridge_started
            bridge_tau = int(self.path[self.pointer].tau)
            record = self.bridge_records[-1]
            if source_type != 1:
                passive_error = np.max(np.abs(passive_difference(
                    self.schema, self.q_p[self.path[self.pointer].vertex], passive
                )) / self.epsilon_p)
                self._passive_bad = self._passive_bad + 1 if passive_error > 3.0 else 0
                if self._passive_bad >= 5:
                    self._set_failure("passive_drift", "R bridge passive coordinates exceeded 3 epsilon_p for 5 steps")
            else:
                record["passive_drift_epsilon"] = float(np.max(np.abs(passive_difference(
                    self.schema, self.q_p[self.path[self.pointer].vertex], passive
                )) / self.epsilon_p))
            if mode != expected_mode:
                self._mode_bad += 1
                if self._mode_bad >= 10:
                    self._set_failure("mode_mismatch", "mode differs from bridge destination for 10 steps")
            else:
                self._mode_bad = 0
            error = self._action_delta(current, self.q_a[target])
            self.follow_error = float(np.max(np.abs(error) / self.epsilon_a))
            landed = np.max(np.abs(error) / self.tracking_tube) <= 1.0
            if landed:
                record["landed"] = True
                record["landed_step"] = self.step_count
                record["elapsed"] = elapsed
                record["landing_ratio"] = float(elapsed / max(bridge_tau, 1))
                self.pointer += 1
                self.bridge_started = -1
                self.stage = "temporal"
                self._active_bad = 0
                self._active_best_tube_error = float("inf")
                self._active_stale_steps = 0
            elif elapsed > bridge_tau + 10:
                self._set_failure("timeout", "bridge landing exceeded tau+10 steps")
        else:
            self.stage = "temporal"
            self._choose_temporal_target(current, passive, mode)
            error = self._action_delta(current, self.q_a[self.target_vertex])
            self.follow_error = float(np.max(np.abs(error) / self.epsilon_a))
            tube_error = float(np.max(np.abs(error) / self.tracking_tube))
            if tube_error < self._active_best_tube_error - 1e-6:
                self._active_best_tube_error = tube_error
                self._active_stale_steps = 0
            else:
                self._active_stale_steps += 1
            self._active_bad = self._active_bad + 1 if tube_error > 3.0 else 0
            if tube_error > 3.0 and self._active_stale_steps >= 15:
                self._set_failure("active_tracking", "tracking error exceeded 3 physical tubes without reducing for 15 steps")
            passive_error = np.max(np.abs(passive_difference(
                self.schema, self.q_p[self.selected_vertex], passive
            )) / self.epsilon_p)
            self._passive_bad = self._passive_bad + 1 if passive_error > 3.0 else 0
            if self._passive_bad >= 5:
                self._set_failure("passive_drift", "passive coordinates exceeded 3 epsilon_p for 5 steps")

        if self.failure_reason is not None:
            return np.zeros(self.action_limits.shape, dtype=np.float32)
        row = int(self.action_row[self.selected_vertex])
        recorded = np.asarray(self.trans.action[row], dtype=np.float64)
        if self.mode == "replay":
            action = recorded
        elif is_bridge or is_free:
            action = recorded.copy()
            action[:4] = np.clip(
                self.feedback_gain * self.response_gain[:4] * error[:4] / self.action_limits[:4],
                -1.0,
                1.0,
            )
        else:
            contact_error = self._action_delta(current, self.q_a[self.selected_vertex])
            action = contact_replay_correction(
                recorded, contact_error, self.action_limits, self.contact_gain
            )
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def status(self) -> dict[str, Any]:
        return {
            "pointer": int(self.pointer),
            "stage": self.stage,
            "selected_vertex": int(self.selected_vertex),
            "target_vertex": int(self.target_vertex),
            "follow_error_epsilon": float(self.follow_error),
            "response_gain": self.response_gain.tolist(),
            "contact_gain": self.contact_gain,
            "tracking_tube": self.tracking_tube.tolist(),
            "bridge_landing": [dict(item) for item in self.bridge_records],
            "failure_reason": self.failure_reason,
            "failure_detail": self.failure_detail,
            "step_count": int(self.step_count),
            "timeout_steps": int(self.max_steps),
        }
