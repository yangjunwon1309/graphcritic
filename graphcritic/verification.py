"""Optional simulator checks for bridge reliability and graph-path replay."""

from __future__ import annotations

from typing import Any, Mapping, Protocol
import sys

import numpy as np
from numpy.typing import NDArray

from graphcritic.config import GraphConfig
from graphcritic.featurize import Featurizer
from graphcritic.modes import MODE_FNS
from graphcritic.solve import solve
from graphcritic.types import EdgeSet, NodeSet, SchemaError, SimStateSet, TransitionSet
from graphcritic.state_schema import get_state_schema
from graphcritic.physical_time import free_space_mask
from graphcritic.weights import step_cost


class SimVerifier(Protocol):
    def rollout(self, state: Mapping[str, NDArray], actions: NDArray) -> Mapping[str, Any]: ...
    def reset_to(self, state: Mapping[str, NDArray]) -> NDArray: ...
    def step(self, action: NDArray) -> tuple[NDArray, float, bool, bool, Mapping[str, Any]]: ...
    def current_state(self) -> Mapping[str, NDArray]: ...


def _bridge_weights(edges: EdgeSet, cfg: GraphConfig, d_step: float) -> NDArray[np.float64]:
    c = step_cost(cfg.gamma)
    radius = cfg.bridge.radius * d_step if cfg.bridge.radius_unit == "step" else cfg.bridge.radius
    penalty = c * (
        cfg.bridge.cost.lambda0_steps
        + cfg.bridge.cost.lambda1_steps * np.square(edges.dist / radius)
    )
    weights = c * edges.kappa.astype(np.float64)
    weights[edges.kind == 1] += penalty[edges.kind == 1]
    return weights


def _used_bridge_edges(
    edges: EdgeSet,
    weights: NDArray[np.float64],
    cost: NDArray[np.float64],
    succ: NDArray[np.int64],
) -> NDArray[np.int64]:
    n = cost.size
    order = np.lexsort((weights, edges.dst, edges.src))
    src, dst = edges.src[order], edges.dst[order]
    first = np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])]
    src, dst, edge_ids = src[first], dst[first], order[first]
    key = src * np.int64(n) + dst
    rows = np.flatnonzero(np.isfinite(cost) & (succ >= 0))
    successor_key = rows * np.int64(n) + succ[rows]
    pos = np.searchsorted(key, successor_key)
    inside = pos < key.size
    match = np.zeros(pos.shape, dtype=np.bool_)
    match[inside] = key[pos[inside]] == successor_key[inside]
    chosen = edge_ids[pos[match]]
    return chosen[edges.kind[chosen] == 1]


def _state_payload(sim_state: SimStateSet, state_id: int) -> dict[str, NDArray]:
    payload = {"qpos": sim_state.qpos[state_id], "qvel": sim_state.qvel[state_id]}
    if sim_state.button_states is not None:
        payload["button_states"] = sim_state.button_states[state_id]
    return payload


def _rollout_mode_and_feature(
    result: Mapping[str, Any], featurizer: Featurizer, cfg: GraphConfig
) -> tuple[NDArray[np.float32], int]:
    if "observation" not in result:
        raise SchemaError("simulator rollout result must contain 'observation'")
    obs = np.asarray(result["observation"], dtype=np.float32).reshape(1, -1)
    feature_aux = {
        key: np.asarray(result[key]).reshape(1, -1)
        for key in ("qpos", "qvel", "button_states") if key in result
    }
    phi = featurizer.transform(obs, aux=feature_aux)[0]
    if cfg.mode_fn == "none":
        mode = 0
    else:
        aux = {
            key: np.asarray(result[key], dtype=np.float32).reshape(1, -1)
            for key in ("qpos", "qvel", "button_states") if key in result
        }
        if cfg.mode_fn not in MODE_FNS:
            raise SchemaError(f"unknown mode function {cfg.mode_fn!r}")
        mode = int(MODE_FNS[cfg.mode_fn](obs, aux)[0])
    return phi, mode


def verify_bridge_edges(
    trans: TransitionSet,
    nodes: NodeSet,
    edges: EdgeSet,
    cfg: GraphConfig,
    featurizer: Featurizer,
    sim_state: SimStateSet,
    modes: NDArray[np.int64],
    *,
    d_step: float,
    verifier: SimVerifier,
) -> tuple[EdgeSet, dict[str, Any]]:
    """Verify only bridges used by current shortest paths, then discard unknowns."""
    if trans.action is None:
        raise SchemaError("sim_verified bridge costs require recorded actions")
    sim_state.validate(nodes.n_nodes)
    current = edges
    n_pos = edges.n_pos.copy()
    n_neg = edges.n_neg.copy()
    alive = np.ones(edges.n_edges, dtype=np.bool_)
    budget = cfg.bridge.cost.sim_budget
    used_steps = 0
    verified = failed = 0
    iterations = 0
    c = step_cost(cfg.gamma)

    while True:
        iterations += 1
        current = EdgeSet(
            src=edges.src[alive], dst=edges.dst[alive], kappa=edges.kappa[alive],
            n_pos=n_pos[alive], n_neg=n_neg[alive], kind=edges.kind[alive],
            dist=edges.dist[alive], anchor=edges.anchor[alive],
            bridge_type=edges.bridge_type[alive],
        )
        active_ids = np.flatnonzero(alive)
        local_weights = c * current.kappa.astype(np.float64)
        bridge_local = current.kind == 1
        local_weights[bridge_local] += c * (
            cfg.bridge.cost.lambda0_steps
            + cfg.bridge.cost.lambda1_steps * np.square(
                current.dist[bridge_local] / (
                    cfg.bridge.radius * d_step if cfg.bridge.radius_unit == "step" else cfg.bridge.radius
                )
            )
        )
        cost, succ = solve(nodes, current, local_weights)
        local_used = _used_bridge_edges(current, local_weights, cost, succ)
        used_ids = active_ids[local_used]
        pending = used_ids[(n_pos[used_ids] == 0) & (n_neg[used_ids] == 0)]
        if pending.size == 0:
            break
        progress = False
        for edge_id in pending:
            length = int(edges.kappa[edge_id])
            if used_steps + length > budget:
                continue
            source = int(edges.src[edge_id])
            anchor = int(edges.anchor[edge_id])
            action_rows = np.arange(anchor, anchor + length, dtype=np.int64)
            result = verifier.rollout(_state_payload(sim_state, source), trans.action[action_rows])
            landed_phi, landed_mode = _rollout_mode_and_feature(result, featurizer, cfg)
            expected = int(edges.dst[edge_id])
            expected_aux = {
                "qpos": sim_state.qpos[expected : expected + 1],
                "qvel": sim_state.qvel[expected : expected + 1],
            }
            if sim_state.button_states is not None:
                expected_aux["button_states"] = sim_state.button_states[expected : expected + 1]
            success = (
                np.linalg.norm(landed_phi - featurizer.transform(
                    nodes.rep_feat[expected : expected + 1], aux=expected_aux
                )[0])
                <= cfg.bridge.cost.land_tol * d_step
                and landed_mode == int(modes[expected])
            )
            used_steps += length
            progress = True
            if success:
                n_pos[edge_id] = 1
                verified += 1
            else:
                alive[edge_id] = False
                failed += 1
        if not progress:
            break

    unverified = alive & (edges.kind == 1) & (n_pos == 0) & (n_neg == 0)
    alive[unverified] = False
    final_edges = EdgeSet(
        src=edges.src[alive], dst=edges.dst[alive], kappa=edges.kappa[alive],
        n_pos=n_pos[alive], n_neg=n_neg[alive], kind=edges.kind[alive],
        dist=edges.dist[alive], anchor=edges.anchor[alive],
        bridge_type=edges.bridge_type[alive],
    )
    return final_edges, {
        "sim_verification_iterations": int(iterations),
        "sim_verification_steps": int(used_steps),
        "sim_bridges_verified": int(verified),
        "sim_bridges_failed": int(failed),
        "sim_bridges_unverified_dropped": int(np.count_nonzero(unverified)),
        "sim_verification_budget": int(budget),
    }


def _chosen_edge_lookup(edges: EdgeSet, weights: NDArray[np.float64]) -> dict[tuple[int, int], int]:
    order = np.lexsort((weights, edges.dst, edges.src))
    lookup: dict[tuple[int, int], int] = {}
    for edge_id in order:
        key = (int(edges.src[edge_id]), int(edges.dst[edge_id]))
        lookup.setdefault(key, int(edge_id))
    return lookup


def evaluate_paths(
    trans: TransitionSet,
    nodes: NodeSet,
    edges: EdgeSet,
    weights: NDArray[np.float64],
    cost: NDArray[np.float64],
    succ: NDArray[np.int64],
    sim_state: SimStateSet,
    verifier: SimVerifier,
    *,
    n_paths: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Replay sampled shortest paths, executing temporal and bridge actions."""
    if trans.action is None:
        raise SchemaError("path execution requires recorded actions")
    if nodes.n_nodes != trans.n_rows + int(np.count_nonzero(trans.terminal)):
        raise SchemaError("path execution requires the identity-assigned state graph")
    sim_state.validate(nodes.n_nodes)
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(np.isfinite(cost) & ~nodes.is_goal)
    if candidates.size == 0:
        return {"path_exec_rate": 0.0, "path_exec_by_switches": {}, "path_exec_n": 0}
    chosen = rng.choice(candidates, size=min(n_paths, candidates.size), replace=False)
    lookup = _chosen_edge_lookup(edges, weights)
    outcomes: list[tuple[int, bool]] = []

    for path_index, start in enumerate(chosen, start=1):
        state_id = int(start)
        if state_id >= trans.n_rows:
            continue
        switches = 0
        success = False
        terminated = False
        verifier.reset_to(_state_payload(sim_state, state_id))
        steps = 0
        while state_id >= 0 and not nodes.is_goal[state_id] and steps < 1_000_000:
            next_id = int(succ[state_id])
            edge_id = lookup.get((state_id, next_id))
            if edge_id is None:
                break
            if edges.kind[edge_id] == 0:
                action_rows = [state_id]
            else:
                switches += 1
                anchor = int(edges.anchor[edge_id])
                action_rows = range(anchor, anchor + int(edges.kappa[edge_id]))
            for row in action_rows:
                obs, reward, terminated, truncated, info = verifier.step(trans.action[int(row)])
                steps += 1
                if bool(info.get("success", False)) or reward > 0:
                    success = True
                    break
                if terminated or truncated:
                    break
            if success or terminated or truncated:
                break
            state_id = next_id
        if not success and not terminated and not truncated:
            for _ in range(10):
                _, reward, terminated, truncated, info = verifier.step(np.zeros_like(trans.action[0]))
                steps += 1
                if bool(info.get("success", False)) or reward > 0:
                    success = True
                    break
                if terminated or truncated:
                    break
        outcomes.append((switches, success))
        if path_index % 50 == 0:
            print(
                f"graphcritic path replay: {path_index}/{len(chosen)} paths",
                file=sys.stderr,
                flush=True,
            )

    by_switches: dict[str, dict[str, float | int]] = {}
    for count in sorted({count for count, _ in outcomes}):
        values = [success for switches, success in outcomes if switches == count]
        by_switches[str(count)] = {"n": len(values), "rate": float(np.mean(values))}
    low_switch = [success for count, success in outcomes if count <= 3]
    return {
        "path_exec_rate": float(np.mean([success for _, success in outcomes])) if outcomes else 0.0,
        "path_exec_switch_le_3": float(np.mean(low_switch)) if low_switch else 0.0,
        "path_exec_by_switches": by_switches,
        "path_exec_n": int(len(outcomes)),
    }


class OGBenchVerifier:
    """State-reset and open-loop verifier for QC-compatible OGBench tasks."""

    def __init__(self, env: Any):
        self.env = env
        self.unwrapped = env.unwrapped

    def reset_to(self, state: Mapping[str, NDArray]) -> NDArray:
        self.env.reset(seed=0)
        qpos = np.asarray(state["qpos"]).copy()
        qvel = np.asarray(state["qvel"]).copy()
        button_states = state.get("button_states")
        setter = getattr(self.unwrapped, "set_state", None)
        if callable(setter):
            if button_states is None:
                setter(qpos, qvel)
            else:
                button_states = np.asarray(button_states, dtype=np.int64).copy()
                try:
                    setter(qpos, qvel, button_states=button_states)
                except TypeError:
                    setter(qpos, qvel, button_states)
        else:
            self.unwrapped.data.qpos[:] = qpos
            self.unwrapped.data.qvel[:] = qvel
            if button_states is not None and hasattr(self.unwrapped, "_cur_button_states"):
                self.unwrapped._cur_button_states = np.asarray(button_states, dtype=np.int64).copy()
            import mujoco
            mujoco.mj_forward(self.unwrapped.model, self.unwrapped.data)
        # The offline sidecar does not store MuJoCo's constraint-solver
        # warm-start vector. Reset it explicitly so repeated restores from the
        # same qpos/qvel have bitwise-identical first steps.
        data = getattr(self.unwrapped, "data", None)
        if data is not None and hasattr(data, "qacc_warmstart"):
            data.qacc_warmstart[:] = 0
        getter = getattr(self.unwrapped, "_get_obs", None)
        if callable(getter):
            return np.asarray(getter(), dtype=np.float32)
        getter = getattr(self.unwrapped, "compute_observation", None)
        if callable(getter):
            return np.asarray(getter(), dtype=np.float32)
        raise SchemaError(
            "OGBench environment exposes neither _get_obs() nor compute_observation()"
        )

    def step(self, action: NDArray) -> tuple[NDArray, float, bool, bool, Mapping[str, Any]]:
        result = self.env.step(np.asarray(action))
        if len(result) != 5:
            raise SchemaError("OGBench environment must return Gymnasium's five step values")
        obs, reward, terminated, truncated, info = result
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info

    def current_state(self) -> Mapping[str, NDArray]:
        state = {
            "qpos": np.asarray(self.unwrapped.data.qpos, dtype=np.float32).copy(),
            "qvel": np.asarray(self.unwrapped.data.qvel, dtype=np.float32).copy(),
        }
        if hasattr(self.unwrapped, "_cur_button_states"):
            state["button_states"] = np.asarray(self.unwrapped._cur_button_states, dtype=np.int64).copy()
        # Physical units for the v0.4 pose-servo controller. These are read
        # directly from MuJoCo's current state, not reconstructed from a
        # potentially scaled observation vector.
        try:
            info = self.unwrapped.compute_ob_info()
            state["eef_position"] = np.asarray(info["proprio/effector_pos"], dtype=np.float32).copy()
            state["eef_yaw"] = np.asarray(info["proprio/effector_yaw"], dtype=np.float32).copy()
            state["gripper_opening"] = np.asarray(info["proprio/gripper_opening"], dtype=np.float32).copy()
            state["gripper_contact"] = np.asarray(info["proprio/gripper_contact"], dtype=np.float32).copy()
        except (AttributeError, KeyError):
            pass
        return state

    def sim_readout(
        self,
        observation: NDArray,
        *,
        schema_name: str = "ogbench_scene",
        press_height: float | None = None,
    ) -> Mapping[str, NDArray]:
        """Return physical coordinates required by ServoTrackingPolicy."""
        state = dict(self.current_state())
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        aux = {key: np.asarray(state[key]).reshape(1, -1) for key in ("qpos", "qvel", "button_states", "eef_position", "eef_yaw", "gripper_opening", "gripper_contact") if key in state}
        schema = get_state_schema(schema_name)
        if schema_name == "ogbench_puzzle" and press_height is not None:
            from dataclasses import replace

            schema = replace(schema, press_height=float(press_height))
        coordinates = schema.coordinates(obs, aux)
        if schema_name == "ogbench_puzzle":
            is_free = coordinates.mode == 0
        else:
            is_free = free_space_mask(
                obs, aux["qpos"], coordinates, distance=0.05,
            )
        return {
            **state,
            "observation": obs[0].copy(),
            "r": coordinates.r[0].copy(),
            "o": coordinates.o[0].copy(),
            "e": coordinates.e[0].copy(),
            "q_p": coordinates.q_p[0].copy(),
            "mode": coordinates.mode[0].copy(),
            "q_a": coordinates.q_a[0].copy(),
            "free": np.asarray(is_free[0]),
        }

    def rollout(self, state: Mapping[str, NDArray], actions: NDArray) -> Mapping[str, Any]:
        observation = self.reset_to(state)
        success = False
        for action in np.asarray(actions):
            observation, reward, terminated, truncated, info = self.step(action)
            success = bool(info.get("success", False)) or reward > 0
            if success or terminated or truncated:
                break
        result: dict[str, Any] = {"observation": observation, "success": success}
        result.update(self.current_state())
        return result
