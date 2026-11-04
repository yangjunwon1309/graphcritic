"""Retrospective v0.3 bridge audit required by the v0.4 development guide."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.spatial import cKDTree

from graphcritic.io import load_graph, load_sim_state, load_transitions
from graphcritic.modes import smooth_mode_changes, state_episode_time
from graphcritic.passive_bridges import physical_tau_lower_bounds
from graphcritic.path import _extract_path_with_table, _successor_edge_table
from graphcritic.state_schema import estimate_passive_tolerance, get_state_schema, passive_difference
from graphcritic.types import SchemaError
from graphcritic.featurize import state_table


def _quantiles(values: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(x.size),
        "p50": float(np.percentile(x, 50)) if x.size else None,
        "p95": float(np.percentile(x, 95)) if x.size else None,
        "max": float(np.max(x)) if x.size else None,
    }


def _coordinates(trans, sim_state, graph):
    schema_name = graph.config.bridge.state_schema or {
        "ogbench_scene_pos": "ogbench_scene", "robomimic_pos": "robomimic"
    }.get(graph.config.feature.name)
    if schema_name is None:
        raise SchemaError("bridge audit requires a recognized physical StateSchema")
    schema = get_state_schema(schema_name)
    states = state_table(trans)[0]
    aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
    if sim_state.button_states is not None:
        aux["button_states"] = sim_state.button_states
    coordinates = schema.coordinates(states, aux)
    episodes, times = state_episode_time(trans)
    modes = smooth_mode_changes(coordinates.mode, episodes, times, graph.config.min_dwell)
    coordinates = replace(coordinates, mode=modes)
    return schema, states, coordinates, episodes, times, modes


def _nearest_cross_episode(schema, coordinates, episodes, modes, epsilon) -> dict[str, float | int | None]:
    q_p = np.asarray(coordinates.q_p, dtype=np.float64)
    epsilon = np.asarray(epsilon, dtype=np.float64)
    z = q_p / epsilon[None, :]
    nearest: list[float] = []
    for mode in np.unique(modes):
        refs = np.flatnonzero(modes == mode)
        if refs.size < 2:
            continue
        tree = cKDTree(z[refs])
        k = min(refs.size, 512)
        for start in range(0, refs.size, 1024):
            src = refs[start : start + 1024]
            _, local = tree.query(z[src], k=k, p=np.inf, workers=-1)
            if k == 1:
                local = local[:, None]
            cand = refs[np.asarray(local, dtype=np.int64)]
            for source, targets in zip(src, cand, strict=True):
                cross = targets[episodes[targets] != episodes[source]]
                if cross.size:
                    delta = passive_difference(schema, q_p[source], q_p[cross])
                    nearest.append(float(np.min(np.max(np.abs(delta) / epsilon[None, :], axis=1))))
    values = np.asarray(nearest, dtype=np.float64)
    return {
        "n": int(values.size),
        "p50_epsilon_units": float(np.percentile(values, 50)) if values.size else None,
        "p95_epsilon_units": float(np.percentile(values, 95)) if values.size else None,
        "frac_le_1": float(np.mean(values <= 1.0)) if values.size else None,
        "epsilon_used": epsilon.tolist(),
    }


def _bridge_physics(graph, schema, coordinates, trans, episodes, modes, epsilon):
    bridge = graph.edges.kind == 1
    ids = np.flatnonzero(bridge)
    if ids.size == 0:
        return {"n": 0, "kinematic_violation": None, "passive_violation": None}
    rows = np.flatnonzero(~trans.terminal)
    rows = rows[(episodes[rows] == episodes[rows + 1]) & (modes[rows] == modes[rows + 1])]
    delta = passive_difference(schema, coordinates.q_p[rows], coordinates.q_p[rows + 1]) if rows.size else np.empty((0, coordinates.q_p.shape[1]))
    speed = np.quantile(np.abs(delta), 0.999, axis=0) if rows.size else np.zeros(coordinates.q_p.shape[1])
    speed = np.maximum(speed, 1e-9)
    kinematic = np.empty(ids.size, dtype=np.int64)
    passive_tau = np.empty(ids.size, dtype=np.int64)
    passive_units = np.empty(ids.size, dtype=np.float64)
    for start in range(0, ids.size, 100_000):
        end = min(start + 100_000, ids.size)
        selected = ids[start:end]
        src, dst = graph.edges.src[selected], graph.edges.dst[selected]
        kinematic[start:end] = schema.kinematic_steps(coordinates.r[src], coordinates.r[dst])
        qpdiff = passive_difference(schema, coordinates.q_p[src], coordinates.q_p[dst])
        passive_tau[start:end] = np.max(np.ceil(np.abs(qpdiff) / speed[None, :]), axis=1).astype(np.int64)
        passive_units[start:end] = np.max(np.abs(qpdiff) / epsilon[None, :], axis=1)
    _, kin_violation, passive_violation = physical_tau_lower_bounds(
        graph.edges.kappa[ids], kinematic, passive_tau,
        settle=graph.config.bridge.kinematic_settle,
    )
    # The v0.3 model predicts total bridge time; compare that prediction with
    # each physical lower bound without applying the v0.4 clamp.
    return {
        "n": int(ids.size),
        "tau_learned": _quantiles(graph.edges.kappa[ids]),
        "tau_kinematic": _quantiles(kinematic),
        "tau_passive": _quantiles(passive_tau),
        "kinematic_violation": float(np.mean(graph.edges.kappa[ids] < kinematic)),
        "passive_violation": float(np.mean(graph.edges.kappa[ids] < passive_tau)),
        "bridge_passive_diff_epsilon_units": _quantiles(passive_units),
    }


def _eval_summary(path, graph, schema, coordinates, epsilon, successor_table):
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("graph_hash") != graph.content_hash:
        raise SchemaError(f"evaluation {path} does not match graph hash")
    outcomes = result.get("outcomes", [])
    zero = [item for item in outcomes if item.get("planned_bridges") == 0]
    one_to_three = [item for item in outcomes if 1 <= item.get("planned_bridges", 0) <= 3]
    zero_rate = float(np.mean([item["success"] for item in zero])) if zero else None
    bridged_rate = float(np.mean([item["success"] for item in one_to_three])) if one_to_three else None
    bridge_attempts = []
    exact = True
    for outcome in outcomes:
        if outcome.get("bridge_landing"):
            for record in outcome["bridge_landing"]:
                bridge_attempts.append({
                    "passive_diff_epsilon": record.get("passive_diff_epsilon"),
                    "landed": bool(record.get("landed")),
                    "tau": record.get("tau"),
                })
            continue
        exact = False
        try:
            path_steps = _extract_path_with_table(graph, int(outcome["start_state"]), *successor_table)
        except Exception:
            continue
        bridges = [step for step in path_steps if step.kind == 1]
        n_attempted = min(len(bridges), int(outcome.get("used_bridges", 0)))
        ratios = outcome.get("bridge_time_ratios", [])
        for index, step in enumerate(bridges[:n_attempted]):
            step_pos = path_steps.index(step)
            dest = path_steps[step_pos + 1].vertex
            delta = passive_difference(schema, coordinates.q_p[step.vertex], coordinates.q_p[dest])
            bridge_attempts.append({
                "passive_diff_epsilon": float(np.max(np.abs(delta) / epsilon)),
                "landed": index < len(ratios),
                "tau": step.tau,
            })
    return {
        "file": str(path),
        "zero_bridge": {"n": len(zero), "success_rate": zero_rate},
        "bridge_1_to_3": {"n": len(one_to_three), "success_rate": bridged_rate},
        "retention": bridged_rate / zero_rate if bridged_rate is not None and zero_rate else None,
        "bridge_attempts": bridge_attempts,
        "landing_attribution_exact": exact,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trans, trans_meta = load_transitions(args.transitions)
    graph = load_graph(args.graph, trans=trans)
    sim_state = load_sim_state(args.sim_state)
    if trans_meta["dataset_id"] != graph.meta["dataset_id"]:
        raise SchemaError("transitions and graph dataset IDs differ")
    schema, states, coordinates, episodes, times, modes = _coordinates(trans, sim_state, graph)
    del states, times
    contact = state_table(trans)[0][:, 18] > 0 if schema.name == "ogbench_scene" else None
    tolerance = estimate_passive_tolerance(
        schema, coordinates.q_p, episodes, modes, coordinates.q_a, contact_mask=contact,
    )
    epsilon = np.asarray(tolerance["epsilon"], dtype=np.float64)
    successor_table = _successor_edge_table(graph)
    result = {
        "dataset_id": graph.meta["dataset_id"],
        "graph_hash": graph.content_hash,
        "passive_epsilon": epsilon.tolist(),
        "nn_passive_cross": _nearest_cross_episode(schema, coordinates, episodes, modes, epsilon),
        "learned_bridge_physics": _bridge_physics(graph, schema, coordinates, trans, episodes, modes, epsilon),
        "pi_g_runs": [
            _eval_summary(path, graph, schema, coordinates, epsilon, successor_table)
            for path in args.eval_json
        ],
        "note": "Historical v0.3 eval files lack per-bridge outcomes; passive-coordinate landing attribution is exact only for files that contain bridge_landing records.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "pi_g_runs"}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
