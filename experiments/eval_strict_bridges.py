"""Direct simulator landing check for sampled R bridge edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphcritic.featurize import state_table
from graphcritic.io import load_graph, load_sim_state, load_transitions
from graphcritic.physical_time import estimate_realized_speed, free_space_mask
from graphcritic.state_schema import OGBenchPuzzleStateSchema, get_state_schema, passive_difference, wrap_angle
from graphcritic.types import SchemaError
from graphcritic.verification import OGBenchVerifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-extra", type=int, default=2)
    args = parser.parse_args()
    if args.trials < 1 or args.settle_extra < 0:
        parser.error("trials must be positive and settle-extra non-negative")

    trans, _ = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    states = state_table(trans)[0]
    sim_state.validate(states.shape[0])
    graph = load_graph(args.graph, trans=trans)
    if graph.config.bridge.strategy not in {"passive_r_strict", "puzzle_r"}:
        raise SchemaError("direct R landing evaluation requires a passive_r_strict or puzzle_r graph")
    edge_ids = np.flatnonzero((graph.edges.kind == 1) & (graph.edges.bridge_type == 0))
    if edge_ids.size == 0:
        raise SchemaError("graph contains no R bridges to evaluate")
    if trans.action is None:
        raise SchemaError("direct bridge evaluation requires recorded actions")

    schema_name = graph.config.bridge.state_schema or "ogbench_scene"
    schema = get_state_schema(schema_name)
    press_height = None
    if schema_name == "ogbench_puzzle":
        press_height = float(graph.meta.get("bridge_puzzle_press_calibration", {}).get("press_height_m", schema.press_height))
        from dataclasses import replace

        schema = replace(schema, press_height=press_height)
    aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
    if sim_state.button_states is not None:
        aux["button_states"] = sim_state.button_states
    coordinates = schema.coordinates(states, aux)
    free = (
        coordinates.mode == 0 if schema_name == "ogbench_puzzle"
        else free_space_mask(states, sim_state.qpos, coordinates, distance=graph.config.bridge.free_distance)
    )
    calibration = estimate_realized_speed(trans, coordinates, free, schema)
    response_gain = np.asarray(calibration["feedback_gain_by_dimension"], dtype=np.float64)
    limits = np.asarray(schema.action_limits, dtype=np.float64)
    tube = np.asarray([0.01, 0.01, 0.01, np.deg2rad(5.0), 0.1], dtype=np.float64)
    epsilon_p = np.asarray([0.5]) if schema_name == "ogbench_puzzle" else np.asarray(graph.meta["bridge_passive_epsilon"])

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(edge_ids, size=args.trials, replace=edge_ids.size < args.trials)
    try:
        import gymnasium as gym
        import ogbench  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("direct bridge landing evaluation requires the bench environment") from exc
    wrapped = gym.make(args.env, max_episode_steps=1_000_000_000)
    verifier = OGBenchVerifier(wrapped.unwrapped)
    outcomes = []
    try:
        for number, edge_id in enumerate(chosen, start=1):
            source = int(graph.edges.src[edge_id])
            target = int(graph.edges.dst[edge_id])
            row = int(graph.nodes.rep_ref[source, 0])
            payload = {"qpos": sim_state.qpos[source], "qvel": sim_state.qvel[source]}
            if sim_state.button_states is not None:
                payload["button_states"] = sim_state.button_states[source]
            obs = verifier.reset_to(payload)
            tau = int(graph.edges.kappa[edge_id])
            budget = tau + args.settle_extra
            landed = False
            passive_drift = False
            final_error = None
            failure_reason = None
            steps = 0
            for steps in range(1, budget + 1):
                current = verifier.sim_readout(obs, schema_name=schema_name, press_height=press_height)
                if schema_name == "ogbench_puzzle" and not np.array_equal(
                    np.rint(current["q_p"]).astype(np.int64),
                    np.rint(coordinates.q_p[source]).astype(np.int64),
                ):
                    passive_drift = True
                    failure_reason = "passive_drift"
                    break
                error = np.asarray(coordinates.q_a[target] - current["q_a"], dtype=np.float64)
                error[3] = float(wrap_angle(np.asarray(error[3])))
                if schema_name == "ogbench_puzzle":
                    action = np.zeros_like(limits)
                    action[:4] = np.clip(response_gain[:4] * error[:4] / limits[:4], -1.0, 1.0)
                    action[4] = 0.0
                else:
                    action = np.asarray(trans.action[row], dtype=np.float64).copy()
                    action[:4] = np.clip(
                        response_gain[:4] * error[:4] / limits[:4], -1.0, 1.0
                    )
                obs, _, terminated, truncated, _ = verifier.step(action.astype(np.float32))
                readout = verifier.sim_readout(obs, schema_name=schema_name, press_height=press_height)
                if schema_name == "ogbench_puzzle" and not np.array_equal(
                    np.rint(readout["q_p"]).astype(np.int64),
                    np.rint(coordinates.q_p[source]).astype(np.int64),
                ):
                    passive_drift = True
                    failure_reason = "passive_drift"
                    break
                final_error = np.asarray(coordinates.q_a[target] - readout["q_a"], dtype=np.float64)
                final_error[3] = float(wrap_angle(np.asarray(final_error[3])))
                landed = bool(np.max(np.abs(final_error) / tube) <= 1.0)
                if landed or terminated or truncated:
                    if (terminated or truncated) and not landed:
                        failure_reason = "timeout"
                    break
            if not landed and failure_reason is None:
                failure_reason = "active_tracking"
            landed = bool(landed and not passive_drift)
            passive_delta = passive_difference(
                schema, coordinates.q_p[source], coordinates.q_p[target]
            )
            same_board = bool(np.array_equal(
                np.asarray(coordinates.q_p[source]), np.asarray(coordinates.q_p[target])
            )) if schema_name == "ogbench_puzzle" else None
            outcomes.append({
                "edge_id": int(edge_id),
                "source": source,
                "target": target,
                "tau_real": tau,
                "steps": int(steps),
                "landed": landed,
                "landing_ratio": float(steps / max(tau, 1)),
                "passive_difference_epsilon": (
                    float(np.max(np.abs(passive_delta) / epsilon_p)) if schema_name != "ogbench_puzzle" else None
                ),
                "same_button_configuration": same_board,
                "passive_drift": passive_drift,
                "failure_reason": failure_reason,
                "final_tube_error": None if final_error is None else float(np.max(np.abs(final_error) / tube)),
            })
            if number % 25 == 0:
                print(f"R direct landing: {number}/{args.trials}", file=sys.stderr, flush=True)
    finally:
        wrapped.close()

    result = {
        "dataset_id": graph.meta["dataset_id"],
        "graph_hash": graph.content_hash,
        "trials": len(outcomes),
        "landing_rate": float(np.mean([item["landed"] for item in outcomes])),
        "failure_counts": {
            reason: int(sum(item["failure_reason"] == reason for item in outcomes))
            for reason in ("passive_drift", "active_tracking", "timeout")
        },
        "gate": 0.8,
        "passed_gate": bool(np.mean([item["landed"] for item in outcomes]) >= 0.8),
        "budget": "tau_real + settle_extra",
        "settle_extra": int(args.settle_extra),
        "tracking_tube": tube.tolist(),
        "response_gain": response_gain.tolist(),
        "calibration": calibration,
        "outcomes": outcomes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "outcomes"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
