"""Evaluate graph-tracking paths in an OGBench simulator (post-build only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphcritic.featurize import StateFeaturizer, state_table
from graphcritic.io import load_graph, load_sim_state, load_transitions
from graphcritic.path import GraphTrackingPolicy, extract_path
from graphcritic.types import SchemaError
from graphcritic.weights import step_cost


def _episode_kind(trans) -> dict[int, bool]:
    result: dict[int, bool] = {}
    for episode in np.unique(trans.episode_id):
        rows = trans.episode_id == episode
        result[int(episode)] = bool(np.any(trans.success_cur[rows]) or np.any(trans.success[rows]))
    return result


def _choose_starts(trans, artifact, count: int, seed: int) -> list[tuple[int, bool]]:
    rng = np.random.default_rng(seed)
    current_cost = artifact.cost[artifact.nodes.assign_cur]
    candidates = np.flatnonzero(np.isfinite(current_cost) & ~trans.success_cur)
    episode_kind = _episode_kind(trans)
    failed = candidates[np.asarray([not episode_kind[int(trans.episode_id[row])] for row in candidates])]
    successful = candidates[np.asarray([episode_kind[int(trans.episode_id[row])] for row in candidates])]
    if failed.size == 0 or successful.size == 0:
        raise SchemaError("graph evaluation needs finite-cost rows from both failed and successful episodes")
    n_failed = count // 2
    n_success = count - n_failed
    result: list[tuple[int, bool]] = []
    for pool, size, label in ((failed, n_failed, True), (successful, n_success, False)):
        chosen = rng.choice(pool, size=size, replace=pool.size < size)
        result.extend((int(row), label) for row in chosen)
    rng.shuffle(result)
    return result


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "p50": None, "p90": None}
    return {
        "n": len(values),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ogbench-dataset")
    source.add_argument("--transitions", type=Path)
    parser.add_argument("--sim-state", type=Path)
    parser.add_argument("--env", help="OGBench Gymnasium environment ID (derived from --ogbench-dataset when omitted)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument(
        "--bridge-tube-dstep", type=float, default=1.0,
        help="bridge arrival tube radius as a multiple of graph d_step (default: 1)",
    )
    parser.add_argument("--max-starts", type=int, help="Debug cap; overrides --starts")
    args = parser.parse_args()
    count = args.max_starts if args.max_starts is not None else args.starts
    if count < 2 or args.window < 0 or not np.isfinite(args.bridge_tube_dstep) or args.bridge_tube_dstep <= 0:
        parser.error("starts must be at least 2, window non-negative, and bridge tube multiplier positive")

    if args.ogbench_dataset:
        from envs.ogbench import _dataset_parts, load_ogbench_transitions
        trans, dataset_id, sim_state = load_ogbench_transitions(
            args.ogbench_dataset, return_sim_state=True
        )
        env_id = args.env or _dataset_parts(args.ogbench_dataset)[0]
    else:
        trans, metadata = load_transitions(args.transitions)
        dataset_id = metadata["dataset_id"]
        if args.sim_state is None:
            parser.error("--sim-state is required with --transitions")
        sim_state = load_sim_state(args.sim_state, n_states=state_table(trans)[0].shape[0])
        if not args.env:
            parser.error("--env is required with --transitions")
        env_id = args.env

    artifact = load_graph(args.graph, trans=trans)
    if artifact.meta["dataset_id"] != dataset_id:
        raise SchemaError("graph and evaluation dataset IDs differ")
    if sim_state is None:
        raise SchemaError("graph tracking requires the aligned simulator state sidecar")
    states = state_table(trans)[0]
    sim_state.validate(states.shape[0])
    featurizer = StateFeaturizer.fit(trans, artifact.config.feature, sim_state=sim_state)
    aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
    if sim_state.button_states is not None:
        aux["button_states"] = sim_state.button_states
    state_features = featurizer.transform(states, aux=aux)
    policy = GraphTrackingPolicy(
        artifact, trans, featurizer, artifact.config.mode_fn,
        sim_state=sim_state, state_features=state_features, window=args.window,
    )
    starts = _choose_starts(trans, artifact, count, args.seed)

    try:
        import ogbench  # noqa: F401
        import gymnasium as gym
        from graphcritic.verification import OGBenchVerifier
    except ImportError as exc:
        raise RuntimeError("graph policy evaluation requires the bench environment") from exc

    wrapped_env = gym.make(env_id, max_episode_steps=1_000_000_000)
    env = wrapped_env.unwrapped  # step the raw environment, bypassing Gym TimeLimit
    verifier = OGBenchVerifier(env)
    c = step_cost(artifact.config.gamma)
    bridge_tube_radius = float(artifact.meta.get("d_step", 1.0)) * args.bridge_tube_dstep
    outcomes: list[dict] = []
    bridge_ratios: list[float] = []
    bridge_targets_reached = 0
    bridge_targets_unreached = 0
    try:
        for number, (start, failed_episode) in enumerate(starts, start=1):
            policy.reset(start)
            path = policy.path
            planned_bridges = sum(step.kind == 1 for step in path)
            graph_steps = float(artifact.cost[start] / c)
            max_steps = int(np.ceil(1.5 * graph_steps) + 20)
            state_payload = {
                "qpos": sim_state.qpos[start], "qvel": sim_state.qvel[start]
            }
            if sim_state.button_states is not None:
                state_payload["button_states"] = sim_state.button_states[start]
            obs = verifier.reset_to(state_payload)
            steps = 0
            drift_steps = 0
            success = False
            failure_reason = None
            status = policy.status()
            active_bridges: list[dict[str, int]] = []
            path_bridge_samples: list[float] = []
            while steps < max_steps:
                current_aux = verifier.current_state()
                try:
                    action = policy.act(obs, current_aux)
                except SchemaError as exc:
                    failure_reason = str(exc)
                    break
                obs, reward, terminated, truncated, info = verifier.step(action)
                steps += 1
                success = bool(info.get("success", False)) or reward > 0
                status = policy.status()
                # A path pointer can skip a bridge source on the first action.
                # Start its clock when it is crossed, but count completion only
                # after the simulated state enters the destination's feature tube.
                for path_pos, tau in status["crossed_bridges"]:
                    active_bridges.append({
                        "target": int(path[path_pos + 1].vertex),
                        "tau": int(tau),
                        "start_step": steps - 1,
                    })
                reached = []
                for bridge in active_bridges:
                    distance = policy.distance_to_vertex(
                        obs, verifier.current_state(), bridge["target"]
                    )
                    if distance <= bridge_tube_radius:
                        elapsed = max(1, steps - bridge["start_step"])
                        ratio = float(elapsed / bridge["tau"])
                        bridge_ratios.append(ratio)
                        path_bridge_samples.append(ratio)
                        reached.append(bridge)
                for bridge in reached:
                    active_bridges.remove(bridge)
                    bridge_targets_reached += 1
                distance = policy.distance_to_selected(obs, verifier.current_state())
                drift_steps = drift_steps + 1 if distance > policy.fail_radius else 0
                if success or terminated or truncated or drift_steps >= 10:
                    if not success and not terminated and not truncated and drift_steps >= 10:
                        failure_reason = "tracking distance exceeded 2r for 10 consecutive steps"
                    elif truncated:
                        failure_reason = "raw environment returned truncated=True"
                    elif terminated and not success:
                        failure_reason = "environment terminated before success"
                    break
            bridge_count = int(status["bridge_use_count"])
            bridge_targets_unreached += len(active_bridges)
            outcomes.append({
                "start_state": start,
                "failed_episode": failed_episode,
                "planned_bridges": planned_bridges,
                "used_bridges": bridge_count,
                "graph_cost_steps": graph_steps,
                "actual_steps": steps,
                "success": bool(success),
                "pessimism_hold": bool(success and steps <= graph_steps),
                "time_ratio": float(steps / graph_steps) if graph_steps > 0 and success else None,
                "bridge_time_ratios": path_bridge_samples,
                "failure_reason": failure_reason,
            })
            if number % 25 == 0:
                print(f"graph policy evaluation: {number}/{len(starts)}", file=sys.stderr, flush=True)
    finally:
        wrapped_env.close()

    by_bridge: dict[str, dict[str, float | int]] = {}
    labels = ["0", "1", "2", "3", "4+"]
    for label in labels:
        selected = [
            item for item in outcomes
            if (str(item["planned_bridges"]) if item["planned_bridges"] < 4 else "4+") == label
        ]
        by_bridge[label] = {
            "n": len(selected),
            "success_rate": float(np.mean([item["success"] for item in selected])) if selected else 0.0,
        }
    successful = [item for item in outcomes if item["success"]]
    bridge_le3 = [item for item in outcomes if item["planned_bridges"] <= 3]
    time_ratios = [item["time_ratio"] for item in successful if item["time_ratio"] is not None]
    result = {
        "dataset_id": dataset_id,
        "graph_hash": artifact.content_hash,
        "tau_model": artifact.config.bridge.cost.tau_model,
        "n_starts": len(outcomes),
        "pi_g_success": float(np.mean([item["success"] for item in outcomes])) if outcomes else 0.0,
        "pi_g_success_bridge_le3": (
            float(np.mean([item["success"] for item in bridge_le3])) if bridge_le3 else 0.0
        ),
        "pi_g_starts_bridge_le3": len(bridge_le3),
        "pi_g_success_by_bridge_count": by_bridge,
        "pessimism_hold": float(np.mean([item["pessimism_hold"] for item in outcomes])) if outcomes else 0.0,
        "time_ratio": _quantiles(time_ratios),
        "bridge_time_ratio": _quantiles(bridge_ratios),
        "bridge_tube_dstep": float(args.bridge_tube_dstep),
        "bridge_tube_radius": bridge_tube_radius,
        "bridge_targets_reached": bridge_targets_reached,
        "bridge_targets_unreached": bridge_targets_unreached,
        "outcomes": outcomes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "outcomes"}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
