"""Evaluate graph-tracking paths in an OGBench simulator (post-build only)."""

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
from scipy.stats import spearmanr

from graphcritic.featurize import StateFeaturizer, state_table
from graphcritic.io import load_graph, load_sim_state, load_targets, load_transitions
from graphcritic.diagnostics import report as graph_report
from graphcritic.path import GraphTrackingPolicy, extract_path
from graphcritic.modes import smooth_mode_changes, state_episode_time
from graphcritic.servo import ServoTrackingPolicy
from graphcritic.state_schema import estimate_passive_tolerance, get_state_schema
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
    parser.add_argument("--targets", type=Path, help="Optional targets pair used for the graph-only cap gate")
    parser.add_argument("--starts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--controller", choices=("replay", "servo", "ff+servo"), default="replay")
    parser.add_argument("--feedback-gain", choices=(0.5, 1.0, 1.5, 2.0, 3.0), type=float, default=1.0,
                        help="pose feedback gain for servo; additive gain for ff+servo")
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
    targets = load_targets(args.targets, graph=artifact, expected_dataset_id=dataset_id) if args.targets else None
    graph_diagnostics = graph_report(trans, artifact, targets)
    states = state_table(trans)[0]
    sim_state.validate(states.shape[0])
    featurizer = StateFeaturizer.fit(trans, artifact.config.feature, sim_state=sim_state)
    aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
    if sim_state.button_states is not None:
        aux["button_states"] = sim_state.button_states
    policy = None
    servo_policy = None
    if args.controller == "replay":
        state_features = featurizer.transform(states, aux=aux)
        policy = GraphTrackingPolicy(
            artifact, trans, featurizer, artifact.config.mode_fn,
            sim_state=sim_state, state_features=state_features, window=args.window,
        )
    else:
        schema_name = artifact.config.bridge.state_schema or {
            "ogbench_scene_pos": "ogbench_scene", "robomimic_pos": "robomimic"
        }.get(artifact.config.feature.name)
        if schema_name is None:
            raise SchemaError("servo evaluation requires a bridge.state_schema or position feature")
        schema = get_state_schema(schema_name)
        coordinates = schema.coordinates(states, aux)
        episode, times = state_episode_time(trans)
        smooth_modes = smooth_mode_changes(
            coordinates.mode, episode, times, artifact.config.min_dwell
        )
        coordinates = replace(
            coordinates, mode=smooth_modes,
        )
        epsilon_p = artifact.meta.get("bridge_passive_epsilon")
        if epsilon_p is None:
            contact = states[:, 18] > 0 if schema_name == "ogbench_scene" and states.shape[1] > 18 else None
            epsilon_p = estimate_passive_tolerance(
                schema, coordinates.q_p, episode, coordinates.mode, coordinates.q_a,
                contact_mask=contact,
            )["epsilon"]
        servo_policy = ServoTrackingPolicy(
            artifact, trans, schema, coordinates,
            mode=args.controller, window=args.window,
            epsilon_p=np.asarray(epsilon_p), feedback_gain=args.feedback_gain,
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
            if servo_policy is None:
                assert policy is not None
                policy.reset(start)
                path = policy.path
            else:
                servo_policy.reset(start)
                path = servo_policy.path
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
            status = policy.status() if policy is not None else servo_policy.status()
            active_bridges: list[dict[str, int]] = []
            path_bridge_samples: list[float] = []
            processed_servo_landings: set[int] = set()
            while steps < max_steps:
                try:
                    if servo_policy is None:
                        current_aux = verifier.current_state()
                        action = policy.act(obs, current_aux)  # type: ignore[union-attr]
                    else:
                        action = servo_policy.act(verifier.sim_readout(obs))
                except SchemaError as exc:
                    failure_reason = str(exc)
                    break
                obs, reward, terminated, truncated, info = verifier.step(action)
                steps += 1
                success = bool(info.get("success", False)) or reward > 0
                status = policy.status() if policy is not None else servo_policy.status()
                if servo_policy is not None:
                    failure_reason = status["failure_reason"]
                    for bridge_index, record in enumerate(status["bridge_landing"]):
                        if bridge_index in processed_servo_landings:
                            continue
                        if record.get("landed") and record.get("landing_ratio") is not None:
                            bridge_ratios.append(float(record["landing_ratio"]))
                            path_bridge_samples.append(float(record["landing_ratio"]))
                            processed_servo_landings.add(bridge_index)
                    if success or terminated or truncated or failure_reason:
                        if truncated and not success:
                            failure_reason = "timeout"
                        elif terminated and not success:
                            failure_reason = "timeout"
                        break
                    continue
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
            bridge_count = (
                int(status["bridge_use_count"])
                if servo_policy is None else len(status["bridge_landing"])
            )
            if servo_policy is None:
                bridge_targets_unreached += len(active_bridges)
            else:
                bridge_targets_unreached += sum(
                    not bool(record.get("landed")) for record in status["bridge_landing"]
                )
            if not success and failure_reason is None and steps >= max_steps:
                failure_reason = "timeout"
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
                "bridge_landing": status.get("bridge_landing", []),
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
    zero_bridge = by_bridge["0"]
    bridged_0_to_3 = [item for item in outcomes if 1 <= item["planned_bridges"] <= 3]
    bridged_rate = float(np.mean([item["success"] for item in bridged_0_to_3])) if bridged_0_to_3 else None
    zero_rate = float(zero_bridge["success_rate"]) if zero_bridge["n"] else None
    relative_retention = bridged_rate / zero_rate if bridged_rate is not None and zero_rate and zero_rate > 0 else None
    rank = None
    if len(successful) >= 2:
        statistic = spearmanr(
            [item["graph_cost_steps"] for item in successful],
            [item["actual_steps"] for item in successful],
        ).statistic
        rank = float(statistic) if np.isfinite(statistic) else None
    conservative_lhs = float(np.mean([
        np.exp(-c * item["actual_steps"]) if item["success"] else 0.0
        for item in outcomes
    ])) if outcomes else 0.0
    conservative_rhs = float(np.mean([
        np.exp(-c * item["graph_cost_steps"]) for item in outcomes
    ])) if outcomes else 0.0
    failure_reasons = {
        label: {
            "count": int(sum(item.get("failure_reason") == label for item in outcomes)),
            "frac": float(np.mean([item.get("failure_reason") == label for item in outcomes])) if outcomes else 0.0,
        }
        for label in ("active_tracking", "passive_drift", "mode_mismatch", "timeout")
    }
    known_failure_reasons = {"active_tracking", "passive_drift", "mode_mismatch", "timeout", None}
    failure_reasons["other"] = {
        "count": int(sum(item.get("failure_reason") not in known_failure_reasons for item in outcomes)),
        "frac": float(np.mean([
            item.get("failure_reason") not in known_failure_reasons for item in outcomes
        ])) if outcomes else 0.0,
    }
    landing_records = [record for item in outcomes for record in item.get("bridge_landing", [])]
    bridge_landing_by_type = {}
    for label, code in (("R", 0), ("M", 1), ("legacy", -1)):
        selected = [record for record in landing_records if int(record.get("bridge_type", -1)) == code]
        landed = [record for record in selected if record.get("landed")]
        passive_groups = {}
        for group, predicate in (
            ("le_1", lambda x: float(x.get("passive_diff_epsilon", 0.0)) <= 1.0),
            ("1_to_3", lambda x: 1.0 < float(x.get("passive_diff_epsilon", 0.0)) <= 3.0),
            ("gt_3", lambda x: float(x.get("passive_diff_epsilon", 0.0)) > 3.0),
        ):
            members = [record for record in selected if predicate(record)]
            n_landed = sum(bool(record.get("landed")) for record in members)
            passive_groups[group] = {
                "attempted": len(members), "landed": int(n_landed),
                "landing_rate": float(n_landed / len(members)) if members else None,
            }
        tau_groups = {}
        for group, predicate in (
            ("le_5", lambda x: int(x.get("tau", 0)) <= 5),
            ("6_to_20", lambda x: 6 <= int(x.get("tau", 0)) <= 20),
            ("gt_20", lambda x: int(x.get("tau", 0)) > 20),
        ):
            members = [record for record in selected if predicate(record)]
            n_landed = sum(bool(record.get("landed")) for record in members)
            tau_groups[group] = {
                "attempted": len(members), "landed": int(n_landed),
                "landing_rate": float(n_landed / len(members)) if members else None,
            }
        bridge_landing_by_type[label] = {
            "attempted": len(selected),
            "landed": len(landed),
            "landing_rate": float(len(landed) / len(selected)) if selected else None,
            "passive_diff_epsilon": _quantiles([float(x.get("passive_diff_epsilon", 0.0)) for x in selected]),
            "tau": _quantiles([float(x.get("tau", 0)) for x in selected]),
            "by_passive_diff": {
                **passive_groups,
            },
            "by_tau": {
                **tau_groups,
            },
        }
    graph_targets = graph_diagnostics.get("targets")
    cap_fraction = graph_targets.get("valid_non_success_cap_frac") if isinstance(graph_targets, dict) else None
    graph_gate_reasons = []
    if cap_fraction is None:
        graph_gate_reasons.append("target cap gate unavailable; pass --targets")
    elif float(cap_fraction) > 0.5:
        graph_gate_reasons.append(f"valid non-success cap fraction {float(cap_fraction):.4f} exceeds 0.5")
    if graph_diagnostics["reachable_frac"] < 0.5:
        graph_gate_reasons.append("reachable fraction is below 0.5")
    if graph_diagnostics["reachable_frac_failed_eps"] < 0.3:
        graph_gate_reasons.append("failed-episode reachable fraction is below 0.3")
    kin_violation = graph_diagnostics["bridge_calibration"].get("kinematic_violation")
    has_m_bridges = bool(np.any(np.asarray(artifact.edges.bridge_type) == 1))
    if has_m_bridges:
        if kin_violation is None or float(kin_violation) > 0.05:
            graph_gate_reasons.append("M-bridge kinematic violation exceeds 0.05 or is unavailable")
    sim_gate_reasons = []
    if zero_rate is None or zero_rate < 0.8:
        sim_gate_reasons.append("zero_bridge_success is below 0.8 or has no samples")
    if relative_retention is None or relative_retention < 0.5:
        sim_gate_reasons.append("bridge retention for 1-3 bridges is below 0.5 or unavailable")
    if rank is None or rank < 0.6:
        sim_gate_reasons.append("successful-start cost/time Spearman is below 0.6 or unavailable")
    v04_gates = {
        "graph": {"passed": not graph_gate_reasons, "reasons": graph_gate_reasons,
                  "valid_non_success_cap_fraction": cap_fraction,
                  "reachable_frac": graph_diagnostics["reachable_frac"],
                  "reachable_frac_failed_eps": graph_diagnostics["reachable_frac_failed_eps"],
                  "kinematic_violation": kin_violation},
        "sim": {"passed": not sim_gate_reasons, "reasons": sim_gate_reasons,
                "zero_bridge_success": zero_rate,
                "bridge_success_1_to_3": bridged_rate,
                "bridge_retention_1_to_3": relative_retention,
                "success_cost_time_spearman": rank},
        "passed": not graph_gate_reasons and not sim_gate_reasons,
    } if servo_policy is not None else None
    if servo_policy is not None:
        bridge_targets_reached = sum(
            bool(record.get("landed"))
            for outcome in outcomes for record in outcome.get("bridge_landing", [])
        )
    result = {
        "dataset_id": dataset_id,
        "graph_hash": artifact.content_hash,
        "tau_model": artifact.config.bridge.cost.tau_model,
        "n_starts": len(outcomes),
        "pi_g_success": float(np.mean([item["success"] for item in outcomes])) if outcomes else 0.0,
        "path_success_by_bridges": by_bridge,
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
        "controller": args.controller,
        "zero_bridge_success": zero_rate,
        "zero_bridge_starts": int(zero_bridge["n"]),
        "bridge_success_1_to_3": bridged_rate,
        "bridge_retention_1_to_3": relative_retention,
        "success_cost_time_spearman": rank,
        "aggregate_conservative_bound": {
            "mean_gamma_T_success": conservative_lhs,
            "mean_exp_minus_C_G": conservative_rhs,
            "satisfied": conservative_lhs >= conservative_rhs,
        },
        "failure_reasons": failure_reasons,
        "bridge_landing_by_type": bridge_landing_by_type,
        "graph_diagnostics": graph_diagnostics,
        "v04_gates": v04_gates,
        "outcomes": outcomes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "outcomes"}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
