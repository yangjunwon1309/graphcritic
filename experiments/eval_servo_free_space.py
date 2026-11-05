"""Check calibrated pose tracking on free-space OGBench Scene or Puzzle states."""

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
from graphcritic.io import load_sim_state, load_transitions
from graphcritic.physical_time import estimate_realized_speed, free_space_mask, realized_travel_steps
from graphcritic.state_schema import (
    OGBenchPuzzleStateSchema,
    estimate_puzzle_press_height,
    get_state_schema,
    wrap_angle,
)
from graphcritic.types import SchemaError
from graphcritic.verification import OGBenchVerifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gain", type=float, default=1.0,
                        help="multiply the clipped pose correction for controller sensitivity checks")
    parser.add_argument("--speed-quantile", choices=("0.95", "0.75"), default="0.95",
                        help="realized speed quantile used for tau_real")
    args = parser.parse_args()
    if args.trials < 1 or not np.isfinite(args.gain) or args.gain <= 0:
        parser.error("--trials must be positive and --gain must be finite and positive")

    trans, _ = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    states = state_table(trans)[0]
    sim_state.validate(states.shape[0])
    aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
    if sim_state.button_states is not None:
        aux["button_states"] = sim_state.button_states
    if sim_state.button_states is not None and sim_state.button_states.shape[1] in {9, 16, 20}:
        final_rows = np.flatnonzero(trans.terminal)
        next_ids = np.empty(trans.n_rows, dtype=np.int64)
        nonterminal = np.flatnonzero(~trans.terminal)
        next_ids[nonterminal] = nonterminal + 1
        next_ids[final_rows] = trans.n_rows + np.arange(final_rows.size, dtype=np.int64)
        press_fit = estimate_puzzle_press_height(
            trans.feat, sim_state.button_states[: trans.n_rows],
            next_observations=trans.next_feat,
            next_button_states=sim_state.button_states[next_ids],
        )
        schema = OGBenchPuzzleStateSchema(press_height=float(press_fit["press_height_m"]))
        coords = schema.coordinates(states, aux)
        free = coords.mode == 0
        candidates_free = free
        schema_name = "ogbench_puzzle"
    else:
        press_fit = None
        schema = get_state_schema("ogbench_scene")
        coords = schema.coordinates(states, aux)
        free = free_space_mask(states, sim_state.qpos, coords, distance=0.05)
        candidates_free = free_space_mask(states, sim_state.qpos, coords, distance=0.15)
        schema_name = "ogbench_scene"
    calibration = estimate_realized_speed(trans, coords, free, schema)
    speed = np.asarray(calibration["speed_quantiles"][args.speed_quantile], dtype=np.float64)
    response_gain = np.asarray(calibration["feedback_gain_by_dimension"], dtype=np.float64)
    candidates = np.flatnonzero(candidates_free)
    if candidates.size < args.trials:
        raise SchemaError(f"only {candidates.size} object-free states are available for {args.trials} trials")

    try:
        import gymnasium as gym
        import ogbench  # noqa: F401
    except ImportError as exc:  # pragma: no cover - bench-only
        raise RuntimeError("free-space servo evaluation requires the bench environment") from exc

    rng = np.random.default_rng(args.seed)
    starts = rng.choice(candidates, size=args.trials, replace=False)
    wrapped = gym.make(args.env, max_episode_steps=1_000_000_000)
    env = wrapped.unwrapped
    verifier = OGBenchVerifier(env)
    limits = np.asarray(schema.action_limits, dtype=np.float64)
    tracking_tube = np.asarray([0.01, 0.01, 0.01, np.deg2rad(5.0), 0.1], dtype=np.float64)
    outcomes = []
    try:
        for start in starts:
            current_q = np.asarray(coords.q_a[start], dtype=np.float64)
            target = current_q.copy()
            target[:3] += rng.uniform(-0.04, 0.04, size=3)
            target[3] += rng.uniform(-0.24, 0.24)
            target[3] = float(wrap_angle(np.asarray(target[3])))
            bounds = getattr(env, "_workspace_bounds", None)
            if bounds is not None:
                target[:3] = np.clip(target[:3], np.asarray(bounds)[0], np.asarray(bounds)[1])
            tau_real = int(realized_travel_steps(
                current_q, target, speed, settle=2,
                periodic=getattr(schema, "periodic_action", ()),
            ))
            budget = tau_real + 2

            payload = {"qpos": sim_state.qpos[start], "qvel": sim_state.qvel[start]}
            if sim_state.button_states is not None:
                payload["button_states"] = sim_state.button_states[start]
            obs = verifier.reset_to(payload)
            passive_drift = False
            for _ in range(budget):
                readout = verifier.sim_readout(
                    obs, schema_name=schema_name,
                    press_height=None if press_fit is None else float(press_fit["press_height_m"]),
                )
                delta = target - np.asarray(readout["q_a"], dtype=np.float64)
                delta[3] = float(wrap_angle(np.asarray(delta[3])))
                action = np.zeros_like(limits, dtype=np.float64)
                action[:4] = np.clip(
                    args.gain * response_gain[:4] * delta[:4] / limits[:4], -1.0, 1.0
                )
                action[4] = 0.0
                action = action.astype(np.float32)
                obs, _, terminated, truncated, _ = verifier.step(action)
                if schema_name == "ogbench_puzzle":
                    current_buttons = np.asarray(verifier.current_state()["button_states"], dtype=np.int64)
                    if not np.array_equal(current_buttons, sim_state.button_states[start]):
                        passive_drift = True
                        break
                if terminated or truncated:
                    break
            actual = np.asarray(verifier.sim_readout(
                obs, schema_name=schema_name,
                press_height=None if press_fit is None else float(press_fit["press_height_m"]),
            )["q_a"], dtype=np.float64)
            error = target - actual
            error[3] = float(wrap_angle(np.asarray(error[3])))
            normalized_error = float(np.max(np.abs(error) / tracking_tube))
            outcomes.append({
                "start_state": int(start), "tau_real": tau_real,
                "budget": budget, "normalized_error": normalized_error,
                "passive_drift": passive_drift,
                "passed": bool(normalized_error <= 1.0 and not passive_drift),
            })
            if len(outcomes) % 25 == 0:
                print(f"free-space servo: {len(outcomes)}/{args.trials}", file=sys.stderr, flush=True)
    finally:
        wrapped.close()

    result = {
        "state_schema": schema_name,
        "puzzle_press_height_calibration": press_fit,
        "n_trials": len(outcomes),
        "success_fraction": float(np.mean([item["passed"] for item in outcomes])),
        "passive_drift_count": int(sum(item["passive_drift"] for item in outcomes)),
        "gate": 0.9,
        "passed_gate": bool(np.mean([item["passed"] for item in outcomes]) >= 0.9),
        "tracking_tube": tracking_tube.tolist(),
        "realized_speed": calibration["speed_quantiles"][args.speed_quantile],
        "speed_quantile": float(args.speed_quantile),
        "response_gain": response_gain.tolist(),
        "calibration": calibration,
        "gain": float(args.gain),
        "seed": int(args.seed),
        "outcomes": outcomes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "outcomes"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
