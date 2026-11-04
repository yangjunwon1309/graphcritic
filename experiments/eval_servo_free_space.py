"""Check v0.4 pose-servo tracking on object-free OGBench Scene states."""

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
from graphcritic.modes import state_episode_time
from graphcritic.state_schema import get_state_schema, wrap_angle
from graphcritic.types import SchemaError
from graphcritic.verification import OGBenchVerifier


def _epsilon_a(q_a: np.ndarray, trans, schema) -> np.ndarray:
    rows = np.flatnonzero(~trans.terminal)
    rows = rows[trans.episode_id[rows] == trans.episode_id[rows + 1]]
    delta = q_a[rows + 1] - q_a[rows]
    for dim in getattr(schema, "periodic_action", ()):
        delta[:, dim] = wrap_angle(delta[:, dim])
    epsilon = np.empty(q_a.shape[1], dtype=np.float64)
    for dim in range(q_a.shape[1]):
        positive = np.abs(delta[:, dim])
        positive = positive[positive > 1e-7]
        epsilon[dim] = np.median(positive) if positive.size else schema.action_limits[dim]
    return epsilon


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
    args = parser.parse_args()
    if args.trials < 1 or not np.isfinite(args.gain) or args.gain <= 0:
        parser.error("--trials must be positive and --gain must be finite and positive")

    trans, _ = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    states = state_table(trans)[0]
    sim_state.validate(states.shape[0])
    schema = get_state_schema("ogbench_scene")
    aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
    if sim_state.button_states is not None:
        aux["button_states"] = sim_state.button_states
    coords = schema.coordinates(states, aux)
    epsilon_a = _epsilon_a(coords.q_a, trans, schema)

    # Stay at least 15 cm from cubes initially. The maximum target offset is
    # 4 cm, keeping these short tests outside the contact region.
    n_cubes = (coords.q_p.shape[1] - 2) // 4
    cube_xyz = coords.q_p[:, : 4 * n_cubes].reshape(states.shape[0], n_cubes, 4)[:, :, :3]
    distance = np.linalg.norm(coords.q_a[:, None, :3] - cube_xyz, axis=2).min(axis=1)
    free = (distance >= 0.15) & (states[:, 18] <= 0.0)
    candidates = np.flatnonzero(free)
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
    outcomes = []
    try:
        for start in starts:
            current_r = np.asarray(coords.r[start], dtype=np.float64)
            current_q = np.asarray(coords.q_a[start], dtype=np.float64)
            target = current_q.copy()
            target[:3] += rng.uniform(-0.04, 0.04, size=3)
            target[3] += rng.uniform(-0.24, 0.24)
            target[3] = float(wrap_angle(np.asarray(target[3])))
            target[4] = np.clip(target[4] + rng.uniform(-0.6, 0.6), 0.0, 1.0)
            bounds = getattr(env, "_workspace_bounds", None)
            if bounds is not None:
                target[:3] = np.clip(target[:3], np.asarray(bounds)[0], np.asarray(bounds)[1])
            target_r = current_r.copy()
            target_r[14:19] = target
            tau_kin = int(schema.kinematic_steps(current_r, target_r))
            budget = tau_kin + 2

            payload = {"qpos": sim_state.qpos[start], "qvel": sim_state.qvel[start]}
            if sim_state.button_states is not None:
                payload["button_states"] = sim_state.button_states[start]
            obs = verifier.reset_to(payload)
            for _ in range(budget):
                readout = verifier.sim_readout(obs)
                delta = target - np.asarray(readout["q_a"], dtype=np.float64)
                delta[3] = float(wrap_angle(np.asarray(delta[3])))
                action = np.clip(args.gain * delta / limits, -1.0, 1.0).astype(np.float32)
                obs, _, terminated, truncated, _ = verifier.step(action)
                if terminated or truncated:
                    break
            actual = np.asarray(verifier.sim_readout(obs)["q_a"], dtype=np.float64)
            error = target - actual
            error[3] = float(wrap_angle(np.asarray(error[3])))
            normalized_error = float(np.max(np.abs(error) / epsilon_a))
            outcomes.append({
                "start_state": int(start), "tau_kin": tau_kin,
                "budget": budget, "normalized_error": normalized_error,
                "passed": bool(normalized_error <= 1.0),
            })
            if len(outcomes) % 25 == 0:
                print(f"free-space servo: {len(outcomes)}/{args.trials}", file=sys.stderr, flush=True)
    finally:
        wrapped.close()

    result = {
        "n_trials": len(outcomes),
        "success_fraction": float(np.mean([item["passed"] for item in outcomes])),
        "gate": 0.95,
        "passed_gate": bool(np.mean([item["passed"] for item in outcomes]) >= 0.95),
        "epsilon_a": epsilon_a.tolist(),
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
