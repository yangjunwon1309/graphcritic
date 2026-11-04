"""Check deterministic resets and feature-space replay for OGBench data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from envs.ogbench import _dataset_parts, load_ogbench_transitions
from envs.ogbench_layout import quaternion_yaw_wxyz
from graphcritic.config import FeatureConfig
from graphcritic.featurize import StateFeaturizer, state_table
from graphcritic.modes import resolve_state_modes
from graphcritic.verification import OGBenchVerifier


def _state_payload(sim_state, state_id: int) -> dict[str, np.ndarray]:
    result = {"qpos": sim_state.qpos[state_id], "qvel": sim_state.qvel[state_id]}
    if sim_state.button_states is not None:
        result["button_states"] = sim_state.button_states[state_id]
    return result


def _candidate_starts(trans, horizon: int) -> np.ndarray:
    eligible = []
    for row in np.flatnonzero(~trans.terminal):
        end = int(row) + horizon
        if end >= trans.n_rows:
            continue
        if np.any(trans.terminal[row : end + 1]):
            continue
        if trans.episode_id[row] != trans.episode_id[end]:
            continue
        eligible.append(int(row))
    return np.asarray(eligible, dtype=np.int64)


def _sequence(verifier, sim_state, trans, start: int, horizon: int) -> np.ndarray:
    verifier.reset_to(_state_payload(sim_state, start))
    result = []
    for row in range(start, start + horizon):
        obs, _, terminated, truncated, _ = verifier.step(trans.action[row])
        result.append(np.asarray(obs).copy())
        if terminated or truncated:
            break
    return np.asarray(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("~/.ogbench/data"))
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--determinism-rows", type=int, default=100)
    parser.add_argument("--determinism-horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature", default="ogbench_scene_pos")
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--yaw-row", type=int, default=525285)
    parser.add_argument("--out", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.rows < 1 or args.determinism_rows < 1 or args.determinism_horizon < 1:
        parser.error("rows and determinism sample sizes must be positive")

    trans, dataset_id, sim_state = load_ogbench_transitions(
        args.dataset, dataset_dir=args.dataset_dir, dataset_path=args.dataset_path,
        split=args.split, return_sim_state=True,
    )
    states, _, _, _ = state_table(trans)
    aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
    if sim_state.button_states is not None:
        aux["button_states"] = sim_state.button_states
    cfg = FeatureConfig(name=args.feature, standardize=True)
    featurizer = StateFeaturizer.fit(trans, cfg, sim_state=sim_state)
    phi = featurizer.transform(states, aux=aux)
    modes = resolve_state_modes(trans, "ogbench_scene", sim_state=sim_state, min_dwell=3)
    step_rows = np.flatnonzero(~trans.terminal)
    step_rows = step_rows[modes[step_rows] == modes[step_rows + 1]]
    step_distance = np.linalg.norm(phi[step_rows + 1] - phi[step_rows], axis=1)
    step_distance = step_distance[np.isfinite(step_distance) & (step_distance > 1e-8)]
    d_step = float(np.median(step_distance)) if step_distance.size else 1.0

    import ogbench  # noqa: F401
    import gymnasium as gym

    wrapped_env = gym.make(_dataset_parts(args.dataset)[0])
    # Run the raw OGBench environment directly, outside Gymnasium's TimeLimit.
    verifier = OGBenchVerifier(wrapped_env.unwrapped)
    rng = np.random.default_rng(args.seed)
    try:
        eligible = np.flatnonzero(~trans.terminal)
        rows = rng.choice(eligible, size=min(args.rows, eligible.size), replace=False)
        feature_errors = np.empty(rows.size, dtype=np.float64)
        raw_errors = np.empty(rows.size, dtype=np.float64)
        for index, row_value in enumerate(rows):
            row = int(row_value)
            result = verifier.rollout(_state_payload(sim_state, row), trans.action[row : row + 1])
            replay_obs = np.asarray(result["observation"], dtype=np.float32)[None, :]
            replay_aux = {
                key: np.asarray(result[key]).reshape(1, -1)
                for key in ("qpos", "qvel", "button_states") if key in result
            }
            next_state_id = row + 1
            replay_phi = featurizer.transform(replay_obs, aux=replay_aux)[0]
            data_phi = featurizer.transform(
                trans.next_feat[row : row + 1],
                aux={key: np.asarray(value[next_state_id : next_state_id + 1]) for key, value in aux.items()},
            )[0]
            feature_errors[index] = np.linalg.norm(replay_phi - data_phi)
            raw_errors[index] = np.max(np.abs(replay_obs[0].astype(np.float64) - trans.next_feat[row]))

        deterministic_starts = _candidate_starts(trans, args.determinism_horizon)
        starts = rng.choice(
            deterministic_starts,
            size=min(args.determinism_rows, deterministic_starts.size),
            replace=False,
        )
        deterministic = []
        deterministic_mismatches = []
        for start in starts:
            first = _sequence(verifier, sim_state, trans, int(start), args.determinism_horizon)
            second = _sequence(verifier, sim_state, trans, int(start), args.determinism_horizon)
            equal = bool(np.array_equal(first, second))
            deterministic.append(equal)
            if not equal and len(deterministic_mismatches) < 20:
                common = min(len(first), len(second))
                delta = np.abs(first[:common].astype(np.float64) - second[:common].astype(np.float64))
                mismatch_rows = np.flatnonzero(np.any(first[:common] != second[:common], axis=1))
                deterministic_mismatches.append({
                    "start_row": int(start),
                    "first_length": int(len(first)),
                    "second_length": int(len(second)),
                    "first_mismatch_step": int(mismatch_rows[0]) if mismatch_rows.size else None,
                    "max_abs_error_common_steps": float(np.max(delta)) if delta.size else None,
                })

        yaw_report = None
        if 0 <= args.yaw_row < trans.n_rows - 1 and not trans.terminal[args.yaw_row]:
            row = int(args.yaw_row)
            result = verifier.rollout(_state_payload(sim_state, row), trans.action[row : row + 1])
            replay_qpos = np.asarray(result["qpos"], dtype=np.float64)
            data_next_qpos = np.asarray(sim_state.qpos[row + 1], dtype=np.float64)
            qpos_cube = 14 + 3
            replay_yaw = float(quaternion_yaw_wxyz(replay_qpos[qpos_cube : qpos_cube + 4]))
            data_yaw = float(quaternion_yaw_wxyz(data_next_qpos[qpos_cube : qpos_cube + 4]))
            wrapped_yaw_error = float(np.abs(np.arctan2(
                np.sin(replay_yaw - data_yaw), np.cos(replay_yaw - data_yaw)
            )))
            expected_obs_yaw = np.asarray(trans.next_feat[row, 26:28], dtype=np.float64)
            replay_obs_yaw = np.asarray(result["observation"][26:28], dtype=np.float64)
            computed_data_yaw = np.asarray([np.cos(data_yaw), np.sin(data_yaw)])
            observation_yaw_error = float(np.max(np.abs(expected_obs_yaw - replay_obs_yaw)))
            yaw_report = {
                "row": row,
                "replay_vs_data_qpos_yaw_abs_rad": wrapped_yaw_error,
                "replay_vs_data_observation_yaw_max_abs": observation_yaw_error,
                "data_observation_vs_qpos_yaw_max_abs": float(np.max(np.abs(expected_obs_yaw - computed_data_yaw))),
                "data_observation_yaw_cos_sin": expected_obs_yaw.tolist(),
                "replay_observation_yaw_cos_sin": replay_obs_yaw.tolist(),
                "data_qpos_yaw_cos_sin": computed_data_yaw.tolist(),
                "yaw_discontinuity_hypothesis_supported": bool(
                    wrapped_yaw_error < 0.01
                    and observation_yaw_error > 0.1
                ),
            }
    finally:
        wrapped_env.close()

    q99 = float(np.quantile(feature_errors, 0.99)) if feature_errors.size else 0.0
    maximum = float(np.max(feature_errors)) if feature_errors.size else 0.0
    report = {
        "dataset": args.dataset,
        "dataset_id": dataset_id,
        "split": args.split,
        "feature": featurizer.meta(),
        "n_rows_checked": int(rows.size),
        "d_step": d_step,
        "feature_replay_error": {
            "q99": q99,
            "max": maximum,
            "q99_over_d_step": q99 / d_step,
            "max_over_d_step": maximum / d_step,
            "q99_limit": 0.1,
            "max_limit": 0.5,
            "passed": bool(q99 <= 0.1 * d_step and maximum <= 0.5 * d_step),
        },
        "raw_observation_error_reference_only": {
            "q99": float(np.quantile(raw_errors, 0.99)) if raw_errors.size else 0.0,
            "max": float(np.max(raw_errors)) if raw_errors.size else 0.0,
        },
        "determinism": {
            "n_start_states": int(len(deterministic)),
            "horizon": int(args.determinism_horizon),
            "bitwise_equal_fraction": float(np.mean(deterministic)) if deterministic else 0.0,
            "passed": bool(all(deterministic)),
            "mismatches_first_20": deterministic_mismatches,
        },
        "yaw_hypothesis": yaw_report,
    }
    serialized = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if report["feature_replay_error"]["passed"] and report["determinism"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
