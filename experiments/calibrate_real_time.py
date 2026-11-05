"""Estimate free-space actuator response and tau_real from saved transitions."""

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
from graphcritic.modes import smooth_mode_changes, state_episode_time
from graphcritic.physical_time import (
    estimate_realized_speed,
    estimate_static_articulation_tolerance,
    free_space_mask,
    realized_time_validation,
)
from graphcritic.state_schema import (
    OGBenchPuzzleStateSchema,
    estimate_puzzle_press_height,
    get_state_schema,
)
from graphcritic.types import SchemaError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--distance", type=float, default=0.05)
    parser.add_argument("--settle", type=int, default=2)
    parser.add_argument("--pairs", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schema", choices=("ogbench_scene", "ogbench_puzzle"), default="ogbench_scene")
    args = parser.parse_args()
    if args.distance <= 0 or args.settle < 0 or args.pairs < 1:
        parser.error("distance/pairs must be positive and settle non-negative")

    trans, transition_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    states, _, success, _ = state_table(trans)
    sim_state.validate(states.shape[0])
    aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
    if sim_state.button_states is not None:
        aux["button_states"] = sim_state.button_states
    episode, times = state_episode_time(trans)
    if args.schema == "ogbench_puzzle":
        if sim_state.button_states is None:
            raise SchemaError("puzzle calibration requires the button_states sidecar")
        final_rows = np.flatnonzero(trans.terminal)
        next_state_ids = np.empty(trans.n_rows, dtype=np.int64)
        nonterminal_rows = np.flatnonzero(~trans.terminal)
        next_state_ids[nonterminal_rows] = nonterminal_rows + 1
        next_state_ids[final_rows] = trans.n_rows + np.arange(final_rows.size, dtype=np.int64)
        press_fit = estimate_puzzle_press_height(
            trans.feat, sim_state.button_states[: trans.n_rows],
            next_observations=trans.next_feat,
            next_button_states=sim_state.button_states[next_state_ids],
        )
        schema = OGBenchPuzzleStateSchema(press_height=float(press_fit["press_height_m"]))
        coordinates = schema.coordinates(states, aux)
        free = coordinates.mode == 0
        static = None
        grasped = np.zeros(states.shape[0], dtype=np.bool_)
    else:
        press_fit = None
        schema = get_state_schema("ogbench_scene")
        coordinates = schema.coordinates(states, aux)
        free = free_space_mask(states, sim_state.qpos, coordinates, distance=args.distance)
        static = estimate_static_articulation_tolerance(
            coordinates.q_p, episode, times, free,
        )
        static["epsilon"] = np.asarray(static["epsilon"]).tolist()
        grasped = coordinates.mode // 64 > 0
    observed_contact = states[:, 18] > 0.0
    non_success_source = ~success
    r_source_free = non_success_source & free
    response = estimate_realized_speed(trans, coordinates, free, schema)
    speed95 = np.asarray(response["speed_quantiles"]["0.95"], dtype=np.float64)
    validation = realized_time_validation(
        trans, coordinates, free, speed95, schema,
        seed=args.seed, max_pairs=args.pairs, settle=args.settle,
    )
    result = {
        "state_schema": args.schema,
        "dataset_id": transition_meta["dataset_id"],
        "free_state_count": int(np.count_nonzero(free)),
        "free_state_fraction": float(np.mean(free)),
        "state_mode_counts": {
            "free": int(np.count_nonzero(free)),
            "contact_observation": int(np.count_nonzero(observed_contact)),
            "grasped_mode": int(np.count_nonzero(grasped)),
            "near_object_or_handle_or_contact": int(np.count_nonzero(~free)),
        },
        "r_strict_source_filter": {
            "non_success_source_states_before_free_filter": int(np.count_nonzero(non_success_source)),
            "eligible_free_source_states": int(np.count_nonzero(r_source_free)),
            "excluded_by_free_condition": int(np.count_nonzero(non_success_source & ~free)),
            "excluded_fraction": float(np.mean(~free[non_success_source])) if np.any(non_success_source) else None,
        },
        "distance_threshold_m": args.distance,
        "real_settle_steps": args.settle,
        "response": response,
        "same_episode_validation": validation,
        "puzzle_press_height_calibration": press_fit,
        "static_articulation_tolerance": static,
        "simulator_used": False,
        "td_checkpoint_used": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
