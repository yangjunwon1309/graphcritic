"""Inspect state-dimension dynamics and recommend slow features for graph distance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphcritic.config import FeatureConfig
from graphcritic.featurize import StateFeaturizer, state_table
from graphcritic.modes import resolve_state_modes, state_episode_time
from graphcritic.types import SimStateSet, SchemaError
from graphcritic.io import load_sim_state, load_transitions


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--transitions", type=Path)
    source.add_argument("--ogbench-dataset")
    source.add_argument("--robomimic-env")
    parser.add_argument("--sim-state", type=Path)
    parser.add_argument("--mode-fn", default="none")
    parser.add_argument(
        "--feature", choices=("identity", "ogbench_scene_pos", "robomimic_pos"),
        default="identity",
    )
    parser.add_argument("--min-dwell", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.tau < 0 or not np.isfinite(args.tau):
        raise ValueError("--tau must be finite and non-negative")
    if args.transitions:
        trans, _ = load_transitions(args.transitions)
        sim_state = load_sim_state(args.sim_state) if args.sim_state else None
    elif args.ogbench_dataset:
        from envs.ogbench import load_ogbench_transitions
        trans, _, sim_state = load_ogbench_transitions(args.ogbench_dataset, return_sim_state=True)
    else:
        from envs.robomimic import load_robomimic_transitions
        trans, _ = load_robomimic_transitions(args.robomimic_env)
        sim_state = None

    x, _, success, _ = state_table(trans)
    feature_cfg = FeatureConfig(name=args.feature, standardize=True)
    featurizer = StateFeaturizer.fit(trans, feature_cfg, sim_state=sim_state)
    aux = None
    if sim_state is not None:
        aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
        if sim_state.button_states is not None:
            aux["button_states"] = sim_state.button_states
    phi = featurizer.transform(x, aux=aux)
    valid_rows = np.flatnonzero(~trans.terminal)
    modes = resolve_state_modes(
        trans, args.mode_fn, sim_state=sim_state, min_dwell=args.min_dwell
    )
    same_mode = modes[valid_rows] == modes[valid_rows + 1]
    valid_rows = valid_rows[same_mode]
    delta = np.abs(phi[valid_rows + 1].astype(np.float64) - phi[valid_rows])
    std = np.std(phi.astype(np.float64), axis=0)
    rho = np.median(delta, axis=0) / np.maximum(std, 1e-12)
    d_step_values = np.linalg.norm(delta, axis=1)
    d_step_values = d_step_values[np.isfinite(d_step_values) & (d_step_values > 1e-8)]
    d_step = float(np.median(d_step_values)) if d_step_values.size else 1.0
    suggested = np.flatnonzero(rho <= args.tau).astype(int).tolist()

    episodes, times = state_episode_time(trans)
    segments = 0
    for ep in np.unique(episodes):
        ids = np.flatnonzero(episodes == ep)
        ids = ids[np.argsort(times[ids], kind="stable")]
        if ids.size:
            segments += 1 + int(np.count_nonzero(modes[ids[1:]] != modes[ids[:-1]]))
    goal_modes, goal_counts = np.unique(modes[success], return_counts=True)
    report = {
        "n_rows": trans.n_rows,
        "n_states": int(x.shape[0]),
        "raw_observation_dim": int(x.shape[1]),
        "feature": featurizer.meta(),
        "feature_dim": int(phi.shape[1]),
        "d_step": d_step,
        "tau": float(args.tau),
        "rho": [float(v) for v in rho],
        "suggested_dims": suggested,
        "mode_fn": args.mode_fn,
        "mode_count": int(np.unique(modes).size),
        "segment_count": int(segments),
        "goal_mode_distribution": {
            str(int(mode)): int(count) for mode, count in zip(goal_modes, goal_counts, strict=True)
        },
    }
    if args.feature == "ogbench_scene_pos":
        if sim_state is None:
            raise SchemaError("ogbench_scene_pos inspection requires a simulator sidecar")
        from envs.ogbench_layout import verify_scene_position_layout
        check_rows = np.arange(min(1000, trans.n_rows), dtype=np.int64)
        report["layout_check"] = verify_scene_position_layout(
            trans.feat, sim_state.qpos[: trans.n_rows], rows=check_rows
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("feature_dim\tstd\trho\tselect")
    for dim, (scale, value) in enumerate(zip(std, rho, strict=True)):
        print(f"{dim}\t{scale:.6g}\t{value:.6g}\t{'keep' if value <= args.tau else 'drop'}")
    print(json.dumps({key: value for key, value in report.items() if key != "rho"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
