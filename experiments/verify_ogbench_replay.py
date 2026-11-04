"""Check one-step OGBench transition replay from the saved MuJoCo sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from envs.ogbench import _dataset_parts, load_ogbench_transitions
from graphcritic.verification import OGBenchVerifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("~/.ogbench/data"))
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    if args.rows < 1 or not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("rows and tolerance must be positive")

    trans, _, sim_state = load_ogbench_transitions(
        args.dataset, dataset_dir=args.dataset_dir, dataset_path=args.dataset_path,
        split=args.split, return_sim_state=True,
    )
    import ogbench  # noqa: F401
    import gymnasium as gym

    env = gym.make(_dataset_parts(args.dataset)[0])
    verifier = OGBenchVerifier(env)
    try:
        eligible = np.flatnonzero(~trans.terminal)
        rng = np.random.default_rng(args.seed)
        rows = rng.choice(eligible, size=min(args.rows, eligible.size), replace=False)
        errors = np.empty(rows.size, dtype=np.float64)
        errors_by_dim = np.zeros(trans.feat.shape[1], dtype=np.float64)
        failures = []
        for index, row in enumerate(rows):
            result = verifier.rollout(
                {"qpos": sim_state.qpos[row], "qvel": sim_state.qvel[row],
                 **({"button_states": sim_state.button_states[row]} if sim_state.button_states is not None else {})},
                trans.action[row : row + 1],
            )
            actual = np.asarray(result["observation"], dtype=np.float64)
            difference = np.abs(actual - trans.next_feat[row])
            errors[index] = np.max(difference)
            errors_by_dim = np.maximum(errors_by_dim, difference)
            if errors[index] > args.tolerance:
                failures.append({
                    "row": int(row),
                    "max_abs_error": float(errors[index]),
                    "dimensions_over_tolerance": np.flatnonzero(
                        difference > args.tolerance
                    ).astype(int).tolist(),
                })
    finally:
        env.close()
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "n_checked": int(rows.size),
        "tolerance": float(args.tolerance),
        "max_abs_error": float(np.max(errors)) if errors.size else 0.0,
        "mean_abs_max_error": float(np.mean(errors)) if errors.size else 0.0,
        "max_abs_error_by_dimension": errors_by_dim.tolist(),
        "failed_rows": failures,
        "passed": bool(np.all(errors <= args.tolerance)),
    }
    print(summary)
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
