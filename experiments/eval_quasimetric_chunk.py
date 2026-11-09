"""Compare state and window quasimetric checkpoints on common chunk metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.train_quasimetric import heldout_metrics
from experiments.v08_runtime import (
    activation_diagnostics,
    load_quasimetric_checkpoint,
    make_activation_diagnostic_fn,
    make_distance_predictor,
    next_state_indices,
)
from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.bridges.windows import puzzle_state_windows
from graphcritic.io import load_sim_state, load_transitions


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, nargs="+", required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--triangles", type=int, default=100_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    trans, transition_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    data = prepare_puzzle_model_data(trans, sim_state, seed=args.seed)
    frame_states = np.concatenate((data.state_cont, data.state_bits), axis=1).astype(np.float32)
    next_index = next_state_indices(trans.terminal)
    reports = []
    for checkpoint in args.ckpt:
        started = time.perf_counter()
        raw_meta = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        window = int(raw_meta.get("window", 1))
        states = (
            puzzle_state_windows(frame_states, trans, data.terminal_state_index, window=window)
            if window > 1 else frame_states
        )
        model, params, metadata = load_quasimetric_checkpoint(
            checkpoint, input_dim=states.shape[1],
            expected_dataset_id=transition_meta["dataset_id"],
        )
        predictor = make_distance_predictor(model, params, batch_size=args.eval_batch_size)
        heldout = heldout_metrics(
            data, states, next_index, predictor,
            step=int(metadata.get("step", 0)), seed=args.seed,
            triangle_count=args.triangles, eval_batch_size=args.eval_batch_size,
            constraint_horizon=5, epsilon=1.25,
        )
        rng = np.random.default_rng(args.seed + 0x51A7)
        rows = rng.choice(
            data.heldout_rows, size=4096,
            replace=len(data.heldout_rows) < 4096,
        )
        diagnostics = activation_diagnostics(
            make_activation_diagnostic_fn(model), params, states[rows],
            activation=str(metadata.get("activation", "relu")),
        )
        reports.append({
            "checkpoint": str(checkpoint),
            "step": int(metadata.get("step", 0)),
            "window": window,
            "lambda": float(metadata.get("lambda_value", float("nan"))),
            "heldout_chunk_constraint": heldout["heldout_chunk_constraint"],
            "same_episode_distance_by_horizon": heldout["same_episode_distance_by_horizon"],
            "diagnostics": diagnostics,
            "triangle_check": heldout["triangle_check"],
            "elapsed_seconds": float(time.perf_counter() - started),
        })

    horizons = (5, 10, 15, 20, 40, 80)
    print(
        "| checkpoint | window | λ | chunk violation >6.25 | changed | unchanged | "
        + " | ".join(f"P(d≤{h})" for h in horizons)
        + " | projector var | triangle violations |"
    )
    print("|:---|---:|---:|---:|---:|---:|" + "---:|" * (len(horizons) + 2))
    for report in reports:
        constraint = report["heldout_chunk_constraint"]
        groups = constraint["button_change_groups"]
        changed = groups["changed"]["constraint_violation_rate_d_gt_limit"]
        unchanged = groups["unchanged"]["constraint_violation_rate_d_gt_limit"]
        probabilities = [
            report["same_episode_distance_by_horizon"][str(h)]["fraction_distance_leq_horizon"]
            for h in horizons
        ]
        print(
            f"| {Path(report['checkpoint']).name} | {report['window']} | {report['lambda']:.6g} | "
            f"{constraint['overall_violation_rate_d_gt_limit']:.4f} | {changed:.4f} | "
            f"{unchanged:.4f} | "
            + " | ".join("n/a" if x is None else f"{x:.4f}" for x in probabilities)
            + f" | {report['diagnostics']['projector_output_mean_variance']:.6g} | "
            f"{report['triangle_check']['violation_count']} |"
        )
    print("\nDetails:")
    print(json.dumps(reports, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
