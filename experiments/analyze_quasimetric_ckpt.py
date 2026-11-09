"""Inspect saved v0.8 quasimetric checkpoints without training or simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
from graphcritic.io import load_sim_state, load_transitions


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, nargs="+", required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--triangles", type=int, default=100_000)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--diagnostic-states", type=int, default=4096)
    return parser.parse_args(argv)


def analyze_checkpoints(args: argparse.Namespace) -> list[dict]:
    trans, transition_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    data = prepare_puzzle_model_data(trans, sim_state, seed=args.seed)
    states = np.concatenate((data.state_cont, data.state_bits), axis=1).astype(np.float32)
    next_index = next_state_indices(trans.terminal)

    rng = np.random.default_rng(args.seed + 0x51A7)
    diagnostic_rows = rng.choice(
        data.heldout_rows,
        size=args.diagnostic_states,
        replace=len(data.heldout_rows) < args.diagnostic_states,
    )
    diagnostic_states = states[diagnostic_rows]
    reports = []
    for checkpoint in args.ckpt:
        model, params, metadata = load_quasimetric_checkpoint(
            checkpoint,
            input_dim=states.shape[1],
            expected_dataset_id=transition_meta["dataset_id"],
        )
        activation = str(metadata.get("activation", "relu"))
        diagnostics = activation_diagnostics(
            make_activation_diagnostic_fn(model), params, diagnostic_states,
            activation=activation,
        )
        predictor = make_distance_predictor(
            model, params, batch_size=args.eval_batch_size
        )
        step = int(metadata.get("step", 0))
        heldout = heldout_metrics(
            data, states, next_index, predictor,
            step=step, seed=args.seed, triangle_count=args.triangles,
            eval_batch_size=args.eval_batch_size,
        )
        reports.append({
            "checkpoint": str(checkpoint),
            "step": step,
            "lambda": float(metadata.get("lambda_value", float("nan"))),
            "activation": activation,
            "layernorm": bool(metadata.get("layernorm", False)),
            "diagnostics": diagnostics,
            "heldout": heldout,
        })
    return reports


def _print_table(reports: list[dict]) -> None:
    print(
        "| step | λ | activation | dead ReLU/GELU units by layer | "
        "encoder var | projector var | heldout d>1.25 | button changed | "
        "button unchanged | P(d≤5), j=5 | triangle violations |"
    )
    print("|---:|---:|:---:|:---|---:|---:|---:|---:|---:|---:|---:|")
    for report in reports:
        heldout = report["heldout"]
        constraint = heldout["heldout_one_step_constraint"]
        groups = constraint["button_change_groups"]
        horizons = heldout["same_episode_distance_by_horizon"]
        h5 = horizons["5"]["fraction_distance_leq_horizon"]
        dead = ", ".join(
            f"{name}={value:.3f}"
            for name, value in report["diagnostics"]["dead_unit_frac_by_layer"].items()
        )
        print(
            f"| {report['step']} | {report['lambda']:.6g} | {report['activation']} | {dead} | "
            f"{report['diagnostics']['encoder_output_mean_variance']:.6g} | "
            f"{report['diagnostics']['projector_output_mean_variance']:.6g} | "
            f"{constraint['overall_violation_rate_d_gt_1_plus_epsilon']:.4f} | "
            f"{groups['changed']['constraint_violation_rate_d_gt_1_plus_epsilon']:.4f} | "
            f"{groups['unchanged']['constraint_violation_rate_d_gt_1_plus_epsilon']:.4f} | "
            f"{h5:.4f} | {heldout['triangle_check']['violation_count']} |"
        )
    print("\nHeld-out distance distributions by same-episode horizon:")
    for report in reports:
        print(json.dumps({
            "step": report["step"],
            "same_episode_distance_by_horizon": report["heldout"]["same_episode_distance_by_horizon"],
        }, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if min(args.triangles, args.eval_batch_size, args.diagnostic_states) < 1:
        raise SystemExit("triangles, eval-batch-size, and diagnostic-states must be positive")
    reports = analyze_checkpoints(args)
    _print_table(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
