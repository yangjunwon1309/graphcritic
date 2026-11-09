"""Train a v0.8 IQE-maxmean quasimetric from offline one-step data."""

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

from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.io import load_sim_state, load_transitions
from experiments.v08_runtime import (
    DEFAULT_COMPONENT_DIM,
    DEFAULT_COMPONENTS,
    DEFAULT_ENCODER_WIDTHS,
    DEFAULT_PROJECTOR_WIDTHS,
    _jax_modules,
    initialize_train_state,
    make_distance_predictor,
    make_quasimetric_model,
    make_train_step,
    next_state_indices,
    provenance,
    save_quasimetric_checkpoint,
)


def _targets_at_offset(rows: np.ndarray, data, offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Return valid same-episode endpoints at an exact transition offset."""
    rows = np.asarray(rows, dtype=np.int64)
    positions = rows + int(offset)
    ends = data.episode_end_rows[rows]
    terminal_positions = ends + 1
    valid = (positions <= ends) | (positions == terminal_positions)
    target = positions.copy()
    terminal_mask = positions == terminal_positions
    if np.any(terminal_mask):
        target[terminal_mask] = data.terminal_state_index[rows[terminal_mask]]
        if np.any(target[terminal_mask] < 0):
            raise ValueError("held-out terminal endpoint is missing from sim_state")
    return rows[valid], target[valid]


def _batched_distances(predict, source: np.ndarray, destination: np.ndarray,
                       batch_size: int) -> np.ndarray:
    if len(source) != len(destination):
        raise ValueError("distance input arrays must align")
    # make_distance_predictor already pads each final batch to its fixed JIT shape.
    del batch_size
    return predict(source, destination)


def heldout_metrics(data, states: np.ndarray, next_index: np.ndarray, predict,
                    *, step: int, seed: int, triangle_count: int = 100_000,
                    eval_batch_size: int = 8192) -> dict:
    """Evaluate time calibration, button-conditioned errors, and triangles."""
    heldout = np.asarray(data.heldout_rows, dtype=np.int64)
    result: dict[str, object] = {
        "step": int(step),
        "same_episode_distance_by_horizon": {},
    }
    for horizon in (*range(1, 21), 40, 80):
        src_rows, dst_rows = _targets_at_offset(heldout, data, horizon)
        distances = _batched_distances(
            predict, states[src_rows], states[dst_rows], eval_batch_size
        )
        result["same_episode_distance_by_horizon"][str(horizon)] = {
            "n": int(len(distances)),
            "mean": float(np.mean(distances)) if len(distances) else None,
            "p10": float(np.quantile(distances, 0.10)) if len(distances) else None,
            "p50": float(np.quantile(distances, 0.50)) if len(distances) else None,
            "p90": float(np.quantile(distances, 0.90)) if len(distances) else None,
            "fraction_distance_leq_horizon": (
                float(np.mean(distances <= horizon)) if len(distances) else None
            ),
        }

    next_rows = next_index[heldout]
    one_step_distance = _batched_distances(
        predict, states[heldout], states[next_rows], eval_batch_size
    )
    button_start = states[heldout, -16:] > 0.5
    button_end = states[next_rows, -16:] > 0.5
    changed = np.any(button_start != button_end, axis=1)
    threshold = 1.0 + 0.25
    per_button = {}
    for label, mask in (("changed", changed), ("unchanged", ~changed)):
        values = one_step_distance[mask]
        per_button[label] = {
            "n": int(np.count_nonzero(mask)),
            "distance_mean": float(np.mean(values)) if len(values) else None,
            "distance_p90": float(np.quantile(values, 0.90)) if len(values) else None,
            "constraint_violation_rate_d_gt_1_plus_epsilon": (
                float(np.mean(values > threshold)) if len(values) else None
            ),
            "constraint_violation_squared_mean": (
                float(np.mean(np.square(np.maximum(values - 1.0, 0.0))))
                if len(values) else None
            ),
        }
    result["heldout_one_step_constraint"] = {
        "n": int(len(one_step_distance)),
        "epsilon": 0.25,
        "threshold": threshold,
        "overall_violation_rate_d_gt_1_plus_epsilon": (
            float(np.mean(one_step_distance > threshold))
            if len(one_step_distance) else None
        ),
        "button_change_groups": per_button,
    }

    rng = np.random.default_rng(seed + step)
    heldout_terminals = data.terminal_state_index[heldout]
    heldout_terminals = np.unique(heldout_terminals[heldout_terminals >= 0])
    heldout_state_pool = np.concatenate((heldout, heldout_terminals)).astype(np.int64)
    if len(heldout_state_pool) < 3:
        raise ValueError("held-out split needs at least three states for triangle checks")
    sampled = rng.choice(heldout_state_pool, size=(triangle_count, 3), replace=True)
    u, v, w = sampled.T
    d_uv = _batched_distances(predict, states[u], states[v], eval_batch_size)
    d_vw = _batched_distances(predict, states[v], states[w], eval_batch_size)
    d_uw = _batched_distances(predict, states[u], states[w], eval_batch_size)
    triangle_violation = d_uw > d_uv + d_vw + 1e-5
    result["triangle_check"] = {
        "n": int(triangle_count),
        "tolerance": 1e-5,
        "violation_count": int(np.count_nonzero(triangle_violation)),
        "violation_rate": float(np.mean(triangle_violation)),
        "max_excess": float(np.max(d_uw - d_uv - d_vw, initial=0.0)),
    }
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--triangles", type=int, default=100_000)
    parser.add_argument("--encoder-widths", type=int, nargs="+",
                        default=list(DEFAULT_ENCODER_WIDTHS))
    parser.add_argument("--projector-widths", type=int, nargs="+",
                        default=list(DEFAULT_PROJECTOR_WIDTHS))
    parser.add_argument("--n-components", type=int, default=DEFAULT_COMPONENTS)
    parser.add_argument("--component-dim", type=int, default=DEFAULT_COMPONENT_DIM)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if min(args.steps, args.save_every, args.batch_size,
           args.eval_batch_size, args.triangles) <= 0:
        raise SystemExit("steps, save-every, batch sizes, and triangles must be positive")

    jax, jnp, _, serialization, _ = _jax_modules()
    trans, transition_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    data = prepare_puzzle_model_data(trans, sim_state, seed=args.seed)
    next_index = next_state_indices(trans.terminal)
    states_np = np.concatenate((data.state_cont, data.state_bits), axis=1).astype(np.float32)
    train_rows_np = np.asarray(data.train_rows, dtype=np.int32)
    if not len(train_rows_np) or not len(data.heldout_rows):
        raise ValueError("episode split must contain training and held-out transitions")

    model = make_quasimetric_model(
        encoder_widths=tuple(args.encoder_widths),
        projector_widths=tuple(args.projector_widths),
        n_components=args.n_components,
        component_dim=args.component_dim,
    )
    train_update, primal_optimizer, dual_optimizer = make_train_step(
        model, batch_size=args.batch_size, epsilon=0.25,
        learning_rate=5e-4, lambda_learning_rate=1e-2,
    )
    params, primal_state, raw_lambda, dual_state, key = initialize_train_state(
        model, seed=args.seed, input_dim=states_np.shape[1],
        primal_optimizer=primal_optimizer, dual_optimizer=dual_optimizer,
    )
    start_step = 0
    if args.resume_from is not None:
        resume_path = args.resume_from
        metadata = json.loads((resume_path / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("dataset_id") != transition_meta["dataset_id"]:
            raise ValueError("resume checkpoint dataset ID does not match training data")
        if not (resume_path / "train_state.msgpack").is_file():
            raise ValueError("resume checkpoint does not contain train_state.msgpack")
        template = {
            "params": params, "primal_state": primal_state,
            "raw_lambda": raw_lambda, "dual_state": dual_state,
            "key": key, "step": np.asarray(0, dtype=np.int32),
        }
        restored = serialization.from_bytes(
            template, (resume_path / "train_state.msgpack").read_bytes()
        )
        params = restored["params"]
        primal_state = restored["primal_state"]
        raw_lambda = restored["raw_lambda"]
        dual_state = restored["dual_state"]
        key = restored["key"]
        start_step = int(restored["step"])

    args.out.mkdir(parents=True, exist_ok=True)
    git_meta = provenance(ROOT)
    common_meta = {
        **git_meta,
        "spec_version": "v0.8",
        "dataset_id": transition_meta["dataset_id"],
        "input_dim": int(states_np.shape[1]),
        "continuous_dim": int(data.n_cont),
        "button_dim": int(data.n_bits),
        "data_split_seed": args.seed,
        "train_fraction": 0.9,
        "train_episode_count": int(len(data.train_episodes)),
        "heldout_episode_count": int(len(data.heldout_episodes)),
        "train_transition_count": int(len(data.train_rows)),
        "heldout_transition_count": int(len(data.heldout_rows)),
        "encoder_widths": list(args.encoder_widths),
        "projector_widths": list(args.projector_widths),
        "iqe_head": "maxmean",
        "n_components": int(args.n_components),
        "component_dim": int(args.component_dim),
        "learning_rate": 5e-4,
        "lambda_learning_rate": 1e-2,
        "lambda_initial": 0.01,
        "epsilon": 0.25,
        "phi": "-softplus(0.01*(500-d))/0.01",
        "batch_size": int(args.batch_size),
        "optimizer": "Adam",
        "planned_steps": int(args.steps),
        "save_every": int(args.save_every),
        "state_mean": data.state_mean.tolist(),
        "state_std": data.state_std.tolist(),
        "epsilon_raw": data.epsilon_raw.tolist(),
        "epsilon_z": data.epsilon_z.tolist(),
        "device_backend": jax.default_backend(),
    }
    (args.out / "data.json").write_text(
        json.dumps({**data.metadata, **git_meta, "dataset_id": transition_meta["dataset_id"],
                    "spec_version": "v0.8"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    states = jnp.asarray(states_np)
    next_indices = jnp.asarray(next_index, dtype=jnp.int32)
    train_rows = jnp.asarray(train_rows_np, dtype=jnp.int32)
    hist_path = args.out / "training.jsonl"
    log_mode = "a" if start_step > 0 and hist_path.exists() else "w"
    started = time.time()
    with hist_path.open(log_mode, encoding="utf-8") as hist:
        for step in range(start_step + 1, args.steps + 1):
            params, primal_state, raw_lambda, dual_state, key, metrics = train_update(
                params, primal_state, raw_lambda, dual_state, key,
                states, next_indices, train_rows,
            )
            if step % args.save_every != 0 and step != args.steps:
                continue
            checkpoint = args.out / f"step_{step:06d}"
            state_payload = {
                "params": params,
                "primal_state": primal_state,
                "raw_lambda": raw_lambda,
                "dual_state": dual_state,
                "key": key,
                "step": np.asarray(step, dtype=np.int32),
            }
            meta = {
                **common_meta,
                "step": int(step),
                "lambda_value": float(jax.device_get(jax.nn.softplus(raw_lambda))),
            }
            save_quasimetric_checkpoint(
                checkpoint, params, meta, train_state=state_payload
            )
            predictor = make_distance_predictor(
                model, params, batch_size=args.eval_batch_size
            )
            report = heldout_metrics(
                data, states_np, next_index, predictor,
                step=step, seed=args.seed, triangle_count=args.triangles,
                eval_batch_size=args.eval_batch_size,
            )
            train_values = {
                key: float(jax.device_get(value))
                for key, value in metrics.items()
            }
            row = {
                **report,
                "training": train_values,
                "checkpoint": str(checkpoint),
                "elapsed_seconds": float(time.time() - started),
            }
            hist.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            hist.flush()
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
