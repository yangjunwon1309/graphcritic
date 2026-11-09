"""Train a v0.8 IQE-maxmean quasimetric from offline one-step data."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
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
    activation_diagnostics,
    collapse_alarm,
    _jax_modules,
    initialize_train_state,
    load_quasimetric_checkpoint,
    make_activation_diagnostic_fn,
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
    parser.add_argument("--steps", type=int, default=200_000,
                        help="number of optimizer steps in this invocation")
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
    parser.add_argument("--activation", choices=("relu", "gelu"), default="gelu")
    norm = parser.add_mutually_exclusive_group()
    norm.add_argument("--layernorm", dest="layernorm", action="store_true")
    norm.add_argument("--no-layernorm", dest="layernorm", action="store_false")
    parser.set_defaults(layernorm=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dual", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--dual-lr", type=float, default=None,
                        help="dual Adam lr (default .01) or projected-SGD eta (default 1.0)")
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--collapse-dead-frac", type=float, default=0.95)
    parser.add_argument("--collapse-min-var", type=float, default=1e-6)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args(argv)


def _source_architecture(metadata: dict) -> dict:
    """Read architecture fields, treating pre-§2.1 checkpoints as legacy ReLU."""
    return {
        "activation": str(metadata.get("activation", "relu")),
        "layernorm": bool(metadata.get("layernorm", False)),
        "encoder_widths": tuple(metadata["encoder_widths"]),
        "projector_widths": tuple(metadata["projector_widths"]),
        "n_components": int(metadata["n_components"]),
        "component_dim": int(metadata["component_dim"]),
    }


def _inverse_softplus(value: float, jnp):
    value = jnp.asarray(max(float(value), 1e-8), dtype=jnp.float32)
    return value + jnp.log(-jnp.expm1(-value))


def _dual_value(parameter, *, dual: str, jax):
    return jax.nn.softplus(parameter) if dual == "adam" else parameter


def _mutable_tree(value):
    if isinstance(value, Mapping):
        return {key: _mutable_tree(item) for key, item in value.items()}
    return value


def _warm_start_parameters(target_params, source_params):
    """Copy Dense tensors and IQE alpha; keep target LayerNorm identity init."""
    params = _mutable_tree(target_params)
    source_params = _mutable_tree(source_params)
    for prefix in ("encoder_", "projector_"):
        for name, target_block in params.items():
            if not name.startswith(prefix) or name.endswith("_norm"):
                continue
            if not isinstance(target_block, Mapping) or "kernel" not in target_block:
                continue
            source_block = source_params.get(name)
            if source_block is None:
                raise ValueError(f"warm-start checkpoint is missing Dense layer {name!r}")
            copied = dict(target_block)
            for field in ("kernel", "bias"):
                if np.shape(source_block[field]) != np.shape(target_block[field]):
                    raise ValueError(f"warm-start shape mismatch for {name}.{field}")
                copied[field] = source_block[field]
            params[name] = copied
    if "raw_alpha" not in source_params:
        raise ValueError("warm-start checkpoint is missing IQE alpha")
    params["raw_alpha"] = source_params["raw_alpha"]
    return params


def _validate_resume_metadata(metadata: dict, *, args, dual_lr: float) -> None:
    arch = _source_architecture(metadata)
    requested = {
        "activation": args.activation,
        "layernorm": bool(args.layernorm),
        "encoder_widths": tuple(args.encoder_widths),
        "projector_widths": tuple(args.projector_widths),
        "n_components": int(args.n_components),
        "component_dim": int(args.component_dim),
    }
    if arch != requested:
        raise ValueError(
            "--resume-from requires an identical model architecture; "
            f"checkpoint={arch}, requested={requested}"
        )
    source_dual = str(metadata.get("dual", "adam"))
    source_dual_lr = float(metadata.get("dual_learning_rate", metadata.get("lambda_learning_rate", 1e-2)))
    if source_dual != args.dual or not np.isclose(source_dual_lr, dual_lr):
        raise ValueError("--resume-from requires the checkpoint's dual optimizer and learning rate")
    if not np.isclose(float(metadata.get("learning_rate", 5e-4)), 5e-4):
        raise ValueError("--resume-from requires the checkpoint's primal learning rate")
    source_clip = float(metadata.get("gradient_clip_global_norm", 0.0))
    if not np.isclose(source_clip, args.grad_clip):
        raise ValueError("--resume-from requires the checkpoint's primal gradient clipping setting")
    source_seed = int(metadata.get("data_split_seed", metadata.get("split_seed", 0)))
    if source_seed != args.seed:
        raise ValueError("--resume-from requires the checkpoint's episode split seed")
    if not np.isclose(float(metadata.get("epsilon", 0.25)), 0.25):
        raise ValueError("--resume-from requires epsilon=0.25")
    if int(metadata.get("batch_size", args.batch_size)) != args.batch_size:
        raise ValueError("--resume-from requires the checkpoint's batch size")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.resume_from is not None and args.init_from is not None:
        raise SystemExit("choose at most one of --resume-from and --init-from")
    dual_lr = args.dual_lr if args.dual_lr is not None else (0.01 if args.dual == "adam" else 1.0)
    if min(args.steps, args.save_every, args.batch_size,
           args.eval_batch_size, args.triangles, args.log_every) <= 0:
        raise SystemExit("steps, save/log intervals, batch sizes, and triangles must be positive")
    if args.grad_clip < 0 or dual_lr <= 0:
        raise SystemExit("grad-clip must be non-negative and dual-lr must be positive")
    if not 0.0 <= args.collapse_dead_frac <= 1.0 or args.collapse_min_var < 0:
        raise SystemExit("collapse thresholds must be non-negative and dead fraction at most 1")
    if len(args.encoder_widths) < 2 or len(args.projector_widths) < 2:
        raise SystemExit("encoder and projector must each have at least one hidden layer")

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
        activation=args.activation,
        layernorm=args.layernorm,
    )
    train_update, primal_optimizer, dual_optimizer = make_train_step(
        model, batch_size=args.batch_size, epsilon=0.25,
        learning_rate=5e-4, lambda_learning_rate=dual_lr,
        grad_clip=args.grad_clip, dual=args.dual,
    )
    params, primal_state, raw_lambda, dual_state, key = initialize_train_state(
        model, seed=args.seed, input_dim=states_np.shape[1],
        primal_optimizer=primal_optimizer, dual_optimizer=dual_optimizer,
    )
    start_step = 0
    init_from_step = None
    if args.resume_from is not None:
        resume_path = args.resume_from
        metadata = json.loads((resume_path / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("dataset_id") != transition_meta["dataset_id"]:
            raise ValueError("resume checkpoint dataset ID does not match training data")
        _validate_resume_metadata(metadata, args=args, dual_lr=dual_lr)
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
    elif args.init_from is not None:
        source_model, source_params, source_metadata = load_quasimetric_checkpoint(
            args.init_from, input_dim=states_np.shape[1],
            expected_dataset_id=transition_meta["dataset_id"],
        )
        del source_model
        params = _warm_start_parameters(params, source_params)
        source_lambda = float(source_metadata.get("lambda_value", 0.01))
        raw_lambda = jnp.asarray(
            _inverse_softplus(source_lambda, jnp) if args.dual == "adam" else source_lambda,
            dtype=jnp.float32,
        )
        init_from_step = int(source_metadata.get("step", 0))
        # Warm starts deliberately begin with fresh optimizer moments.
        primal_state = primal_optimizer.init(params)
        dual_state = (
            dual_optimizer.init(raw_lambda) if dual_optimizer is not None
            else jnp.asarray(0.0, dtype=jnp.float32)
        )

    end_step = start_step + int(args.steps)
    lambda_start_value = float(jax.device_get(_dual_value(raw_lambda, dual=args.dual, jax=jax)))

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
        "activation": args.activation,
        "layernorm": bool(args.layernorm),
        "iqe_head": "maxmean",
        "n_components": int(args.n_components),
        "component_dim": int(args.component_dim),
        "learning_rate": 5e-4,
        "dual": args.dual,
        "dual_parameterization": "softplus" if args.dual == "adam" else "projected_direct",
        "dual_learning_rate": float(dual_lr),
        "lambda_learning_rate": float(dual_lr) if args.dual == "adam" else None,
        "lambda_start_value": lambda_start_value,
        "gradient_clip_global_norm": float(args.grad_clip),
        "epsilon": 0.25,
        "phi": "-softplus(0.01*(500-d))/0.01",
        "batch_size": int(args.batch_size),
        "optimizer": "Adam",
        "planned_steps": int(args.steps),
        "resume_start_step": int(start_step),
        "invocation_end_step": int(end_step),
        "init_from": str(args.init_from) if args.init_from is not None else None,
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
        "warm_start_source_step": init_from_step,
        "save_every": int(args.save_every),
        "log_every": int(args.log_every),
        "collapse_dead_frac": float(args.collapse_dead_frac),
        "collapse_min_var": float(args.collapse_min_var),
        "diagnostic_state_count": 4096,
        "diagnostic_state_seed": int(args.seed + 0x51A7),
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
    diag_rng = np.random.default_rng(args.seed + 0x51A7)
    diag_rows = diag_rng.choice(
        data.heldout_rows, size=4096,
        replace=len(data.heldout_rows) < 4096,
    ).astype(np.int64)
    diagnostic_states = states_np[diag_rows]
    diagnose = make_activation_diagnostic_fn(model)
    last_projector_layer = f"projector_{len(args.projector_widths) - 2}"
    hist_path = args.out / "training.jsonl"
    train_hist_path = args.out / "train_metrics.jsonl"
    log_mode = "a" if args.resume_from is not None and hist_path.exists() else "w"
    train_log_mode = "a" if args.resume_from is not None and train_hist_path.exists() else "w"
    started = time.time()
    with hist_path.open(log_mode, encoding="utf-8") as hist, train_hist_path.open(
        train_log_mode, encoding="utf-8"
    ) as train_hist:
        for step in range(start_step + 1, end_step + 1):
            run_step = step - start_step
            params, primal_state, raw_lambda, dual_state, key, metrics = train_update(
                params, primal_state, raw_lambda, dual_state, key,
                states, next_indices, train_rows,
            )
            train_values = {
                key: float(jax.device_get(value))
                for key, value in metrics.items()
            }
            monitor_now = run_step % args.log_every == 0 or run_step == args.steps
            checkpoint_now = run_step % args.save_every == 0 or run_step == args.steps
            alarm_reason = None
            diagnostics = None
            if monitor_now:
                diagnostics = activation_diagnostics(
                    diagnose, params, diagnostic_states, activation=args.activation,
                )
                alarm, alarm_reason = collapse_alarm(
                    diagnostics, last_projector_layer=last_projector_layer,
                    dead_fraction_limit=args.collapse_dead_frac,
                    min_projector_variance=args.collapse_min_var,
                )
                train_row = {
                    "step": int(step),
                    "run_step": int(run_step),
                    "training": train_values,
                    "diagnostics": diagnostics,
                    "collapse_alarm": bool(alarm),
                    "collapse_alarm_reason": alarm_reason,
                    "elapsed_seconds": float(time.time() - started),
                }
                train_hist.write(json.dumps(train_row, sort_keys=True, allow_nan=False) + "\n")
                train_hist.flush()
                if alarm:
                    checkpoint_now = True

            checkpoint = args.out / f"step_{step:06d}" if checkpoint_now else None
            if checkpoint is not None:
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
                    "lambda_value": float(jax.device_get(
                        _dual_value(raw_lambda, dual=args.dual, jax=jax)
                    )),
                    "collapse_alarm_reason": alarm_reason,
                }
                save_quasimetric_checkpoint(
                    checkpoint, params, meta, train_state=state_payload
                )

            if checkpoint is not None and alarm_reason is None:
                predictor = make_distance_predictor(
                    model, params, batch_size=args.eval_batch_size
                )
                report = heldout_metrics(
                    data, states_np, next_index, predictor,
                    step=step, seed=args.seed, triangle_count=args.triangles,
                    eval_batch_size=args.eval_batch_size,
                )
                row = {
                    **report,
                    "run_step": int(run_step),
                    "training": train_values,
                    "checkpoint": str(checkpoint),
                    "elapsed_seconds": float(time.time() - started),
                }
                hist.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                hist.flush()
                print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

            if alarm_reason is not None:
                print(
                    f"collapse alarm at step {step}: {alarm_reason}; "
                    f"checkpoint saved to {checkpoint}", file=sys.stderr, flush=True,
                )
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
