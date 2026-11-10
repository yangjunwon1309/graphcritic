"""Train the v0.8.2 Puzzle object event-count or motion-time quasimetric."""

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

from experiments.v08_runtime import (
    activation_diagnostics,
    collapse_alarm,
    _jax_modules,
    initialize_train_state,
    make_activation_diagnostic_fn,
    make_distance_predictor,
    make_quasimetric_model,
    make_train_step,
    provenance,
    save_quasimetric_checkpoint,
)
from graphcritic.bridges.object_state import (
    chunk_constraint_costs,
    prepare_puzzle_object_data,
)
from graphcritic.io import load_sim_state, load_transitions
from graphcritic.puzzle import minimum_press_solution, toggle_matrix
from graphcritic.types import SchemaError


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "n": int(len(values)), "mean": float(np.mean(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _exact_press_counts(pairs: np.ndarray, boards: np.ndarray) -> np.ndarray:
    matrix = toggle_matrix(4, 4)
    memo: dict[int, int] = {}
    result = np.empty(len(pairs), dtype=np.int16)
    for i, (source, destination) in enumerate(np.asarray(pairs, dtype=np.int64)):
        difference = boards[int(source)] ^ boards[int(destination)]
        key = sum(int(bit) << j for j, bit in enumerate(difference))
        if key not in memo:
            try:
                solution = minimum_press_solution(matrix, difference)
            except SchemaError:
                memo[key] = -1
            else:
                memo[key] = int(np.count_nonzero(solution))
        result[i] = memo[key]
    return result


def _heldout_candidate_pairs(args, data, trans, dataset_id: str) -> np.ndarray:
    if args.candidate_cache is not None:
        with np.load(args.candidate_cache, allow_pickle=False) as cache:
            if "dataset_id" in cache.files and str(cache["dataset_id"].item()) != dataset_id:
                raise ValueError("candidate cache dataset ID does not match training data")
            pairs = np.asarray(cache["pairs"], dtype=np.int64)
        heldout_nodes = np.zeros(trans.n_rows + int(np.count_nonzero(trans.terminal)), dtype=np.bool_)
        heldout_nodes[data.heldout_rows] = True
        terminal_rows = np.flatnonzero(trans.terminal)
        heldout_terminal = terminal_rows[np.isin(trans.episode_id[terminal_rows], data.heldout_episodes)]
        heldout_nodes[trans.n_rows + np.searchsorted(terminal_rows, heldout_terminal)] = True
        keep = heldout_nodes[pairs[:, 0]] & heldout_nodes[pairs[:, 1]]
        pairs = pairs[keep]
    else:
        rng = np.random.default_rng(args.seed + 0xE71)
        heldout = np.asarray(data.heldout_rows, dtype=np.int64)
        episode_by_row = trans.episode_id
        sampled: list[tuple[int, int]] = []
        by_episode = [heldout[episode_by_row[heldout] == ep] for ep in data.heldout_episodes]
        nonempty = [rows for rows in by_episode if len(rows)]
        if len(nonempty) < 2:
            return np.empty((0, 2), dtype=np.int64)
        n_pairs = min(args.max_eval_pairs, max(10_000, len(heldout)))
        for _ in range(n_pairs):
            a, b = rng.choice(len(nonempty), size=2, replace=False)
            sampled.append((int(rng.choice(nonempty[a])), int(rng.choice(nonempty[b]))))
        pairs = np.asarray(sampled, dtype=np.int64)
    rng = np.random.default_rng(args.seed + 0xCA7)
    if len(pairs) > args.max_eval_pairs:
        pairs = pairs[np.sort(rng.choice(len(pairs), args.max_eval_pairs, replace=False))]
    return pairs


def _heldout_motion_pairs(data, trans, *, max_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build same-episode pairs and their moved-chunk-count × 5 labels."""
    pairs: list[tuple[int, int]] = []
    labels: list[float] = []
    valid_rows = np.zeros(trans.n_rows, dtype=np.bool_)
    valid_rows[data.valid_chunk_rows] = True
    terminal_rows = np.flatnonzero(trans.terminal)
    for source in data.heldout_rows:
        source = int(source)
        end = int(data.episode_end_rows[source])
        for offset in (5, 10, 15, 20, 25, 30, 35, 40):
            raw_destination = source + offset
            if raw_destination > end + 1:
                break
            if raw_destination <= end:
                destination = raw_destination
            else:
                # The terminal occurrence is the final state at t=end+1.
                ordinal = np.searchsorted(terminal_rows, end)
                destination = trans.n_rows + int(ordinal)
            if destination >= trans.n_rows and trans.episode_id[end] not in data.heldout_episodes:
                continue
            if offset > 5:
                if destination >= trans.n_rows and destination != trans.n_rows + int(np.searchsorted(terminal_rows, end)):
                    continue
            chunk_starts = source + np.arange(offset // 5, dtype=np.int64) * 5
            if np.any(~valid_rows[chunk_starts]):
                continue
            pairs.append((source, destination))
            labels.append(float(np.count_nonzero(data.moved[chunk_starts]) * 5))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(pairs), size=max_pairs, replace=False))
        pairs = [pairs[i] for i in chosen]
        labels = [labels[i] for i in chosen]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2), np.asarray(labels, dtype=np.float32)


def _event_report(scores: np.ndarray, press_counts: np.ndarray) -> dict:
    by_count = {}
    for count in sorted(np.unique(press_counts).tolist()):
        by_count["infinity" if count < 0 else str(int(count))] = _summary(scores[press_counts == count])
    predicted = scores <= 1.5
    actual = (press_counts >= 0) & (press_counts <= 1)
    return {
        "e_phi_by_minimum_press_count": by_count,
        "threshold_1_5_vs_n_p_le_1": {
            "true_positive": int(np.count_nonzero(predicted & actual)),
            "false_positive": int(np.count_nonzero(predicted & ~actual)),
            "false_negative": int(np.count_nonzero(~predicted & actual)),
            "true_negative": int(np.count_nonzero(~predicted & ~actual)),
        },
        "candidate_pair_count": int(len(scores)),
    }


def _heldout_report(args, data, trans, sim_state, model, params, predictor, *, step: int) -> dict:
    costs = chunk_constraint_costs(data, args.cost)
    valid = np.asarray(data.valid_chunk_rows, dtype=np.int64)
    heldout_mask = np.zeros(trans.n_rows, dtype=np.bool_)
    heldout_mask[data.heldout_rows] = True
    chunk_rows = valid[heldout_mask[valid]]
    endpoints = data.chunk_endpoint[chunk_rows]
    distances = predictor(data.windows[chunk_rows], data.windows[endpoints])
    chunk_cost = costs[chunk_rows]
    epsilon_g = 0.25 if args.cost == "events" else 1.25
    report: dict[str, object] = {
        "step": int(step),
        "cost": args.cost,
        "heldout_chunk_count": int(len(chunk_rows)),
        "chunk_constraint_violation_rate": float(np.mean(distances > chunk_cost)) if len(distances) else None,
        "chunk_constraint_violation_rate_beyond_epsilon": (
            float(np.mean(distances > chunk_cost + epsilon_g)) if len(distances) else None
        ),
        "distance_by_chunk_cost": {
            str(float(value)): _summary(distances[chunk_cost == value])
            for value in np.unique(chunk_cost)
        },
        "collapse_metrics": activation_diagnostics(
            make_activation_diagnostic_fn(model), params,
            data.windows[np.asarray(data.heldout_rows[:min(4096, len(data.heldout_rows))])],
            activation="gelu",
        ),
        "moved_fraction": data.movement_report["moved_fraction"],
        "q90_assumption_warning": data.movement_report["q90_assumption_warning"],
        "T_I": data.onset_report["p1_frames"],
    }
    if args.cost == "events":
        pairs = _heldout_candidate_pairs(args, data, trans, args.dataset_id)
        scores = predictor(data.windows[pairs[:, 0]], data.windows[pairs[:, 1]]) if len(pairs) else np.empty(0)
        press_counts = _exact_press_counts(pairs, np.asarray(sim_state.button_states, dtype=np.uint8)) if len(pairs) else np.empty(0, dtype=np.int16)
        report["event_candidate_comparison"] = _event_report(scores, press_counts)
    else:
        pairs, motion_time = _heldout_motion_pairs(
            data, trans, max_pairs=args.max_eval_pairs, seed=args.seed + step
        )
        scores = predictor(data.windows[pairs[:, 0]], data.windows[pairs[:, 1]]) if len(pairs) else np.empty(0)
        report["motion_time_comparison"] = {
            "m_psi": _summary(scores),
            "moved_chunk_count_times_5": _summary(motion_time),
            "spearman": _spearman(scores, motion_time),
        }
    return report


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    from scipy.stats import rankdata
    ar = rankdata(np.asarray(a), method="average")
    br = rankdata(np.asarray(b), method="average")
    if np.std(ar) == 0 or np.std(br) == 0:
        return None
    return float(np.corrcoef(ar, br)[0, 1])


def _parse(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cost", choices=("events", "motion"), required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--max-eval-pairs", type=int, default=50_000)
    parser.add_argument("--collapse-dead-frac", type=float, default=0.95)
    parser.add_argument("--collapse-min-var", type=float, default=1e-6)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.window != 5 or min(args.steps, args.save_every, args.log_every,
                               args.batch_size, args.eval_batch_size, args.max_eval_pairs) <= 0:
        raise SystemExit("v0.8.2 fixes window=5; step, batch, and interval values must be positive")
    jax, jnp, _, _, _ = _jax_modules()
    trans, transition_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    data = prepare_puzzle_object_data(trans, sim_state, seed=args.seed, window=args.window)
    if not len(data.train_rows) or not len(data.heldout_rows):
        raise ValueError("episode split must contain training and held-out rows")
    train_mask = np.zeros(trans.n_rows, dtype=np.bool_)
    train_mask[data.train_rows] = True
    constraint_rows = data.valid_chunk_rows[train_mask[data.valid_chunk_rows]]
    if not len(constraint_rows):
        raise ValueError("training split contains no valid five-frame object chunks")
    next_index = np.full(trans.n_rows, -1, dtype=np.int64)
    next_index[constraint_rows] = data.chunk_endpoint[constraint_rows]
    constraint_costs = chunk_constraint_costs(data, args.cost)
    epsilon_g = 0.25 if args.cost == "events" else 1.25
    encoder_widths = (256, 256, 128)
    projector_widths = (256, 512)
    n_components, component_dim = 16, 32
    model = make_quasimetric_model(
        encoder_widths=encoder_widths, projector_widths=projector_widths,
        n_components=n_components, component_dim=component_dim,
        activation="gelu", layernorm=True,
    )
    update, primal_opt, dual_opt = make_train_step(
        model, batch_size=args.batch_size, epsilon=epsilon_g,
        learning_rate=5e-4, lambda_learning_rate=1e-2, grad_clip=1.0,
        dual="adam", constraint_horizon=0.0,
        independent_goal_is_next_state=False,
    )
    params, primal_state, raw_lambda, dual_state, key = initialize_train_state(
        model, seed=args.seed, input_dim=data.windows.shape[1],
        primal_optimizer=primal_opt, dual_optimizer=dual_opt,
    )
    states = jnp.asarray(data.windows)
    next_jax = jnp.asarray(next_index, dtype=jnp.int32)
    train_jax = jnp.asarray(data.train_rows, dtype=jnp.int32)
    constraint_rows_jax = jnp.asarray(constraint_rows, dtype=jnp.int32)
    costs_jax = jnp.asarray(constraint_costs, dtype=jnp.float32)
    args.dataset_id = transition_meta["dataset_id"]
    args.out.mkdir(parents=True, exist_ok=True)
    common_meta = {
        **provenance(ROOT),
        "spec_version": "v0.8.2",
        "model_kind": "object-quasimetric-v0.8.2",
        "object_cost": args.cost,
        "dataset_id": args.dataset_id,
        "object_state_schema": "puzzle-button-onehot-depth-velocity-v1",
        **data.metadata,
        "data_split_seed": int(args.seed),
        "train_fraction": 0.9,
        "train_transition_count": int(len(data.train_rows)),
        "heldout_transition_count": int(len(data.heldout_rows)),
        "encoder_widths": list(encoder_widths),
        "projector_widths": list(projector_widths),
        "n_components": n_components,
        "component_dim": component_dim,
        "activation": "gelu",
        "layernorm": True,
        "iqe_head": "maxmean",
        "window": int(args.window),
        "input_dim": int(data.windows.shape[1]),
        "frame_dim": 64,
        "constraint_cost_definition": "onset(t)" if args.cost == "events" else "5*moved(t)",
        "constraint_cost_values": [0.0, 1.0] if args.cost == "events" else [0.0, 5.0],
        "epsilon": epsilon_g,
        "learning_rate": 5e-4,
        "dual": "adam",
        "dual_learning_rate": 1e-2,
        "gradient_clip_global_norm": 1.0,
        "batch_size": int(args.batch_size),
        "planned_steps": int(args.steps),
        "save_every": int(args.save_every),
        "log_every": int(args.log_every),
        "T_I": data.onset_report["p1_frames"],
        "event_max_auto": (
            int(np.ceil(20.0 / data.onset_report["p1_frames"]))
            if data.onset_report["p1_frames"] else None
        ),
        "device_backend": jax.default_backend(),
    }
    (args.out / "data.json").write_text(
        json.dumps({**data.metadata, **provenance(ROOT), "dataset_id": args.dataset_id},
                   sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    diag_rng = np.random.default_rng(args.seed + 0x51A7)
    diag_rows = diag_rng.choice(data.heldout_rows, min(4096, len(data.heldout_rows)), replace=False)
    diagnostic_states = data.windows[diag_rows]
    diagnose = make_activation_diagnostic_fn(model)
    last_projector_layer = "projector_0"
    log_path = args.out / "training.jsonl"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            params, primal_state, raw_lambda, dual_state, key, metrics = update(
                params, primal_state, raw_lambda, dual_state, key,
                states, next_jax, train_jax, constraint_rows_jax, next_jax,
                costs_jax,
            )
            training_metrics = {
                name: float(jax.device_get(value)) for name, value in metrics.items()
            }
            should_save = step % args.save_every == 0 or step == args.steps
            should_log = step % args.log_every == 0 or should_save
            diagnostics = None
            alarm_reason = None
            if should_log:
                diagnostics = activation_diagnostics(
                    diagnose, params, diagnostic_states, activation="gelu"
                )
                alarm, alarm_reason = collapse_alarm(
                    diagnostics, last_projector_layer=last_projector_layer,
                    dead_fraction_limit=args.collapse_dead_frac,
                    min_projector_variance=args.collapse_min_var,
                )
                should_save = should_save or alarm
                row = {
                    "step": int(step), "training": training_metrics,
                    "diagnostics": diagnostics,
                    "collapse_alarm": bool(alarm),
                    "collapse_alarm_reason": alarm_reason,
                    "movement": data.movement_report,
                    "T_I": data.onset_report,
                    "elapsed_seconds": float(time.time() - started),
                }
                log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                log.flush()
                print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
            checkpoint = args.out / f"step_{step:06d}" if should_save else None
            if checkpoint is not None:
                meta = {
                    **common_meta, "step": int(step),
                    "lambda_value": float(jax.device_get(jax.nn.softplus(raw_lambda))),
                    "collapse_alarm_reason": alarm_reason,
                }
                state_payload = {
                    "params": params, "primal_state": primal_state,
                    "raw_lambda": raw_lambda, "dual_state": dual_state,
                    "key": key, "step": np.asarray(step, dtype=np.int32),
                }
                save_quasimetric_checkpoint(
                    checkpoint, params, meta, train_state=state_payload
                )
                predictor = make_distance_predictor(
                    model, params, batch_size=args.eval_batch_size
                )
                report = _heldout_report(
                    args, data, trans, sim_state, model, params, predictor, step=step
                )
                report.update({
                    "training": training_metrics,
                    "collapse_metrics": diagnostics,
                    "checkpoint": str(checkpoint),
                    "elapsed_seconds": float(time.time() - started),
                })
                with (args.out / "heldout.jsonl").open("a", encoding="utf-8") as heldout_log:
                    heldout_log.write(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
                print(json.dumps({"heldout": report}, sort_keys=True, allow_nan=False), flush=True)
            if alarm_reason is not None:
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
