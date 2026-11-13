"""Train a v0.8 IQE-maxmean quasimetric from offline one-step data."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphcritic.bridges.models import prepare_cube_robot_model_data, prepare_puzzle_model_data
from graphcritic.bridges.object_state import load_or_build_object_pair_index
from graphcritic.bridges.windows import puzzle_state_windows
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
    make_embedding_cache_predictor,
    make_object_consistency_train_step,
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
        # `terminal_state_index` is keyed by the episode's final transition
        # row, not by every earlier chunk start that lands on that state.
        target[terminal_mask] = data.terminal_state_index[ends[terminal_mask]]
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


def _chunk_constraint_report(rows: np.ndarray, data, states: np.ndarray, predict,
                             *, horizon: int, epsilon: float, seed: int,
                             max_pairs: int = 100_000) -> dict:
    """Measure a deterministic sample of same-episode chunk constraints."""
    source_rows, target_rows = _targets_at_offset(rows, data, horizon)
    if len(source_rows) > max_pairs:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(source_rows), size=max_pairs, replace=False))
        source_rows, target_rows = source_rows[selected], target_rows[selected]
    distances = np.asarray(predict(states[source_rows], states[target_rows]), dtype=np.float32)
    threshold = float(horizon) + float(epsilon)
    return {
        "n": int(len(distances)),
        "constraint_horizon": int(horizon),
        "epsilon": float(epsilon),
        "threshold": threshold,
        "sample_seed": int(seed),
        "overall_violation_rate_d_gt_limit": (
            float(np.mean(distances > threshold)) if len(distances) else None
        ),
        "distance_p50": float(np.quantile(distances, 0.50)) if len(distances) else None,
        "distance_p90": float(np.quantile(distances, 0.90)) if len(distances) else None,
    }


def heldout_metrics(data, states: np.ndarray, next_index: np.ndarray, predict,
                    *, step: int, seed: int, triangle_count: int = 100_000,
                    eval_batch_size: int = 8192, constraint_horizon: int = 1,
                    epsilon: float = 0.25) -> dict:
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

    constraint_src_rows, next_rows = _targets_at_offset(
        heldout, data, constraint_horizon
    )
    one_step_distance = _batched_distances(
        predict, states[constraint_src_rows], states[next_rows], eval_batch_size
    )
    # A chunk is button-changing if any recorded transition inside it changes a
    # bit, including a bit that flips back before the chunk endpoint.
    one_step_button_change = np.any(
        data.state_bits[:len(next_index)]
        != data.state_bits[np.asarray(next_index, dtype=np.int64)],
        axis=1,
    )
    change_prefix = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(one_step_button_change, dtype=np.int64),
    ))
    changed = (
        change_prefix[constraint_src_rows + constraint_horizon]
        - change_prefix[constraint_src_rows]
    ) > 0
    threshold = float(constraint_horizon) + float(epsilon)
    per_button = {}
    for label, mask in (("changed", changed), ("unchanged", ~changed)):
        values = one_step_distance[mask]
        group_report = {
            "n": int(np.count_nonzero(mask)),
            "distance_mean": float(np.mean(values)) if len(values) else None,
            "distance_p90": float(np.quantile(values, 0.90)) if len(values) else None,
            "constraint_violation_rate_d_gt_limit": (
                float(np.mean(values > threshold)) if len(values) else None
            ),
            "constraint_violation_squared_mean": (
                float(np.mean(np.square(np.maximum(values - constraint_horizon, 0.0))))
                if len(values) else None
            ),
        }
        if constraint_horizon == 1:
            group_report["constraint_violation_rate_d_gt_1_plus_epsilon"] = (
                group_report["constraint_violation_rate_d_gt_limit"]
            )
        per_button[label] = group_report
    constraint_report = {
        "n": int(len(one_step_distance)),
        "constraint_horizon": int(constraint_horizon),
        "epsilon": float(epsilon),
        "threshold": threshold,
        "overall_violation_rate_d_gt_limit": (
            float(np.mean(one_step_distance > threshold))
            if len(one_step_distance) else None
        ),
        "button_change_groups": per_button,
    }
    if constraint_horizon == 1:
        constraint_report["overall_violation_rate_d_gt_1_plus_epsilon"] = (
            constraint_report["overall_violation_rate_d_gt_limit"]
        )
        result["heldout_one_step_constraint"] = constraint_report
    else:
        result["heldout_chunk_constraint"] = constraint_report

    if constraint_horizon > 1:
        result["train_chunk_constraint"] = _chunk_constraint_report(
            np.asarray(data.train_rows, dtype=np.int64), data, states, predict,
            horizon=constraint_horizon, epsilon=epsilon,
            seed=int(seed + 0x7A1),
        )

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
    parser.add_argument("--transitions", type=Path)
    parser.add_argument("--sim-state", type=Path)
    parser.add_argument("--cube-double-dataset", type=Path,
                        help="standard cube-double-play-v0 archive for --input-part=robot")
    parser.add_argument("--input-part", choices=("all", "robot"), default="all",
                        help="model input: existing puzzle vector or only its continuous robot part")
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
    parser.add_argument("--window", type=int, default=1,
                        help="number of trailing normalized frames per model input")
    parser.add_argument("--constraint-horizon", type=int, default=1,
                        help="frame-step separation used by the quasimetric constraint")
    parser.add_argument("--epsilon", type=float, default=0.25,
                        help="constraint slack measured in steps")
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
    parser.add_argument("--match-constraint", choices=("none", "object"), default="none")
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--epsilon-match", type=float, default=1.25)
    parser.add_argument("--unmatched-max-frac", type=float, default=0.5)
    return parser.parse_args(argv)


def _state_episode_ids(trans) -> np.ndarray:
    terminal_rows = np.flatnonzero(trans.terminal)
    return np.concatenate((trans.episode_id, trans.episode_id[terminal_rows])).astype(np.int64)


def _load_object_pairs(candidate_cache: Path, dataset_id: str, data, trans):
    """Load cached object-match indexes and split both endpoints by episode."""
    pairs, matched_idx, unmatched_idx, index_path, cache_hit = load_or_build_object_pair_index(
        candidate_cache, data.state_bits, dataset_id=dataset_id,
    )
    episodes = _state_episode_ids(trans)
    pair_eps = episodes[pairs]
    if np.any(pair_eps[:, 0] == pair_eps[:, 1]):
        raise ValueError("object-match training requires cross-episode candidate pairs")
    train_only = np.isin(pair_eps[:, 0], data.train_episodes) & np.isin(
        pair_eps[:, 1], data.train_episodes
    )
    heldout_only = np.isin(pair_eps[:, 0], data.heldout_episodes) & np.isin(
        pair_eps[:, 1], data.heldout_episodes
    )
    train_matched_idx = matched_idx[train_only[matched_idx]]
    train_unmatched_idx = unmatched_idx[train_only[unmatched_idx]]
    heldout_matched_idx = matched_idx[heldout_only[matched_idx]]
    heldout_unmatched_idx = unmatched_idx[heldout_only[unmatched_idx]]
    if not len(train_matched_idx) or not len(train_unmatched_idx):
        raise ValueError("training episode split needs both matched and unmatched candidate pairs")
    return {
        "pairs": pairs,
        "train_matched": pairs[train_matched_idx],
        "train_unmatched": pairs[train_unmatched_idx],
        "heldout_matched": pairs[heldout_matched_idx],
        "heldout_unmatched": pairs[heldout_unmatched_idx],
        "index_path": index_path,
        "cache_hit": cache_hit,
        "candidate_pair_count": int(len(pairs)),
        "matched_pair_count": int(len(matched_idx)),
        "unmatched_pair_count": int(len(unmatched_idx)),
        "train_matched_count": int(len(train_matched_idx)),
        "train_unmatched_count": int(len(train_unmatched_idx)),
        "heldout_matched_count": int(len(heldout_matched_idx)),
        "heldout_unmatched_count": int(len(heldout_unmatched_idx)),
    }


def _reference_match_distances(
    checkpoint: Path,
    *,
    dataset_id: str,
    state_features: np.ndarray,
    candidate_nodes: np.ndarray,
    train_pairs: np.ndarray,
    heldout_pairs: np.ndarray,
    expected_state_mean: np.ndarray,
    expected_state_std: np.ndarray,
    output_dir: Path,
):
    """Precompute directed d_R caps for train and held-out matched pairs."""
    model, params, metadata = load_quasimetric_checkpoint(
        checkpoint, input_dim=state_features.shape[1], expected_dataset_id=dataset_id,
    )
    if int(metadata.get("window", -1)) != 5 or int(metadata.get("constraint_horizon", -1)) != 5:
        raise ValueError("reference checkpoint must be the fixed v0.8 window-5, horizon-5 model")
    if int(metadata.get("data_split_seed", metadata.get("split_seed", -1))) != 0:
        raise ValueError("reference checkpoint must use the prescribed seed-0 episode split")
    reference_mean = metadata.get("state_mean")
    reference_std = metadata.get("state_std")
    if reference_mean is None or reference_std is None:
        raise ValueError("reference checkpoint is missing state normalization metadata")
    if (not np.allclose(np.asarray(reference_mean), expected_state_mean, atol=1e-10, rtol=0)
            or not np.allclose(np.asarray(reference_std), expected_state_std, atol=1e-10, rtol=0)):
        raise ValueError("reference checkpoint normalization does not match the prepared puzzle data")
    checkpoint_bytes = (checkpoint / "params.msgpack").read_bytes()
    reference_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    if reference_sha != metadata.get("params_sha256"):
        raise ValueError("reference checkpoint params SHA256 does not match metadata")
    cache_path = output_dir / "reference_embeddings" / (
        f"embeddings-{reference_sha[:16]}-{hashlib.sha256(candidate_nodes.tobytes()).hexdigest()[:16]}.npy"
    )
    predict, report = make_embedding_cache_predictor(
        model, params, state_features[candidate_nodes], candidate_nodes,
        params_sha256=reference_sha, dataset_id=dataset_id,
        cache_path=cache_path, pair_batch_size=4096,
    )

    def both_directions(pairs):
        if not len(pairs):
            empty = np.empty(0, dtype=np.float32)
            return empty, empty
        return predict(pairs), predict(pairs[:, ::-1])

    train_forward, train_reverse = both_directions(train_pairs)
    heldout_forward, heldout_reverse = both_directions(heldout_pairs)
    del predict, params, model
    gc.collect()
    try:
        import jax
        jax.clear_caches()
    except Exception:  # pragma: no cover - runtime cleanup is best effort
        pass
    return (train_forward, train_reverse, heldout_forward, heldout_reverse,
            reference_sha, report)


def _heldout_object_match_metrics(predict, states, pairs, reference_forward,
                                 reference_reverse, *, epsilon_match: float,
                                 batch_size: int) -> dict:
    """Measure P2 in both directions on held-out cross-episode pairs."""
    if not len(pairs):
        return {
            "pair_count": 0,
            "directed_pair_count": 0,
            "violation_rate": None,
            "distance_p50": None,
            "reference_distance_p50": None,
        }
    directed_source = np.concatenate((pairs[:, 0], pairs[:, 1]))
    directed_target = np.concatenate((pairs[:, 1], pairs[:, 0]))
    reference = np.concatenate((reference_forward, reference_reverse))
    distance = _batched_distances(
        predict, states[directed_source], states[directed_target], batch_size
    )
    return {
        "pair_count": int(len(pairs)),
        "directed_pair_count": int(len(distance)),
        "epsilon_match": float(epsilon_match),
        "violation_rate": float(np.mean(distance > reference + epsilon_match)),
        "distance_p50": float(np.median(distance)),
        "reference_distance_p50": float(np.median(reference)),
    }


def _heldout_unmatched_metrics(predict, states, pairs, *, batch_size: int) -> dict:
    """Report d+ scale and false-accept rate on held-out object-mismatched pairs."""
    if not len(pairs):
        return {
            "pair_count": 0,
            "d_plus_p10": None,
            "d_plus_p50": None,
            "d_plus_p90": None,
            "fraction_d_plus_le_20": None,
        }
    distances = _batched_distances(
        predict, states[pairs[:, 0]], states[pairs[:, 1]], batch_size
    )
    return {
        "pair_count": int(len(distances)),
        "d_plus_p10": float(np.quantile(distances, 0.1)),
        "d_plus_p50": float(np.quantile(distances, 0.5)),
        "d_plus_p90": float(np.quantile(distances, 0.9)),
        "fraction_d_plus_le_20": float(np.mean(distances <= 20.0)),
    }


def _source_architecture(metadata: dict) -> dict:
    """Read architecture fields, treating pre-§2.1 checkpoints as legacy ReLU."""
    return {
        "activation": str(metadata.get("activation", "relu")),
        "layernorm": bool(metadata.get("layernorm", False)),
        "encoder_widths": tuple(metadata["encoder_widths"]),
        "projector_widths": tuple(metadata["projector_widths"]),
        "window": int(metadata.get("window", 1)),
        "constraint_horizon": int(metadata.get("constraint_horizon", 1)),
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
        "window": int(args.window),
        "constraint_horizon": int(args.constraint_horizon),
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
    if not np.isclose(float(metadata.get("epsilon", 0.25)), args.epsilon):
        raise ValueError("--resume-from requires the checkpoint's epsilon")
    if int(metadata.get("batch_size", args.batch_size)) != args.batch_size:
        raise ValueError("--resume-from requires the checkpoint's batch size")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    object_mode = args.match_constraint == "object"
    if args.resume_from is not None and args.init_from is not None:
        raise SystemExit("choose at most one of --resume-from and --init-from")
    if object_mode:
        if args.candidate_cache is None or args.reference_checkpoint is None:
            raise SystemExit("--match-constraint object requires --candidate-cache and --reference-checkpoint")
        if args.resume_from is not None or args.init_from is not None:
            raise SystemExit("object-match v1.0 training must initialize from scratch")
        if args.window != 5 or args.constraint_horizon != 5 or args.dual != "adam":
            raise SystemExit("object-match v1.0 fixes window=5, horizon=5, and Adam duals")
        if not 0.0 < args.unmatched_max_frac <= 0.5 or args.epsilon_match < 0:
            raise SystemExit("unmatched-max-frac must be in (0,0.5] and epsilon-match non-negative")
    elif args.candidate_cache is not None or args.reference_checkpoint is not None:
        raise SystemExit("--candidate-cache and --reference-checkpoint require --match-constraint object")
    dual_lr = args.dual_lr if args.dual_lr is not None else (0.01 if args.dual == "adam" else 1.0)
    if min(args.steps, args.save_every, args.batch_size,
           args.eval_batch_size, args.triangles, args.log_every) <= 0:
        raise SystemExit("steps, save/log intervals, batch sizes, and triangles must be positive")
    if args.grad_clip < 0 or dual_lr <= 0:
        raise SystemExit("grad-clip must be non-negative and dual-lr must be positive")
    if args.window < 1 or args.constraint_horizon < 1 or args.epsilon < 0:
        raise SystemExit("window and constraint-horizon must be positive; epsilon non-negative")
    if not 0.0 <= args.collapse_dead_frac <= 1.0 or args.collapse_min_var < 0:
        raise SystemExit("collapse thresholds must be non-negative and dead fraction at most 1")
    if len(args.encoder_widths) < 2 or len(args.projector_widths) < 2:
        raise SystemExit("encoder and projector must each have at least one hidden layer")

    jax, jnp, _, serialization, _ = _jax_modules()
    if args.cube_double_dataset is not None:
        if args.transitions is not None or args.sim_state is not None:
            raise SystemExit("--cube-double-dataset cannot be combined with --transitions/--sim-state")
        if args.input_part != "robot":
            raise SystemExit("cube-double training requires --input-part=robot")
        trans, transition_meta, data = prepare_cube_robot_model_data(
            str(args.cube_double_dataset), seed=args.seed,
        )
    else:
        if args.transitions is None or args.sim_state is None:
            raise SystemExit("puzzle training requires both --transitions and --sim-state")
        trans, transition_meta = load_transitions(args.transitions)
        sim_state = load_sim_state(args.sim_state)
        data = prepare_puzzle_model_data(trans, sim_state, seed=args.seed)
    next_index = next_state_indices(trans.terminal)
    frame_states_np = (
        data.state_cont if args.input_part == "robot"
        else np.concatenate((data.state_cont, data.state_bits), axis=1)
    ).astype(np.float32)
    states_np = (
        puzzle_state_windows(
            frame_states_np, trans, data.terminal_state_index, window=args.window
        )
        if args.window > 1 else frame_states_np
    )
    train_rows_np = np.asarray(data.train_rows, dtype=np.int32)
    constraint_rows_np, constraint_targets_np = _targets_at_offset(
        train_rows_np, data, args.constraint_horizon
    )
    if not len(constraint_rows_np):
        raise ValueError("training split has no valid constraint pairs at the selected horizon")
    constraint_next_index = np.full(trans.n_rows, -1, dtype=np.int64)
    constraint_next_index[constraint_rows_np] = constraint_targets_np
    if not len(train_rows_np) or not len(data.heldout_rows):
        raise ValueError("episode split must contain training and held-out transitions")

    object_pairs = None
    train_match_reference_forward = train_match_reference_reverse = None
    heldout_match_reference_forward = heldout_match_reference_reverse = None
    reference_sha = None
    reference_cache_report = None
    candidate_nodes = None
    if object_mode:
        object_pairs = _load_object_pairs(
            args.candidate_cache, transition_meta["dataset_id"], data, trans,
        )
        with np.load(args.candidate_cache, allow_pickle=False) as candidate_archive:
            candidate_nodes = np.asarray(candidate_archive["node_ids"], dtype=np.int64)
        (train_match_reference_forward, train_match_reference_reverse,
         heldout_match_reference_forward, heldout_match_reference_reverse,
         reference_sha, reference_cache_report) = _reference_match_distances(
            args.reference_checkpoint,
            dataset_id=transition_meta["dataset_id"], state_features=states_np,
            candidate_nodes=candidate_nodes,
            train_pairs=object_pairs["train_matched"],
            heldout_pairs=object_pairs["heldout_matched"],
            expected_state_mean=data.state_mean,
            expected_state_std=data.state_std,
            output_dir=args.out,
        )

    model = make_quasimetric_model(
        encoder_widths=tuple(args.encoder_widths),
        projector_widths=tuple(args.projector_widths),
        n_components=args.n_components,
        component_dim=args.component_dim,
        activation=args.activation,
        layernorm=args.layernorm,
    )
    if object_mode:
        train_update, primal_optimizer, dual_optimizer = make_object_consistency_train_step(
            model, batch_size=args.batch_size, epsilon_chunk=args.epsilon,
            epsilon_match=args.epsilon_match,
            unmatched_max_frac=args.unmatched_max_frac,
            learning_rate=5e-4, lambda_learning_rate=dual_lr,
            grad_clip=args.grad_clip, constraint_horizon=args.constraint_horizon,
        )
    else:
        train_update, primal_optimizer, dual_optimizer = make_train_step(
            model, batch_size=args.batch_size, epsilon=args.epsilon,
            learning_rate=5e-4, lambda_learning_rate=dual_lr,
            grad_clip=args.grad_clip, dual=args.dual,
            constraint_horizon=args.constraint_horizon,
        )
    params, primal_state, raw_lambda, dual_state, key = initialize_train_state(
        model, seed=args.seed, input_dim=states_np.shape[1],
        primal_optimizer=primal_optimizer, dual_optimizer=dual_optimizer,
    )
    if object_mode:
        raw_lambda_match = jnp.asarray(_inverse_softplus(0.01, jnp), dtype=jnp.float32)
        dual_state_match = dual_optimizer.init(raw_lambda_match)
        train_match_pairs_jax = jnp.asarray(object_pairs["train_matched"], dtype=jnp.int32)
        train_match_forward_jax = jnp.asarray(train_match_reference_forward, dtype=jnp.float32)
        train_match_reverse_jax = jnp.asarray(train_match_reference_reverse, dtype=jnp.float32)
        train_unmatched_pairs_jax = jnp.asarray(object_pairs["train_unmatched"], dtype=jnp.int32)
        heldout_match_pairs = object_pairs["heldout_matched"]
        heldout_unmatched_pairs = object_pairs["heldout_unmatched"]
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
        "spec_version": "v1.0" if object_mode else "v0.8",
        "dataset_id": transition_meta["dataset_id"],
        "input_dim": int(states_np.shape[1]),
        "frame_dim": int(frame_states_np.shape[1]),
        "window": int(args.window),
        "input_part": args.input_part,
        "constraint_horizon": int(args.constraint_horizon),
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
        "epsilon": float(args.epsilon),
        "phi": "-softplus(0.01*(500-d))/0.01",
        "batch_size": int(args.batch_size),
        "optimizer": "Adam",
        "planned_steps": int(args.steps),
        "resume_start_step": int(start_step),
        "invocation_end_step": int(end_step),
        "init_from": str(args.init_from) if args.init_from is not None else None,
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
        "resume_optimizer_state_restored": args.resume_from is not None,
        "resume_dual_state_restored": args.resume_from is not None,
        "resume_restored_step": int(start_step) if args.resume_from is not None else None,
        "resume_train_state_sha256": (
            hashlib.sha256((args.resume_from / "train_state.msgpack").read_bytes()).hexdigest()
            if args.resume_from is not None else None
        ),
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
    if object_mode:
        common_meta.update({
            "match_constraint": "object",
            "candidate_cache": str(args.candidate_cache.resolve()),
            "candidate_pair_index_cache": str(object_pairs["index_path"].resolve()),
            "candidate_pair_index_cache_hit": bool(object_pairs["cache_hit"]),
            "candidate_pair_count": object_pairs["candidate_pair_count"],
            "matched_pair_count": object_pairs["matched_pair_count"],
            "unmatched_pair_count": object_pairs["unmatched_pair_count"],
            "train_matched_pair_count": object_pairs["train_matched_count"],
            "train_unmatched_pair_count": object_pairs["train_unmatched_count"],
            "heldout_matched_pair_count": object_pairs["heldout_matched_count"],
            "reference_checkpoint": str(args.reference_checkpoint.resolve()),
            "reference_checkpoint_sha256": reference_sha,
            "reference_embedding_cache": reference_cache_report,
            "epsilon_match": float(args.epsilon_match),
            "unmatched_max_frac": float(args.unmatched_max_frac),
            "maximization_batch_size": int(args.batch_size),
            "maximization_random_count": int(round(args.batch_size * (1 - args.unmatched_max_frac))),
            "maximization_unmatched_count": int(round(args.batch_size * args.unmatched_max_frac)),
            "lambda_chunk_start_value": 0.01,
            "lambda_match_start_value": 0.01,
        })
    (args.out / "data.json").write_text(
        json.dumps({**data.metadata, **git_meta, "dataset_id": transition_meta["dataset_id"],
                    "spec_version": "v1.0" if object_mode else "v0.8"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    states = jnp.asarray(states_np)
    next_indices = jnp.asarray(next_index, dtype=jnp.int32)
    constraint_next_indices = jnp.asarray(constraint_next_index, dtype=jnp.int32)
    train_rows = jnp.asarray(train_rows_np, dtype=jnp.int32)
    constraint_rows = jnp.asarray(constraint_rows_np, dtype=jnp.int32)
    if object_mode:
        object_train_rows = jnp.asarray(train_rows_np, dtype=jnp.int32)
        object_chunk_rows = jnp.asarray(constraint_rows_np, dtype=jnp.int32)
        object_chunk_targets = jnp.asarray(constraint_targets_np, dtype=jnp.int32)
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
            gate_failure = None
            if object_mode:
                (params, primal_state, raw_lambda, dual_state,
                 raw_lambda_match, dual_state_match, key, metrics) = train_update(
                    params, primal_state, raw_lambda, dual_state,
                    raw_lambda_match, dual_state_match, key,
                    states, next_indices, object_train_rows,
                    object_chunk_rows, object_chunk_targets,
                    train_match_pairs_jax, train_match_forward_jax,
                    train_match_reverse_jax, train_unmatched_pairs_jax,
                )
            else:
                params, primal_state, raw_lambda, dual_state, key, metrics = train_update(
                    params, primal_state, raw_lambda, dual_state, key,
                    states, next_indices, train_rows, constraint_rows,
                    constraint_next_indices,
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
                if (object_mode and run_step == args.steps and not alarm
                        and any(value != 0.0 for value in diagnostics["dead_unit_frac_by_layer"].values())):
                    gate_failure = "v1.0 S1 gate requires zero dead units"
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
                if object_mode:
                    state_payload.update({
                        "raw_lambda_match": raw_lambda_match,
                        "dual_state_match": dual_state_match,
                    })
                meta = {
                    **common_meta,
                    "step": int(step),
                    "lambda_value": float(jax.device_get(
                        _dual_value(raw_lambda, dual=args.dual, jax=jax)
                    )),
                    "collapse_alarm_reason": alarm_reason,
                }
                if object_mode:
                    meta.update({
                        "lambda_chunk": float(jax.device_get(
                            _dual_value(raw_lambda, dual=args.dual, jax=jax)
                        )),
                        "lambda_match": float(jax.device_get(
                            jax.nn.softplus(raw_lambda_match)
                        )),
                        "lambda_value": float(jax.device_get(
                            _dual_value(raw_lambda, dual=args.dual, jax=jax)
                        )),
                    })
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
                    constraint_horizon=args.constraint_horizon, epsilon=args.epsilon,
                )
                if object_mode:
                    report["heldout_object_match_constraint"] = _heldout_object_match_metrics(
                        predictor, states_np, object_pairs["heldout_matched"],
                        heldout_match_reference_forward, heldout_match_reference_reverse,
                        epsilon_match=args.epsilon_match,
                        batch_size=args.eval_batch_size,
                    )
                    report["heldout_unmatched_object_pairs"] = _heldout_unmatched_metrics(
                        predictor, states_np, heldout_unmatched_pairs,
                        batch_size=args.eval_batch_size,
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
            if gate_failure is not None:
                print(f"training gate failed at step {step}: {gate_failure}", file=sys.stderr, flush=True)
                return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
