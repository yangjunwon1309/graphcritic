"""Train the v0.9 action-conditioned local quasimetric ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.v08_runtime import (
    _jax_modules,
    load_quasimetric_checkpoint,
    make_quasimetric_embedder,
    provenance,
)
from graphcritic.bridges.models import prepare_puzzle_model_data
from graphcritic.bridges.windows import puzzle_state_windows
from graphcritic.io import load_sim_state, load_transitions
from graphcritic.direct_q import _make_action_head, action_qrl_loss_components


def _parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--state-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--epsilon-c1", type=float, default=1.25)
    parser.add_argument("--epsilon-c2", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--eval-batch-size", type=int, default=8_192)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--collapse-dead-frac", type=float, default=0.95)
    parser.add_argument("--collapse-min-var", type=float, default=1e-6)
    return parser.parse_args(argv)


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def flatten_state_windows(windows):
    """Flatten (N,W,F) windows in the same time-major C order as v0.8."""
    if windows.ndim == 2:
        return windows
    if windows.ndim != 3:
        raise ValueError("state windows must be (N,W,F) or flattened (N,W*F)")
    return windows.reshape((windows.shape[0], -1))


def embed_state_windows(embedder, state_params, windows):
    """Embed windows after applying the canonical v0.8 window flattening."""
    return embedder.apply(
        {"params": state_params}, flatten_state_windows(windows)
    )


def gather_endpoint_windows(all_windows, endpoint_rows):
    """Resolve endpoint row IDs to state windows for action-head training."""
    return all_windows[endpoint_rows]


def _build_data(args):
    trans, transition_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    data = prepare_puzzle_model_data(trans, sim_state, seed=0)
    frame = np.concatenate((data.state_cont, data.state_bits), axis=1).astype(np.float32)
    windows = puzzle_state_windows(frame, trans, data.terminal_state_index, window=args.window)
    starts = np.arange(trans.n_rows, dtype=np.int64)
    endpoint = starts + args.horizon
    valid = endpoint <= data.episode_end_rows
    # Exclude a chunk that includes a terminal action, as required by v0.9.
    valid &= np.logical_not(np.isin(starts, np.flatnonzero(trans.terminal)))
    rows = starts[valid]
    endpoints = endpoint[valid]
    train = np.isin(trans.episode_id[rows], data.train_episodes)
    heldout = np.isin(trans.episode_id[rows], data.heldout_episodes)
    return trans, transition_meta, data, windows, rows[train], endpoints[train], rows[heldout], endpoints[heldout]


def _make_train_step(head, embedder, raw_alpha, *, batch_size, eps1, eps2, horizon):
    jax, jnp, _, _, optax = _jax_modules()
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-4))
    dual_optimizer = optax.adam(1e-2)

    def distances(action_params, state_params, windows, actions, endpoints, goals):
        windows = flatten_state_windows(windows)
        endpoints = flatten_state_windows(endpoints)
        goals = flatten_state_windows(goals)
        action_input = jnp.concatenate((windows, actions.reshape((len(actions), -1))), axis=-1)
        phi = head.apply({"params": action_params}, action_input).reshape((-1, 64, 32))
        psi_end = embed_state_windows(embedder, state_params, endpoints).reshape((-1, 64, 32))
        psi_start = embed_state_windows(embedder, state_params, windows).reshape((-1, 64, 32))
        psi_goal = embed_state_windows(embedder, state_params, goals).reshape((-1, 64, 32))
        c1 = iqe_distance_from_embeddings(phi, psi_end, raw_alpha, n_components=64, component_dim=32, jnp=jnp)
        c2 = iqe_distance_from_embeddings(psi_start, phi, raw_alpha, n_components=64, component_dim=32, jnp=jnp)
        goal = iqe_distance_from_embeddings(phi, psi_goal, raw_alpha, n_components=64, component_dim=32, jnp=jnp)
        return c1, c2, goal

    from experiments.v08_runtime import iqe_distance_from_embeddings

    @jax.jit
    def update(params, opt_state, raw_lambdas, dual_state, key, all_windows,
               all_actions, all_endpoints, train_rows, state_params):
        key, k1, k2, k3 = jax.random.split(key, 4)
        current_rows = train_rows[jax.random.randint(k1, (batch_size,), 0, len(train_rows))]
        goal_rows = train_rows[jax.random.randint(k2, (batch_size,), 0, len(train_rows))]
        current = all_windows[current_rows]
        actions = all_actions[current_rows]
        endpoint = gather_endpoint_windows(all_windows, all_endpoints[current_rows])
        goal = all_windows[goal_rows]

        def loss_fn(action_params):
            c1, c2, d_goal = distances(action_params, state_params, current, actions, endpoint, goal)
            lambdas = jax.lax.stop_gradient(jax.nn.softplus(raw_lambdas))
            total, maximize, gap1, gap2 = action_qrl_loss_components(
                c1, c2, d_goal, lambdas[0], lambdas[1],
                horizon=horizon, epsilon_c1=eps1, epsilon_c2=eps2,
            )
            gaps = jnp.stack((gap1, gap2))
            return total, (c1, c2, gaps, maximize)

        (loss, (c1, c2, gaps, maximize)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        def dual_loss(raw):
            return -jnp.sum(jax.nn.softplus(raw) * jax.lax.stop_gradient(gaps))

        dual_grad = jax.grad(dual_loss)(raw_lambdas)
        dual_updates, dual_state = dual_optimizer.update(dual_grad, dual_state, raw_lambdas)
        raw_lambdas = optax.apply_updates(raw_lambdas, dual_updates)
        metrics = {
            "loss": loss, "distance_maximization_loss": maximize,
            "c1_mean": jnp.mean(c1), "c1_violation": jnp.mean(c1 > float(horizon) + eps1),
            "c2_mean": jnp.mean(c2), "c2_violation": jnp.mean(c2 > eps2),
            "lambda1": jax.nn.softplus(raw_lambdas[0]),
            "lambda2": jax.nn.softplus(raw_lambdas[1]),
            "grad_norm_preclip": grad_norm,
        }
        return params, opt_state, raw_lambdas, dual_state, key, metrics

    return update, optimizer, dual_optimizer, distances


def _evaluate_head(head, params, state_params, raw_alpha, embedder, data, windows,
                   rows, endpoints, actions, horizon, epsilon_c1, epsilon_c2, seed, batch_size):
    jax, jnp, _, _, _ = _jax_modules()
    from experiments.v08_runtime import iqe_distance_from_embeddings
    del seed
    if not len(rows):
        raise ValueError("held-out split has no valid five-action chunks")
    rng = np.random.default_rng(0xD1EC7)
    if len(rows) > 20_000:
        selected = rng.choice(len(rows), 20_000, replace=False)
        rows, endpoints = rows[selected], endpoints[selected]
    c1_all, c2_all = [], []
    offset_d = {k: [] for k in (5, 10, 15, 20)}
    predict_phi = jax.jit(lambda x: head.apply({"params": params}, x).reshape((-1, 64, 32)))
    predict_psi = jax.jit(lambda x: embedder.apply({"params": state_params}, x).reshape((-1, 64, 32)))
    for start in range(0, len(rows), batch_size):
        r = rows[start : start + batch_size]
        e = endpoints[start : start + batch_size]
        w = jnp.asarray(windows[r])
        a = jnp.asarray(actions[r].reshape(len(r), -1))
        phi = predict_phi(jnp.concatenate((w, a), axis=1))
        psi_start = predict_psi(w)
        psi_end = predict_psi(jnp.asarray(windows[e]))
        c1 = iqe_distance_from_embeddings(phi, psi_end, raw_alpha, n_components=64, component_dim=32, jnp=jnp)
        c2 = iqe_distance_from_embeddings(psi_start, phi, raw_alpha, n_components=64, component_dim=32, jnp=jnp)
        c1_all.append(np.asarray(jax.device_get(c1)))
        c2_all.append(np.asarray(jax.device_get(c2)))
        for k in offset_d:
            target = r + k
            ok = target <= data.episode_end_rows[r]
            positions = np.flatnonzero(ok)
            target_ids = target[positions]
            if len(target_ids):
                psi_k = predict_psi(jnp.asarray(windows[target_ids]))
                dk = iqe_distance_from_embeddings(phi[jnp.asarray(positions)], psi_k, raw_alpha, n_components=64, component_dim=32, jnp=jnp)
                offset_d[k].extend(np.asarray(jax.device_get(dk)).tolist())
    c1_values = np.concatenate(c1_all)
    c2_values = np.concatenate(c2_all)
    return {
        "heldout_chunks": int(len(rows)),
        "c1_violation_rate_d_gt_h_plus_epsilon": float(np.mean(c1_values > horizon + epsilon_c1)),
        "c2_violation_rate_d_gt_epsilon": float(np.mean(c2_values > epsilon_c2)),
        "c1_median": float(np.median(c1_values)),
        "c2_median": float(np.median(c2_values)),
        "distance_median_by_offset": {str(k): float(np.median(v)) if v else None for k, v in offset_d.items()},
        "c1_p95": float(np.quantile(c1_values, 0.95)),
    }


def _collapse_diagnostics(head, params, windows, actions, rows, *, seed,
                          dead_fraction_limit=0.95, min_output_variance=1e-6):
    jax, jnp, _, _, _ = _jax_modules()
    rng = np.random.default_rng(seed + 0xF3)
    selected = rng.choice(rows, min(2048, len(rows)), replace=False)
    input_values = np.concatenate(
        (windows[selected], actions[selected].reshape(len(selected), -1)), axis=1
    )
    output = head.apply({"params": params}, jnp.asarray(input_values), return_diagnostics=True)
    diagnostic = {name: np.asarray(jax.device_get(value)) for name, value in output["preactivations"].items()}
    output_embedding = np.asarray(jax.device_get(output["embedding"]))
    dead_by_layer = {
        name: float(np.mean(np.var(value, axis=0) < 1e-8))
        for name, value in diagnostic.items()
    }
    max_dead = max(dead_by_layer.values(), default=0.0)
    output_variance = float(np.var(output_embedding))
    alarm = max_dead > dead_fraction_limit or output_variance < min_output_variance
    return {
        "dead_unit_fraction_by_layer": dead_by_layer,
        "max_dead_unit_fraction": float(max_dead),
        "output_variance": output_variance,
        "collapse_alarm": bool(alarm),
    }


def main(argv=None):
    args = _parse(argv)
    if args.window != 5 or args.horizon != 5 or args.epsilon_c1 <= 0 or args.epsilon_c2 < 0:
        raise SystemExit("v0.9 fixes window and horizon to 5; epsilons must be non-negative")
    if min(args.steps, args.log_every, args.save_every, args.batch_size, args.eval_batch_size) <= 0:
        raise SystemExit("steps, intervals, and batch sizes must be positive")
    jax, jnp, _, serialization, optax = _jax_modules()
    trans, dataset_meta, data, windows, train_rows, train_endpoints, held_rows, held_endpoints = _build_data(args)
    if not len(train_rows) or not len(held_rows):
        raise ValueError("train and held-out episode splits both need valid chunks")
    state_model, state_params_all, state_meta = load_quasimetric_checkpoint(
        args.state_checkpoint, input_dim=windows.shape[1], expected_dataset_id=dataset_meta["dataset_id"]
    )
    if int(state_meta.get("window", -1)) != 5 or int(state_meta.get("constraint_horizon", -1)) != 5:
        raise ValueError("v0.9 requires the fixed window-5, horizon-5 state checkpoint")
    params_sha = hashlib.sha256((args.state_checkpoint / "params.msgpack").read_bytes()).hexdigest()
    if params_sha != state_meta.get("params_sha256"):
        raise ValueError("state checkpoint parameter hash does not match metadata")
    embedder = make_quasimetric_embedder(state_model)
    state_params = {k: v for k, v in state_params_all.items() if k != "raw_alpha"}
    experiment_version = "v1.0" if state_meta.get("spec_version") == "v1.0" else "v0.9"
    head = _make_action_head()
    input_dim = windows.shape[1] + args.horizon * trans.action.shape[1]
    endpoints_lookup = np.full(trans.n_rows, -1, dtype=np.int64)
    endpoints_lookup[train_rows] = train_endpoints
    endpoints_lookup[held_rows] = held_endpoints
    actions = np.zeros((trans.n_rows, args.horizon, trans.action.shape[1]), dtype=np.float32)
    for ep in np.unique(trans.episode_id):
        rows = np.flatnonzero(trans.episode_id == ep)
        starts = rows[: max(0, len(rows) - args.horizon)]
        if len(starts):
            action_windows = np.lib.stride_tricks.sliding_window_view(
                trans.action[rows], args.horizon, axis=0
            )
            action_windows = np.moveaxis(action_windows, -1, 1)
            actions[starts] = action_windows[: len(starts)]

    train_update, optimizer, dual_optimizer, _ = _make_train_step(
        head, embedder, state_params_all["raw_alpha"], batch_size=args.batch_size,
        eps1=args.epsilon_c1, eps2=args.epsilon_c2, horizon=args.horizon,
    )
    all_windows = jax.device_put(jnp.asarray(windows))
    all_actions = jax.device_put(jnp.asarray(actions))
    all_endpoints = jax.device_put(jnp.asarray(np.where(endpoints_lookup >= 0, endpoints_lookup, 0), dtype=jnp.int32))
    train_rows_jax = jax.device_put(jnp.asarray(train_rows, dtype=jnp.int32))
    state_params_jax = jax.device_put(state_params)
    args.out.mkdir(parents=True, exist_ok=True)
    hist = args.out / "training.jsonl"
    started = time.time()
    final_gate = {}
    for seed in args.seeds:
        seed_dir = args.out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        params = head.init(jax.random.PRNGKey(seed), jnp.zeros((1, input_dim), jnp.float32))["params"]
        opt_state = optimizer.init(params)
        raw_lambdas = jnp.asarray([_inverse_softplus(0.01)] * 2, dtype=jnp.float32)
        dual_state = dual_optimizer.init(raw_lambdas)
        key = jax.random.PRNGKey(seed + 0xA917)
        with hist.open("a", encoding="utf-8") as log:
            for step in range(1, args.steps + 1):
                params, opt_state, raw_lambdas, dual_state, key, metrics = train_update(
                    params, opt_state, raw_lambdas, dual_state, key,
                    all_windows, all_actions, all_endpoints, train_rows_jax, state_params_jax,
                )
                if step % args.log_every == 0 or step == args.steps:
                    values = {k: float(jax.device_get(v)) for k, v in metrics.items()}
                    record = {"seed": int(seed), "step": int(step), "metrics": values,
                              "elapsed_seconds": float(time.time() - started)}
                    log.write(json.dumps(record, sort_keys=True) + "\n")
                    log.flush()
                if step % args.save_every == 0 or step == args.steps:
                    path = seed_dir / f"step_{step:06d}"
                    path.mkdir(parents=True, exist_ok=True)
                    payload = serialization.to_bytes(params)
                    (path / "params.msgpack").write_bytes(payload)
                    collapse = _collapse_diagnostics(
                        head, params, windows, actions, held_rows,
                        seed=seed, dead_fraction_limit=args.collapse_dead_frac,
                        min_output_variance=args.collapse_min_var,
                    )
                    meta = {
                        **provenance(ROOT), "spec_version": experiment_version,
                        "model_kind": f"action-conditioned-quasimetric-{experiment_version}",
                        "dataset_id": dataset_meta["dataset_id"],
                        "state_checkpoint": str(args.state_checkpoint),
                        "state_checkpoint_sha256": params_sha,
                        "state_checkpoint_spec_version": state_meta.get("spec_version", "v0.8"),
                        "window": args.window, "horizon": args.horizon,
                        "action_dim": int(trans.action.shape[1]), "input_dim": int(input_dim),
                        "epsilon_c1": float(args.epsilon_c1), "epsilon_c2": float(args.epsilon_c2),
                        "encoder_widths": [1024, 1024, 256, 1024, 1024, 2048],
                        "activation": "gelu", "layernorm": True,
                        "n_components": 64, "component_dim": 32,
                        "iqe_head": "fixed-v0.8-maxmean", "seed": int(seed), "step": int(step),
                        "learning_rate": 5e-4, "dual": "adam", "dual_learning_rate": 1e-2,
                        "lambda1": float(jax.device_get(jax.nn.softplus(raw_lambdas[0]))),
                        "lambda2": float(jax.device_get(jax.nn.softplus(raw_lambdas[1]))),
                        "batch_size": args.batch_size,
                        "params_sha256": hashlib.sha256(payload).hexdigest(),
                        **collapse,
                    }
                    (path / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
                    heldout = _evaluate_head(
                        head, params, state_params_jax, state_params_all["raw_alpha"], embedder, data, windows,
                        held_rows, held_endpoints, actions, args.horizon,
                        args.epsilon_c1, args.epsilon_c2, seed, args.eval_batch_size,
                    )
                    (path / "heldout.json").write_text(json.dumps(heldout, indent=2, sort_keys=True) + "\n")
                    final_gate[str(seed)] = {**heldout, **collapse, "lambda1": meta["lambda1"], "lambda2": meta["lambda2"], "step": step}
                    if collapse["collapse_alarm"]:
                        summary = {
                            **provenance(ROOT), "spec_version": experiment_version,
                            "dataset_id": dataset_meta["dataset_id"],
                            "state_checkpoint_sha256": params_sha,
                            "heads": final_gate,
                            "stopped_on_collapse": True,
                            "elapsed_seconds": float(time.time() - started),
                        }
                        (args.out / "summary.json").write_text(
                            json.dumps(summary, indent=2, sort_keys=True) + "\n"
                        )
                        return 3
                    if step == args.steps and (
                        collapse["max_dead_unit_fraction"] > 0.0
                        or heldout["c1_violation_rate_d_gt_h_plus_epsilon"] > 0.05
                    ):
                        summary = {
                            **provenance(ROOT), "spec_version": experiment_version,
                            "dataset_id": dataset_meta["dataset_id"],
                            "state_checkpoint_sha256": params_sha,
                            "heads": final_gate,
                            "stopped_on_gate": True,
                            "elapsed_seconds": float(time.time() - started),
                        }
                        (args.out / "summary.json").write_text(
                            json.dumps(summary, indent=2, sort_keys=True) + "\n"
                        )
                        return 4
    summary = {
        **provenance(ROOT), "spec_version": experiment_version, "dataset_id": dataset_meta["dataset_id"],
        "state_checkpoint_sha256": params_sha, "heads": final_gate,
        "elapsed_seconds": float(time.time() - started),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

