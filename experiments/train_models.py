"""Train v0.7 goal-conditioned flow policy and five-step transition model."""

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
from experiments.v07_runtime import _flax_modules, save_checkpoint


def _endpoint_index(data, row: np.ndarray | int, offset: np.ndarray | int):
    rows = np.asarray(row, dtype=np.int64)
    offsets = np.asarray(offset, dtype=np.int64)
    ends = data.episode_end_rows[rows]
    current = rows + offsets
    clipped = current > ends
    if np.any(clipped):
        current = current.copy()
        current[clipped] = data.terminal_state_index[rows[clipped]]
        if np.any(current[clipped] < 0):
            raise ValueError("terminal endpoint missing from aligned sidecar")
    return current


def _eligible_starts(rows: np.ndarray, data, h: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    return rows[rows + h - 1 <= data.episode_end_rows[rows]]


def _report_metrics(data, f_model, f_params, pi_model, pi_params, *, step: int, seed: int):
    jax, jnp, _, _, _, _, _ = _flax_modules()
    h = 5
    heldout = _eligible_starts(data.heldout_rows, data, h)
    rng = np.random.default_rng(seed + step)
    n_cont = data.n_cont

    @jax.jit
    def f_apply(x):
        return f_model.apply({"params": f_params}, x)

    one_err, button_correct, changed_correct, exact = [], [], [], []
    for offset in range(0, len(heldout), 4096):
        ids = heldout[offset:offset + 4096]
        endpoint = _endpoint_index(data, ids, np.full(ids.shape, h, dtype=np.int64))
        action_ids = ids[:, None] + np.arange(h, dtype=np.int64)[None, :]
        actions = data.actions[action_ids].reshape(len(ids), 5 * h)
        inp = np.concatenate((data.state_cont[ids], data.state_bits[ids], actions), axis=1)
        pred = np.asarray(f_apply(jnp.asarray(inp)))
        pc = data.state_cont[ids] + pred[:, :n_cont] * (data.delta_std / data.state_std)
        truth = data.state_cont[endpoint]
        ratio = np.abs(pc - truth) * data.state_std / data.epsilon_raw
        pb = pred[:, n_cont:] > 0.0
        tb = data.state_bits[endpoint] > 0.5
        changed = tb != (data.state_bits[ids] > 0.5)
        one_err.append(ratio)
        button_correct.append(pb == tb)
        changed_correct.append((pb == tb)[changed])
        exact.append(np.all(pb == tb, axis=1))
    ratios = np.concatenate(one_err)
    bacc = np.concatenate(button_correct)
    cacc = np.concatenate(changed_correct) if any(x.size for x in changed_correct) else np.empty(0)
    metrics = {
        "step": int(step),
        "f_one_chunk": {
            "n": int(len(heldout)),
            "continuous_epsilon_ratio_mean": np.mean(ratios, axis=0).tolist(),
            "continuous_epsilon_ratio_p50": np.median(ratios, axis=0).tolist(),
            "continuous_epsilon_ratio_p90": np.quantile(ratios, .90, axis=0).tolist(),
            "button_accuracy": float(np.mean(bacc)),
            "changed_button_accuracy": float(np.mean(cacc)) if cacc.size else None,
            "exact_board_accuracy": float(np.mean(np.concatenate(exact))),
        },
        "f_rollout": {},
    }
    for length in range(1, 5):
        valid = data.heldout_rows[
            data.heldout_rows + length * h - 1 <= data.episode_end_rows[data.heldout_rows]
        ]
        if valid.size > 10_000:
            valid = np.sort(rng.choice(valid, 10_000, replace=False))
        if valid.size == 0:
            metrics["f_rollout"][str(length)] = {"n": 0}
            continue
        state_c = data.state_cont[valid].copy()
        state_b = data.state_bits[valid].copy()
        for chunk in range(length):
            action_ids = valid[:, None] + chunk * h + np.arange(h, dtype=np.int64)[None, :]
            acts = data.actions[action_ids]
            inp = np.concatenate((state_c, state_b, acts.reshape(len(valid), 5 * h)), axis=1)
            out = np.asarray(f_apply(jnp.asarray(inp)))
            state_c += out[:, :n_cont] * (data.delta_std / data.state_std)
            state_b = (out[:, n_cont:] > 0.0).astype(np.float32)
        endpoint = _endpoint_index(data, valid, np.full(valid.shape, length * h, dtype=np.int64))
        ratios_l = np.abs(state_c - data.state_cont[endpoint]) * data.state_std / data.epsilon_raw
        bit_acc = state_b == data.state_bits[endpoint]
        metrics["f_rollout"][str(length)] = {
            "n": int(len(valid)),
            "continuous_epsilon_ratio_mean": np.mean(ratios_l, axis=0).tolist(),
            "continuous_epsilon_ratio_p90": np.quantile(ratios_l, .90, axis=0).tolist(),
            "button_accuracy": float(np.mean(bit_acc)),
            "exact_board_accuracy": float(np.mean(np.all(bit_acc, axis=1))),
        }

    # π diagnostic: for each horizon, measure individual sampled endpoints on
    # held-out source/goal pairs, rather than an any-of-64 score.
    starts = _eligible_starts(data.heldout_rows, data, h)
    if starts.size > 64:
        starts = np.sort(rng.choice(starts, 64, replace=False))
    rates: dict[str, float | None] = {}
    if starts.size:
        n_samples = 64
        @jax.jit
        def pi_apply(x):
            return pi_model.apply({"params": pi_params}, x)
        @jax.jit
        def f_endpoint(x):
            return f_model.apply({"params": f_params}, x)
        for horizon in range(1, 21):
            target_ids = _endpoint_index(data, starts, np.minimum(horizon, data.episode_end_rows[starts] - starts + 1))
            s = np.repeat(np.concatenate((data.state_cont[starts], data.state_bits[starts]), axis=1), n_samples, axis=0)
            g = np.repeat(np.concatenate((data.state_cont[target_ids], data.state_bits[target_ids]), axis=1), n_samples, axis=0)
            key = jax.random.PRNGKey(seed + step * 31 + horizon)
            action_noise = np.asarray(jax.random.normal(key, (len(s), h * 5))).copy()
            for flow_step in range(10):
                tau = np.full((len(s), 1), flow_step / 10.0, dtype=np.float32)
                flow_input = np.concatenate((s, g, action_noise, tau), axis=1)
                action_noise += np.asarray(pi_apply(jnp.asarray(flow_input))) / 10.0
            action_noise = np.clip(action_noise, -1, 1)
            pred = np.asarray(f_endpoint(jnp.asarray(np.concatenate((s, action_noise), axis=1))))
            pc = s[:, :n_cont] + pred[:, :n_cont] * (data.delta_std / data.state_std)
            pb = pred[:, n_cont:] > 0.0
            error = np.max(np.abs(pc - g[:, :n_cont]) / data.epsilon_z, axis=1)
            good = (error <= 1.0) & np.all(pb == (g[:, n_cont:] > 0.5), axis=1)
            rates[str(horizon)] = float(np.mean(good))
    else:
        rates = {str(horizon): None for horizon in range(1, 21)}
    metrics["pi_heldout_endpoint_rate_by_j"] = rates
    metrics["pi_endpoint_eval_sampling"] = {
        "heldout_pairs_per_j": int(starts.size),
        "generated_chunks_per_pair": 64,
        "flow_euler_steps": 10,
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.steps <= 0 or args.save_every <= 0 or args.batch_size <= 0:
        parser.error("steps, save-every, and batch-size must be positive")

    jax, jnp, _, _, optax, MLP, TransitionMLP = _flax_modules()
    trans, trans_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    data = prepare_puzzle_model_data(trans, sim_state, seed=args.seed)
    h = 5
    train_starts = _eligible_starts(data.train_rows, data, h)
    valid_starts = _eligible_starts(data.heldout_rows, data, h)
    if not train_starts.size or not valid_starts.size:
        raise ValueError("episode split must contain complete five-action chunks in both splits")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data.json").write_text(json.dumps({
        **data.metadata, "dataset_id": trans_meta["dataset_id"],
        "n_train_chunks": int(train_starts.size), "n_heldout_chunks": int(valid_starts.size),
        "h": h, "L": 4, "N": 64, "flow_euler_steps": 10,
        "architecture": "4x512 GELU plus LayerNorm", "learning_rate": 3e-4,
        "batch_size": args.batch_size, "optimizer": "Adam",
    }, indent=2, sort_keys=True) + "\n")

    n_cont, n_bits = data.n_cont, data.n_bits
    state_dim = n_cont + n_bits
    pi_model = MLP(output_dim=h * 5)
    f_model = TransitionMLP(n_cont=n_cont, n_bits=n_bits)
    pi_key, f_key, step_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    pi_params = pi_model.init(pi_key, jnp.zeros((1, 2 * state_dim + h * 5 + 1), jnp.float32))["params"]
    f_params = f_model.init(f_key, jnp.zeros((1, state_dim + h * 5), jnp.float32))["params"]
    pi_opt = optax.adam(3e-4)
    f_opt = optax.adam(3e-4)
    pi_opt_state, f_opt_state = pi_opt.init(pi_params), f_opt.init(f_params)

    @jax.jit
    def pi_update(params, opt_state, start, goal, action, key):
        key_noise, key_tau = jax.random.split(key)
        a0 = jax.random.normal(key_noise, action.shape)
        tau = jax.random.uniform(key_tau, (action.shape[0], 1))
        x_tau = (1.0 - tau) * a0 + tau * action
        inp = jnp.concatenate((start, goal, x_tau, tau), axis=1)
        target = action - a0
        def loss_fn(p):
            pred = pi_model.apply({"params": p}, inp)
            return jnp.mean(jnp.square(pred - target))
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_state = pi_opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), next_state, loss

    @jax.jit
    def f_update(params, opt_state, start, action, delta, bits):
        inp = jnp.concatenate((start, action), axis=1)
        def loss_fn(p):
            pred = f_model.apply({"params": p}, inp)
            continuous = jnp.mean(jnp.square(pred[:, :n_cont] - delta))
            binary = jnp.mean(optax.sigmoid_binary_cross_entropy(pred[:, n_cont:], bits))
            return continuous + binary
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_state = f_opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), next_state, loss

    all_states = np.concatenate((data.state_cont, data.state_bits), axis=1)
    rng = np.random.default_rng(args.seed)
    hist_path = args.out / "training.jsonl"
    started = time.time()
    for step in range(1, args.steps + 1):
        ids = rng.choice(train_starts, size=args.batch_size, replace=True)
        # f uses every valid training start and the recorded five-action chunk.
        endpoint = _endpoint_index(data, ids, np.full(ids.shape, h, dtype=np.int64))
        action_ids = ids[:, None] + np.arange(h, dtype=np.int64)[None, :]
        action = data.actions[action_ids].reshape(args.batch_size, h * 5)
        raw_delta = data.state_raw[endpoint, :n_cont] - data.state_raw[ids, :n_cont]
        delta = raw_delta / data.delta_std
        f_params, f_opt_state, f_loss = f_update(
            f_params, f_opt_state, jnp.asarray(all_states[ids]), jnp.asarray(action),
            jnp.asarray(delta), jnp.asarray(data.state_bits[endpoint]),
        )

        js = rng.integers(1, 21, size=args.batch_size)
        ends = data.episode_end_rows[ids]
        max_j = ends - ids + 1
        js = np.minimum(js, max_j)
        goals_idx = _endpoint_index(data, ids, js)
        pi_params, pi_opt_state, pi_loss = pi_update(
            pi_params, pi_opt_state, jnp.asarray(all_states[ids]),
            jnp.asarray(all_states[goals_idx]), jnp.asarray(action), step_key,
        )
        step_key, _ = jax.random.split(step_key)

        if step % args.save_every == 0 or step == args.steps:
            checkpoint = args.out / f"step_{step:06d}"
            metadata = {"dataset_id": trans_meta["dataset_id"], "seed": args.seed,
                        "h": h, "n_cont": n_cont, "n_bits": n_bits,
                        "state_dim": state_dim, "epsilon_z": data.epsilon_z.tolist(),
                        "state_std": data.state_std.tolist(), "delta_std": data.delta_std.tolist(),
                        "data_split_seed": args.seed, "architecture": "4x512 GELU LayerNorm"}
            save_checkpoint(checkpoint / "pi", pi_params, step, {**metadata, "model": "goal_flow_matching"})
            save_checkpoint(checkpoint / "f", f_params, step, {**metadata, "model": "five_step_transition"})
            eval_metrics = _report_metrics(data, f_model, f_params, pi_model, pi_params,
                                           step=step, seed=args.seed)
            eval_metrics["train"] = {"pi_loss": float(pi_loss), "f_loss": float(f_loss)}
            eval_metrics["elapsed_seconds"] = float(time.time() - started)
            with hist_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(eval_metrics, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps({"step": step, "pi_loss": float(pi_loss), "f_loss": float(f_loss),
                              "f_endpoint_exact_board": eval_metrics["f_one_chunk"]["exact_board_accuracy"],
                              "pi_j5_rate": eval_metrics["pi_heldout_endpoint_rate_by_j"].get("5"),
                              "checkpoint": str(checkpoint)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
