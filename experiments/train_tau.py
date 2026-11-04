"""Train and calibrate the bridge-only frame-gap classifier (no TD targets)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import rankdata, spearmanr

from graphcritic.config import GraphConfig
from graphcritic.featurize import StateFeaturizer
from graphcritic.io import load_sim_state, load_transitions
from graphcritic.modes import resolve_state_modes
from graphcritic.tau_model import FrameGapModel
from graphcritic.types import SchemaError, SimStateSet, TransitionSet


@dataclass
class PairSampler:
    trans: TransitionSet
    modes: np.ndarray
    phi: np.ndarray
    episode_split: dict[int, str]
    horizon: int
    distance_threshold: float
    rng: np.random.Generator

    def __post_init__(self):
        n = self.trans.n_rows
        idx = np.arange(n, dtype=np.int64)
        valid_step = np.zeros(n, dtype=np.bool_)
        if n > 1:
            valid_step[:-1] = (
                ~self.trans.terminal[:-1]
                & (self.trans.episode_id[:-1] == self.trans.episode_id[1:])
                & (self.modes[: n - 1] == self.modes[1:n])
            )
        max_delta = np.zeros(n, dtype=np.int64)
        alive = np.ones(n, dtype=np.bool_)
        for delta in range(1, self.horizon + 1):
            move = np.zeros(n, dtype=np.bool_)
            if delta <= n:
                move[: n - delta + 1] = valid_step[delta - 1 : n]
            alive &= move
            max_delta[alive] = delta
        self.max_delta = max_delta
        episode_starts = np.r_[
            0, np.flatnonzero(self.trans.episode_id[1:] != self.trans.episode_id[:-1]) + 1
        ].astype(np.int64, copy=False)
        episode_ends = np.r_[episode_starts[1:], n]
        self.episode_rows = {
            int(self.trans.episode_id[start]): np.arange(start, end, dtype=np.int64)
            for start, end in zip(episode_starts, episode_ends, strict=True)
        }
        self.episode_end_by_row = np.empty(n, dtype=np.int64)
        for ep, episode_rows in self.episode_rows.items():
            self.episode_end_by_row[episode_rows] = episode_rows[-1]
        self.split_rows: dict[str, np.ndarray] = {}
        for split in ("train", "calibration", "test"):
            eps = [ep for ep, label in self.episode_split.items() if label == split]
            self.split_rows[split] = np.flatnonzero(np.isin(self.trans.episode_id, eps)).astype(np.int64)
        self.positive_starts = {
            split: rows[self.max_delta[rows] > 0]
            for split, rows in self.split_rows.items()
        }
        self.positive_starts_by_delta = {
            split: {
                delta: rows[self.max_delta[rows] >= delta]
                for delta in range(1, self.horizon + 1)
            }
            for split, rows in self.split_rows.items()
        }
        self.long_starts = {
            split: rows[self.episode_end_by_row[rows] - rows > self.horizon]
            for split, rows in self.split_rows.items()
        }

    def _sample_positive(self, split: str, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        by_delta = self.positive_starts_by_delta[split]
        deltas = np.asarray([delta for delta, rows in by_delta.items() if rows.size], dtype=np.int64)
        if deltas.size == 0:
            raise SchemaError(f"split {split!r} has no valid within-trajectory near pairs")
        # Sample the frame gap uniformly first, then sample a valid start for
        # that gap. This avoids over-representing short gaps from long episodes.
        delta = self.rng.choice(deltas, size=count, replace=True)
        u = np.empty(count, dtype=np.int64)
        for value in deltas:
            positions = np.flatnonzero(delta == value)
            u[positions] = self.rng.choice(by_delta[int(value)], size=positions.size, replace=True)
        v = u + delta
        return u, v, delta

    def _sample_cross_far(
        self, split: str, count: int, *, threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = self.split_rows[split]
        if np.unique(self.trans.episode_id[rows]).size < 2:
            raise SchemaError(f"split {split!r} needs at least two episodes for cross-trajectory pairs")
        cutoff = self.distance_threshold if threshold is None else threshold
        us: list[np.ndarray] = []
        vs: list[np.ndarray] = []
        remaining = count
        attempts = 0
        while remaining and attempts < 100:
            batch = max(remaining * 2, 512)
            u = self.rng.choice(rows, size=batch, replace=True)
            v = self.rng.choice(rows, size=batch, replace=True)
            d = np.linalg.norm(self.phi[u] - self.phi[v], axis=1)
            keep = (self.trans.episode_id[u] != self.trans.episode_id[v]) & (d > cutoff)
            if np.any(keep):
                take = np.flatnonzero(keep)[:remaining]
                us.append(u[take])
                vs.append(v[take])
                remaining -= take.size
            attempts += 1
        if remaining:
            raise SchemaError(
                f"could only sample {count - remaining}/{count} cross-episode pairs beyond distance {cutoff:.4g}"
            )
        return np.concatenate(us), np.concatenate(vs)

    def _sample_long_far(self, split: str, count: int) -> tuple[np.ndarray, np.ndarray]:
        starts = self.long_starts[split]
        if starts.size == 0:
            raise SchemaError(f"split {split!r} has no within-trajectory pairs beyond H={self.horizon}")
        us: list[np.ndarray] = []
        vs: list[np.ndarray] = []
        remaining = count
        attempts = 0
        while remaining and attempts < 100:
            batch = max(remaining * 3, 512)
            u = self.rng.choice(starts, size=batch, replace=True)
            ends = self.episode_end_by_row[u]
            min_delta = self.horizon + 1
            delta = self.rng.integers(min_delta, ends - u + 1)
            v = u + delta
            d = np.linalg.norm(self.phi[u] - self.phi[v], axis=1)
            keep = d > self.distance_threshold
            if np.any(keep):
                take = np.flatnonzero(keep)[:remaining]
                us.append(u[take])
                vs.append(v[take])
                remaining -= take.size
            attempts += 1
        if remaining:
            raise SchemaError(
                f"could only sample {count - remaining}/{count} within-trajectory far pairs"
            )
        return np.concatenate(us), np.concatenate(vs)

    def sample(self, split: str, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_positive = count // 2
        n_cross = (3 * count) // 8
        n_long = count - n_positive - n_cross
        u_pos, v_pos, d_pos = self._sample_positive(split, n_positive)
        u_cross, v_cross = self._sample_cross_far(split, n_cross)
        u_long, v_long = self._sample_long_far(split, n_long)
        u = np.concatenate((u_pos, u_cross, u_long))
        v = np.concatenate((v_pos, v_cross, v_long))
        labels = np.concatenate((d_pos - 1, np.full(n_cross + n_long, self.horizon)))
        order = self.rng.permutation(u.size)
        return u[order], v[order], labels[order].astype(np.int32)

    def sample_near(self, split: str, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u, v, delta = self._sample_positive(split, count)
        order = self.rng.permutation(u.size)
        return u[order], v[order], (delta[order] - 1).astype(np.int32)


def _episode_splits(trans: TransitionSet, seed: int) -> dict[int, str]:
    eps = np.unique(trans.episode_id)
    rng = np.random.default_rng(seed)
    eps = rng.permutation(eps)
    if eps.size < 3:
        raise SchemaError("frame-gap training requires at least three episodes for 80/10/10 splits")
    n_train = min(eps.size - 2, max(1, int(np.floor(0.8 * eps.size))))
    n_cal = min(eps.size - n_train - 1, max(1, int(np.floor(0.1 * eps.size))))
    result = {int(ep): "train" for ep in eps[:n_train]}
    result.update({int(ep): "calibration" for ep in eps[n_train : n_train + n_cal]})
    result.update({int(ep): "test" for ep in eps[n_train + n_cal :]})
    return result


def _r_h(phi: np.ndarray, starts: np.ndarray, horizon: int, seed: int) -> float:
    eligible = starts
    if eligible.size == 0:
        raise SchemaError(f"no same-trajectory pairs exist at Δ=H={horizon}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=min(100_000, eligible.size), replace=False)
    return float(np.quantile(np.linalg.norm(phi[chosen] - phi[chosen + horizon], axis=1), 0.99))


def _init_params(input_dim: int, horizon: int, seed: int, pair_delta_indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pair_dim = 2 * input_dim + (input_dim if pair_delta_indices is None else len(pair_delta_indices))
    dims = [pair_dim, 512, 512, 512, horizon + 1]
    params: dict[str, np.ndarray] = {}
    for layer in range(4):
        fan_in, fan_out = dims[layer], dims[layer + 1]
        scale = np.sqrt(2.0 / fan_in) if layer < 3 else np.sqrt(1.0 / fan_in)
        params[f"w{layer}"] = rng.normal(0.0, scale, (fan_in, fan_out)).astype(np.float32)
        params[f"b{layer}"] = np.zeros(fan_out, dtype=np.float32)
        if layer < 3:
            params[f"ln_scale{layer}"] = np.ones(fan_out, dtype=np.float32)
            params[f"ln_bias{layer}"] = np.zeros(fan_out, dtype=np.float32)
    return params


def _prepare_pairs(
    u: np.ndarray,
    v: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    pair_delta_indices: np.ndarray | None = None,
) -> np.ndarray:
    zu = (u.astype(np.float32) - mean) / std
    zv = (v.astype(np.float32) - mean) / std
    indices = np.arange(zu.shape[1], dtype=np.int64) if pair_delta_indices is None else np.asarray(pair_delta_indices, dtype=np.int64)
    return np.concatenate((zu, zv, (zv - zu)[:, indices]), axis=1).astype(np.float32, copy=False)


def _episode_bootstrap_interval(
    covered: np.ndarray,
    episode_ids: np.ndarray,
    *,
    seed: int,
    replicates: int = 1000,
) -> dict[str, float | int | None]:
    """Cluster bootstrap by source episode to preserve within-episode dependence."""
    values = np.asarray(covered, dtype=np.float64)
    groups = np.asarray(episode_ids, dtype=np.int64)
    if values.ndim != 1 or groups.shape != values.shape or values.size == 0:
        return {"n_pairs": int(values.size), "n_episodes": int(np.unique(groups).size), "low": None, "high": None, "width": None}
    unique = np.unique(groups)
    by_episode = [values[groups == episode] for episode in unique]
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for b in range(replicates):
        sample = rng.integers(0, len(unique), size=len(unique))
        numerator = sum(float(np.sum(by_episode[i])) for i in sample)
        denominator = sum(int(by_episode[i].size) for i in sample)
        estimates[b] = numerator / max(denominator, 1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "n_pairs": int(values.size),
        "n_episodes": int(unique.size),
        "low": float(low),
        "high": float(high),
        "width": float(high - low),
        "replicates": int(replicates),
    }


def _calibrate_quantile(probabilities: np.ndarray, labels: np.ndarray, nominal: float) -> tuple[float, float, float]:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        raise SchemaError("quantile calibration requires at least one near pair")
    if probs.ndim != 2 or probs.shape[0] != labels.size or probs.shape[1] < 2:
        raise SchemaError("quantile calibration probabilities and labels are misaligned")
    horizon = probs.shape[1] - 1
    if np.any(labels < 0) or np.any(labels >= horizon):
        raise SchemaError("quantile calibration accepts only near-pair labels in [0, H)")
    # labels are zero-based (Δ=1 has label 0).  An upper time quantile covers
    # the observed gap exactly when its threshold has not already been crossed
    # before Δ: F(Δ-1) < q.  Scoring F(Δ) reverses this event and calibrates
    # underestimation instead of the requested upper bound.
    cdf = np.cumsum(probs[:, :-1], axis=1)
    scores = np.zeros(labels.size, dtype=np.float64)
    positive = labels > 0
    rows = np.arange(labels.size, dtype=np.int64)
    scores[positive] = cdf[rows[positive], labels[positive] - 1]
    target_count = min(labels.size, int(np.ceil((labels.size + 1) * nominal)))
    nominal_coverage = float(np.mean(scores < nominal))
    if abs(nominal_coverage - nominal) < 0.02:
        return float(nominal), nominal_coverage, float(target_count / labels.size)
    # Split-conformal order statistic: the next representable float includes
    # ties at the target score under the strict coverage condition score < q'.
    q_prime = float(np.nextafter(np.partition(scores, target_count - 1)[target_count - 1], np.inf))
    q_prime = min(q_prime, float(np.nextafter(1.0, 0.0)))
    actual = float(np.mean(scores < q_prime))
    return q_prime, actual, float(target_count / labels.size)


def _roc_auc(pos_score: np.ndarray, neg_score: np.ndarray) -> float | None:
    if pos_score.size == 0 or neg_score.size == 0:
        return None
    score = np.concatenate((pos_score, neg_score))
    ranks = rankdata(score, method="average")
    n_pos = pos_score.size
    n_neg = neg_score.size
    return float((np.sum(ranks[:n_pos]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _jax_train(
    sampler: PairSampler,
    raw_states: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    horizon: int,
    steps: int,
    batch_size: int,
    seed: int,
    eval_pairs: tuple[np.ndarray, np.ndarray, np.ndarray],
    eval_every: int,
    patience: int,
    pair_delta_indices: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as exc:  # pragma: no cover - bench-only workflow
        raise RuntimeError("train_tau requires JAX and Optax in the bench environment") from exc
    params_np = _init_params(raw_states.shape[1], horizon, seed, pair_delta_indices)
    params = jax.tree_util.tree_map(jnp.asarray, params_np)
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)

    def forward(p, x):
        for layer in range(3):
            x = x @ p[f"w{layer}"] + p[f"b{layer}"]
            x = (x - jnp.mean(x, axis=1, keepdims=True)) * jax.lax.rsqrt(
                jnp.mean(jnp.square(x - jnp.mean(x, axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-5
            )
            x = jax.nn.relu(x * p[f"ln_scale{layer}"] + p[f"ln_bias{layer}"])
        return x @ p["w3"] + p["b3"]

    @jax.jit
    def train_step(p, state, x, labels):
        def objective(current):
            logits = forward(current, x)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        loss, grads = jax.value_and_grad(objective)(p)
        updates, state = optimizer.update(grads, state, p)
        return optax.apply_updates(p, updates), state, loss

    eval_u, eval_v, eval_y = eval_pairs
    eval_x = jnp.asarray(_prepare_pairs(raw_states[eval_u], raw_states[eval_v], mean, std, pair_delta_indices))
    eval_y_jax = jnp.asarray(eval_y, dtype=jnp.int32)
    best_params = params
    best_loss = float("inf")
    best_step = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    device = str(jax.default_backend())
    for step in range(1, steps + 1):
        u, v, labels = sampler.sample("train", batch_size)
        x = jnp.asarray(_prepare_pairs(raw_states[u], raw_states[v], mean, std, pair_delta_indices))
        params, opt_state, loss = train_step(
            params, opt_state, x, jnp.asarray(labels, dtype=jnp.int32)
        )
        if step % eval_every == 0 or step == 1:
            eval_loss = float(jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                forward(params, eval_x), eval_y_jax
            )))
            history.append({"step": step, "train_loss": float(loss), "calibration_loss": eval_loss})
            if step % max(eval_every * 10, 1000) == 0:
                print(
                    f"tau training step={step} train_loss={float(loss):.4f} "
                    f"calibration_loss={eval_loss:.4f}",
                    file=sys.stderr, flush=True,
                )
            if eval_loss < best_loss - 1e-6:
                best_loss, best_step, best_params, stale = eval_loss, step, params, 0
            else:
                stale += eval_every
            if stale >= patience:
                break
    best_np = {key: np.asarray(value, dtype=np.float32) for key, value in best_params.items()}
    return best_np, {
        "optimizer": "Adam",
        "learning_rate": 3e-4,
        "batch_size": batch_size,
        "requested_steps": steps,
        "steps_run": step,
        "early_stopping_patience": patience,
        "best_step": best_step,
        "best_calibration_loss": best_loss,
        "backend": device,
        "history": history,
    }


def _predict(model: FrameGapModel, u: np.ndarray, v: np.ndarray, raw_states: np.ndarray) -> np.ndarray:
    result = []
    for start in range(0, len(u), 8192):
        end = min(start + 8192, len(u))
        result.append(model.predict_proba_pairs(raw_states[u[start:end]], raw_states[v[start:end]], backend="auto"))
    return np.concatenate(result, axis=0) if result else np.empty((0, model.horizon + 1), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ogbench-dataset")
    src.add_argument("--transitions", type=Path)
    parser.add_argument("--sim-state", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10_000)
    parser.add_argument("--calibration-pairs", type=int, default=20_000)
    parser.add_argument("--test-pairs", type=int, default=20_000)
    parser.add_argument("--far-exclusion-radius", choices=("r", "r_H"), default="r")
    parser.add_argument("--state-schema", choices=("none", "ogbench_scene", "robomimic"), default="none",
                        help="Train on the physical [r,o,e] StateSchema representation")
    parser.add_argument("--quantiles", type=float, nargs="+", default=(0.8, 0.9))
    args = parser.parse_args()
    if min(args.horizon, args.steps, args.batch_size, args.eval_every,
           args.patience, args.calibration_pairs, args.test_pairs) < 1:
        parser.error("horizon, steps, batch size, eval interval, patience, and pair counts must be positive")
    if any(not np.isfinite(q) or not 0 < q < 1 for q in args.quantiles):
        parser.error("quantiles must lie strictly between zero and one")

    cfg = GraphConfig.from_yaml(args.config)
    if args.ogbench_dataset:
        from envs.ogbench import load_ogbench_transitions
        trans, dataset_id, sim_state = load_ogbench_transitions(
            args.ogbench_dataset, return_sim_state=True
        )
    else:
        trans, transition_meta = load_transitions(args.transitions)
        dataset_id = transition_meta["dataset_id"]
        sim_state = load_sim_state(args.sim_state) if args.sim_state else None
    states = np.asarray(trans.feat, dtype=np.float32)
    if cfg.feature.name == "ogbench_scene_pos" and sim_state is None:
        raise SchemaError("Scene tau training with position features requires --sim-state")
    featurizer = StateFeaturizer.fit(trans, cfg.feature, sim_state=sim_state)
    feature_aux = None
    if sim_state is not None:
        feature_aux = {"qpos": sim_state.qpos, "qvel": sim_state.qvel}
        if sim_state.button_states is not None:
            feature_aux["button_states"] = sim_state.button_states
    state_values = np.concatenate(
        (trans.feat, trans.next_feat[np.flatnonzero(trans.terminal)]), axis=0
    )
    all_state_features = featurizer.transform(state_values, aux=feature_aux)
    state_schema_name = args.state_schema
    if state_schema_name == "none" and cfg.bridge.strategy == "passive_rm":
        state_schema_name = cfg.bridge.state_schema or {
            "ogbench_scene_pos": "ogbench_scene", "robomimic_pos": "robomimic"
        }.get(cfg.feature.name, "none")
    pair_delta_indices = None
    model_states = state_values
    if state_schema_name != "none":
        from graphcritic.state_schema import get_state_schema

        if state_schema_name == "ogbench_scene" and sim_state is None:
            raise SchemaError("ogbench_scene StateSchema training requires qpos/button sidecar data")
        schema = get_state_schema(state_schema_name)
        if sim_state is not None:
            sim_state.validate(state_values.shape[0])
        coords = schema.coordinates(state_values, feature_aux)
        model_states = np.concatenate((coords.r, coords.o, coords.e), axis=1).astype(np.float32)
        pair_delta_indices = np.arange(coords.r.shape[1], model_states.shape[1], dtype=np.int64)
    modes = resolve_state_modes(trans, cfg.mode_fn, sim_state=sim_state, min_dwell=cfg.min_dwell)
    mode_move = modes[: trans.n_rows - 1] == modes[1 : trans.n_rows]
    allowed_starts = np.flatnonzero(
        (~trans.terminal[:-1])
        & (trans.episode_id[:-1] == trans.episode_id[1:])
        & mode_move
    )
    dstep_rows = allowed_starts
    dstep = np.linalg.norm(
        all_state_features[dstep_rows + 1] - all_state_features[dstep_rows], axis=1
    )
    dstep = dstep[np.isfinite(dstep) & (dstep > 1e-8)]
    d_step = float(np.median(dstep)) if dstep.size else 1.0
    radius = cfg.bridge.radius * d_step if cfg.bridge.radius_unit == "step" else cfg.bridge.radius
    episode_split = _episode_splits(trans, args.seed)
    modes_current = modes[: trans.n_rows]
    phi = all_state_features[: trans.n_rows]
    rng = np.random.default_rng(args.seed + 1)
    sampler = PairSampler(
        trans, modes, phi, episode_split, args.horizon, radius, rng
    )
    valid_h_starts = sampler.positive_starts["train"]
    valid_h_starts = valid_h_starts[sampler.max_delta[valid_h_starts] >= args.horizon]
    r_h = _r_h(phi, valid_h_starts, args.horizon, args.seed)
    far_threshold = radius if args.far_exclusion_radius == "r" else r_h
    sampler.distance_threshold = far_threshold
    train_rows = sampler.split_rows["train"]
    train_obs = model_states[train_rows]
    obs_mean = np.mean(train_obs.astype(np.float64), axis=0).astype(np.float32)
    obs_std = np.maximum(np.std(train_obs.astype(np.float64), axis=0), 1e-6).astype(np.float32)

    eval_u, eval_v, eval_y = sampler.sample("calibration", args.calibration_pairs)
    params, train_meta = _jax_train(
        sampler, model_states, obs_mean, obs_std, horizon=args.horizon, steps=args.steps,
        batch_size=args.batch_size, seed=args.seed, eval_pairs=(eval_u, eval_v, eval_y),
        eval_every=args.eval_every, patience=args.patience,
        pair_delta_indices=pair_delta_indices,
    )
    model_meta: dict[str, Any] = {
        "model": "frame_gap_mlp_v1",
        "horizon": int(args.horizon),
        "input": "standardized_r_o_e_u_v_delta_o_e" if pair_delta_indices is not None else "standardized_xu_xv_xv_minus_xu",
        "observation_dim": int(model_states.shape[1]),
        "state_schema": state_schema_name if state_schema_name != "none" else None,
        "pair_delta_indices": None if pair_delta_indices is None else pair_delta_indices.tolist(),
        "pair_input_layout": (
            "[r_u,o_u,e_u,r_v,o_v,e_v,delta_o,delta_e]"
            if pair_delta_indices is not None else "[x_u,x_v,delta_x]"
        ),
        "dataset_id": dataset_id,
        "split_seed": int(args.seed),
        "episode_split_counts": {
            split: int(sum(label == split for label in episode_split.values()))
            for split in ("train", "calibration", "test")
        },
        "episode_splits": {str(episode): split for episode, split in episode_split.items()},
        "feature": featurizer.meta(),
        "d_pos": "euclidean standardized position features with group inverse-sqrt scaling",
        "d_step": d_step,
        "bridge_radius": radius,
        "r_H": r_h,
        "far_exclusion_radius": args.far_exclusion_radius,
        "far_exclusion_radius_value": float(far_threshold),
        "positive_gap_sampling": "uniform_delta_then_uniform_valid_start",
        "training": train_meta,
        "calibrated_quantiles": {},
    }
    model = FrameGapModel(params, obs_mean, obs_std, args.horizon, model_meta)
    cal_u, cal_v, cal_y = sampler.sample_near("calibration", args.calibration_pairs)
    cal_prob = _predict(model, cal_u, cal_v, model_states)
    calibration_metrics = {}
    for q in args.quantiles:
        qprime, coverage, target = _calibrate_quantile(cal_prob, cal_y, float(q))
        calibration_metrics[f"{q:.6g}"] = {
            "q_prime": qprime,
            "calibration_coverage": coverage,
            "target_coverage": target,
        }
        model_meta["calibrated_quantiles"][f"{q:.6g}"] = qprime

    test_u, test_v, test_y = sampler.sample("test", args.test_pairs)
    test_prob = _predict(model, test_u, test_v, model_states)
    median_tau, _ = model.predict_tau(test_prob, 0.5)
    near_mask = test_y < args.horizon
    with np.errstate(invalid="ignore"):
        rho_value = float(spearmanr(median_tau[near_mask], test_y[near_mask] + 1).statistic)
        rho_spearman = rho_value if np.isfinite(rho_value) else None
    q_metrics = {}
    near_episodes = trans.episode_id[test_u[near_mask]]
    for q in args.quantiles:
        qprime = model.calibrated_quantile(float(q))
        tau_q, f_h = model.predict_tau(test_prob[near_mask], float(q))
        covered = (tau_q <= args.horizon) & ((test_y[near_mask] + 1) <= tau_q)
        coverage = float(np.mean(covered))
        interval = _episode_bootstrap_interval(
            covered, near_episodes, seed=args.seed + int(round(float(q) * 1000))
        )
        q_metrics[f"{q:.6g}"] = {
            "q_prime": qprime,
            "test_coverage": coverage,
            "episode_bootstrap_95": interval,
            "bootstrap_gate": bool(interval["low"] is not None and interval["low"] <= q <= interval["high"]),
        }
    cross_u, cross_v = sampler._sample_cross_far("test", args.test_pairs, threshold=radius)
    cross_prob = _predict(model, cross_u, cross_v, model_states)
    f_h_cross = np.sum(cross_prob[:, : args.horizon], axis=1)
    auc = _roc_auc(test_prob[near_mask, : args.horizon].sum(axis=1), f_h_cross)

    conflict_u, conflict_v = sampler._sample_cross_far("test", args.test_pairs, threshold=radius)
    conflict_dist = np.linalg.norm(phi[conflict_u] - phi[conflict_v], axis=1)
    conflict_mask = conflict_dist <= r_h
    conflict_prob = _predict(model, conflict_u[conflict_mask], conflict_v[conflict_mask], model_states)
    conflict_near_rate = (
        float(np.mean(conflict_prob[:, : args.horizon].sum(axis=1) > 0.5))
        if conflict_prob.shape[0] else None
    )
    metrics = {
        "test_spearman_median_tau_delta": rho_spearman,
        "test_coverage": q_metrics,
        "test_far_auroc_f_h": auc,
        "label_conflict": {
            "r": {
                "pairs_labeled_far": int(conflict_dist.size),
                "collision_pairs_in_r_to_r_H": int(np.count_nonzero(conflict_mask)),
                "collision_predicted_near_fraction": conflict_near_rate,
            },
            "r_H": {
                "pairs_excluded_from_far_label": int(np.count_nonzero(conflict_mask)),
                "collision_predicted_near_fraction": conflict_near_rate,
            },
        },
        "calibration": calibration_metrics,
        "test_pairs": int(args.test_pairs),
    }
    model_meta["metrics"] = metrics
    model = FrameGapModel(params, obs_mean, obs_std, args.horizon, model_meta)
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "tau_model.npz")
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"model": str(args.out / "tau_model.npz"), "metrics": metrics}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
