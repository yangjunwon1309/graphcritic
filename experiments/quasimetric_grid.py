"""Train and evaluate the QRL quasimetric on a small exact grid."""

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
from scipy.stats import spearmanr

from experiments.v08_runtime import (
    initialize_train_state,
    make_distance_predictor,
    make_quasimetric_model,
    make_train_step,
)


def train_grid_quasimetric(*, side: int = 5, steps: int = 6000, seed: int = 0) -> dict:
    """Fit the same QRL objective on every directed edge of an open square grid."""
    import jax
    import jax.numpy as jnp

    if side < 2 or steps < 1:
        raise ValueError("side must be at least two and steps must be positive")
    coordinates = np.asarray(
        [(x, y) for y in range(side) for x in range(side)], dtype=np.float32
    )
    # Match the production input convention: normalize continuous coordinates
    # using training-set statistics, and leave any discrete values unscaled.
    coordinates = (coordinates - coordinates.mean(axis=0)) / np.maximum(
        coordinates.std(axis=0), 1e-6
    )
    state_rows = []
    destination_vertices = []
    for y in range(side):
        for x in range(side):
            source = y * side + x
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < side and 0 <= ny < side:
                    state_rows.append(source)
                    destination_vertices.append(ny * side + nx)
    state_rows = np.asarray(state_rows, dtype=np.int32)
    destination_vertices = np.asarray(destination_vertices, dtype=np.int32)
    n_transitions = len(state_rows)
    # Transition rows index their source occurrence. Append one canonical state
    # per vertex so each one-step endpoint can refer to its destination exactly.
    states_table = np.concatenate((coordinates[state_rows], coordinates), axis=0)
    next_ids = n_transitions + destination_vertices
    train_rows = np.arange(n_transitions, dtype=np.int32)

    # Narrow dimensions keep this a fast structural learning test while using
    # exactly the production model and primal/dual optimizer implementation.
    model = make_quasimetric_model(
        encoder_widths=(64, 32), projector_widths=(64, 64, 64),
        n_components=8, component_dim=8,
    )
    batch_size = 128
    update, primal_optimizer, dual_optimizer = make_train_step(
        model, batch_size=batch_size, epsilon=0.25,
        learning_rate=5e-4, lambda_learning_rate=1e-2,
    )
    params, primal_state, raw_lambda, dual_state, key = initialize_train_state(
        model, seed=seed, input_dim=2,
        primal_optimizer=primal_optimizer, dual_optimizer=dual_optimizer,
    )
    states = jnp.asarray(states_table)
    state_rows_jax = jnp.asarray(train_rows)
    next_ids_jax = jnp.asarray(next_ids)
    final_metrics = {}
    started = time.time()
    for _ in range(steps):
        params, primal_state, raw_lambda, dual_state, key, final_metrics = update(
            params, primal_state, raw_lambda, dual_state, key,
            states, next_ids_jax, state_rows_jax,
        )
    predict = make_distance_predictor(model, params, batch_size=256)
    count = side * side
    source_ids, target_ids = np.meshgrid(np.arange(count), np.arange(count), indexing="ij")
    predicted = predict(
        coordinates[source_ids.reshape(-1)], coordinates[target_ids.reshape(-1)]
    ).reshape(count, count)
    raw_positions = np.asarray(
        [(x, y) for y in range(side) for x in range(side)], dtype=np.int32
    )
    exact = np.abs(raw_positions[:, None, :] - raw_positions[None, :, :]).sum(axis=-1)
    off_diagonal = ~np.eye(count, dtype=np.bool_)
    report = {
        "environment": "open_square_grid",
        "side": int(side),
        "n_states": int(count),
        "n_directed_transitions": int(len(state_rows)),
        "steps": int(steps),
        "seed": int(seed),
        "device_backend": jax.default_backend(),
        "distance_mae_steps": float(np.mean(np.abs(predicted[off_diagonal] - exact[off_diagonal]))),
        "distance_rmse_steps": float(np.sqrt(np.mean(np.square(
            predicted[off_diagonal] - exact[off_diagonal]
        )))),
        "distance_spearman": float(spearmanr(
            predicted[off_diagonal], exact[off_diagonal]
        ).statistic),
        "one_step_distance_mean": float(np.mean(
            predicted[state_rows, destination_vertices]
        )),
        "final_training_metrics": {
            name: float(jax.device_get(value)) for name, value in final_metrics.items()
        },
        "elapsed_seconds": float(time.time() - started),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=5)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = train_grid_quasimetric(side=args.side, steps=args.steps, seed=args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
