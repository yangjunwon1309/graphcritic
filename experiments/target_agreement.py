"""Compare graph cost targets with an existing TD QC critic (diagnostic only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import spearmanr

from graphcritic.io import load_graph, load_targets, load_transitions
from graphcritic.path import _successor_edge_table
from graphcritic.types import SchemaError


def _td_agent(checkpoint: Path, flags_path: Path, obs_dim: int, action_dim: int, horizon: int):
    qc_root = ROOT / "third_party" / "qc"
    if str(qc_root) not in sys.path:
        sys.path.insert(0, str(qc_root))
    import jax.numpy as jnp
    from agents.acfql import ACFQLAgent, get_config
    from utils.flax_utils import restore_agent_with_file

    try:
        saved_flags = json.loads(flags_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"cannot read TD checkpoint flags at {flags_path}") from exc
    config = dict(get_config().to_dict())
    config.update(saved_flags.get("agent", {}))
    if config.get("critic_mode", "td") != "td":
        raise SchemaError("target agreement requires a TD-mode checkpoint")
    config["horizon_length"] = int(horizon)
    config["action_dim"] = int(action_dim)
    config["ob_dims"] = [1, int(obs_dim)]
    agent = ACFQLAgent.create(
        int(saved_flags.get("seed", 0)),
        jnp.zeros((1, obs_dim), dtype=jnp.float32),
        jnp.zeros((1, action_dim), dtype=jnp.float32),
        config,
    )
    return restore_agent_with_file(agent, str(checkpoint))


def _bridge_counts(artifact) -> np.ndarray:
    kind_by_src, _ = _successor_edge_table(artifact)
    count = np.zeros(artifact.nodes.n_nodes, dtype=np.int32)
    order = np.argsort(artifact.cost, kind="stable")
    for vertex in order:
        successor = int(artifact.succ[vertex])
        if successor >= 0:
            count[vertex] = count[successor] + int(kind_by_src[vertex] == 1)
    return count


def _q_td(agent, observations: np.ndarray, action_chunks: np.ndarray, batch_size: int) -> np.ndarray:
    import jax.numpy as jnp

    output = np.empty(observations.shape[0], dtype=np.float32)
    critic = agent.network.select("critic")
    for start in range(0, observations.shape[0], batch_size):
        stop = min(start + batch_size, observations.shape[0])
        q_ensemble = critic(
            jnp.asarray(observations[start:stop]),
            actions=jnp.asarray(action_chunks[start:stop]),
        )
        output[start:stop] = np.asarray(q_ensemble).mean(axis=0)
    return output


def _group_stats(q_graph: np.ndarray, q_td: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    x, y = q_graph[mask], q_td[mask]
    if x.size == 0:
        return {"n": 0, "spearman": None, "median_optimism_qg_minus_qtd": None}
    corr = spearmanr(x, y).statistic
    return {
        "n": int(x.size),
        "spearman": float(corr) if np.isfinite(corr) else None,
        "median_optimism_qg_minus_qtd": float(np.median(x - y)),
        "q_graph_p50": float(np.median(x)),
        "q_td_p50": float(np.median(y)),
    }


def _graph_report(name: str, graph_path: Path, trans, q_td: np.ndarray, gamma: float) -> dict[str, Any]:
    artifact = load_graph(graph_path, trans=trans)
    if artifact.config.graph_kind != "state":
        raise SchemaError(f"{name} is not a state graph")
    target_path = graph_path.with_name("targets")
    targets = load_targets(target_path, graph=artifact, expected_dataset_id=artifact.meta["dataset_id"])
    if targets.h != 5 or targets.unit != "steps":
        raise SchemaError(f"{name} targets must be h=5 in step units")

    n_rows = trans.n_rows
    starts = np.arange(n_rows, dtype=np.int64)
    endpoint_rows = starts + targets.h - 1
    endpoint_nodes = np.full(n_rows, -1, dtype=np.int64)
    in_range = endpoint_rows < n_rows
    endpoint_nodes[in_range] = artifact.nodes.assign_next[endpoint_rows[in_range]]
    b_nodes = _bridge_counts(artifact)
    bridge_count = np.zeros(n_rows, dtype=np.int32)
    bridge_count[in_range] = b_nodes[endpoint_nodes[in_range]]
    q_graph = -(
        1.0 - np.power(float(gamma), np.maximum(targets.y.astype(np.float64) - 1.0, 0.0))
    ) / (1.0 - float(gamma))
    valid = targets.valid
    groups = {
        "g0_chunk_success": valid & (targets.source == 0),
        "g1_finite_b0": valid & (targets.source == 1) & (bridge_count == 0),
        "g2_finite_bge1": valid & (targets.source == 1) & (bridge_count >= 1),
        "g3_cap": valid & (targets.source == 2),
    }
    return {
        "graph": name,
        "graph_hash": artifact.content_hash,
        "bridge_edges": int(np.count_nonzero(artifact.edges.kind == 1)),
        "bridge_count_by_row": {
            "p50": float(np.median(bridge_count[valid])) if np.any(valid) else None,
            "p95": float(np.percentile(bridge_count[valid], 95)) if np.any(valid) else None,
            "max": int(np.max(bridge_count[valid])) if np.any(valid) else None,
        },
        "groups": {key: _group_stats(q_graph, q_td, mask) for key, mask in groups.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--td-checkpoint", type=Path, required=True)
    parser.add_argument("--td-flags", type=Path, required=True)
    parser.add_argument("--graph", action="append", required=True, metavar="NAME=PATH",
                        help="Graph NPZ path; reads matching targets.npz in its directory")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 < args.gamma < 1:
        parser.error("batch size must be positive and gamma must be in (0,1)")

    trans, transition_meta = load_transitions(args.transitions)
    if trans.action is None:
        raise SchemaError("transition artifact does not contain actions")
    h = 5
    action_chunks = np.lib.stride_tricks.sliding_window_view(trans.action, h, axis=0)
    # Sliding windows are [row, action_dim, horizon] on NumPy; QC flattens
    # [horizon, action_dim] in chronological order.
    action_chunks = np.moveaxis(action_chunks[: trans.n_rows - h + 1], -1, 1)
    action_chunks = action_chunks.reshape(action_chunks.shape[0], -1).astype(np.float32)
    observations = trans.feat[: action_chunks.shape[0]].astype(np.float32, copy=False)
    td_values_valid = _td_agent(
        args.td_checkpoint, args.td_flags, observations.shape[1], trans.action.shape[1], h
    )
    # Match the target validity mask rather than trusting sliding-window tails.
    prefix = np.r_[0, np.cumsum(trans.terminal, dtype=np.int64)]
    start_count = trans.n_rows - h + 1
    valid = (prefix[np.arange(start_count) + h - 1] - prefix[np.arange(start_count)]) == 0
    # Ensure chunks never cross flattened episode boundaries. The terminal on
    # the chunk's final row is allowed, matching QC's sample_sequence rule.
    for offset in range(1, h):
        valid &= trans.episode_id[:start_count] == trans.episode_id[offset : offset + start_count]
    q_td = np.full(trans.n_rows, np.nan, dtype=np.float32)
    q_td[:start_count][valid] = _q_td(
        td_values_valid,
        observations[valid],
        action_chunks[valid],
        args.batch_size,
    )
    graph_specs: list[tuple[str, Path]] = []
    for spec in args.graph:
        if "=" not in spec:
            parser.error(f"--graph must be NAME=PATH, found {spec!r}")
        name, path = spec.split("=", 1)
        graph_specs.append((name, Path(path)))
    reports = [
        _graph_report(name, path, trans, q_td, args.gamma)
        for name, path in graph_specs
    ]
    result = {
        "dataset_id": transition_meta["dataset_id"],
        "td_checkpoint": str(args.td_checkpoint),
        "horizon": h,
        "gamma": float(args.gamma),
        "q_td_rows": int(np.count_nonzero(np.isfinite(q_td))),
        "graphs": reports,
        "interpretation": "development-only diagnostic; do not use to select the reported graph configuration",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
