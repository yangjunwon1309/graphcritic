"""Recompute R_strict shortest paths for the v0.5 rho sensitivity grid."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphcritic.config import TargetConfig
from graphcritic.diagnostics import report
from graphcritic.io import load_graph, load_transitions, save_graph, save_targets
from graphcritic.solve import solve
from graphcritic.targets import critic_targets
from graphcritic.types import GraphArtifact, SchemaError
from graphcritic.weights import edge_weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True,
                        help="Base R_strict graph built with rho=1")
    parser.add_argument("--rho", type=float, nargs="+", default=[0.9, 0.7, 0.5])
    parser.add_argument("--h", type=int, default=5)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if any(not 0 < rho <= 1 for rho in args.rho) or args.h < 1:
        parser.error("rho values must be in (0,1] and h must be positive")
    trans, transition_meta = load_transitions(args.transitions)
    base = load_graph(args.graph, trans=trans)
    if base.config.bridge.strategy != "passive_r_strict":
        raise SchemaError("rho reweighting requires a passive_r_strict graph")
    if base.meta["dataset_id"] != transition_meta["dataset_id"]:
        raise SchemaError("graph and transition dataset IDs differ")

    reports = []
    for rho in args.rho:
        penalty = -math.log(rho) / -math.log(base.config.gamma)
        cost_cfg = replace(
            base.config.bridge.cost,
            bridge_success_rho=float(rho),
            lambda0_steps=float(penalty),
        )
        bridge_cfg = replace(base.config.bridge, cost=cost_cfg)
        config = replace(base.config, bridge=bridge_cfg)
        weights = edge_weights(
            base.edges, config,
            d_step=float(base.meta.get("d_step", 1.0)),
            meta=base.meta,
        )
        cost, succ = solve(base.nodes, base.edges, weights)
        meta = dict(base.meta)
        meta["bridge_success_rho"] = float(rho)
        meta["bridge_lambda_rho_steps"] = float(penalty)
        artifact = GraphArtifact(
            config=config,
            feature_stats=base.feature_stats,
            nodes=base.nodes,
            edges=base.edges,
            weights=weights,
            cost=cost,
            succ=succ,
            meta=meta,
        )
        out = args.out_root / f"rstrict-rho-{rho:g}"
        out.mkdir(parents=True, exist_ok=True)
        graph_hash = save_graph(artifact, out / "graph.npz")
        targets = critic_targets(trans, artifact, TargetConfig(h=args.h, unit="steps"))
        save_targets(targets, out / "targets.npz")
        summary = report(trans, artifact, targets)
        gate = summary.get("training_gate", {})
        gate_reasons = []
        if float(gate.get("cap_fraction", 1.0)) > 0.8:
            gate_reasons.append("valid non-success cap fraction exceeds 0.8")
        if float(gate.get("reachable_fraction_failed_eps", 0.0)) <= 0.1:
            gate_reasons.append("failed-episode reachable fraction must be greater than 0.1")
        if artifact.meta.get("bridge_r_tau_real", {}).get("kinematic_violation_count", 0) != 0:
            gate_reasons.append("tau_real is below tau_kin on a retained R bridge")
        summary["training_gate_v05"] = {"passed": not gate_reasons, "reasons": gate_reasons}
        (out / "diagnostics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        reports.append({
            "rho": rho,
            "lambda_steps": penalty,
            "graph_hash": graph_hash,
            "output": str(out),
            "bridge_edges": int((artifact.edges.kind == 1).sum()),
            "training_gate_v05": summary["training_gate_v05"],
            "graph": summary.get("graph", summary),
            "targets": summary.get("targets"),
        })
    result = {"dataset_id": transition_meta["dataset_id"], "rho_sweep": reports}
    index = args.out_root / "rho_sweep.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
