"""Build graph artifacts and critic targets from saved transitions or toy data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from envs.toy import make_chain, make_grid, make_stitch
from graphcritic.config import GraphConfig, TargetConfig
from graphcritic.diagnostics import report
from graphcritic.io import load_transitions, save_graph, save_targets
from graphcritic.pipeline import build_artifact
from graphcritic.targets import critic_targets


def _toy_data(name: str, seed: int):
    if name == "chain":
        # The CLI chain is a deterministic shortest-path sanity case. The
        # generator itself remains stochastic by default for other tests.
        return make_chain(seed=seed, p_forward=1.0)
    if name == "grid":
        return make_grid(seed=seed)
    if name == "stitch":
        return make_stitch(seed=seed)
    raise ValueError(f"unsupported toy dataset {name!r}")


def _toy_id(name: str, seed: int, trans) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in (
        trans.feat,
        trans.next_feat,
        trans.episode_id,
        trans.t,
        trans.success_cur,
        trans.success,
        trans.terminal,
        trans.row_idx,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    return f"toy:{name}:seed{seed}:{digest.hexdigest()}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--transitions", type=Path, help="TransitionSet NPZ path or pair stem")
    source.add_argument("--toy", choices=("chain", "grid", "stitch"), help="Generate a small CPU toy dataset")
    parser.add_argument("--config", type=Path, required=True, help="GraphConfig JSON/YAML file")
    parser.add_argument("--h", type=int, nargs="+", required=True, help="One or more chunk horizons")
    parser.add_argument("--unit", choices=("steps", "nats"), default="steps")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Toy data seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if len(set(args.h)) != len(args.h) or any(h < 1 for h in args.h):
            raise ValueError("--h values must be unique positive integers")
        cfg = GraphConfig.from_yaml(args.config)
        if args.toy is not None:
            trans, toy_info = _toy_data(args.toy, args.seed)
            dataset_id = _toy_id(args.toy, args.seed, trans)
            source_name = f"toy:{args.toy}"
        else:
            trans, transition_meta = load_transitions(args.transitions)
            dataset_id = transition_meta["dataset_id"]
            source_name = str(transition_meta.get("source") or args.transitions)
            toy_info = None

        artifact = build_artifact(
            trans,
            cfg,
            dataset_id=dataset_id,
            source=source_name,
            extra_meta={"toy": args.toy, "toy_seed": args.seed} if toy_info is not None else None,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        graph_path = args.out / "graph.npz"
        graph_hash = save_graph(artifact, graph_path)

        built_targets = {}
        for horizon in args.h:
            targets = critic_targets(trans, artifact, TargetConfig(h=horizon, unit=args.unit))
            target_stem = args.out / ("targets" if len(args.h) == 1 else f"targets_h{horizon}")
            save_targets(targets, target_stem)
            built_targets[str(horizon)] = targets

        if len(built_targets) == 1:
            only_targets = next(iter(built_targets.values()))
            diagnostics = report(trans, artifact, only_targets)
        else:
            diagnostics = {
                "graph": report(trans, artifact),
                "targets": {
                    f"h{h}": report(trans, artifact, target)["targets"]
                    for h, target in ((int(k), v) for k, v in built_targets.items())
                },
            }
        diagnostics_path = args.out / "diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(diagnostics, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "graph_hash": graph_hash,
            "graph": str(graph_path),
            "targets": [
                str(args.out / ("targets" if len(args.h) == 1 else f"targets_h{h}")) + ".npz"
                for h in args.h
            ],
            "diagnostics": str(diagnostics_path),
            "summary": diagnostics,
        }, sort_keys=True, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"build_graph: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
