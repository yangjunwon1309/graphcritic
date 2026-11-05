"""Build graph artifacts and critic targets from saved transitions or toy data."""

from __future__ import annotations

import argparse
import hashlib
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
from graphcritic.io import load_sim_state, load_transitions, save_graph, save_sim_state, save_targets, save_transitions
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
        trans.mode,
        trans.mode_next,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    if trans.action is not None:
        digest.update(np.ascontiguousarray(trans.action).tobytes())
    return f"toy:{name}:seed{seed}:{digest.hexdigest()}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--transitions", type=Path, help="TransitionSet NPZ path or pair stem")
    source.add_argument("--toy", choices=("chain", "grid", "stitch"), help="Generate a small CPU toy dataset")
    source.add_argument("--ogbench-dataset", help="Load a QC OGBench dataset ID directly")
    parser.add_argument("--dataset-path", type=Path, help="Immutable OGBench train NPZ (paired -val.npz required)")
    parser.add_argument("--dataset-tag", help="Include a provenance tag in the dataset ID")
    parser.add_argument("--sim-state", type=Path, help="Aligned sim_state.npz for a saved transition set")
    parser.add_argument("--ogbench-env", help="OGBench env ID used for simulator verification and path replay")
    parser.add_argument("--path-exec", action="store_true", help="Replay sampled graph paths in the simulator")
    parser.add_argument("--config", type=Path, required=True, help="GraphConfig JSON/YAML file")
    parser.add_argument("--h", type=int, nargs="+", required=True, help="One or more chunk horizons")
    parser.add_argument("--unit", choices=("steps", "nats"), default="steps")
    parser.add_argument("--target-kind", choices=("cost", "regret"), default="regret")
    parser.add_argument("--regret-max", type=float, default=50.0)
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Toy data seed")
    parser.add_argument("--force", action="store_true", help="Write artifacts despite a failed training gate")
    parser.add_argument("--gate-cap-frac", type=float, default=0.5)
    parser.add_argument("--gate-reachable", type=float, default=0.5)
    parser.add_argument("--gate-failed-reachable", type=float, default=0.3)
    parser.add_argument("--gate-path-exec", type=float, default=0.7)
    parser.add_argument("--gate-profile", choices=("legacy", "v05"), default="legacy",
                        help="legacy v0.4 gates or v0.5 necessary-only graph gates")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    verifier_env = None
    try:
        if len(set(args.h)) != len(args.h) or any(h < 1 for h in args.h):
            raise ValueError("--h values must be unique positive integers")
        cfg = GraphConfig.from_yaml(args.config)
        if args.path_exec and cfg.bridge.cost.policy == "time":
            raise ValueError(
                "time-policy paths must be evaluated with experiments/eval_graph_policy.py; "
                "--path-exec is the legacy open-loop diagnostic"
            )
        sim_state = None
        if args.toy is not None:
            trans, toy_info = _toy_data(args.toy, args.seed)
            dataset_id = _toy_id(args.toy, args.seed, trans)
            source_name = f"toy:{args.toy}"
        elif args.ogbench_dataset is not None:
            from envs.ogbench import load_ogbench_transitions

            dataset_tag = args.dataset_tag
            if args.dataset_path is not None:
                dataset_path = args.dataset_path.expanduser().resolve()
                metadata_path = dataset_path.with_suffix(".json")
                if metadata_path.is_file():
                    dataset_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                    expected_hash = dataset_meta.get("dataset_sha256")
                    if expected_hash is not None:
                        digest = hashlib.sha256()
                        with dataset_path.open("rb") as stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b""):
                                digest.update(block)
                        if digest.hexdigest() != expected_hash:
                            raise ValueError("dataset archive SHA256 does not match adjacent metadata")
                    artifact_tag = dataset_meta.get("dataset_tag")
                    if dataset_tag is None:
                        dataset_tag = artifact_tag
                    elif artifact_tag is not None and dataset_tag != artifact_tag:
                        raise ValueError("--dataset-tag does not match the adjacent dataset metadata")
            trans, dataset_id, sim_state = load_ogbench_transitions(
                args.ogbench_dataset,
                dataset_path=dataset_path if args.dataset_path is not None else None,
                return_sim_state=True,
                dataset_tag=dataset_tag,
            )
            source_name = f"ogbench:{args.ogbench_dataset}"
            toy_info = None
        else:
            if args.dataset_path is not None or args.dataset_tag is not None:
                raise ValueError("--dataset-path/--dataset-tag require --ogbench-dataset")
            trans, transition_meta = load_transitions(args.transitions)
            dataset_id = transition_meta["dataset_id"]
            source_name = str(transition_meta.get("source") or args.transitions)
            toy_info = None
            if args.sim_state is not None:
                from graphcritic.featurize import state_table

                sim_state = load_sim_state(args.sim_state, n_states=state_table(trans)[0].shape[0])

        env_id = args.ogbench_env
        if env_id is None and args.ogbench_dataset is not None:
            from envs.ogbench import _dataset_parts
            env_id = _dataset_parts(args.ogbench_dataset)[0]
        verifier = None
        wants_verifier = cfg.bridge.cost.policy == "sim_verified" or args.path_exec
        if wants_verifier:
            if env_id is None:
                raise ValueError("simulator verification requires --ogbench-env or --ogbench-dataset")
            try:
                import ogbench  # noqa: F401
                import gymnasium as gym
                from graphcritic.verification import OGBenchVerifier
            except ImportError as exc:
                raise RuntimeError("OGBench simulator verification requires the bench environment") from exc
            verifier_env = gym.make(env_id)
            verifier = OGBenchVerifier(verifier_env)
        if cfg.bridge.cost.policy == "sim_verified" and sim_state is None:
            raise ValueError("sim_verified policy requires an aligned --sim-state sidecar")

        args.out.mkdir(parents=True, exist_ok=True)
        if args.ogbench_dataset is not None:
            save_transitions(
                trans, args.out / "transitions", dataset_id=dataset_id, source=source_name
            )
            if sim_state is not None:
                save_sim_state(sim_state, args.out / "sim_state.npz")

        artifact = build_artifact(
            trans,
            cfg,
            dataset_id=dataset_id,
            source=source_name,
            extra_meta={"toy": args.toy, "toy_seed": args.seed} if toy_info is not None else None,
            sim_state=sim_state,
            verifier=verifier,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        graph_path = args.out / "graph.npz"
        graph_hash = save_graph(artifact, graph_path)
        if sim_state is not None:
            save_sim_state(sim_state, args.out / "sim_state.npz")

        built_targets = {}
        for horizon in args.h:
            targets = critic_targets(
                trans, artifact,
                TargetConfig(h=horizon, unit=args.unit, kind=args.target_kind, regret_max=args.regret_max),
            )
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
        cap_fracs = [
            float(value["targets"]["valid_non_success_cap_frac"])
            for value in diagnostics.get("targets_by_horizon", {}).values()
        ]
        if not cap_fracs and "targets" in diagnostics:
            cap_fracs = [float(diagnostics["targets"].get("valid_non_success_cap_frac", 0.0))]
        if len(built_targets) > 1:
            diagnostics["targets_by_horizon"] = {
                f"h{h}": report(trans, artifact, target)["targets"]
                for h, target in ((int(k), v) for k, v in built_targets.items())
            }
            cap_fracs = [
                float(value["valid_non_success_cap_frac"])
                for value in diagnostics["targets_by_horizon"].values()
            ]
        cap_frac = max(cap_fracs, default=0.0)
        graph_diagnostics = diagnostics["graph"] if "graph" in diagnostics else diagnostics
        reachable = float(graph_diagnostics["reachable_frac"])
        failed_reachable = float(graph_diagnostics["reachable_frac_failed_eps"])
        failed_reachability_applicable = int(graph_diagnostics.get("n_failed_episodes", 0)) > 0
        bridge_use = float(graph_diagnostics["bridge_use_frac"])
        stitch_gain = float(graph_diagnostics["stitch_gain"]["frac_gt_1_step"])
        gate_reasons = []
        if args.gate_profile == "v05":
            if cap_frac > 0.8:
                gate_reasons.append(f"valid non-success cap fraction {cap_frac:.4f} exceeds 0.8000")
            if failed_reachability_applicable and failed_reachable <= 0.1:
                gate_reasons.append(
                    f"failed-episode reachable fraction {failed_reachable:.4f} "
                    "must be greater than 0.1000"
                )
            if cfg.bridge.strategy in {"passive_r_real", "passive_r_strict", "passive_rm_real"}:
                tau_check = artifact.meta.get("bridge_r_tau_real", {})
                if int(tau_check.get("kinematic_violation_count", 0)) != 0:
                    gate_reasons.append("at least one retained R bridge has tau_real < tau_kin")
            if cfg.bridge.strategy in {"passive_rm", "passive_rm_real"} or cfg.bridge.include_m_bridges:
                kin_violation = float(artifact.meta.get("bridge_tau_kinematic_violation", 1.0))
                if kin_violation > 0.05:
                    gate_reasons.append(
                        f"M-bridge kinematic violation {kin_violation:.4f} exceeds 0.0500"
                    )
        else:
            if cap_frac > args.gate_cap_frac:
                gate_reasons.append(f"valid non-success cap fraction {cap_frac:.4f} exceeds {args.gate_cap_frac:.4f}")
            if reachable < args.gate_reachable:
                gate_reasons.append(f"reachable fraction {reachable:.4f} is below {args.gate_reachable:.4f}")
        if args.gate_profile == "legacy" and cfg.graph_kind == "state" and cfg.bridge.enabled and cfg.bridge.cost.policy == "time":
            if failed_reachable < args.gate_failed_reachable:
                gate_reasons.append(
                    f"failed-episode reachable fraction {failed_reachable:.4f} "
                    f"is below {args.gate_failed_reachable:.4f}"
                )
            if bridge_use <= 0:
                gate_reasons.append("no finite shortest path uses a bridge")
            if stitch_gain <= 0:
                gate_reasons.append("no successful-episode state gains more than one step from stitching")
        if args.gate_profile == "legacy" and cfg.bridge.strategy == "passive_rm":
            kin_violation = float(artifact.meta.get("bridge_tau_kinematic_violation", 1.0))
            if kin_violation > 0.05:
                gate_reasons.append(
                    f"M-bridge kinematic violation {kin_violation:.4f} exceeds 0.0500"
                )
        path_rate = (
            diagnostics.get("graph", {}).get("path_exec_switch_le_3")
            if "graph" in diagnostics else diagnostics.get("path_exec_switch_le_3")
        )
        if path_rate is not None and float(path_rate) < args.gate_path_exec:
            gate_reasons.append(
                f"path execution rate for paths with <=3 switches {float(path_rate):.4f} "
                f"is below {args.gate_path_exec:.4f}"
            )
        diagnostics["training_gate"] = {
            "passed": not gate_reasons,
            "reasons": gate_reasons,
            "cap_fraction": cap_frac,
            "reachable_fraction": reachable,
            "reachable_fraction_failed_eps": failed_reachable,
            "failed_episode_reachability_applicable": failed_reachability_applicable,
            "bridge_use_frac": bridge_use,
            "stitch_gain_frac_gt_1_step": stitch_gain,
            "force": bool(args.force),
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
        if verifier_env is not None:
            verifier_env.close()
        return 0 if not gate_reasons or args.force else 2
    except Exception as exc:
        if verifier_env is not None:
            verifier_env.close()
        print(f"build_graph: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
