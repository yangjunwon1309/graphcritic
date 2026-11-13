#!/usr/bin/env python3
"""Inventory and safely reorganize the Graph Critic output archive.

This is a one-time, manifest-driven migration.  The default invocation only
writes an inventory and dry-run plan.  Pass ``--apply`` to move retained
artifacts and runs, then move all remaining output content to
``outputs/_delete_pending``.  Nothing is deleted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EXTERNAL_WANDB = {
    "A2_qc_graph": Path("/tmp/tmp15c_klrf/wandb/offline-run-20261112_172749-14iib6uc"),
    "A3_dqc_td": Path("/tmp/tmpr2y3jvd_/wandb/offline-run-20261113_173502-jh97rg0c"),
    "A4_dqc_graph_distill": Path("/tmp/tmpie9fscqx/wandb/offline-run-20261113_180457-cdy8bg6t"),
    "A5_fql_td": Path("/tmp/tmpvpb0k9h9/wandb/offline-run-20261112_214832-psrwxs8n"),
    "A6_fql_graph": Path("/tmp/tmphkfr5key/wandb/offline-run-20261112_220927-9g7usfi8"),
    "A7_qcfql_td": Path("/tmp/tmpfvgm5ooo/wandb/offline-run-20261113_013218-d081qq21"),
    "A8_qcfql_graph": Path("/tmp/tmp1s4thuip/wandb/offline-run-20261113_015603-c98wnqzo"),
    "A9_ablation_L_cost": Path("/tmp/tmplvrnm8bj/wandb/offline-run-20261110_220336-9kwpxwl3"),
    "A10_ablation_O_cost": Path("/tmp/tmpsrkinqt6/wandb/offline-run-20261110_222508-ugo0pwcj"),
    "B1_qc_td": Path("/tmp/tmpdbo7isb4/wandb/offline-run-20261112_231151-z36sqyf3"),
    "B2_qc_graph": Path("/tmp/tmpysqzr484/wandb/offline-run-20261112_225028-kctarosz"),
    "B3_dqc_td": Path("/tmp/tmpl84l6fo2/wandb/offline-run-20261113_183325-d7ibzzhj"),
    "B4_dqc_graph_distill": Path("/tmp/tmptip0_e8m/wandb/offline-run-20261113_193502-bphp16mo"),
    "B5_fql_td": Path("/tmp/tmpyv_3ajt8/wandb/offline-run-20261113_154202-08ckmax6"),
    "B6_fql_graph": Path("/tmp/tmpqnd1k2l7/wandb/offline-run-20261113_160500-zf3c5jgv"),
    "B4_incomplete": Path("/tmp/tmp082_5i0o/wandb/offline-run-20261113_190626-2uhfrpe9"),
}

RUN_SPECS: dict[str, dict[str, Any]] = {
    "A1_qc_td": {"task": "puzzle-3x3-task3", "expected": [0.10, 1.00, 0.98, 1.00], "max_step": 200000, "method": "QC TD", "extract": "best-of-N", "target": "TD"},
    "A2_qc_graph": {"task": "puzzle-3x3-task3", "expected": [0.44, 0.78, 0.96, 1.00, 0.98, 0.94], "method": "QC graph critic", "extract": "best-of-N", "target": "discounted cost, h=5", "commit": "root 3d539149dee8d755f78cbb79091df17a81508f06; QC f12e624ff62a23c77366e751832361fe8a017f79"},
    "A3_dqc_td": {"task": "puzzle-3x3-task3", "expected": [0.00, 0.34, 0.98, 1.00, 1.00, 0.98], "method": "DQC TD", "extract": "DQC policy", "target": "TD", "commit": "root ef92576f5df310785820a737fe27bb88778f8f44; DQC 39d41719a344928f0ba8890afd02975d569c5169"},
    "A4_dqc_graph_distill": {"task": "puzzle-3x3-task3", "expected": [0.34, 0.40, 0.64, 0.54, 0.62, 0.64], "method": "DQC graph distillation", "extract": "DQC policy distillation", "target": "gamma_pow, h=25", "commit": "root ef92576f5df310785820a737fe27bb88778f8f44; DQC 39d41719a344928f0ba8890afd02975d569c5169"},
    "A5_fql_td": {"task": "puzzle-3x3-task3", "expected": [0.00, 0.00, 0.38, 0.66, 0.86, 0.94], "method": "FQL TD", "extract": "distill-ddpg", "target": "TD", "commit": "root 39055bc8b54559eaa170309acf5e5e9c9e61c3d3; QC f12e624ff62a23c77366e751832361fe8a017f79"},
    "A6_fql_graph": {"task": "puzzle-3x3-task3", "expected": [0.16, 0.42, 0.38, 0.54, 0.60, 0.52], "method": "FQL graph critic", "extract": "distill-ddpg", "target": "discounted cost, h=1", "commit": "root 39055bc8b54559eaa170309acf5e5e9c9e61c3d3; QC f12e624ff62a23c77366e751832361fe8a017f79"},
    "A7_qcfql_td": {"task": "puzzle-3x3-task3", "expected": [0.00, 0.02, 0.08, 0.12, 0.08, 0.04], "method": "QC-FQL TD", "extract": "distill-ddpg, 5-step chunk", "target": "TD", "commit": "root 3cc634d0db50ef02eb3bed5a8819c870deca22f6; QC f12e624ff62a23c77366e751832361fe8a017f79"},
    "A8_qcfql_graph": {"task": "puzzle-3x3-task3", "expected": [0.08, 0.00, 0.00, 0.00, 0.00, 0.00], "method": "QC-FQL graph critic", "extract": "distill-ddpg, 5-step chunk", "target": "discounted cost, h=5", "commit": "root 3cc634d0db50ef02eb3bed5a8819c870deca22f6; QC f12e624ff62a23c77366e751832361fe8a017f79"},
    "A9_ablation_L_cost": {"task": "puzzle-3x3-task3", "expected": [0.00, 0.06, 0.02, 0.04, 0.02, 0.06], "method": "L graph ablation QC", "extract": "best-of-N", "target": "cost, h=5", "commit": "root 199314afa34bc3d39ecad1a11b8687ee3716cd6d; QC f12e624ff62a23c77366e751832361fe8a017f79"},
    "A10_ablation_O_cost": {"task": "puzzle-3x3-task3", "expected": [0.04, 0.24, 0.26, 0.16, 0.28, 0.30], "method": "O graph ablation QC", "extract": "best-of-N", "target": "cost, h=5", "commit": "root 199314afa34bc3d39ecad1a11b8687ee3716cd6d; QC f12e624ff62a23c77366e751832361fe8a017f79"},
    "A11_bc": {"task": "puzzle-3x3-task3", "expected": None, "method": "BC", "extract": "behavior cloning", "target": "none", "standalone_bc_only": True},
    "B1_qc_td": {"task": "puzzle-4x4-p13-task2", "expected": [0.04, 0.56, 0.14, 0.10, 0.60, 0.46], "method": "QC TD", "extract": "best-of-N", "target": "TD", "commit": "root 00d50a2a19342df539bf9b7de80eddcff8d05701; QC f12e624ff62a23c77366e751832361fe8a017f79"},
    "B2_qc_graph": {"task": "puzzle-4x4-p13-task2", "expected": [0.68, 0.64, 0.80, 0.78, 0.82, 0.78], "method": "QC graph critic", "extract": "best-of-N", "target": "discounted cost, h=5", "commit": "root 00d50a2a19342df539bf9b7de80eddcff8d05701; QC f12e624ff62a23c77366e751832361fe8a017f79"},
    "B3_dqc_td": {"task": "puzzle-4x4-p13-task2", "expected": [0.70, 0.80, 0.78, 0.84, 0.88, 0.96], "method": "DQC TD", "extract": "DQC policy", "target": "TD", "commit": "root ef92576f5df310785820a737fe27bb88778f8f44; DQC 39d41719a344928f0ba8890afd02975d569c5169"},
    "B4_dqc_graph_distill": {"task": "puzzle-4x4-p13-task2", "expected": [0.70, 0.78, 0.88, 0.90, 0.88, 0.94], "method": "DQC graph distillation", "extract": "DQC policy distillation", "target": "gamma_pow, h=25", "commit": "root ef92576f5df310785820a737fe27bb88778f8f44; DQC 39d41719a344928f0ba8890afd02975d569c5169"},
    "B5_fql_td": {"task": "puzzle-4x4-p13-task2", "expected": [0.08, 0.40, 0.32, 0.82, 0.48, 0.82], "method": "FQL TD", "extract": "distill-ddpg", "target": "TD", "commit": "root b98208dee273cf3086bf71931122d0052b720954; QC f12e624ff62a23c77366e751832361fe8a017f79"},
    "B6_fql_graph": {"task": "puzzle-4x4-p13-task2", "expected": [0.02, 0.32, 0.32, 0.30, 0.38, 0.34], "method": "FQL graph critic", "extract": "distill-ddpg", "target": "discounted cost, h=1", "commit": "root b98208dee273cf3086bf71931122d0052b720954; QC f12e624ff62a23c77366e751832361fe8a017f79"},
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(OUT.resolve()).as_posix()
    except ValueError:
        return str(path)


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stat_times(path: Path) -> tuple[int, float, str]:
    stats = path.stat()
    total = 0
    mtimes: list[float] = []
    if path.is_file():
        total, mtimes = stats.st_size, [stats.st_mtime]
    else:
        for base, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
            for name in files:
                f = Path(base) / name
                try:
                    st = f.stat()
                except OSError:
                    continue
                if f.is_file():
                    total += st.st_size
                    mtimes.append(st.st_birthtime if hasattr(st, "st_birthtime") else st.st_mtime)
    if not mtimes:
        stamp = stats.st_birthtime if hasattr(stats, "st_birthtime") else stats.st_mtime
    else:
        stamp = min(mtimes)
    basis = "birthtime" if hasattr(stats, "st_birthtime") else "earliest_file_mtime_proxy"
    return total, stamp, basis


def find_run_dir(eval_csv: Path) -> Path:
    for parent in eval_csv.parents:
        if parent.name.startswith("sd") and parent.name[2:].isdigit():
            return parent
    return eval_csv.parent


def read_eval(path: Path) -> tuple[list[int], list[float], list[float | None]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], [], []
    fields = rows[0].keys()
    success_key = "success" if "success" in fields else next(
        (k for k in fields if k.endswith("_success") and "overall" not in k), None
    )
    wrong_key = "episode.final_wrong_buttons" if "episode.final_wrong_buttons" in fields else next(
        (k for k in fields if k.endswith("episode.final_wrong_buttons")), None
    )
    steps: list[int] = []
    successes: list[float] = []
    wrong: list[float | None] = []
    for row in rows:
        try:
            step = int(float(row.get("step", "")))
            success = float(row[success_key]) if success_key else float("nan")
        except (TypeError, ValueError, KeyError):
            continue
        steps.append(step)
        successes.append(success)
        try:
            wrong.append(float(row[wrong_key]) if wrong_key else None)
        except (TypeError, ValueError):
            wrong.append(None)
    return steps, successes, wrong


def success_curve(eval_csv: Path, max_step: int | None = None) -> tuple[list[float], list[int], list[float | None]]:
    steps, values, wrong = read_eval(eval_csv)
    wanted_steps = [50000, 100000, 150000, 200000] if max_step == 200000 else [50000, 100000, 150000, 200000, 250000, 300000]
    by_step = {step: (value, w) for step, value, w in zip(steps, values, wrong)}
    return ([by_step[s][0] for s in wanted_steps if s in by_step], [s for s in wanted_steps if s in by_step], [by_step[s][1] for s in wanted_steps if s in by_step])


def task_from_path(path: Path) -> str | None:
    value = path.as_posix()
    if "standard-3x3-task3" in value:
        return "puzzle-3x3-task3"
    if "p13-4x4-task2-100-generated" in value:
        return "puzzle-4x4-p13-task2"
    return None


def selected_runs() -> tuple[dict[str, Path], dict[str, list[Path]]]:
    matches: dict[str, list[Path]] = defaultdict(list)
    all_eval = sorted(OUT.rglob("eval.csv"))
    for spec_name, spec in RUN_SPECS.items():
        if spec.get("standalone_bc_only"):
            # eval_bc.csv beside a direct-Q run is an ablation, not a standalone BC run.
            continue
        for eval_csv in all_eval:
            if task_from_path(eval_csv) != spec["task"]:
                continue
            curve, steps, _ = success_curve(eval_csv, spec.get("max_step"))
            expected = spec.get("expected")
            if expected is None or len(curve) != len(expected) or len(steps) != len(expected):
                continue
            if [round(x, 2) for x in curve] == [round(float(x), 2) for x in expected]:
                matches[spec_name].append(find_run_dir(eval_csv))
    chosen: dict[str, Path] = {}
    for name, candidates in matches.items():
        completed = []
        for candidate in candidates:
            eval_path = candidate / "eval.csv"
            steps, _, _ = read_eval(eval_path)
            complete = max(steps, default=-1) >= RUN_SPECS[name].get("max_step", 300000)
            completed.append((complete, eval_path.stat().st_mtime, candidate))
        chosen[name] = max(completed, key=lambda item: (item[0], item[1]))[2]
    return chosen, dict(matches)


def provenance(path: Path) -> str:
    values: set[str] = set()
    jsons = [path] if path.is_file() and path.suffix == ".json" else list(path.glob("*.json")) if path.is_dir() else []
    for p in jsons:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        def walk(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    if k.lower() in {"git_commit", "commit", "root_commit", "execution_commit"} and isinstance(v, str):
                        values.add(v)
                    walk(v, k)
            elif isinstance(value, list):
                for v in value:
                    walk(v, key)
        walk(obj)
    return ";".join(sorted(values))


def artifact_kind(path: Path) -> str | None:
    names = {p.name for p in path.iterdir()} if path.is_dir() else set()
    if "params.msgpack" in names and "metadata.json" in names:
        return "d_theta_or_model_checkpoint"
    if "graph.npz" in names:
        return "graph_and_targets" if "targets.npz" in names else "graph"
    if "targets.npz" in names:
        return "targets"
    if any(n.startswith("candidates") and n.endswith(".npz") for n in names):
        return "candidate_cache"
    if "train.npz" in names or "train-val.npz" in names:
        return "generated_dataset"
    if any(n.startswith("sweep") and n.endswith(".json") for n in names):
        return "sweep"
    if any(n.startswith("stats") and n.endswith(".json") for n in names):
        return "stats"
    return None


def inventory() -> list[dict[str, str]]:
    chosen, matches = selected_runs()
    rows: list[dict[str, str]] = []
    seen: set[Path] = set()
    for eval_csv in sorted(OUT.rglob("eval.csv")):
        run = find_run_dir(eval_csv)
        if run in seen:
            continue
        seen.add(run)
        size, created, basis = stat_times(run)
        steps, curve, _ = read_eval(eval_csv)
        row = {
            "item_type": "run",
            "path": rel(run),
            "size_bytes": str(size),
            "created": datetime.fromtimestamp(created, timezone.utc).isoformat(),
            "created_basis": basis,
            "git_commit": provenance(run),
            "success_50k_to_300k": "/".join(f"{v:.2f}" for s, v in zip(steps, curve) if 50000 <= s <= 300000),
            "notes": ";".join(k for k, v in chosen.items() if v == run),
        }
        rows.append(row)
    artifact_dirs: set[Path] = set()
    for dirpath, dirs, _files in os.walk(OUT):
        d = Path(dirpath)
        if d.name == "_delete_pending":
            dirs[:] = []
            continue
        try:
            kind = artifact_kind(d)
        except OSError:
            continue
        if kind:
            artifact_dirs.add(d)
    # Include all files as distinct artifact rows when they are a standalone dataset/stat item.
    for path in sorted(artifact_dirs):
        size, created, basis = stat_times(path)
        rows.append({
            "item_type": artifact_kind(path) or "artifact",
            "path": rel(path),
            "size_bytes": str(size),
            "created": datetime.fromtimestamp(created, timezone.utc).isoformat(),
            "created_basis": basis,
            "git_commit": provenance(path),
            "success_50k_to_300k": "",
            "notes": "",
        })
    all_wandb = {p for p in Path("/tmp").glob("tmp*/wandb/offline-run-*") if p.is_dir()}
    aliases_by_path = {str(path): alias for alias, path in EXTERNAL_WANDB.items()}
    for path in sorted(all_wandb):
        if path.exists():
            size, created, basis = stat_times(path)
            rows.append({"item_type": "wandb_offline", "path": str(path), "size_bytes": str(size),
                         "created": datetime.fromtimestamp(created, timezone.utc).isoformat(),
                         "created_basis": basis, "git_commit": "", "success_50k_to_300k": "", "notes": aliases_by_path.get(str(path), "unselected external W&B folder")})
    return sorted(rows, key=lambda r: (r["item_type"], r["path"]))


def add_dir_moves(moves: list[dict[str, str]], source: str, dest: str) -> None:
    src = OUT / source
    if not src.exists():
        raise FileNotFoundError(f"required retained source missing: {src}")
    moves.append({"source": str(src), "destination": str(OUT / dest), "kind": "directory" if src.is_dir() else "file", "category": "retained"})


def split_graph_targets(moves: list[dict[str, str]], mappings: dict[str, str], source_rel: str,
                        graph_dest_rel: str, target_dest_rel: str) -> None:
    """Move graph and target files into the requested artifact subtrees."""
    source = OUT / source_rel
    if not source.is_dir():
        raise FileNotFoundError(f"required graph/target artifact missing: {source}")
    graph_files = {"graph.npz", "graph.json", "judge_candidates.npz", "layered_paths.npz", "result.json", "diagnostics.json"}
    target_files = {"targets.npz", "targets.json", "targets_source.npy"}
    for file in sorted(source.iterdir()):
        if file.name in target_files:
            dest = OUT / target_dest_rel / file.name
        elif file.name in graph_files:
            dest = OUT / graph_dest_rel / file.name
        else:
            # Preserve any additional sidecar with the graph package.
            dest = OUT / graph_dest_rel / file.name
        moves.append({"source": str(file), "destination": str(dest), "kind": "file", "category": "retained"})
        mappings[str(file)] = str(dest)
    # Directory references in old result/config JSONs resolve to the new graph directory.
    mappings[str(source)] = str(OUT / graph_dest_rel)
    moves.append({"source": str(source), "destination": str(OUT / graph_dest_rel), "kind": "reference_alias", "category": "reference_alias"})


def artifact_split(moves: list[dict[str, str]], old: str, base: str, *, split_target: bool = True) -> None:
    source = OUT / old
    if not source.is_dir():
        raise FileNotFoundError(f"required graph/target artifact missing: {source}")
    graph_names = {"graph.npz", "graph.json", "judge_candidates.npz", "layered_paths.npz", "result.json", "diagnostics.json"}
    target_names = {"targets.npz", "targets.json", "targets_source.npy"}
    for f in sorted(source.iterdir()):
        if f.name in target_names and split_target:
            dest = OUT / base / "targets" / f.name
        else:
            dest = OUT / base / ("graph" if split_target else "") / f.name if split_target else OUT / base / f.name
        moves.append({"source": str(f), "destination": str(dest), "kind": "directory" if f.is_dir() else "file", "category": "retained"})
    moves.append({"source": str(source), "destination": str(OUT / base / "graph"), "kind": "logical_split", "category": "retained", "split": "graph files and target files separated"})


def build_moves(chosen: dict[str, Path]) -> tuple[list[dict[str, str]], dict[str, str]]:
    moves: list[dict[str, str]] = []
    # Runs are selected by exact rounded success curves; absent A1/A11 are recorded explicitly.
    run_dest = {
        "A1_qc_td": "puzzle-3x3-task3/runs/A1_qc_td",
        "A2_qc_graph": "puzzle-3x3-task3/runs/A2_qc_graph",
        "A3_dqc_td": "puzzle-3x3-task3/runs/A3_dqc_td",
        "A4_dqc_graph_distill": "puzzle-3x3-task3/runs/A4_dqc_graph_distill",
        "A5_fql_td": "puzzle-3x3-task3/runs/A5_fql_td",
        "A6_fql_graph": "puzzle-3x3-task3/runs/A6_fql_graph",
        "A7_qcfql_td": "puzzle-3x3-task3/runs/A7_qcfql_td",
        "A8_qcfql_graph": "puzzle-3x3-task3/runs/A8_qcfql_graph",
        "A9_ablation_L_cost": "puzzle-3x3-task3/runs/A9_ablation_L_cost",
        "A10_ablation_O_cost": "puzzle-3x3-task3/runs/A10_ablation_O_cost",
        "B1_qc_td": "puzzle-4x4-p13-task2/runs/B1_qc_td",
        "B2_qc_graph": "puzzle-4x4-p13-task2/runs/B2_qc_graph",
        "B3_dqc_td": "puzzle-4x4-p13-task2/runs/B3_dqc_td",
        "B4_dqc_graph_distill": "puzzle-4x4-p13-task2/runs/B4_dqc_graph_distill",
        "B5_fql_td": "puzzle-4x4-p13-task2/runs/B5_fql_td",
        "B6_fql_graph": "puzzle-4x4-p13-task2/runs/B6_fql_graph",
    }
    mappings: dict[str, str] = {}
    for alias, dest in run_dest.items():
        if alias in chosen:
            source = chosen[alias]
            moves.append({"source": str(source), "destination": str(OUT / dest), "kind": "directory", "category": "retained_run", "alias": alias})
            mappings[str(source)] = str(OUT / dest)
        else:
            # A missing match is not guessed; create a small marker during apply.
            moves.append({"source": "", "destination": str(OUT / dest), "kind": "missing_marker", "category": "missing_run", "alias": alias})
    # Dataset roots and graph-building sources.
    add_dir_moves(moves, "puzzle/standard-3x3-task3/g0", "puzzle-3x3-task3/artifacts/reference_graphs/g0")
    add_dir_moves(moves, "puzzle/standard-3x3-task3/gr-rho09", "puzzle-3x3-task3/artifacts/reference_graphs/gr-rho09")
    add_dir_moves(moves, "puzzle/standard-3x3-task3/v08/check/window5_chunk/step_020000", "puzzle-3x3-task3/artifacts/d_theta/window5_chunk/step_020000")
    split_graph_targets(moves, mappings, "puzzle/standard-3x3-task3/v12/Lsame", "puzzle-3x3-task3/artifacts/graph_Lsame", "puzzle-3x3-task3/artifacts/targets_qc_h5")
    split_graph_targets(moves, mappings, "puzzle/standard-3x3-task3/v13/Lsame_h1", "puzzle-3x3-task3/artifacts/graph_h1", "puzzle-3x3-task3/artifacts/targets_h1")
    split_graph_targets(moves, mappings, "puzzle/standard-3x3-task3/v16/Lsame_h25_gp", "puzzle-3x3-task3/artifacts/graph_dqc_h25_gp", "puzzle-3x3-task3/artifacts/targets_dqc_h25_gp")
    add_dir_moves(moves, "puzzle/standard-3x3-task3/v08/L_lam10.5_cost", "puzzle-3x3-task3/artifacts/graph_ablation_L/L_lam10.5_cost")
    add_dir_moves(moves, "puzzle/standard-3x3-task3/v08/O_np1_cost", "puzzle-3x3-task3/artifacts/graph_ablation_O/O_np1_cost")
    for name in ("sweep_same.json", "sweep_any.json"):
        add_dir_moves(moves, f"puzzle/standard-3x3-task3/v12/{name}", f"puzzle-3x3-task3/artifacts/b_max_sweep/{name}")
    # v11 candidate cache, metadata, sidecar, and embedding cache used by the Lsame family.
    for name in ("candidates_obj.npz", "candidates_obj.json", "candidates_audit.json", "change_set.json"):
        add_dir_moves(moves, f"puzzle/standard-3x3-task3/v11/{name}", f"puzzle-3x3-task3/artifacts/candidates/v11/{name}")
    v11_embedding = "puzzle/standard-3x3-task3/v11/embeddings/embeddings-9adaf65cb96691db-d8e00c609a016926.npy"
    v11_embedding_meta = v11_embedding + ".json"
    add_dir_moves(moves, v11_embedding, "puzzle-3x3-task3/artifacts/candidates/embedding_cache_v11/embeddings-9adaf65cb96691db-d8e00c609a016926.npy")
    add_dir_moves(moves, v11_embedding_meta, "puzzle-3x3-task3/artifacts/candidates/embedding_cache_v11/embeddings-9adaf65cb96691db-d8e00c609a016926.npy.json")
    # Keep the v0.8 cache too: retained v0.8 ablation results point to that exact
    # cache path, while the Lsame family points to the v11 path.
    v08_embedding = OUT / "puzzle/standard-3x3-task3/v08/embeddings/embeddings-9adaf65cb96691db-d8e00c609a016926.npy"
    v08_embedding_meta = Path(str(v08_embedding) + ".json")
    v11_embedding_path = OUT / v11_embedding
    if v08_embedding.exists() and v11_embedding_path.exists() and sha256_file(v08_embedding) != sha256_file(v11_embedding_path):
        raise RuntimeError("v0.8 and v11 embedding caches differ; both must be retained separately")
    add_dir_moves(moves, "puzzle/standard-3x3-task3/v08/embeddings/embeddings-9adaf65cb96691db-d8e00c609a016926.npy", "puzzle-3x3-task3/artifacts/candidates/embedding_cache_v08/embeddings-9adaf65cb96691db-d8e00c609a016926.npy")
    add_dir_moves(moves, "puzzle/standard-3x3-task3/v08/embeddings/embeddings-9adaf65cb96691db-d8e00c609a016926.npy.json", "puzzle-3x3-task3/artifacts/candidates/embedding_cache_v08/embeddings-9adaf65cb96691db-d8e00c609a016926.npy.json")
    for name in ("candidates_window5.npz", "candidates_window5.json", "candidates_window5.object_match_indices.npz"):
        add_dir_moves(moves, f"puzzle/standard-3x3-task3/v08/{name}", f"puzzle-3x3-task3/artifacts/candidates/v08/{name}")

    # P13 task dataset: preserve source files and all generated-data metadata.
    p13 = OUT / "puzzle/p13-4x4-task2-100-generated"
    if not p13.is_dir():
        raise FileNotFoundError(f"required generated data root missing: {p13}")
    for f in sorted(p13.iterdir()):
        if f.is_file() and (f.name in {"train.npz", "train.json", "train-val.npz", "inspect.json"} or f.name.endswith("-build.log")):
            moves.append({"source": str(f), "destination": str(OUT / "puzzle-4x4-p13-task2/dataset" / f.name), "kind": "file", "category": "retained_dataset"})
    add_dir_moves(moves, "puzzle/p13-4x4-task2-100-generated/g0", "puzzle-4x4-p13-task2/artifacts/reference_graphs/g0")
    add_dir_moves(moves, "puzzle/p13-4x4-task2-100-generated/gr-rho09", "puzzle-4x4-p13-task2/artifacts/reference_graphs/gr-rho09")
    add_dir_moves(moves, "puzzle/p13-4x4-task2-100-generated/v08/check/window5_chunk/step_020000", "puzzle-4x4-p13-task2/artifacts/d_theta/window5_chunk/step_020000")
    split_graph_targets(moves, mappings, "puzzle/p13-4x4-task2-100-generated/v14/Lsame", "puzzle-4x4-p13-task2/artifacts/graph_Lsame", "puzzle-4x4-p13-task2/artifacts/targets_qc_h5")
    split_graph_targets(moves, mappings, "puzzle/p13-4x4-task2-100-generated/v14b/Lsame_h1", "puzzle-4x4-p13-task2/artifacts/graph_h1", "puzzle-4x4-p13-task2/artifacts/targets_h1")
    split_graph_targets(moves, mappings, "puzzle/p13-4x4-task2-100-generated/v16/Lsame_h25_gp", "puzzle-4x4-p13-task2/artifacts/graph_dqc_h25_gp", "puzzle-4x4-p13-task2/artifacts/targets_dqc_h25_gp")
    add_dir_moves(moves, "puzzle/p13-4x4-task2-100-generated/v14/sweep_same.json", "puzzle-4x4-p13-task2/artifacts/b_max_sweep/sweep_same.json")
    for name in ("candidates_obj.npz", "candidates_obj.json", "change_set.json"):
        add_dir_moves(moves, f"puzzle/p13-4x4-task2-100-generated/v14/{name}", f"puzzle-4x4-p13-task2/artifacts/candidates/v14/{name}")
    add_dir_moves(moves, "puzzle/p13-4x4-task2-100-generated/v14/embeddings/embeddings-a686b44eeabd868e-89f95c3b13de1b44.npy", "puzzle-4x4-p13-task2/artifacts/candidates/embedding_cache_v14/embeddings-a686b44eeabd868e-89f95c3b13de1b44.npy")
    add_dir_moves(moves, "puzzle/p13-4x4-task2-100-generated/v14/embeddings/embeddings-a686b44eeabd868e-89f95c3b13de1b44.npy.json", "puzzle-4x4-p13-task2/artifacts/candidates/embedding_cache_v14/embeddings-a686b44eeabd868e-89f95c3b13de1b44.npy.json")
    # Dataset metadata and the sweep are retained as explicit provenance artifacts.
    add_dir_moves(moves, "cube-double", "cube-double-play-stats")
    # DQC W&B offline folders are part of the selected full run artifacts.
    for alias, source in EXTERNAL_WANDB.items():
        if alias == "B4_incomplete":
            continue
        if alias in chosen and source.exists():
            dest_rel = run_dest.get(alias)
            if dest_rel:
                moves.append({"source": str(source), "destination": str(OUT / dest_rel / "wandb" / source.name), "kind": "directory", "category": "retained_wandb", "alias": alias})
    # The incomplete R4 attempt is intentionally not retained; mark its temp W&B for pending.
    if EXTERNAL_WANDB["B4_incomplete"].exists():
        moves.append({"source": str(EXTERNAL_WANDB["B4_incomplete"]), "destination": str(OUT / "_delete_pending" / "external_tmp" / "wandb" / EXTERNAL_WANDB["B4_incomplete"].name), "kind": "directory", "category": "delete_pending_external", "alias": "B4_incomplete"})
    for move in moves:
        if move["source"] and Path(move["source"]).exists() and move.get("category") != "reference_alias":
            mappings[move["source"]] = move["destination"]
    return moves, mappings


def is_under(path: Path, ancestor: Path) -> bool:
    try:
        path.relative_to(ancestor)
        return True
    except ValueError:
        return False


def pending_roots(retained_sources: list[Path]) -> list[Path]:
    control = {"_inventory.csv", "_plan.md", "MIGRATION.json", "INDEX.md", "results.csv", "_delete_pending"}
    new_roots = {"puzzle-3x3-task3", "puzzle-4x4-p13-task2", "cube-double-play-stats"}
    roots: list[Path] = []
    def walk(directory: Path) -> None:
        for child in sorted(directory.iterdir()):
            if child.name in control and directory == OUT:
                continue
            if child.name in new_roots and directory == OUT:
                continue
            if child.is_dir() and not any(child.iterdir()):
                continue
            existing_retained = [source for source in retained_sources if source.exists()]
            if any(child == source or is_under(child, source) for source in existing_retained):
                # This item is itself retained or is inside a retained directory.
                continue
            if any(is_under(source, child) for source in existing_retained):
                # This item is an old container holding retained descendants.
                if child.is_dir():
                    walk(child)
                else:
                    continue
            else:
                roots.append(child)
    walk(OUT)
    return roots


def category(path: Path) -> str:
    p = path.as_posix().lower()
    if "190626" in p or "incomplete" in p:
        return "R4 incomplete"
    if "direct_q" in p or "/direct" in p or "action_heads" in p:
        return "direct Q / action head"
    if "dplus" in p or "v10" in p:
        return "v1.0 d+"
    if "v11" in p or "lobj" in p:
        return "v1.1 Lobj"
    if "v13c" in p or "alpha1000" in p or "a1000" in p:
        return "FQL a1000"
    if "/v08/qc/" in p and not p.endswith("_cost"):
        return "regret"
    if "regret" in p or "rho" in p:
        return "regret"
    return "other intermediate"


def _write_inventory(rows: list[dict[str, str]]) -> None:
    path = OUT / "_inventory.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["item_type", "path", "size_bytes", "created", "created_basis", "git_commit", "success_50k_to_300k", "notes"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plan(rows: list[dict[str, str]], moves: list[dict[str, str]], chosen: dict[str, Path], matches: dict[str, list[Path]]) -> None:
    retained_sources = [Path(m["source"]) for m in moves if m.get("source") and m["category"].startswith("retained") and Path(m["source"]).exists()]
    pending = pending_roots(retained_sources)
    pending_info = []
    pending_bytes = defaultdict(int)
    for p in pending:
        size, _, _ = stat_times(p)
        cl = category(p)
        pending_info.append((p, size, cl))
        pending_bytes[cl] += size
    all_bytes = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file() and p.name not in {"_inventory.csv", "_plan.md"} and "_delete_pending" not in p.parts)
    retained_bytes = 0
    for m in moves:
        if m.get("source") and m.get("category", "").startswith("retained") and Path(m["source"]).exists():
            retained_bytes += stat_times(Path(m["source"]))[0]
    # Unique-file retained footprint avoids double-counting overlapping explicit references.
    retained_file_set: set[Path] = set()
    for source in retained_sources:
        if source.is_file():
            retained_file_set.add(source)
        elif source.is_dir():
            retained_file_set.update(p for p in source.rglob("*") if p.is_file())
    retained_unique = sum(p.stat().st_size for p in retained_file_set)
    text = [
        "# Output reorganization dry-run plan", "",
        f"- Inventory rows: {len(rows)}; total current outputs size: {human_size(all_bytes)}.",
        f"- Retained source footprint (unique files): {human_size(retained_unique)}.",
        f"- Delete-pending footprint: {human_size(sum(x[1] for x in pending_info))}.",
        "- Operation: same-filesystem move; no file deletion. Existing run directories move in full.",
        "", "## Run selection", "",
        "Matches are based on success values rounded to two decimal places; the latest completed match is selected.", "",
        "| Alias | Selected run | Other matching candidates | Status |", "|---|---|---|---|",
    ]
    for alias in RUN_SPECS:
        if alias in chosen:
            others = [rel(p) for p in matches.get(alias, []) if p != chosen[alias]]
            text.append(f"| {alias} | `{rel(chosen[alias])}` | {', '.join(f'`{p}`' for p in others) or '—'} | matched |")
        else:
            reason = "standalone BC run not found" if alias == "A11_bc" else "no exact success-curve match found"
            text.append(f"| {alias} | — | — | **not found** ({reason}) |")
    text += ["", "## Retained move list", "", "| Source | Destination | Size | Purpose |", "|---|---|---:|---|"]
    for m in moves:
        if not m.get("source"):
            continue
        src = Path(m["source"])
        size = stat_times(src)[0] if src.exists() else 0
        text.append(f"| `{m['source']}` | `{m['destination']}` | {human_size(size)} | {m.get('alias', m.get('category','retained'))} |")
    text += ["", "## Delete-pending move list", "", "| Original path | Size | Classification |", "|---|---:|---|"]
    for p, size, cl in pending_info:
        text.append(f"| `{rel(p)}` | {human_size(size)} | {cl} |")
    incomplete_wandb = EXTERNAL_WANDB["B4_incomplete"]
    if incomplete_wandb.exists():
        size, _, _ = stat_times(incomplete_wandb)
        pending_bytes["R4 incomplete"] += size
        text.append(f"| `{incomplete_wandb}` | {human_size(size)} | R4 incomplete W&B offline data |")
    text += ["", "## Delete-pending totals by classification", "", "| Classification | Size |", "|---|---:|"]
    for cl, size in sorted(pending_bytes.items()):
        text.append(f"| {cl} | {human_size(size)} |")
    (OUT / "_plan.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def update_json_paths(path: Path, mappings: dict[str, str], old_rel: str) -> None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    source_pairs = sorted(mappings.items(), key=lambda pair: len(pair[0]), reverse=True)
    def rewrite(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: rewrite(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v, key) for v in value]
        if not isinstance(value, str) or any(part in key.lower() for part in ("hash", "sha", "dataset_id")):
            return value
        out = value
        for source, dest in source_pairs:
            candidates = [source]
            if source.startswith(str(OUT) + os.sep):
                candidates.append("outputs/" + Path(source).relative_to(OUT).as_posix())
            for old in candidates:
                if out == old or out.startswith(old + os.sep):
                    out = dest + out[len(old):]
                    break
        return out
    obj = rewrite(obj)
    if isinstance(obj, dict) and (path.name in {"metadata.json", "result.json", "graph.json", "targets.json", "transitions.json", "diagnostics.json", "stats_v14.json"} or "metadata" in path.name.lower()):
        obj.setdefault("relocated_from", old_rel)
    path.write_text(json.dumps(obj, sort_keys=False, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def apply_moves(moves: list[dict[str, str]], mappings: dict[str, str]) -> None:
    # Detect overlapping retained source paths; only exact duplicate source->destination is allowed.
    active = [m for m in moves if m.get("source") and m.get("category") not in {"delete_pending_external", "reference_alias"}]
    paths = [Path(m["source"]) for m in active if Path(m["source"]).exists()]
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            if a != b and (is_under(a, b) or is_under(b, a)):
                raise RuntimeError(f"overlapping move sources: {a} and {b}")
    # Move retained runs/artifacts first. mv is atomic on this same filesystem.
    for move in moves:
        if not move.get("source") or move.get("category", "").startswith("delete_pending") or move.get("category") == "reference_alias":
            continue
        source, dest = Path(move["source"]), Path(move["destination"])
        if not source.exists():
            raise FileNotFoundError(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError(f"destination already exists: {dest}")
        subprocess.run(["mv", "--", str(source), str(dest)], check=True)
    # Keep complete DQC W&B directories with their runs. An incomplete R4 W&B folder
    # is retained under delete-pending for review, never removed.
    for move in moves:
        if move.get("category") == "delete_pending_external":
            source, dest = Path(move["source"]), Path(move["destination"])
            if source.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["mv", "--", str(source), str(dest)], check=True)
    # Write missing-run markers after directory moves.
    for move in moves:
        if move.get("kind") == "missing_marker":
            dest = Path(move["destination"])
            dest.mkdir(parents=True, exist_ok=True)
            reason = "standalone BC run not found" if move["alias"] == "A11_bc" else "No output run matched the requested evaluation curve."
            (dest / "NOT_FOUND.md").write_text(f"# {move['alias']}\n\n{reason}\n", encoding="utf-8")
    # Move every remaining output item to _delete_pending/<original relative path>.
    retained_sources = [Path(m["source"]) for m in moves if m.get("source") and m.get("category", "").startswith("retained")]
    pending = pending_roots([p for p in retained_sources if p.exists()])
    for source in pending:
        if not source.exists():
            continue
        dest = OUT / "_delete_pending" / rel(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError(dest)
        subprocess.run(["mv", "--", str(source), str(dest)], check=True)
        mappings[str(source)] = str(dest)
    # Remove only now-empty original container directories.
    for p in sorted(OUT.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if p.is_dir() and p.name != "_delete_pending":
            try:
                p.rmdir()
            except OSError:
                pass
    # Rewrite path metadata after all moves. Keep hash/dataset id fields byte-values unchanged.
    retained_roots = [Path(m["destination"]) for m in moves if m.get("destination") and m.get("category", "").startswith("retained")]
    metadata_paths: set[Path] = set()
    for root in retained_roots:
        if root.is_file() and root.suffix == ".json":
            metadata_paths.add(root)
        elif root.is_dir():
            metadata_paths.update(root.rglob("*.json"))
    for path in sorted(metadata_paths):
        if path.exists():
            # Reverse lookup original path for this retained destination.
            old = next((s for s, d in mappings.items() if d == str(path.parent) or d == str(path)), str(path))
            old_rel = old.replace(str(OUT) + os.sep, "")
            update_json_paths(path, mappings, old_rel)
    # external W&B metadata and paths should also reflect the new archive path.
    for move in moves:
        if move.get("category") == "retained_wandb":
            root = Path(move["destination"])
            for path in root.rglob("*.json"):
                update_json_paths(path, mappings, str(move["source"]))
    # Store canonical relocation map; source path keys are absolute for unambiguous provenance.
    (OUT / "MIGRATION.json").write_text(json.dumps({"schema_version": 1, "moves": mappings}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_indexes(chosen: dict[str, Path], matches: dict[str, list[Path]], moves: list[dict[str, str]]) -> None:
    move_by_alias = {m.get("alias"): m for m in moves if m.get("alias") and m.get("category") == "retained_run"}
    dest_by_alias = {m.get("alias"): Path(m["destination"]) for m in moves if m.get("alias") and m.get("category") == "retained_run"}
    rows: list[dict[str, str]] = []
    for alias, spec in RUN_SPECS.items():
        source = chosen.get(alias)
        if source:
            run_path = dest_by_alias.get(alias, source)
            steps, curve, _ = read_eval(run_path / "eval.csv")
            old_path = rel(source)
            target_path = "outputs/" + Path(move_by_alias[alias]["destination"]).relative_to(OUT).as_posix() if alias in move_by_alias else ""
            success = "/".join(f"{x:.2f}" for x in curve)
            mean = f"{sum(curve) / len(curve):.4f}" if curve else ""
            flags_path = run_path / "flags.json"
            commit = provenance(run_path) or spec.get("commit", "not recorded in run metadata")
            if flags_path.exists():
                try:
                    flags = json.loads(flags_path.read_text())
                    agent = flags.get("agent", {})
                    if not isinstance(agent, dict): agent = {}
                    if spec["method"].startswith("QC") or spec["method"].startswith("FQL"):
                        extraction = str(agent.get("actor_type", spec["extract"]))
                    else:
                        extraction = spec["extract"]
                    target = spec["target"]
                    if flags.get("graph_targets"):
                        target = str(flags.get("graph_target_kind", agent.get("graph_target_kind", target))) + "; " + target
                except (OSError, json.JSONDecodeError):
                    extraction, target = spec["extract"], spec["target"]
            else:
                extraction, target = spec["extract"], spec["target"]
        else:
            old_path, target_path, success, mean, commit = "not found", "not created", "", "", "—"
            extraction, target = spec["extract"], spec["target"]
        rows.append({"task": spec["task"], "run_id": alias, "method": spec["method"], "extraction": extraction,
                     "target": target, "success_50k_to_300k": success, "mean_success": mean,
                     "new_path": target_path, "original_path": old_path, "commit": commit})
    c1 = {"task": "cube-double-play-stats", "run_id": "C1", "method": "dataset statistics", "extraction": "not applicable",
          "target": "not applicable", "success_50k_to_300k": "not applicable", "mean_success": "not applicable",
          "new_path": "outputs/cube-double-play-stats/", "original_path": "outputs/cube-double/", "commit": "stats artifact provenance as stored"}
    rows.append(c1)
    fields = list(rows[0])
    with (OUT / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Retained run index", "", "C1 is retained as a complete stats artifact under `cube-double-play-stats/`. Missing runs are reported explicitly.", "",
             "| Task | Run ID | Method | Extraction | Target | Success 50k–300k | Mean | New path | Original path | Commit provenance |",
             "|---|---|---|---|---|---|---:|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(row[k].replace("|", "\\|") for k in fields) + " |")
    lines += ["", "## C1 — cube-double statistics", "", "- Path: `outputs/cube-double-play-stats/`", "- Source: `outputs/cube-double/`", "- Statistics only; no training, graph construction, or QC run.", ""]
    (OUT / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def verify_graph_targets() -> list[dict[str, str]]:
    sys.path.insert(0, str(ROOT))
    from graphcritic.io import load_graph, load_targets

    checks: list[dict[str, str]] = []
    datasets = {
        "puzzle-3x3-task3": OUT / "puzzle-3x3-task3/artifacts",
        "puzzle-4x4-p13-task2": OUT / "puzzle-4x4-p13-task2/artifacts",
    }
    graph_targets = {
        "puzzle-3x3-task3": [
            ("reference_graphs/g0", "reference_graphs/g0", "targets"),
            ("reference_graphs/gr-rho09", "reference_graphs/gr-rho09", "targets"),
            ("graph_Lsame", "targets_qc_h5", "split"),
            ("graph_h1", "targets_h1", "split"),
            ("graph_dqc_h25_gp", "targets_dqc_h25_gp", "split"),
            ("graph_ablation_L/L_lam10.5_cost", "graph_ablation_L/L_lam10.5_cost", "targets"),
            ("graph_ablation_O/O_np1_cost", "graph_ablation_O/O_np1_cost", "targets"),
        ],
        "puzzle-4x4-p13-task2": [
            ("reference_graphs/g0", "reference_graphs/g0", "targets"),
            ("reference_graphs/gr-rho09", "reference_graphs/gr-rho09", "targets"),
            ("graph_Lsame", "targets_qc_h5", "split"),
            ("graph_h1", "targets_h1", "split"),
            ("graph_dqc_h25_gp", "targets_dqc_h25_gp", "split"),
        ],
    }
    expected_hashes = {
        "puzzle-3x3-task3/graph_Lsame": "3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa",
        "puzzle-3x3-task3/graph_h1": "3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa",
        "puzzle-3x3-task3/graph_dqc_h25_gp": "3daea899774c21ade4cf1a2542300cec135d82c2059d52cf1d1dc8e61fd8f7aa",
        "puzzle-4x4-p13-task2/graph_Lsame": "6bd8f696e65e362770facddea65caca03f57abdb78927f054a730f32afbc01aa",
        "puzzle-4x4-p13-task2/graph_h1": "6bd8f696e65e362770facddea65caca03f57abdb78927f054a730f32afbc01aa",
        "puzzle-4x4-p13-task2/graph_dqc_h25_gp": "6bd8f696e65e362770facddea65caca03f57abdb78927f054a730f32afbc01aa",
    }
    for task, artifacts in datasets.items():
        for graph_rel, target_rel, target_mode in graph_targets[task]:
            graph_path = artifacts / graph_rel / "graph"
            graph = load_graph(graph_path)
            graph_key = task + "/" + graph_rel
            if graph_key in expected_hashes and graph.content_hash != expected_hashes[graph_key]:
                raise RuntimeError(f"graph hash mismatch for {graph_key}: {graph.content_hash}")
            target_path = artifacts / target_rel / "targets"
            if target_mode == "targets":
                target_path = artifacts / target_rel / "targets"
            target_meta = json.loads(target_path.with_suffix(".json").read_text(encoding="utf-8"))
            dataset_id = target_meta["config"]["dataset_id"]
            targets = load_targets(target_path, graph=graph, expected_dataset_id=dataset_id)
            checks.append({"task": task, "graph": graph_rel, "graph_hash": graph.content_hash,
                           "target": target_rel, "dataset_id": targets.dataset_id, "status": "passed"})
            del graph, targets
    return checks


def verify_fork_restore(fork: str) -> list[str]:
    """Restore all retained checkpoints with the corresponding fork's restore API."""
    fork_root = ROOT / "third_party" / fork
    sys.path.insert(0, str(fork_root))
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import jax
    from ml_collections import ConfigDict
    from pathlib import Path as _Path

    if fork == "qc":
        from agents import agents
        from envs.ogbench_utils import make_ogbench_env_and_datasets
        from utils.flax_utils import restore_agent_with_file
        from utils.datasets import Dataset
        aliases = [a for a in RUN_SPECS if a.startswith(("A2", "A5", "A6", "A7", "A8", "A9", "A10", "B1", "B2", "B5", "B6"))]
        dataset_cache: dict[tuple[str, str | None], Any] = {}
        restored: list[str] = []
        for alias in aliases:
            task = RUN_SPECS[alias]["task"]
            run = OUT / task / "runs" / alias
            flags = json.loads((run / "flags.json").read_text(encoding="utf-8"))
            dataset_path = flags.get("ogbench_dataset_path")
            key = (flags["env_name"], dataset_path)
            if key not in dataset_cache:
                env, eval_env, train_data, val_data = make_ogbench_env_and_datasets(
                    flags["env_name"], dataset_path=dataset_path, compact_dataset=False
                )
                dataset_cache[key] = (env, eval_env, train_data)
            env, eval_env, train_data = dataset_cache[key]
            if isinstance(train_data, dict):
                train_data = Dataset.create(**train_data)
            example = train_data.sample(())
            config = ConfigDict(flags["agent"])
            config["horizon_length"] = int(flags.get("horizon_length", 5))
            config["discount"] = float(flags.get("discount", 0.99))
            agent = agents[config["agent_name"]].create(
                int(flags.get("seed", 0)), example["observations"], example["actions"], config
            )
            checkpoints = sorted(run.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))
            if not checkpoints:
                raise FileNotFoundError(f"no checkpoints in {run}")
            restore_agent_with_file(agent, str(checkpoints[-1]))
            restored.append(f"{alias}: {checkpoints[-1].name}")
        for env, eval_env, _ in dataset_cache.values():
            for obj in (env, eval_env):
                if hasattr(obj, "close"):
                    obj.close()
        return restored

    if fork == "dqc":
        from agents import agents
        from envs.env_utils import make_ogbench_env_and_datasets
        from utils.datasets import Dataset, make_task_dataset
        from utils.flax_utils import restore_agent
        from graphcritic.io import load_targets
        aliases = [a for a in RUN_SPECS if a.startswith(("A3", "A4", "B3", "B4"))]
        restored = []
        for alias in aliases:
            task = RUN_SPECS[alias]["task"]
            run = OUT / task / "runs" / alias
            flags = json.loads((run / "flags.json").read_text(encoding="utf-8"))
            dataset_path = flags.get("ogbench_dataset_path")
            env, train_data, val_data = make_ogbench_env_and_datasets(
                flags["env_name"], dataset_path=dataset_path, compact_dataset=True, add_info=True
            )
            config = ConfigDict(flags["agent"])
            config["critic_mode"] = flags.get("critic_mode", "td")
            data = dict(train_data)
            if config["critic_mode"] == "graph":
                targets = load_targets(flags["graph_targets"], expected_dataset_id=flags["graph_dataset_id"])
                data["graph_target"] = targets.y
                data["graph_target_valid"] = targets.valid
                data["graph_target_loss_mask"] = targets.loss_mask
            data["actions"] = np.clip(data["actions"], -1 + 1e-5, 1 - 1e-5)
            task_dataset = make_task_dataset("single", Dataset.create(**data), config)
            np.random.seed(int(flags.get("seed", 0)))
            example = task_dataset.sample(1)
            agent = agents[config["agent_name"]].create(int(flags.get("seed", 0)), example, config)
            checkpoints = sorted(run.glob("params_*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))
            if not checkpoints:
                raise FileNotFoundError(f"no checkpoints in {run}")
            epoch = int(checkpoints[-1].stem.split("_")[-1])
            restore_agent(agent, str(run), epoch)
            restored.append(f"{alias}: {checkpoints[-1].name}")
            if hasattr(env, "close"):
                env.close()
        return restored
    raise ValueError(f"unknown fork: {fork}")


def write_docs(rows: list[dict[str, str]], chosen: dict[str, Path], matches: dict[str, list[Path]], moves: list[dict[str, str]]) -> None:
    # Documentation is written after verification; this creates a detailed draft with inventory and move accounting.
    missing = [name for name in RUN_SPECS if name not in chosen]
    pending_root = OUT / "_delete_pending"
    pending_total = sum(stat_times(p)[0] for p in pending_root.iterdir()) if pending_root.exists() else 0
    hardcoded = subprocess.run(["rg", "-n", "outputs/", "--glob", "*.py", "--glob", "*.md", "--glob", "*.yaml", "--glob", "!docs/results_v*.md", "--glob", "!experiments/reorganize_outputs.py", str(ROOT)], capture_output=True, text=True).stdout.splitlines()
    # Limit to references likely affected by the old roots; records are intentionally excluded.
    relevant = [line for line in hardcoded if "standard-3x3-task3" in line or "p13-4x4-task2-100-generated" in line or "cube-double" in line]
    doc = [
        "# Outputs cleanup and migration report", "",
        "## 0. Inventory", "",
        f"Inventory written to `outputs/_inventory.csv` with {len(rows)} run/artifact rows. `st_birthtime` is unavailable on this filesystem, so `created` is the earliest file mtime proxy. Total initial outputs size: recorded in `outputs/_plan.md` before moves.",
        "", "## 1. Run selection", "",
        "Success curves were read from the run `eval.csv` files and compared after rounding to two decimal places. The most recent completed exact match was selected where duplicates existed.", "",
    ]
    for alias in RUN_SPECS:
        if alias in chosen:
            alternatives = [rel(p) for p in matches.get(alias, []) if p != chosen[alias]]
            doc.append(f"- **{alias}:** `{rel(chosen[alias])}`" + (f"; other exact matches: {', '.join(alternatives)}." if alternatives else "."))
        else:
            doc.append(f"- **{alias}:** not found. No run was inferred.")
    doc += ["", "A11 note: the `eval_bc.csv` paired with a direct-Q run is not a standalone BC run and was not counted as A11.", "",
            "## 2. Dependency closure", "",
            "Required 3×3 and P13 graph, target, dθ, candidate, dataset, and sweep inputs were present and retained. The complete C1 `cube-double` stats directory was retained. DQC offline W&B directories were moved into their corresponding retained run directories; the incomplete first R4 attempt and its W&B folder are pending review.", "",
            "## 3–4. Plan and moves", "",
            "The dry-run plan is `outputs/_plan.md`. Moves used `mv` on the same filesystem. Retained run folders moved intact. JSON path references were rewritten from the relocation map; graph hashes and dataset IDs were preserved. Unselected outputs were moved to `_delete_pending` without deletion.", "",
            "## 5. Verification", "",
            "Verification records are written below from the completed read-only checks.", "",
            "## 6. Delete-pending", "",
            f"Pending tree: `outputs/_delete_pending/`; measured bytes: {human_size(pending_total)}. The data remains recoverable. After reviewing the pending list, the requested deletion command is `rm -rf {OUT}/_delete_pending`.", ""]
    plan_text = (OUT / "_plan.md").read_text(encoding="utf-8")
    pending_section = plan_text.split("## Delete-pending move list", 1)[-1].split("## Delete-pending totals by classification", 1)[0].strip()
    doc += ["", "### Delete-pending item list", "", pending_section, "", "### Hard-coded output path references", ""]
    doc += [f"- `{line}`" for line in relevant] if relevant else ["- None found in Python, Markdown, and YAML files outside results records."]
    root_head = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h %s"], capture_output=True, text=True, check=True).stdout.strip()
    doc += ["", "## Push 명령", "",
            "- QC fork (`graph-critic`): 변경 없음.", "- FQL fork: 변경 없음.", "- DQC fork (`graph-critic`): 변경 없음.",
            f"- Root (`quasi_develop`): `{root_head}` (implementation script); report/document commit will be the subsequent root commit.", "",
            "```bash", "cd /home/url/graphcritic && git status -sb && git log --oneline origin/quasi_develop..HEAD && git push origin quasi_develop", "```", ""]
    (ROOT / "docs/results_cleanup.md").write_text("\n".join(doc), encoding="utf-8")
    (ROOT / "docs/results_index.md").write_text((OUT / "INDEX.md").read_text(encoding="utf-8"), encoding="utf-8")


def assert_clean_repositories() -> dict[str, dict[str, str]]:
    repos = {"root": (ROOT, "quasi_develop"), "QC": (ROOT / "third_party/qc", "graph-critic"),
             "FQL": (ROOT / "third_party/fql", None), "DQC": (ROOT / "third_party/dqc", "graph-critic")}
    state = {}
    for name, (repo, expected_branch) in repos.items():
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
        branch = subprocess.run(["git", "-C", str(repo), "branch", "--show-current"], check=True, capture_output=True, text=True).stdout.strip()
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        if status:
            raise RuntimeError(f"{name} repository is dirty before apply: {status}")
        if expected_branch and branch != expected_branch:
            raise RuntimeError(f"{name} branch is {branch!r}; expected {expected_branch!r}")
        state[name] = {"branch": branch or "detached", "head": head, "git_dirty": "false"}
    return state


def verify_all(chosen: dict[str, Path], matches: dict[str, list[Path]], moves: list[dict[str, str]], rows: list[dict[str, str]]) -> dict[str, Any]:
    verification: dict[str, Any] = {"repositories": assert_clean_repositories(), "graphs_and_targets": verify_graph_targets()}
    py = ROOT / ".pixi/envs/bench/bin/python"
    if not py.exists():
        raise FileNotFoundError(f"benchmark Python interpreter missing: {py}")
    env = dict(os.environ)
    env.update({"JAX_PLATFORMS": "cpu", "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "MUJOCO_GL": "egl"})
    restores = {}
    for fork in ("qc", "dqc"):
        run = subprocess.run([str(py), str(Path(__file__).resolve()), "--verify-fork", fork], cwd=ROOT / "third_party" / fork,
                             env=env, check=True, capture_output=True, text=True)
        restores[fork] = [line for line in run.stdout.splitlines() if line.startswith("RESTORED ")]
        expected_count = 11 if fork == "qc" else 4
        if len(restores[fork]) != expected_count:
            raise RuntimeError(f"{fork} restore count {len(restores[fork])}; expected {expected_count}. Output: {run.stdout[-2000:]} {run.stderr[-2000:]}")
    verification["checkpoint_restores"] = restores
    indexed: dict[str, str] = {}
    with (OUT / "results.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["run_id"] in RUN_SPECS and row["success_50k_to_300k"]:
                indexed[row["run_id"]] = row["success_50k_to_300k"]
    run_checks = []
    for alias, old_source in chosen.items():
        spec = RUN_SPECS[alias]
        run = OUT / spec["task"] / "runs" / alias
        curve, steps, _ = success_curve(run / "eval.csv", spec.get("max_step"))
        actual = "/".join(f"{x:.2f}" for x in curve)
        if actual != indexed.get(alias):
            raise RuntimeError(f"INDEX curve mismatch for {alias}: {indexed.get(alias)} != {actual}")
        expected = [round(float(x), 2) for x in spec["expected"]]
        if [round(x, 2) for x in curve] != expected:
            raise RuntimeError(f"relocated run curve mismatch for {alias}: {curve}")
        run_checks.append({"run_id": alias, "steps": steps, "success": actual, "status": "passed"})
    verification["index_curve_checks"] = run_checks
    (OUT / "_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return verification


def write_verification_into_report(verification: dict[str, Any]) -> None:
    report = ROOT / "docs/results_cleanup.md"
    content = report.read_text(encoding="utf-8")
    repo_summary = ", ".join(
        "{} `{}` at `{}` ({})".format(name, value["git_dirty"], value["head"], value["branch"])
        for name, value in verification["repositories"].items()
    )
    section = ["### Verification results", "",
               f"- Repository cleanliness at apply/verification: {repo_summary}.",
               f"- Retained graph/target pairs loaded: {len(verification['graphs_and_targets'])}; all stored graph hashes, dataset IDs, and target graph links validated.",
               f"- Checkpoint restores: QC {len(verification['checkpoint_restores']['qc'])}/{len(verification['checkpoint_restores']['qc'])}; DQC {len(verification['checkpoint_restores']['dqc'])}/{len(verification['checkpoint_restores']['dqc'])}.",
               f"- INDEX success curves matched relocated eval CSVs: {len(verification['index_curve_checks'])}/{len(RUN_SPECS)-2}; all selected curves matched.", ""]
    marker = "## 5. Verification\n\nVerification records are written below from the completed read-only checks.\n"
    content = content.replace(marker, marker + "\n".join(section) + "\n")
    report.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="perform the planned same-filesystem moves")
    parser.add_argument("--verify-fork", choices=("qc", "dqc"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_fork:
        for line in verify_fork_restore(args.verify_fork):
            print("RESTORED " + line)
        return 0
    if not OUT.is_dir():
        raise FileNotFoundError(OUT)
    rows = inventory()
    _write_inventory(rows)
    chosen, matches = selected_runs()
    moves, mappings = build_moves(chosen)
    write_plan(rows, moves, chosen, matches)
    print(f"Inventory: {OUT / '_inventory.csv'} ({len(rows)} rows)")
    print(f"Plan: {OUT / '_plan.md'}")
    print(f"Matched runs: {len(chosen)}/{len(RUN_SPECS)}")
    if not args.apply:
        print("Dry run only; no outputs were moved.")
        return 0
    repo_state = assert_clean_repositories()
    # All explicitly required artifact sources must exist before any move starts.
    for move in moves:
        if move.get("category", "").startswith("retained") and move.get("source") and not Path(move["source"]).exists():
            raise FileNotFoundError(f"dependency closure incomplete: {move['source']}")
    out_device = OUT.stat().st_dev
    for move in moves:
        if move.get("source") and move.get("category") != "reference_alias" and Path(move["source"]).exists():
            if Path(move["source"]).stat().st_dev != out_device:
                raise RuntimeError(f"same-filesystem move requirement failed for {move['source']}")
    apply_moves(moves, mappings)
    make_indexes(chosen, matches, moves)
    # Keep the original-path inventory written before moves; MIGRATION.json provides new locations.
    write_docs(rows, chosen, matches, moves)
    verification = verify_all(chosen, matches, moves, rows)
    write_verification_into_report(verification)
    print(f"Moved retained artifacts and runs; pending tree: {OUT / '_delete_pending'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
