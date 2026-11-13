#!/usr/bin/env python3
"""Enforce the v1.8 robot-only quasimetric training gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _last_jsonl(path: Path) -> dict:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no records in {path}")
    value = json.loads(rows[-1])
    if not isinstance(value, dict):
        raise ValueError(f"last record in {path} is not a JSON object")
    return value


def inspect_run(run: Path, *, step: int, violation_gate: str) -> dict:
    checkpoint = run / f"step_{step:06d}"
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata.get("step", -1)) != step:
        raise ValueError(f"checkpoint metadata step does not match requested step {step}")
    training = _last_jsonl(run / "train_metrics.jsonl")
    heldout = _last_jsonl(run / "training.jsonl")
    violation = heldout.get("heldout_chunk_constraint", {}).get(
        "overall_violation_rate_d_gt_limit"
    )
    diagnostics = training.get("diagnostics") or {}
    dead_by_layer = diagnostics.get("dead_unit_frac_by_layer") or {}
    dead_unit_max = max((float(value) for value in dead_by_layer.values()), default=None)
    f3_passed = violation is not None and float(violation) <= 0.05
    f3_reported = not f3_passed and violation_gate == "report"
    report = {
        "input_part": metadata.get("input_part"),
        "frame_dim": metadata.get("frame_dim"),
        "input_dim": metadata.get("input_dim"),
        "step": metadata.get("step"),
        "collapse_alarm": training.get("collapse_alarm"),
        "collapse_alarm_reason": training.get("collapse_alarm_reason"),
        "dead_unit_fraction_max": dead_unit_max,
        "heldout_chunk_violation_d_gt_6_25": violation,
        "train_chunk_violation_d_gt_6_25": heldout.get(
            "train_chunk_constraint", {}
        ).get("overall_violation_rate_d_gt_limit"),
        "F3_passed": bool(f3_passed),
        "F3_violation_reported": bool(f3_reported),
        "violation_gate": violation_gate,
    }
    hard_gates_pass = bool(
        report["input_part"] == "robot"
        and report["frame_dim"] == 19
        and report["input_dim"] == 95
        and report["step"] == step
        and training.get("collapse_alarm") is False
        and dead_unit_max == 0.0
    )
    report["gate_passed"] = bool(hard_gates_pass and (f3_passed or f3_reported))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--step", type=int, default=20_000)
    parser.add_argument("--violation-gate", choices=("stop", "report"), default="stop")
    args = parser.parse_args(argv)
    if args.step <= 0:
        parser.error("--step must be positive")
    report = inspect_run(args.run, step=args.step, violation_gate=args.violation_gate)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
