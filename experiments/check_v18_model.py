#!/usr/bin/env python3
"""Enforce the v1.8 robot-only quasimetric training gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _last_jsonl(path: Path) -> dict:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no records in {path}")
    value = json.loads(rows[-1])
    if not isinstance(value, dict):
        raise ValueError(f"last record in {path} is not a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = json.loads((args.run / "step_020000" / "metadata.json").read_text(encoding="utf-8"))
    training = _last_jsonl(args.run / "train_metrics.jsonl")
    heldout = _last_jsonl(args.run / "training.jsonl")
    violation = heldout.get("heldout_chunk_constraint", {}).get("overall_violation_rate_d_gt_limit")
    report = {
        "input_part": metadata.get("input_part"),
        "frame_dim": metadata.get("frame_dim"),
        "input_dim": metadata.get("input_dim"),
        "step": metadata.get("step"),
        "collapse_alarm": training.get("collapse_alarm"),
        "collapse_alarm_reason": training.get("collapse_alarm_reason"),
        "heldout_chunk_violation_d_gt_6_25": violation,
        "F3_passed": violation is not None and float(violation) <= 0.05,
    }
    report["gate_passed"] = bool(
        report["input_part"] == "robot"
        and report["frame_dim"] == 19
        and report["input_dim"] == 95
        and report["step"] == 20_000
        and training.get("collapse_alarm") is False
        and report["F3_passed"]
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
