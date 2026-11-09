"""Print 5k-step summaries and collapse alarms from v0.8 training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_log(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"no training rows found in {path}")
    return rows


def summarize_logs(paths: list[Path], *, interval: int = 5000) -> list[dict]:
    if interval < 1:
        raise ValueError("interval must be positive")
    summaries = []
    for path in paths:
        rows = _read_log(path)
        selected = [row for row in rows if int(row["step"]) % interval == 0]
        if rows[-1] not in selected:
            selected.append(rows[-1])
        alarm_rows = [row for row in rows if row.get("collapse_alarm")]
        summaries.append({
            "path": path,
            "rows": selected,
            "alarm_steps": [int(row["step"]) for row in alarm_rows],
        })
    return summaries


def _last_projector_dead(row: dict) -> float | None:
    dead = row.get("diagnostics", {}).get("dead_unit_frac_by_layer", {})
    names = [name for name in dead if name.startswith("projector_")]
    if not names:
        return None
    names.sort(key=lambda name: int(name.split("_")[1]))
    return float(dead[names[-1]])


def print_summaries(summaries: list[dict]) -> None:
    print(
        "| run | step | d(s,g) | d(s,s') | λ | gap | grad norm preclip | "
        "last projector dead | projector var | alarm |"
    )
    print("|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for summary in summaries:
        run_name = summary["path"].parent.name
        for row in summary["rows"]:
            train = row.get("training", {})
            diagnostics = row.get("diagnostics", {})
            values = (
                train.get("distance_goal_mean"),
                train.get("distance_step_mean"),
                train.get("lambda"),
                train.get("constraint_gap"),
                train.get("gradient_norm_preclip"),
                _last_projector_dead(row),
                diagnostics.get("projector_output_mean_variance"),
            )
            formatted = ["n/a" if value is None else f"{float(value):.6g}" for value in values]
            alarm = row.get("collapse_alarm_reason") or "—"
            print(
                f"| {run_name} | {int(row['step'])} | "
                + " | ".join(formatted)
                + f" | {alarm} |"
            )
        alarms = ", ".join(map(str, summary["alarm_steps"])) or "none"
        print(f"\n{run_name}: collapse alarm step(s): {alarms}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--compare", type=Path, nargs="*", default=[])
    parser.add_argument("--interval", type=int, default=5000)
    args = parser.parse_args(argv)
    summaries = summarize_logs([args.log, *args.compare], interval=args.interval)
    print_summaries(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
