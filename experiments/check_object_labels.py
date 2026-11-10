"""Run the v0.8.2 D0 offline object-contact label audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphcritic.bridges.object_state import prepare_puzzle_object_data
from graphcritic.io import load_sim_state, load_transitions


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    if args.window != 5:
        raise SystemExit("v0.8.2 object labels require --window 5")
    trans, transition_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    data = prepare_puzzle_object_data(
        trans, sim_state, seed=args.seed, window=args.window
    )
    result = {
        "diagnostic": "v0.8.2_object_contact_label_check",
        "dataset_id": transition_meta["dataset_id"],
        "transitions": str(args.transitions),
        "sim_state": str(args.sim_state),
        "window": int(args.window),
        **data.label_report,
        "position_q999_displacement": data.movement_report["position_q999_displacement"],
        "epsilon_position": data.movement_report["epsilon_position"],
        "movement_report": data.movement_report,
        "event_window_pairs": {
            "count": int(len(data.event_source_ids)),
            "coverage_fraction": data.label_report["event_pair_interval_fraction"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0 if data.label_report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
