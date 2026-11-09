"""Compare v0.8 flow and quasimetric decisions on identical candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare_candidate_decisions(flow_path: Path, quasi_path: Path) -> dict:
    with np.load(flow_path, allow_pickle=False) as flow, np.load(
        quasi_path, allow_pickle=False
    ) as quasi:
        for name in ("pairs", "board_changed", "dataset_id"):
            if not np.array_equal(flow[name], quasi[name]):
                raise ValueError(f"flow and quasimetric candidate field {name!r} differs")
        if str(flow["judge"].item()) != "flow" or str(quasi["judge"].item()) != "quasimetric":
            raise ValueError("decision artifacts have incorrect judge labels")
        flow_accept = np.asarray(flow["accept"], dtype=np.bool_)
        quasi_accept = np.asarray(quasi["accept"], dtype=np.bool_)
        changed = np.asarray(flow["board_changed"], dtype=np.bool_)
        dataset_id = str(flow["dataset_id"].item())
        if flow_accept.shape != quasi_accept.shape or changed.shape != flow_accept.shape:
            raise ValueError("candidate decision arrays do not align")

    def cells(mask: np.ndarray) -> dict[str, int | float]:
        f = flow_accept[mask]
        q = quasi_accept[mask]
        return {
            "both_accept": int(np.count_nonzero(f & q)),
            "flow_only": int(np.count_nonzero(f & ~q)),
            "quasimetric_only": int(np.count_nonzero(~f & q)),
            "both_reject": int(np.count_nonzero(~f & ~q)),
            "n": int(np.count_nonzero(mask)),
            "agreement_fraction": float(np.mean(f == q)) if np.any(mask) else None,
        }

    all_pairs = np.ones(len(flow_accept), dtype=np.bool_)
    return {
        "dataset_id": dataset_id,
        "candidate_count": int(len(flow_accept)),
        "board_unchanged": cells(~changed),
        "board_changed": cells(changed),
        "all": cells(all_pairs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-candidates", type=Path, required=True)
    parser.add_argument("--quasimetric-candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_candidate_decisions(args.flow_candidates, args.quasimetric_candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
