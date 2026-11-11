"""Build the observed Puzzle object-change set from aligned offline data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from graphcritic.bridges.object_state import observed_object_change_counts
from graphcritic.io import load_sim_state, load_transitions
from graphcritic.modes import state_episode_time


def build_change_set(button_states, episode_ids, times, *, max_horizon=20, min_count=10):
    patterns, counts, minimum_horizon = observed_object_change_counts(
        button_states, episode_ids, times, max_horizon=max_horizon,
    )
    keep = counts >= int(min_count)
    retained = []
    for code, count, min_k in zip(patterns[keep], counts[keep], minimum_horizon[keep], strict=True):
        changed = [
            index for index in range(button_states.shape[1])
            if int(code) & (1 << index)
        ]
        retained.append({
            "code": int(code), "changed_buttons": changed,
            "count": int(count), "min_k": int(min_k),
        })
    omitted = ~keep
    return {
        "schema": "puzzle-observed-object-change-set-v1",
        "button_count": int(button_states.shape[1]),
        "max_horizon": int(max_horizon),
        "min_count": int(min_count),
        "patterns": retained,
        "n_patterns_observed": int(len(patterns)),
        "n_patterns_retained": int(np.count_nonzero(keep)),
        "n_patterns_omitted_below_min_count": int(np.count_nonzero(omitted)),
        "occurrences_omitted_below_min_count": int(np.sum(counts[omitted], dtype=np.int64)),
        "occurrences_all_patterns": int(np.sum(counts, dtype=np.int64)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--sim-state", type=Path, required=True)
    parser.add_argument("--max-horizon", type=int, default=20)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_horizon < 1 or args.min_count < 1:
        parser.error("max-horizon and min-count must be positive")
    started = time.time()
    trans, trans_meta = load_transitions(args.transitions)
    sim_state = load_sim_state(args.sim_state)
    episode, times = state_episode_time(trans)
    sim_state.validate(len(episode))
    if sim_state.button_states is None:
        raise ValueError("object-change set requires aligned button_states")
    report = build_change_set(
        np.asarray(sim_state.button_states, dtype=np.uint8), episode, times,
        max_horizon=args.max_horizon, min_count=args.min_count,
    )
    report.update({
        "dataset_id": trans_meta["dataset_id"],
        "n_states": int(len(episode)),
        "elapsed_seconds": float(time.time() - started),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
