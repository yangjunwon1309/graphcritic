#!/usr/bin/env python3
"""Score v1.8 cube-double object-matched candidates with a shared d_theta model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphcritic.bridges.cube_graph import score_cube_candidate_cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-double-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair-batch-size", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.pair_batch_size < 1:
        parser.error("--pair-batch-size must be positive")
    report = score_cube_candidate_cache(
        args.cube_double_dataset, args.checkpoint, args.candidate_cache,
        args.out, pair_batch_size=args.pair_batch_size,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
