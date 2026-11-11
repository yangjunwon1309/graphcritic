"""Judge candidate bridges once, then summarize every bridge budget."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-b", type=int, required=True)
    parser.add_argument("--out", required=True)
    args, passthrough = parser.parse_known_args(argv)
    if args.max_b < 0:
        parser.error("--max-b must be non-negative")
    if "--b-max" in passthrough or "--sweep-only" in passthrough:
        parser.error("--max-b controls the internal path budget")
    return args, passthrough


def main(argv=None):
    args, passthrough = _parse(argv)
    # The graph builder owns candidate validation, optimistic C0 filtering,
    # model inference, and the single layered Dijkstra invocation. Sweep-only
    # mode writes its compact JSON directly to --out without graph artifacts.
    from experiments.build_judged_graph import main as build_main

    return build_main(passthrough + [
        "--out", args.out, "--b-max", str(args.max_b), "--sweep-only",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
