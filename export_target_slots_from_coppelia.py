"""Export the phase-to-ring-slot table once; fast training then needs no simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coppeliasim_attractor_closed_loop import CoppeliaAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--phases", type=int, default=64)
    parser.add_argument("--ring-size", type=int, default=32)
    args = parser.parse_args()
    adapter = CoppeliaAdapter(
        args.host,
        args.port,
        ["/MTB/axis", "/MTB/link/axis"],
        args.phases,
        args.ring_size,
    )
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "ring_size": args.ring_size,
            "phases": args.phases,
            "ring0_target_slots": [value + 1 for value in adapter.target_slots[0]],
            "ring1_target_slots": [value + 1 for value in adapter.target_slots[1]],
            "ik_angle_ranges": adapter.ik_angle_ranges,
            "source": "CoppeliaSim IK range export",
        }, indent=2), encoding="utf-8")
    finally:
        adapter.stop()
    print(args.output)


if __name__ == "__main__":
    main()
