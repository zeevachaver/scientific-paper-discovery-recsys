#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STEPS = [
    "scripts/00_match_seed_titles.py",
    "scripts/01_collect_bulk_pool.py",
    "scripts/02_select_anchors.py",
    "scripts/03_crawl_edges.py",
    "scripts/04_enrich_papers.py",
    "scripts/05_build_raw_inputs.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Semantic Scholar collection pipeline.")
    parser.add_argument("--config", default="config_collect.yaml")
    parser.add_argument("--start-step", type=int, default=0, help="0-based step index")
    parser.add_argument("--stop-step", type=int, default=len(STEPS) - 1, help="0-based step index")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    for i, step in enumerate(STEPS):
        if i < args.start_step or i > args.stop_step:
            continue
        print(f"\n========== STEP {i}: {step} ==========")
        cmd = [sys.executable, str(root / step), "--config", args.config]
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
