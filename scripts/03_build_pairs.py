from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import read_jsonl, read_table, write_table
from src.pair_builder import build_candidate_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    papers = read_jsonl(cfg.path("paths.cleaned_papers"))
    edges = read_table(cfg.path("paths.graph_edges"))
    pairs = build_candidate_pairs(
        papers,
        edges,
        n_targets=cfg.get("pair_building.n_targets", 50),
        candidates_per_target=cfg.get("pair_building.candidates_per_target", 200),
        include_two_hop=cfg.get("pair_building.include_two_hop", True),
        max_two_hop_per_target=cfg.get("pair_building.max_two_hop_per_target", 100),
        negatives_per_target=cfg.get("pair_building.negatives_per_target", 50),
        random_state=cfg.get("pair_building.random_state", 172),
    )
    write_table(pairs, cfg.path("paths.candidate_pairs"))
    print(f"Candidate pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
