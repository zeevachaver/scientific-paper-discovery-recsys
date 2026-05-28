from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import read_jsonl, read_table, write_table
from src.rankers import RankingWeights, rank_candidate_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--relation_scores", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    papers = read_jsonl(cfg.path("paths.cleaned_papers"))
    pairs = read_table(cfg.path("paths.candidate_pairs"))
    relation_scores = None
    relation_path = Path(args.relation_scores) if args.relation_scores else cfg.path("paths.relation_scores")
    if relation_path.exists():
        relation_scores = read_table(relation_path)

    ranked = rank_candidate_pairs(
        pairs,
        papers,
        relation_scores=relation_scores,
        desired_relation=cfg.get("ranking.desired_relation", "extension"),
        weights=RankingWeights.from_dict(cfg.get("ranking.weights", {})),
        mmr_lambda=cfg.get("ranking.mmr_lambda", 0.75),
    )
    write_table(ranked, cfg.path("paths.ranked_results"))
    print(f"Ranked pairs: {len(ranked)}")


if __name__ == "__main__":
    main()
