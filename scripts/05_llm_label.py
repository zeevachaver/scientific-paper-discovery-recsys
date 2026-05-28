from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import read_table, write_table
from src.relation_scorer import build_prompt, score_pairs_heuristic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--mode", choices=["heuristic", "prompts"], default="heuristic")
    args = parser.parse_args()
    cfg = load_config(args.config)

    pairs = read_table(cfg.path("paths.candidate_pairs"))
    if args.mode == "heuristic":
        scores = score_pairs_heuristic(pairs)
        write_table(scores, cfg.path("paths.relation_scores"))
        print(f"Relation scores written: {len(scores)}")
    else:
        prompts = pairs[["target_paper_id", "candidate_paper_id"]].copy()
        prompts["prompt"] = pairs.apply(build_prompt, axis=1)
        out = cfg.base_dir / "outputs" / "relation_prompts.csv"
        write_table(prompts, out)
        print(f"Prompts written to {out}")


if __name__ == "__main__":
    main()
