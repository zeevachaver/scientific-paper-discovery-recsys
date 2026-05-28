from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.graph import build_graph_stats
from src.io_utils import read_jsonl, read_table, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    papers = read_jsonl(cfg.path("paths.cleaned_papers"))
    edges = read_table(cfg.path("paths.graph_edges"))
    stats = build_graph_stats(papers, edges)
    write_table(stats, cfg.base_dir / "outputs" / "graph_stats.csv")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
