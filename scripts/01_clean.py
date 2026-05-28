from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clean import clean_edges, clean_papers
from src.config import load_config
from src.io_utils import read_jsonl, read_table, write_jsonl, write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    papers = read_jsonl(cfg.path("paths.raw_papers"))
    edges = read_table(cfg.path("paths.raw_edges"))
    cleaned, report = clean_papers(
        papers,
        min_abstract_chars=cfg.get("cleaning.min_abstract_chars", 50),
        low_connectivity_citation_threshold=cfg.get("cleaning.low_connectivity_citation_threshold", 5),
        low_connectivity_reference_threshold=cfg.get("cleaning.low_connectivity_reference_threshold", 5),
    )
    cleaned_edges = clean_edges(edges, valid_paper_ids=set(cleaned["paperId"]))

    write_jsonl(cleaned, cfg.path("paths.cleaned_papers"))
    write_table(cleaned_edges, cfg.path("paths.graph_edges"))
    write_table(report, cfg.base_dir / "outputs" / "cleaning_report.csv")
    print(f"Cleaned papers: {len(cleaned)}")
    print(f"Cleaned edges: {len(cleaned_edges)}")


if __name__ == "__main__":
    main()
