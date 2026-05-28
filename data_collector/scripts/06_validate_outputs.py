#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_config, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate output schema for downstream recommender pipeline.")
    parser.add_argument("--config", default="config_collect.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(config["project"].get("output_dir", "data"))
    papers_path = out_dir / "raw_papers.jsonl"
    edges_path = out_dir / "raw_edges.csv"

    papers = read_jsonl(papers_path)
    edges = pd.read_csv(edges_path)
    paper_ids = {p.get("paperId") for p in papers if p.get("paperId")}

    required_paper_cols = {"paperId", "title", "abstract", "year", "citationCount", "referenceCount"}
    required_edge_cols = {"source_paper_id", "target_paper_id", "edge_type", "hop", "citation_path"}
    actual_paper_cols = set(papers[0].keys()) if papers else set()
    actual_edge_cols = set(edges.columns)

    missing_papers = required_paper_cols - actual_paper_cols
    missing_edges = required_edge_cols - actual_edge_cols
    if missing_papers:
        raise ValueError(f"raw_papers.jsonl missing columns: {sorted(missing_papers)}")
    if missing_edges:
        raise ValueError(f"raw_edges.csv missing columns: {sorted(missing_edges)}")

    bad_source = (~edges["source_paper_id"].isin(paper_ids)).sum()
    bad_target = (~edges["target_paper_id"].isin(paper_ids)).sum()
    abstract_count = sum(1 for p in papers if isinstance(p.get("abstract"), str) and len(p["abstract"]) >= 50)

    print("Validation passed.")
    print(f"papers: {len(papers):,}")
    print(f"edges: {len(edges):,}")
    print(f"papers with abstracts >= 50 chars: {abstract_count:,}")
    print(f"edges with source missing metadata: {bad_source:,}")
    print(f"edges with target missing metadata: {bad_target:,}")


if __name__ == "__main__":
    main()
