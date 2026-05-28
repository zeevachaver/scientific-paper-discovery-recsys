#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import dedupe_papers, ensure_dir, load_config, normalize_paper_id, read_jsonl, safe_authors, write_jsonl


def compact_paper(p: dict[str, Any]) -> dict[str, Any]:
    """Flatten only the fields needed by the downstream recommender pipeline."""
    external_ids = p.get("externalIds") if isinstance(p.get("externalIds"), dict) else {}
    publication_venue = p.get("publicationVenue") if isinstance(p.get("publicationVenue"), dict) else {}
    journal = p.get("journal") if isinstance(p.get("journal"), dict) else {}
    open_pdf = p.get("openAccessPdf") if isinstance(p.get("openAccessPdf"), dict) else {}
    return {
        "paperId": normalize_paper_id(p.get("paperId")),
        "corpusId": p.get("corpusId") or external_ids.get("CorpusId"),
        "doi": external_ids.get("DOI"),
        "url": p.get("url"),
        "title": p.get("title"),
        "abstract": p.get("abstract"),
        "year": p.get("year"),
        "venue": p.get("venue"),
        "publicationVenue": publication_venue.get("name") or publication_venue.get("id"),
        "publicationDate": p.get("publicationDate"),
        "journal": journal.get("name") or journal.get("volume"),
        "citationCount": p.get("citationCount"),
        "referenceCount": p.get("referenceCount"),
        "influentialCitationCount": p.get("influentialCitationCount"),
        "isOpenAccess": p.get("isOpenAccess"),
        "openAccessPdfUrl": open_pdf.get("url"),
        "fieldsOfStudy": json.dumps(p.get("fieldsOfStudy"), ensure_ascii=False),
        "s2FieldsOfStudy": json.dumps(p.get("s2FieldsOfStudy"), ensure_ascii=False),
        "publicationTypes": json.dumps(p.get("publicationTypes"), ensure_ascii=False),
        "authors": safe_authors(p.get("authors")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/raw_papers.jsonl and data/raw_edges.csv for the recommender pipeline.")
    parser.add_argument("--config", default="config_collect.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_dir(config["project"].get("output_dir", "data"))
    keep_edges_without_metadata = bool(config.get("build_outputs", {}).get("keep_edges_without_metadata", False))

    source_path = out_dir / "enriched_papers.jsonl"
    if not source_path.exists():
        source_path = out_dir / "bulk_papers.jsonl"
    papers = dedupe_papers(read_jsonl(source_path))
    compact = [compact_paper(p) for p in papers]
    compact = [p for p in compact if p.get("paperId")]
    write_jsonl(out_dir / "raw_papers.jsonl", compact)

    edges_path = out_dir / "collected_edges.csv"
    if not edges_path.exists():
        raise FileNotFoundError("collected_edges.csv not found. Run scripts/03_crawl_edges.py first.")
    edges = pd.read_csv(edges_path)
    required = ["source_paper_id", "target_paper_id", "edge_type", "hop", "citation_path"]
    for col in required:
        if col not in edges.columns:
            raise ValueError(f"Missing required edge column: {col}")
    edges["source_paper_id"] = edges["source_paper_id"].map(normalize_paper_id)
    edges["target_paper_id"] = edges["target_paper_id"].map(normalize_paper_id)
    edges = edges.dropna(subset=["source_paper_id", "target_paper_id"])
    edges = edges[edges["source_paper_id"] != edges["target_paper_id"]].copy()
    edges = edges.drop_duplicates(subset=["source_paper_id", "target_paper_id", "edge_type"])

    if not keep_edges_without_metadata:
        valid_ids = {p["paperId"] for p in compact}
        edges = edges[edges["source_paper_id"].isin(valid_ids) & edges["target_paper_id"].isin(valid_ids)].copy()

    output_cols = ["source_paper_id", "target_paper_id", "edge_type", "hop", "citation_path"]
    # Keep diagnostic columns too, but put required columns first.
    extra_cols = [c for c in edges.columns if c not in output_cols]
    edges[output_cols + extra_cols].to_csv(out_dir / "raw_edges.csv", index=False)

    # Dataset statistics for final report.
    stats = {
        "num_raw_papers": len(compact),
        "num_raw_edges": len(edges),
        "num_papers_with_abstract": sum(1 for p in compact if isinstance(p.get("abstract"), str) and len(p.get("abstract", "")) > 0),
        "num_papers_with_title": sum(1 for p in compact if isinstance(p.get("title"), str) and len(p.get("title", "")) > 0),
    }
    pd.DataFrame([stats]).to_csv(out_dir / "collection_stats.csv", index=False)
    print(f"[DONE] wrote {out_dir / 'raw_papers.jsonl'} ({len(compact):,} papers)")
    print(f"[DONE] wrote {out_dir / 'raw_edges.csv'} ({len(edges):,} edges)")
    print(f"[DONE] wrote {out_dir / 'collection_stats.csv'}")


if __name__ == "__main__":
    main()
