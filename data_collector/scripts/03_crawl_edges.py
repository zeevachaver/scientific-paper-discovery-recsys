#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import append_jsonl, ensure_dir, load_config, normalize_paper_id, write_csv
from src.s2_client import S2Client


def _paper_from_edge_item(item: dict[str, Any], key: str) -> dict[str, Any] | None:
    paper = item.get(key)
    if not isinstance(paper, dict):
        return None
    pid = normalize_paper_id(paper.get("paperId"))
    if not pid:
        return None
    paper = dict(paper)
    paper["paperId"] = pid
    return paper


def _crawl_one_endpoint(
    client: S2Client,
    anchor_id: str,
    endpoint: str,
    max_items: int,
    limit: int,
    fields: str,
    publication_date_or_year: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return papers, edges, and citation context rows for one anchor endpoint."""
    assert endpoint in {"citations", "references"}
    offset = 0
    papers: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    paper_key = "citingPaper" if endpoint == "citations" else "citedPaper"

    while offset < max_items:
        page_limit = min(limit, max_items - offset)
        params = {
            "fields": fields,
            "offset": offset,
            "limit": page_limit,
            "publicationDateOrYear": publication_date_or_year,
        }
        data = client.get(f"/paper/{anchor_id}/{endpoint}", params=params)
        if not isinstance(data, dict) or data.get("error"):
            break
        batch = data.get("data") or []
        if not batch:
            break

        for item in batch:
            paper = _paper_from_edge_item(item, paper_key)
            if paper is None:
                continue
            neighbor_id = paper["paperId"]
            paper["_source_endpoint"] = f"paper/{endpoint}"
            paper["_anchor_paper_id"] = anchor_id
            papers.append(paper)

            if endpoint == "citations":
                # Actual citation direction: citing paper -> cited paper/anchor.
                source = neighbor_id
                target = anchor_id
                edge_type = "citation"
                citation_path = f"{neighbor_id}|{anchor_id}"
            else:
                # Anchor cites referenced/cited paper.
                source = anchor_id
                target = neighbor_id
                edge_type = "reference"
                citation_path = f"{anchor_id}|{neighbor_id}"

            edges.append({
                "source_paper_id": source,
                "target_paper_id": target,
                "edge_type": edge_type,
                "hop": 1,
                "citation_path": citation_path,
                "anchor_paper_id": anchor_id,
                "neighbor_paper_id": neighbor_id,
            })
            contexts.append({
                "source_paper_id": source,
                "target_paper_id": target,
                "edge_type": edge_type,
                "anchor_paper_id": anchor_id,
                "neighbor_paper_id": neighbor_id,
                "contexts": item.get("contexts"),
                "intents": item.get("intents"),
                "isInfluential": item.get("isInfluential"),
            })

        next_offset = data.get("next")
        if next_offset is None or next_offset == offset:
            break
        offset = int(next_offset)

    return papers, edges, contexts


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl citation/reference edges for anchor papers.")
    parser.add_argument("--config", default="config_collect.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_dir(config["project"].get("output_dir", "data"))
    client = S2Client(config)
    ecfg = config["edges"]
    fields = config["fields"]["edge_paper"]

    anchors_path = out_dir / "anchor_papers.csv"
    if not anchors_path.exists():
        raise FileNotFoundError("anchor_papers.csv not found. Run scripts/02_select_anchors.py first.")
    anchors = pd.read_csv(anchors_path)
    anchor_ids = [normalize_paper_id(x) for x in anchors["paperId"].tolist()]
    anchor_ids = [x for x in anchor_ids if x]

    neighbor_path = out_dir / "edge_neighbor_papers.jsonl"
    if neighbor_path.exists():
        neighbor_path.unlink()

    all_edges: list[dict[str, Any]] = []
    all_contexts: list[dict[str, Any]] = []
    limit = int(ecfg.get("endpoint_limit", 100))
    publication_date_or_year = str(ecfg.get("publicationDateOrYear", "") or "")

    for anchor_id in tqdm(anchor_ids, desc="crawl anchors"):
        c_papers, c_edges, c_contexts = _crawl_one_endpoint(
            client,
            anchor_id,
            "citations",
            int(ecfg.get("citations_per_anchor", 200)),
            limit,
            fields,
            publication_date_or_year,
        )
        r_papers, r_edges, r_contexts = _crawl_one_endpoint(
            client,
            anchor_id,
            "references",
            int(ecfg.get("references_per_anchor", 200)),
            limit,
            fields,
            publication_date_or_year,
        )
        append_jsonl(neighbor_path, c_papers + r_papers)
        all_edges.extend(c_edges + r_edges)
        all_contexts.extend(c_contexts + r_contexts)

    edges_df = pd.DataFrame(all_edges).drop_duplicates(subset=["source_paper_id", "target_paper_id", "edge_type"])
    contexts_df = pd.DataFrame(all_contexts).drop_duplicates(subset=["source_paper_id", "target_paper_id", "edge_type"])
    write_csv(out_dir / "collected_edges.csv", edges_df)
    write_csv(out_dir / "citation_contexts.csv", contexts_df)
    print(f"[DONE] edges={len(edges_df):,} -> {out_dir / 'collected_edges.csv'}")
    print(f"[DONE] neighbor paper metadata -> {neighbor_path}")


if __name__ == "__main__":
    main()
