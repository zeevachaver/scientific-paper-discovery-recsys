#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import dedupe_papers, ensure_dir, load_config, write_jsonl
from src.s2_client import S2Client


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a large CS paper pool using /paper/search/bulk.")
    parser.add_argument("--config", default="config_collect.yaml")
    parser.add_argument("--pages-per-query", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_dir(config["project"].get("output_dir", "data"))
    client = S2Client(config)

    bcfg = config["bulk_search"]
    fields = config["fields"]["paper"]
    pages_per_query = args.pages_per_query or int(bcfg.get("pages_per_query", 5))

    all_papers = []
    query_summaries = []

    for query in bcfg["queries"]:
        token = None
        total_seen_for_query = 0
        print(f"\n[BULK] query={query!r}")
        for page in tqdm(range(pages_per_query), desc=f"bulk:{query[:25]}"):
            params = {
                "query": query,
                "fields": fields,
                "sort": bcfg.get("sort", "paperId:asc"),
                "year": bcfg.get("year"),
                "fieldsOfStudy": bcfg.get("fieldsOfStudy"),
                "publicationTypes": bcfg.get("publicationTypes"),
                "minCitationCount": bcfg.get("minCitationCount"),
                "token": token,
            }
            data = client.get("/paper/search/bulk", params=params)
            if not isinstance(data, dict):
                break
            batch = data.get("data") or []
            for p in batch:
                p["_source_query"] = query
                p["_source_endpoint"] = "paper/search/bulk"
            all_papers.extend(batch)
            total_seen_for_query += len(batch)
            token = data.get("token")
            if not token or not batch:
                break
        query_summaries.append({"query": query, "papers_collected_raw": total_seen_for_query})

    deduped = dedupe_papers(all_papers)
    write_jsonl(out_dir / "bulk_papers.jsonl", deduped)
    pd.DataFrame(query_summaries).to_csv(out_dir / "bulk_query_summary.csv", index=False)
    print(f"[DONE] raw={len(all_papers):,}, deduped={len(deduped):,}")
    print(f"[DONE] wrote {out_dir / 'bulk_papers.jsonl'}")


if __name__ == "__main__":
    main()
