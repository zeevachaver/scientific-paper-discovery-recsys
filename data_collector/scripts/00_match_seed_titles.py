#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import ensure_dir, load_config, write_jsonl
from src.s2_client import S2Client


def main() -> None:
    parser = argparse.ArgumentParser(description="Match seed paper titles to Semantic Scholar paperIds.")
    parser.add_argument("--config", default="config_collect.yaml")
    parser.add_argument("--seeds", default="data/seeds.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_dir(config["project"].get("output_dir", "data"))
    client = S2Client(config)
    fields = config["fields"]["paper"]

    seeds = pd.read_csv(args.seeds)
    if "title" not in seeds.columns:
        raise ValueError("seeds.csv must contain a 'title' column")

    rows = []
    paper_rows = []
    for r in seeds.itertuples(index=False):
        title = str(getattr(r, "title")).strip()
        if not title:
            continue
        print(f"[MATCH] {title}")
        data = client.get("/paper/search/match", params={"query": title, "fields": fields})
        if isinstance(data, dict) and data.get("paperId"):
            paper = dict(data)
            paper_rows.append(paper)
            rows.append({
                "seed_id": getattr(r, "seed_id", ""),
                "input_title": title,
                "topic": getattr(r, "topic", ""),
                "paperId": paper.get("paperId"),
                "matched_title": paper.get("title"),
                "matchScore": paper.get("matchScore"),
                "year": paper.get("year"),
                "citationCount": paper.get("citationCount"),
            })
        else:
            rows.append({
                "seed_id": getattr(r, "seed_id", ""),
                "input_title": title,
                "topic": getattr(r, "topic", ""),
                "paperId": "",
                "matched_title": "",
                "matchScore": "",
                "year": "",
                "citationCount": "",
            })

    pd.DataFrame(rows).to_csv(out_dir / "seed_papers.csv", index=False)
    write_jsonl(out_dir / "seed_papers.jsonl", paper_rows)
    print(f"[DONE] wrote {out_dir / 'seed_papers.csv'} and {out_dir / 'seed_papers.jsonl'}")


if __name__ == "__main__":
    main()
