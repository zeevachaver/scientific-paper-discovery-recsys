#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import ensure_dir, load_config, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Select anchor papers for citation/reference crawling.")
    parser.add_argument("--config", default="config_collect.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_dir(config["project"].get("output_dir", "data"))
    acfg = config["anchors"]
    rng = np.random.default_rng(int(acfg.get("random_seed", 172)))

    papers = pd.DataFrame(read_jsonl(out_dir / "bulk_papers.jsonl"))
    if papers.empty:
        raise FileNotFoundError("No bulk_papers.jsonl found. Run scripts/01_collect_bulk_pool.py first.")

    papers = papers.dropna(subset=["paperId"]).drop_duplicates("paperId").copy()
    papers["citationCount"] = pd.to_numeric(papers.get("citationCount", 0), errors="coerce").fillna(0)
    papers["year"] = pd.to_numeric(papers.get("year", np.nan), errors="coerce")
    papers["abstract_len"] = papers.get("abstract", "").fillna("").astype(str).str.len()

    # Prefer papers with abstracts, because relation scoring needs target abstracts.
    pool = papers[(papers["abstract_len"] >= 50) & (papers["title"].fillna("").astype(str).str.len() > 0)].copy()
    if pool.empty:
        pool = papers.copy()

    n = int(acfg.get("n_anchors", 200))
    high_n = int(n * float(acfg.get("high_citation_fraction", 0.4)))
    recent_n = int(n * float(acfg.get("recent_fraction", 0.3)))
    random_n = n - high_n - recent_n
    selected_ids: list[str] = []

    if bool(acfg.get("include_seed_papers", True)) and (out_dir / "seed_papers.csv").exists():
        seed_df = pd.read_csv(out_dir / "seed_papers.csv")
        selected_ids.extend([pid for pid in seed_df.get("paperId", []) if isinstance(pid, str) and pid])

    high = pool[~pool["paperId"].isin(selected_ids)].sort_values("citationCount", ascending=False).head(high_n)
    selected_ids.extend(high["paperId"].tolist())

    recent_year = int(acfg.get("recent_year_threshold", 2021))
    recent_pool = pool[(pool["year"] >= recent_year) & (~pool["paperId"].isin(selected_ids))]
    if len(recent_pool) > 0:
        recent = recent_pool.sample(n=min(recent_n, len(recent_pool)), random_state=int(acfg.get("random_seed", 172)))
        selected_ids.extend(recent["paperId"].tolist())

    remaining = pool[~pool["paperId"].isin(selected_ids)]
    if len(remaining) > 0 and random_n > 0:
        random_df = remaining.sample(n=min(random_n, len(remaining)), random_state=int(acfg.get("random_seed", 172)) + 1)
        selected_ids.extend(random_df["paperId"].tolist())

    # Fill up if seed papers created duplicates or pools were small.
    selected_ids = list(dict.fromkeys(selected_ids))
    if len(selected_ids) < n:
        extra = pool[~pool["paperId"].isin(selected_ids)].head(n - len(selected_ids))["paperId"].tolist()
        selected_ids.extend(extra)

    anchors = papers[papers["paperId"].isin(selected_ids[:n])].copy()
    anchors["anchor_rank"] = anchors["paperId"].map({pid: i for i, pid in enumerate(selected_ids[:n])})
    anchors = anchors.sort_values("anchor_rank")
    keep_cols = [c for c in ["anchor_rank", "paperId", "title", "year", "citationCount", "referenceCount", "venue", "abstract"] if c in anchors.columns]
    anchors[keep_cols].to_csv(out_dir / "anchor_papers.csv", index=False)
    print(f"[DONE] selected {len(anchors):,} anchors -> {out_dir / 'anchor_papers.csv'}")


if __name__ == "__main__":
    main()
