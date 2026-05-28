#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import dedupe_papers, ensure_dir, load_config, normalize_paper_id, read_jsonl, write_jsonl
from src.s2_client import S2Client


def chunks(xs: list[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich all collected paper IDs using /paper/batch.")
    parser.add_argument("--config", default="config_collect.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_dir(config["project"].get("output_dir", "data"))
    client = S2Client(config)
    fields = config["fields"]["paper"]
    batch_size = int(config.get("enrich", {}).get("batch_size", 500))

    id_set: set[str] = set()
    base_papers = []
    for name in ["bulk_papers.jsonl", "seed_papers.jsonl", "edge_neighbor_papers.jsonl"]:
        rows = read_jsonl(out_dir / name)
        base_papers.extend(rows)
        for p in rows:
            pid = normalize_paper_id(p.get("paperId"))
            if pid:
                id_set.add(pid)

    if (out_dir / "collected_edges.csv").exists():
        edges = pd.read_csv(out_dir / "collected_edges.csv")
        for col in ["source_paper_id", "target_paper_id", "anchor_paper_id", "neighbor_paper_id"]:
            if col in edges.columns:
                for value in edges[col].dropna().tolist():
                    pid = normalize_paper_id(value)
                    if pid:
                        id_set.add(pid)

    ids = sorted(id_set)
    print(f"[ENRICH] unique paper IDs: {len(ids):,}; batch_size={batch_size}")
    enriched = []
    for batch in tqdm(list(chunks(ids, batch_size)), desc="paper/batch"):
        data = client.post("/paper/batch", params={"fields": fields}, json_body={"ids": batch})
        if isinstance(data, list):
            enriched.extend([p for p in data if isinstance(p, dict) and p.get("paperId")])

    # Merge in any metadata we already had if batch returned null fields.
    by_id = {p["paperId"]: p for p in dedupe_papers(base_papers)}
    for p in enriched:
        pid = normalize_paper_id(p.get("paperId"))
        if not pid:
            continue
        old = by_id.get(pid, {})
        merged = dict(old)
        merged.update({k: v for k, v in p.items() if v is not None})
        by_id[pid] = merged

    final = dedupe_papers(by_id.values())
    write_jsonl(out_dir / "enriched_papers.jsonl", final)
    print(f"[DONE] enriched papers={len(final):,} -> {out_dir / 'enriched_papers.jsonl'}")


if __name__ == "__main__":
    main()
