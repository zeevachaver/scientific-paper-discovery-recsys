# Semantic Scholar Collector for ECS172 Project

This folder contains the **data acquisition scripts** for the relation-aware paper recommendation project.
It collects exactly the files needed by the downstream recommender pipeline:

```text
data/raw_papers.jsonl
data/raw_edges.csv
```

The scripts use these Semantic Scholar Graph API endpoints:

```text
/paper/search/match          # match seed paper titles to paperId
/paper/search/bulk           # collect a large CS paper pool
/paper/{paper_id}/citations  # collect papers that cite an anchor paper
/paper/{paper_id}/references # collect papers cited by an anchor paper
/paper/batch                 # enrich all collected paper IDs with full metadata
```

## Setup

```bash
cd semantic_scholar_collector
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then edit `.env` and put your Semantic Scholar API key:

```text
S2_API_KEY=your_new_key_here
```

Do not commit `.env` to GitHub.

## Quick test run

The default config is intentionally moderate so you can test the pipeline first:

```bash
python collect_all.py --config config_collect.yaml
```

This creates:

```text
data/seed_papers.csv
data/bulk_papers.jsonl
data/anchor_papers.csv
data/collected_edges.csv
data/citation_contexts.csv
data/enriched_papers.jsonl
data/raw_papers.jsonl
data/raw_edges.csv
data/collection_stats.csv
```

Validate the final files:

```bash
python scripts/06_validate_outputs.py --config config_collect.yaml
```

## Larger final run

For the final experiment, edit `config_collect.yaml`:

```yaml
bulk_search:
  pages_per_query: 30

anchors:
  n_anchors: 1000

edges:
  citations_per_anchor: 500
  references_per_anchor: 500
  endpoint_limit: 500
```

With an API key and 1 request/second pacing, this is slow but feasible. The code caches every API response in `raw_cache/`, so interrupted runs can be resumed.

## How edges are represented

`raw_edges.csv` uses actual citation direction:

```text
source_paper_id,target_paper_id,edge_type,hop,citation_path
```

For `/citations`, the API returns papers that cite the anchor, so the edge is:

```text
citing_paper_id -> anchor_paper_id
```

For `/references`, the API returns papers cited by the anchor, so the edge is:

```text
anchor_paper_id -> cited_paper_id
```

The downstream graph code can still find both forward and reverse neighbors.

## Recommended step-by-step commands

```bash
python scripts/00_match_seed_titles.py --config config_collect.yaml
python scripts/01_collect_bulk_pool.py --config config_collect.yaml
python scripts/02_select_anchors.py --config config_collect.yaml
python scripts/03_crawl_edges.py --config config_collect.yaml
python scripts/04_enrich_papers.py --config config_collect.yaml
python scripts/05_build_raw_inputs.py --config config_collect.yaml
python scripts/06_validate_outputs.py --config config_collect.yaml
```

After that, copy these two files into the recommender project:

```text
data/raw_papers.jsonl
data/raw_edges.csv
```

