# ECS172 Relation-aware Citation Graph Recommender

This code starts **after data acquisition**. It assumes you already have:

- `data/raw_papers.jsonl`: paper metadata from Semantic Scholar or another source
- `data/raw_edges.csv`: citation/reference edges

It implements:

1. data cleaning
2. citation graph construction
3. candidate pair construction
4. TF-IDF semantic similarity baseline
5. graph-distance baseline
6. relation scoring interface, with executable heuristic fallback
7. final ranking
8. evaluation metrics
9. dataset/result summary tables

## Expected input format

### `data/raw_papers.jsonl`
One JSON object per line:

```json
{"paperId":"P1","title":"...","abstract":"...","year":2024,"citationCount":12,"referenceCount":30,"venue":"..."}
```

Required columns:

- `paperId`
- `title`
- `abstract`

Recommended columns:

- `year`
- `citationCount`
- `referenceCount`
- `venue`
- `fieldsOfStudy`
- `s2FieldsOfStudy`

### `data/raw_edges.csv`

```csv
source_paper_id,target_paper_id,edge_type,hop,citation_path
P1,P2,citation,1,"P1|P2"
P1,P3,reference,1,"P1|P3"
```

Meaning:

- `source_paper_id`: usually the target/anchor paper
- `target_paper_id`: candidate paper
- `edge_type`: `citation`, `reference`, `two_hop`, or `negative`
- `hop`: 1, 2, etc.

## Run full pipeline

```bash
pip install -r requirements.txt
python run_pipeline.py --config config.yaml
```

## Run step by step

```bash
python scripts/01_clean.py --config config.yaml
python scripts/02_build_graph.py --config config.yaml
python scripts/03_build_pairs.py --config config.yaml
python scripts/04_rank.py --config config.yaml
python scripts/05_llm_label.py --config config.yaml --mode heuristic
python scripts/06_evaluate.py --config config.yaml
```

## Main outputs

- `data/cleaned_papers.jsonl`
- `data/graph_edges.csv`
- `data/candidate_pairs.csv`
- `outputs/ranked_results.csv`
- `outputs/relation_scores.csv`
- `outputs/metrics.csv`
- `outputs/dataset_stats.csv`

## Important project framing

You can collect a large graph, but the later cleaning, ranking, relation scoring, and evaluation logic is the same. The expensive part is labeling/evaluation, so you can rank over a large candidate pool but evaluate a sampled target-candidate set.
