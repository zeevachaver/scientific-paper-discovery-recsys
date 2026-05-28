from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.clean import clean_edges, clean_papers
from src.config import load_config
from src.graph import build_graph_stats
from src.io_utils import read_jsonl, read_table, write_jsonl, write_table
from src.pair_builder import build_candidate_pairs
from src.rankers import RankingWeights, rank_candidate_pairs
from src.relation_scorer import score_pairs_heuristic
from src.evaluation import evaluate_ranking, evaluate_relation_classification
from src.report_tables import dataset_statistics, top_recommendation_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip_relation_scoring", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("[1/6] Cleaning papers and edges...")
    papers = read_jsonl(cfg.path("paths.raw_papers"))
    edges = read_table(cfg.path("paths.raw_edges"))
    cleaned, cleaning_report = clean_papers(
        papers,
        min_abstract_chars=cfg.get("cleaning.min_abstract_chars", 50),
        low_connectivity_citation_threshold=cfg.get("cleaning.low_connectivity_citation_threshold", 5),
        low_connectivity_reference_threshold=cfg.get("cleaning.low_connectivity_reference_threshold", 5),
    )
    cleaned_edges = clean_edges(edges, valid_paper_ids=set(cleaned["paperId"]))
    write_jsonl(cleaned, cfg.path("paths.cleaned_papers"))
    write_table(cleaned_edges, cfg.path("paths.graph_edges"))
    write_table(cleaning_report, cfg.base_dir / "outputs" / "cleaning_report.csv")

    print("[2/6] Building candidate pairs...")
    pairs = build_candidate_pairs(
        cleaned,
        cleaned_edges,
        n_targets=cfg.get("pair_building.n_targets", 50),
        candidates_per_target=cfg.get("pair_building.candidates_per_target", 200),
        include_two_hop=cfg.get("pair_building.include_two_hop", True),
        max_two_hop_per_target=cfg.get("pair_building.max_two_hop_per_target", 100),
        negatives_per_target=cfg.get("pair_building.negatives_per_target", 50),
        random_state=cfg.get("pair_building.random_state", 172),
    )
    write_table(pairs, cfg.path("paths.candidate_pairs"))

    print("[3/6] Relation scoring...")
    relation_scores = None
    if not args.skip_relation_scoring:
        relation_scores = score_pairs_heuristic(pairs)
        write_table(relation_scores, cfg.path("paths.relation_scores"))

    print("[4/6] Ranking...")
    weights = RankingWeights.from_dict(cfg.get("ranking.weights", {}))
    ranked = rank_candidate_pairs(
        pairs,
        cleaned,
        relation_scores=relation_scores,
        desired_relation=cfg.get("ranking.desired_relation", "extension"),
        weights=weights,
        mmr_lambda=cfg.get("ranking.mmr_lambda", 0.75),
    )
    write_table(ranked, cfg.path("paths.ranked_results"))

    print("[5/6] Evaluation...")
    metrics = evaluate_ranking(
        ranked,
        k_values=cfg.get("evaluation.k_values", [5, 10, 20]),
        desired_relation=cfg.get("ranking.desired_relation", None),
        positive_labels=cfg.get("evaluation.positive_labels", ["critique", "extension", "application"]),
    )
    write_table(metrics, cfg.path("paths.metrics"))
    relation_metrics = evaluate_relation_classification(ranked)
    if not relation_metrics.empty:
        write_table(relation_metrics, cfg.base_dir / "outputs" / "relation_classification_metrics.csv")

    print("[6/6] Report tables...")
    write_table(dataset_statistics(cleaned, cleaned_edges, pairs), cfg.base_dir / "outputs" / "dataset_stats.csv")
    write_table(build_graph_stats(cleaned, cleaned_edges), cfg.base_dir / "outputs" / "graph_stats.csv")
    write_table(top_recommendation_examples(ranked), cfg.base_dir / "outputs" / "recommendation_examples.csv")

    print("Done. Check outputs/ for result tables.")


if __name__ == "__main__":
    main()
