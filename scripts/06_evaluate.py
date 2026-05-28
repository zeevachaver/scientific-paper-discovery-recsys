from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation import evaluate_ranking, evaluate_relation_classification
from src.io_utils import read_table, write_table
from src.report_tables import top_recommendation_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)

    ranked = read_table(cfg.path("paths.ranked_results"))
    metrics = evaluate_ranking(
        ranked,
        k_values=cfg.get("evaluation.k_values", [5, 10, 20]),
        desired_relation=cfg.get("ranking.desired_relation", None),
        positive_labels=cfg.get("evaluation.positive_labels", ["critique", "extension", "application"]),
    )
    write_table(metrics, cfg.path("paths.metrics"))

    rel_metrics = evaluate_relation_classification(ranked)
    if not rel_metrics.empty:
        write_table(rel_metrics, cfg.base_dir / "outputs" / "relation_classification_metrics.csv")
    write_table(top_recommendation_examples(ranked), cfg.base_dir / "outputs" / "recommendation_examples.csv")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
