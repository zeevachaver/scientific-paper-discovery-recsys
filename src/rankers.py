from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .features import TextSimilarityModel, graph_score_from_hop, log_normalize_count, mmr_diversity_scores
from .relation_scorer import merge_relation_scores


@dataclass
class RankingWeights:
    semantic: float = 0.35
    graph: float = 0.25
    relation: float = 0.25
    citation: float = 0.10
    diversity: float = 0.05

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "RankingWeights":
        return cls(**{k: float(v) for k, v in d.items() if hasattr(cls, k)})


def rank_candidate_pairs(
    pairs: pd.DataFrame,
    papers: pd.DataFrame,
    relation_scores: Optional[pd.DataFrame] = None,
    desired_relation: str = "extension",
    weights: RankingWeights | None = None,
    mmr_lambda: float = 0.75,
) -> pd.DataFrame:
    """Rank target-candidate paper pairs.

    Returns one row per pair with component scores and final_score.
    """
    if weights is None:
        weights = RankingWeights()

    df = pairs.copy()
    if df.empty:
        return df

    df["target_text"] = (df["target_title"].fillna("") + ". " + df["target_abstract"].fillna("")).str.strip()
    df["candidate_text"] = (df["candidate_title"].fillna("") + ". " + df["candidate_abstract"].fillna("")).str.strip()

    corpus = pd.concat([
        papers.get("text_for_retrieval", papers["title"].fillna("") + ". " + papers["abstract"].fillna("")),
        df["target_text"],
        df["candidate_text"],
    ], ignore_index=True).fillna("").astype(str)
    text_model = TextSimilarityModel.fit(corpus)

    df["semantic_score"] = text_model.pairwise_scores(df["target_text"], df["candidate_text"])
    df["graph_score"] = graph_score_from_hop(df["hop"])
    df["citation_score"] = log_normalize_count(df.get("candidate_citationCount", pd.Series([0] * len(df))))

    df = merge_relation_scores(df, relation_scores=relation_scores, desired_relation=desired_relation)

    # Diversity score is computed within each target's candidate set.
    df["diversity_score"] = 0.0
    for target_id, idx in df.groupby("target_paper_id").groups.items():
        sub = df.loc[idx]
        relevance = sub["semantic_score"].to_numpy(dtype=float)
        div_scores = mmr_diversity_scores(
            relevance_scores=relevance,
            candidate_texts=sub["candidate_text"].fillna("").astype(str).tolist(),
            vectorizer=text_model.vectorizer,
            lambda_=mmr_lambda,
        )
        df.loc[idx, "diversity_score"] = div_scores

    df["final_score"] = (
        weights.semantic * df["semantic_score"].astype(float)
        + weights.graph * df["graph_score"].astype(float)
        + weights.relation * df["relation_score"].astype(float)
        + weights.citation * df["citation_score"].astype(float)
        + weights.diversity * df["diversity_score"].astype(float)
    )

    df["rank"] = df.groupby("target_paper_id")["final_score"].rank(method="first", ascending=False).astype(int)
    df = df.sort_values(["target_paper_id", "rank"]).reset_index(drop=True)
    return df


def make_baseline_rankings(pairs: pd.DataFrame, papers: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return separate baseline rankings for ablation studies."""
    baselines = {}
    for name, weights in {
        "semantic_only": RankingWeights(semantic=1, graph=0, relation=0, citation=0, diversity=0),
        "graph_only": RankingWeights(semantic=0, graph=1, relation=0, citation=0, diversity=0),
        "citation_only": RankingWeights(semantic=0, graph=0, relation=0, citation=1, diversity=0),
        "semantic_graph": RankingWeights(semantic=0.6, graph=0.4, relation=0, citation=0, diversity=0),
    }.items():
        ranked = rank_candidate_pairs(pairs, papers, relation_scores=None, desired_relation="extension", weights=weights)
        ranked["baseline"] = name
        baselines[name] = ranked
    return baselines
