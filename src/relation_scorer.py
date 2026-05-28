from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


RELATION_LABELS = ["critique", "extension", "application", "background", "unrelated"]


@dataclass
class RelationPrediction:
    label: str
    confidence: float
    reason: str


class HeuristicRelationScorer:
    """Executable fallback scorer.

    This is not meant to replace the LLM. It gives the team a runnable baseline
    and a debugging tool before connecting an actual LLM API or manual labels.
    """

    critique_terms = {
        "limitation", "limitations", "fail", "fails", "failure", "error", "bias", "robustness",
        "critique", "challenge", "challenges", "problem", "problems", "weakness", "weaknesses",
        "inconsistent", "misleading", "hallucination", "overestimate", "underestimate"
    }
    extension_terms = {
        "extend", "extends", "extension", "improve", "improves", "improved", "enhance", "enhances",
        "build", "builds", "upon", "generalize", "generalizes", "new method", "variant", "framework"
    }
    application_terms = {
        "apply", "applies", "application", "use", "uses", "case study", "experiment", "dataset",
        "evaluation", "benchmark", "empirical", "real-world", "domain", "task"
    }
    background_terms = {
        "survey", "overview", "review", "introduction", "foundation", "foundational", "background",
        "tutorial", "taxonomy"
    }

    def score_pair(self, row: pd.Series) -> RelationPrediction:
        text = f"{row.get('candidate_title', '')} {row.get('candidate_abstract', '')}".lower()
        counts = {
            "critique": self._count_terms(text, self.critique_terms),
            "extension": self._count_terms(text, self.extension_terms),
            "application": self._count_terms(text, self.application_terms),
            "background": self._count_terms(text, self.background_terms),
        }
        best_label, best_count = max(counts.items(), key=lambda x: x[1])
        if best_count == 0:
            return RelationPrediction("unrelated", 0.45, "No clear relation-specific textual signal was found.")
        total = sum(counts.values())
        confidence = min(0.95, 0.50 + 0.10 * best_count + 0.05 * (best_count / max(total, 1)))
        return RelationPrediction(best_label, float(confidence), f"Detected {best_label}-related terms in the candidate abstract/title.")

    @staticmethod
    def _count_terms(text: str, terms: set[str]) -> int:
        count = 0
        for term in terms:
            if " " in term:
                count += text.count(term)
            else:
                count += len(re.findall(rf"\b{re.escape(term)}\b", text))
        return count


def build_prompt(row: pd.Series) -> str:
    return f"""Given a target paper and a candidate paper, classify their relationship.

Labels:
1. critique
2. extension
3. application
4. background
5. unrelated

Target title:
{row.get('target_title', '')}

Target abstract:
{row.get('target_abstract', '')}

Candidate title:
{row.get('candidate_title', '')}

Candidate abstract:
{row.get('candidate_abstract', '')}

Citation path:
{row.get('citation_path', '')}

Return JSON only:
{{
  "label": "critique|extension|application|background|unrelated",
  "confidence": 0.0,
  "reason": "one short evidence-based reason"
}}
"""


def parse_relation_response(text: str) -> RelationPrediction:
    """Parse a JSON response from an LLM relation classifier."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return RelationPrediction("unrelated", 0.0, "Could not parse model response.")
        data = json.loads(match.group(0))

    label = str(data.get("label", "unrelated")).strip().lower()
    if label not in RELATION_LABELS:
        label = "unrelated"
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(data.get("reason", "")).strip()
    return RelationPrediction(label, confidence, reason)


def score_pairs_heuristic(pairs: pd.DataFrame) -> pd.DataFrame:
    scorer = HeuristicRelationScorer()
    rows = []
    for row in pairs.itertuples(index=False):
        s = pd.Series(row._asdict())
        pred = scorer.score_pair(s)
        rows.append({
            "target_paper_id": s.get("target_paper_id"),
            "candidate_paper_id": s.get("candidate_paper_id"),
            "predicted_relation": pred.label,
            "relation_confidence": pred.confidence,
            "relation_reason": pred.reason,
        })
    return pd.DataFrame(rows)


def merge_relation_scores(pairs: pd.DataFrame, relation_scores: Optional[pd.DataFrame], desired_relation: str) -> pd.DataFrame:
    df = pairs.copy()
    if relation_scores is not None and not relation_scores.empty:
        keys = ["target_paper_id", "candidate_paper_id"]
        df = df.merge(relation_scores, on=keys, how="left")
    if "predicted_relation" not in df.columns:
        df["predicted_relation"] = ""
    if "relation_confidence" not in df.columns:
        df["relation_confidence"] = 0.0
    df["relation_confidence"] = pd.to_numeric(df["relation_confidence"], errors="coerce").fillna(0.0)
    df["relation_score"] = (df["predicted_relation"].astype(str).str.lower() == desired_relation.lower()).astype(float) * df["relation_confidence"]
    return df
