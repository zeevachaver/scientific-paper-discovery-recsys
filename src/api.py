from __future__ import annotations

from pathlib import Path
import sys
import os
from functools import lru_cache
import re

from flask import Flask, request, jsonify, send_from_directory
import requests

# Ensure the repo root is on sys.path when running this file directly from src/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.io_utils import read_table, read_jsonl
from src.features import TextSimilarityModel
from src.pair_builder import build_candidate_pairs
from src.rankers import rank_candidate_pairs, RankingWeights

import parse as parser


APP = Flask(__name__, static_folder="..")


def _first_nonempty(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return None


def _paper_lookup_frame(papers):
    if papers.empty or "paperId" not in papers.columns:
        return {}
    return papers.set_index("paperId", drop=False).to_dict(orient="index")


def _normalize_title(value: object) -> str:
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _build_title_index(papers):
    title_index: dict[str, list[str]] = {}
    if papers.empty or "paperId" not in papers.columns or "title" not in papers.columns:
        return title_index
    for paper_id, title in papers[["paperId", "title"]].itertuples(index=False):
        normalized = _normalize_title(title)
        if not normalized:
            continue
        title_index.setdefault(normalized, []).append(paper_id)
    return title_index


def _resolve_target_paper_ids(papers, parsed: dict, max_targets: int = 5) -> list[str]:
    title_index = _build_title_index(papers)
    normalized_titles = list(title_index.items())

    query_terms: list[str] = []
    target_paper = parsed.get("target_paper")
    if target_paper:
        query_terms.append(target_paper)
    query_terms.extend(parsed.get("search_terms") or [])

    resolved: list[str] = []
    seen: set[str] = set()

    def add_ids(ids: list[str]) -> None:
        for paper_id in ids:
            if paper_id not in seen:
                resolved.append(paper_id)
                seen.add(paper_id)

    for term in query_terms:
        normalized = _normalize_title(term)
        if not normalized:
            continue
        exact_ids = title_index.get(normalized, [])
        if exact_ids:
            add_ids(exact_ids)
        if len(resolved) >= max_targets:
            return resolved[:max_targets]

    if target_paper and len(resolved) < max_targets:
        normalized_target = _normalize_title(target_paper)
        for title_norm, paper_ids in normalized_titles:
            if normalized_target in title_norm or title_norm in normalized_target:
                add_ids(paper_ids)
                if len(resolved) >= max_targets:
                    break

    if len(resolved) < max_targets:
        for term in parsed.get("search_terms") or []:
            normalized_term = _normalize_title(term)
            if not normalized_term:
                continue
            for title_norm, paper_ids in normalized_titles:
                if normalized_term in title_norm:
                    add_ids(paper_ids)
                    if len(resolved) >= max_targets:
                        return resolved[:max_targets]

    if len(resolved) < max_targets:
        corpus = (papers.get("title", "").fillna("") + ". " + papers.get("abstract", "").fillna("")).astype(str).tolist()
        if corpus:
            query_text = " ".join(parsed.get("search_terms") or [target_paper or ""]).strip()
            if query_text:
                model = TextSimilarityModel.fit(corpus)
                scores = model.query_to_candidates(query_text, corpus)
                top_idx = list(scores.argsort()[::-1][:max_targets])
                for idx in top_idx:
                    paper_id = papers.iloc[idx].get("paperId")
                    if paper_id:
                        add_ids([paper_id])
                    if len(resolved) >= max_targets:
                        return resolved[:max_targets]

    return resolved[:max_targets]


@lru_cache(maxsize=2048)
def _fetch_semantic_scholar_link(paper_id: str) -> str | None:
    base_url = "https://api.semanticscholar.org/graph/v1"
    fields = "url,openAccessPdf,externalIds,title"
    headers = {"Accept": "application/json"}
    api_key = os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = requests.get(
            f"{base_url}/paper/{paper_id}",
            params={"fields": fields},
            headers=headers,
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None

    url = _first_nonempty(
        data.get("url"),
        (data.get("openAccessPdf") or {}).get("url"),
    )
    if url:
        return url

    doi = _first_nonempty((data.get("externalIds") or {}).get("DOI"))
    if doi:
        return f"https://doi.org/{doi.lstrip('https://doi.org/').strip()}"

    return None


@lru_cache(maxsize=2048)
def _fetch_semantic_scholar_link_by_title(title: str) -> str | None:
    base_url = "https://api.semanticscholar.org/graph/v1"
    fields = "url,openAccessPdf,externalIds,title"
    headers = {"Accept": "application/json"}
    api_key = os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = requests.get(
            f"{base_url}/paper/search/match",
            params={"query": title, "fields": fields},
            headers=headers,
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None

    candidate = data.get("data") or []
    if not candidate:
        return None

    first = candidate[0] if isinstance(candidate, list) else None
    if not isinstance(first, dict):
        return None

    url = _first_nonempty(
        first.get("url"),
        (first.get("openAccessPdf") or {}).get("url"),
    )
    if url:
        return url

    doi = _first_nonempty((first.get("externalIds") or {}).get("DOI"))
    if doi:
        return f"https://doi.org/{doi.lstrip('https://doi.org/').strip()}"

    return None


def resolve_paper_url(result_row, paper_meta: dict | None) -> str | None:
    paper_meta = paper_meta or {}
    raw_doi = _first_nonempty(
        result_row.get("candidate_doi"),
        result_row.get("doi"),
        paper_meta.get("doi"),
        paper_meta.get("DOI"),
        paper_meta.get("externalIds", {}).get("DOI") if isinstance(paper_meta.get("externalIds"), dict) else None,
    )
    url = _first_nonempty(
        result_row.get("candidate_url"),
        result_row.get("paper_url"),
        result_row.get("url"),
        result_row.get("openAccessPdfUrl"),
        paper_meta.get("url"),
        paper_meta.get("paper_url"),
        paper_meta.get("openAccessPdfUrl"),
    )
    if url:
        return url
    if raw_doi:
        return f"https://doi.org/{raw_doi.lstrip('https://doi.org/').strip()}"

    paper_id = _first_nonempty(result_row.get("candidate_paper_id"), paper_meta.get("paperId"))
    if paper_id:
        url = _fetch_semantic_scholar_link(paper_id)
        if url:
            return url

    title = _first_nonempty(result_row.get("candidate_title"), paper_meta.get("title"))
    if title:
        return _fetch_semantic_scholar_link_by_title(title)

    return None


def load_data(cfg_path: str | Path):
    cfg = load_config(cfg_path)
    # Load the raw pipeline inputs used by the recommender.
    papers = read_jsonl(cfg.path("paths.raw_papers"))

    # Load the raw citation/reference edges for pair building.
    edges = read_table(cfg.path("paths.raw_edges"))

    return cfg, papers, edges


@APP.route("/", methods=["GET"])
def root_index():
    # Serve the local frontend.html if present
    root = Path(__file__).resolve().parents[1]
    return send_from_directory(str(root), "frontend.html")


@APP.route("/api/search", methods=["POST"])
def api_search():
    payload = request.get_json() or {}
    query = payload.get("query", "")
    top_k = int(payload.get("top_k", 10))

    if not query:
        return jsonify({"error": "empty query"}), 400

    # Load data (cached per request for simplicity)
    cfg, papers, edges = load_data(REPO_ROOT / "config.yaml")
    paper_index = _paper_lookup_frame(papers)

    # Basic text for retrieval
    papers = papers.copy()
    papers["text_for_retrieval"] = (papers.get("title", "").fillna("") + ". " + papers.get("abstract", "").fillna("")).astype(str)

    # Try LLM parsing
    parsed = parser.llm_parse_user_message(query)
    desired_relation = parsed.get("relationship")
    target_ids = _resolve_target_paper_ids(papers, parsed)

    if not target_ids or edges is None:
        return jsonify({
            "results": [],
            "parsed": parsed,
            "message": "Could not resolve a target paper from the query, so retrieval was not run.",
        })

    pairs = build_candidate_pairs(
        papers,
        edges,
        target_ids=target_ids,
        candidates_per_target=200,
        include_two_hop=True,
        negatives_per_target=50,
        random_state=cfg.get("pair_building.random_state", 172),
    )

    if pairs.empty:
        return jsonify({
            "results": [],
            "parsed": parsed,
            "message": "Retrieval returned no candidate pairs for the resolved target paper(s).",
        })

    # Rank pairs
    weights = RankingWeights.from_dict(cfg.get("ranking.weights", {}))
    ranked = rank_candidate_pairs(pairs, papers, relation_scores=None, desired_relation=desired_relation or cfg.get("ranking.desired_relation", "extension"), weights=weights, mmr_lambda=cfg.get("ranking.mmr_lambda", 0.75))
    if not ranked.empty and "candidate_paper_id" in ranked.columns:
        ranked = (
            ranked.sort_values(["final_score", "rank"], ascending=[False, True])
            .drop_duplicates(subset=["candidate_paper_id"], keep="first")
            .sort_values(["final_score", "rank"], ascending=[False, True])
            .reset_index(drop=True)
        )

    # Prepare response
    results = []
    for _, r in ranked.head(top_k).iterrows():
        title = str(r.get("candidate_title") or "")
        paper_meta = paper_index.get(r.get("candidate_paper_id"), {})
        paper_url = resolve_paper_url(r, paper_meta)
        results.append(
            {
                "candidate_paper_id": r.get("candidate_paper_id"),
                "title": title,
                "abstract": r.get("candidate_abstract"),
                "score": float(r.get("final_score", 0)),
                "rank": int(r.get("rank", 0)),
                "paper_url": paper_url,
                "link": paper_url,
            }
        )

    return jsonify({"results": results, "parsed": parsed})


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=5000, debug=True)
