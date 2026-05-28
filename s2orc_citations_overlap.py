import json
import re
from rapidfuzz import process, fuzz

# -----------------------------
# NORMALIZATION
# -----------------------------
def norm(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def norm_doi(doi):
    if not doi:
        return None
    doi = doi.lower().strip()
    doi = doi.replace("https://doi.org/", "")
    return doi

# -----------------------------
# LOAD DATA
# -----------------------------
with open("citations_pre2024.json", "r") as f:
    citations = json.load(f)

print(f"Loaded {len(citations)} citations")

# with open("s2orc_doi_index.json", "r") as f:
#     doi_index = json.load(f)

# with open("s2orc_title_index.json", "r") as f:
#     title_index = json.load(f)


# -----------------------------
# LOAD DOI INDEX 
# -----------------------------
doi_index = {}

with open("s2orc_doi_index.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        doi = norm_doi(obj["doi"])
        doi_index[doi] = obj["record"]


print(f"Loaded {len(doi_index)} DOI records")

# -----------------------------
# OUTPUT STORAGE
# -----------------------------
enriched = []

doi_matches = 0
title_matches = 0
no_matches = 0


# -----------------------------
# MATCHING
# -----------------------------
for item in citations:

    paper = item.get("citingPaper", {})

    title = paper.get("title", "")
    doi = (paper.get("externalIds") or {}).get("DOI")

    match = None
    match_method = None
    norm_title = norm(title)

    # -------------------------
    # DOI MATCH 
    # -------------------------
    if doi and doi in doi_index:

        match = doi_index[doi]
        match_method = "doi"
        doi_matches += 1

    # -------------------------
    # TITLE MATCH 
    # -------------------------
    # else:
    # #elif norm_title in title_index:

    #     #fuzzy match against ALL indexed titles
    #     best = process.extractOne(
    #         norm_title,
    #         title_index.keys(),
    #         scorer=fuzz.token_set_ratio
    #     )

    #     if best:
    #         best_key, score, _ = best

    #         if score >= 90:   # threshold (you can tune 85–92)
    #             match = title_index[best_key][0]
    #             match_method = "title_fuzzy"
    #             title_matches += 1

    # -------------------------
    # STORE
    # -------------------------
    if match:
        enriched.append({
            "citation_title": title,
            "citation_doi": doi,
            "match_method": match_method,
            "matched_title": match.get("title"),
            "year": match.get("year")
        })
    else:
        no_matches += 1




# -----------------------------
# SAVE OUTPUT
# -----------------------------
with open("enriched_citations.json", "w") as f:
    json.dump(enriched, f, indent=2)


# -----------------------------
# SUMMARY
# -----------------------------
print("\nMATCH SUMMARY")
print("------------------")
print(f"DOI matches:   {doi_matches}")
print(f"Title matches: {title_matches}")
print(f"No matches:    {no_matches}")
print(f"Total matched: {len(enriched)}")

print("\nSaved to enriched_citations.json")