from datasets import load_dataset
import json
import re

# -----------------------------
# STORAGE
# -----------------------------
doi_index = {}
#arxiv_index = {}
title_index = {}

# -----------------------------
# CLEANING
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
# ROBUST FIELD EXTRACTORS
# -----------------------------
def get_doi(p):
    return (
        p.get("DOI")
        or p.get("doi")
        or (p.get("externalIds") or {}).get("DOI")
        or (p.get("external_ids") or {}).get("DOI")
    )

def get_arxiv(p):
    return (
        p.get("ArXiv")
        or p.get("arxiv")
        or (p.get("externalIds") or {}).get("ArXiv")
        or (p.get("external_ids") or {}).get("ArXiv")
    )

def get_corpus_id(p):
    return (
        p.get("CorpusId")
        or p.get("corpus_id")
        or (p.get("externalIds") or {}).get("CorpusId")
        or (p.get("external_ids") or {}).get("CorpusId")
    )


# -----------------------------
# LOAD DATASET
# -----------------------------
ds = load_dataset(
    "AlgorithmicResearchGroup/s2orc-cs-enriched",
    streaming=True,
    split="train"
)

MAX_ROWS = 1000000
print("Building index...")

for i, paper in enumerate(ds):

    if i >= MAX_ROWS:
        break

    title = norm(paper.get("parsed_title"))

    # if not get_doi(paper):
    #     continue

    doi = norm_doi(get_doi(paper))
    #arxiv = get_arxiv(paper)
    corpus_id = get_corpus_id(paper)

    # minimal stored record (IMPORTANT: keep small)
    record = {
        #"paperId": paper.get("paperId"), # null 
        "corpus_id": paper.get("corpus_id"),
        "title": paper.get("parsed_title"),
        #"methods": paper.get("methods"),
        #"datasets": paper.get("datasets"), # low coverage 
        #"metrics": paper.get("metrics"), # low coverage
        "gpu_hours": paper.get("estimated_gpu_hours"),
        "year": paper.get("year")
        # author?
        # abstract for sure
    }

    # -----------------------------
    # DOI INDEX
    # -----------------------------

    if doi:
        doi_index[doi] = record
    
    # -----------------------------
    # ARXIV INDEX
    # -----------------------------
    # if arxiv:
    #     arxiv_index[arxiv] = record

    # -----------------------------
    # TITLE INDEX (FIXED: no overwrite)
    # -----------------------------
    if title:
        title_index.setdefault(title, []).append(record)

    if i % 1000 == 0:
        print(f"Indexed {i} papers...")


# -----------------------------
# SAVE TO DISK
# -----------------------------
# with open("s2orc_doi_index.json", "w") as f:
#     json.dump(doi_index, f, indent=2)

# with open("s2orc_arxiv_index.json", "w") as f:
#     json.dump(arxiv_index, f)

# with open("s2orc_title_index.json", "w") as f:
#     json.dump(title_index, f)

with open("s2orc_doi_index.jsonl", "w") as f:
    for doi, record in doi_index.items():
        f.write(json.dumps({"doi": doi, "record": record}) + "\n")

# with open("s2orc_doi_index.jsonl") as f:
#     for line in f:
#         obj = json.loads(line)fil
#         doi_index[obj["doi"]] = obj["record"]

# with open("s2orc_arxiv_index.jsonl", "w") as f:
#     for arxiv, record in arxiv_index.items():
#         f.write(json.dumps({"arxiv": arxiv, "record": record}) + "\n")

with open("s2orc_title_index.jsonl", "w") as f:
    for title, records in title_index.items():
        f.write(json.dumps({"title": title, "records": records}) + "\n")

print("Done building indexes.")