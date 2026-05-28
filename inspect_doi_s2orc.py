from datasets import load_dataset

ds = load_dataset(
    "AlgorithmicResearchGroup/s2orc-cs-enriched",
    streaming=True,
    split="train"
)

def safe_get(d, key):
    try:
        return d.get(key)
    except:
        return None

def inspect_paper(paper, idx):
    doi_top = paper.get("DOI")

    parsed_ext = paper.get("parsed_external_ids") or {}
    meta_ext = paper.get("metadata_externalids") or {}

    doi_parsed = parsed_ext.get("doi")
    doi_meta = meta_ext.get("DOI")

    print("\n" + "="*80)
    print(f"PAPER {idx}")
    print("="*80)

    print("\nParsed title:")
    print(paper.get("parsed_title"))

    print("\nDOI FIELDS:")
    print(f"  paper['DOI']:                      {doi_top}")
    print(f"  parsed_external_ids['doi']:        {doi_parsed}")
    print(f"  metadata_externalids['DOI']:       {doi_meta}")

    print("\nRAW STRUCTURES:")
    print("parsed_external_ids:")
    print(parsed_ext)

    print("\nmetadata_externalids:")
    print(meta_ext)

    print("\nCONSISTENCY CHECK:")

    all_dois = [doi_top, doi_parsed, doi_meta]
    non_null = [d for d in all_dois if d]

    if len(set(non_null)) <= 1:
        print("  ✔ consistent or single-source DOI")
    else:
        print("   MISMATCH between DOI fields:")
        print(f"  unique values: {set(non_null)}")


# -------------------------
# RUN ON FIRST 3 PAPERS
# -------------------------
for i, paper in enumerate(ds):
    inspect_paper(paper, i)
    if i >= 2:
        break