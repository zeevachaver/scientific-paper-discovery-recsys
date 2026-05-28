import json

with open("enriched_citations.json", "r") as f:
    data = json.load(f)

print("SAMPLE MATCHES\n")

for i, item in enumerate(data[:50]):
    print("=" * 40)
    print("Citation title:", item["citation_title"])
    print("Matched title:", item["matched_title"])
    print("Year:", item["year"])
    print("Match method:", item["match_method"])