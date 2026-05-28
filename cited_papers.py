import os
import json
import requests
from dotenv import load_dotenv
import time 

# Load environment variables from .env
load_dotenv()

API_KEY = os.getenv("API_KEY")

# test paper 1: no citations before 2023
# https://www.semanticscholar.org/paper/A-Digital-Recommendation-System-for-Personalized-to-DHANANJAYAG-Goudar/88e85e760067f7003d5d0f97f5fc3cf90f6c642f
# paper_id = "88e85e760067f7003d5d0f97f5fc3cf90f6c642f"

# test paper 2: https://www.semanticscholar.org/paper/Graph-Convolutional-Neural-Networks-for-Web-Scale-Ying-He/6c96c2d4a3fbd572fef2d59cb856521ee1746789

# Input paper ID
paper_id = '6c96c2d4a3fbd572fef2d59cb856521ee1746789'

# Semantic Scholar citations endpoint
url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"

# Pagination settings
limit = 100 #10
offset = 0

# Store all citation results
all_results = []

MAX_PAPERS = 2000 # editable

while len(all_results) < MAX_PAPERS:
    print(f"Fetching citations {offset} to {offset + limit}...")

    response = requests.get(
        url,
        params={
            "fields": (
                "citingPaper.paperId,"
                "citingPaper.title,"
                "citingPaper.year,"
                "citingPaper.abstract,"
                "citingPaper.citationCount,"
                "citingPaper.authors,"
                 "citingPaper.externalIds"
            ),
            "limit": limit,
            "offset": offset
        },
        headers={
            "x-api-key": API_KEY
        }
    )

    # rate limited
    if response.status_code == 429:
        print("Rate limited. Sleeping 30 seconds...")
        time.sleep(30)
        continue

    # Raise error if request failed
    response.raise_for_status()

    # Parse JSON response
    result = response.json()

    # Get current page data
    data = result.get("data", [])

    # Stop if no more results
    if not data:
        break

    # Add results to master list
    all_results.extend(data)

    # Move to next page
    offset += limit
    time.sleep(1) # time.sleep(2)

# Print summary
print(f"\nRetrieved {len(all_results)} citing papers.")

# -----------------------------
# YEAR ANALYSIS
# -----------------------------
years = []

for item in all_results:

    paper = item.get("citingPaper", {})
    year = paper.get("year")

    if year:
        years.append(year)

if years:
    print("\nYEAR DISTRIBUTION")
    print("------------------")
    print(f"Min year: {min(years)}")
    print(f"Max year: {max(years)}")
    #print(Counter(years))

# -----------------------------
# FILTER TO S2ORC-COMPATIBLE YEARS
# -----------------------------
filtered_results = []

for item in all_results:

    paper = item.get("citingPaper", {})
    year = paper.get("year")

    # adjust threshold if needed
    if year and year <= 2023:
        filtered_results.append(item)

print(f"\nFiltered to {len(filtered_results)} papers from 2023 or earlier.")

# -----------------------------
# SAVE RAW RESULTS
# -----------------------------
with open("citations.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("Saved all citations to citations.json")

# -----------------------------
# SAVE FILTERED RESULTS
# -----------------------------
with open("citations_pre2024.json", "w") as f:
    json.dump(filtered_results, f, indent=2)

print("Saved filtered citations to citations_pre2024.json")