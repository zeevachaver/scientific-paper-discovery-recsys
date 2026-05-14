import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Specify the search term
query = "computer science"

# Define the API endpoint URL
url = "http://api.semanticscholar.org/graph/v1/paper/search" # maximum 100

all_fields = (
    "paperId,corpusId,url,title,abstract,venue,publicationVenue,"
    "year,referenceCount,citationCount,influentialCitationCount,"
    "isOpenAccess,openAccessPdf,fieldsOfStudy,s2FieldsOfStudy,"
    "publicationTypes,publicationDate,journal,citationStyles,authors,tldr"
)

# Define the query parameters
query_params = {
    "query": query,
    "fields": all_fields,
    "fieldsOfStudy": "Computer Science",
    "limit": 100,

}

# Directly define the API key (Reminder: Securely handle API keys in production environments)
api_key = os.getenv("API_KEY") 

# Define headers with API key
headers = {"x-api-key": api_key}

print(f"API key loaded: {'yes (' + api_key[:6] + '...)' if api_key else 'NO - check .env file'}")

# Send the API request
response = requests.get(url, params=query_params, headers=headers).json()

with open("cs_search_results.json", "w") as f:
    json.dump(response, f, indent=2)

print(f"Got {len(response.get('data', []))} papers")