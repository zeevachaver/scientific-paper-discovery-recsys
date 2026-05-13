import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Specify the search term
query = '"computer science"'

# Define the API endpoint URL
url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"

# Define the query parameters
query_params = {
    "query": '"computer science"',
    "fields": "title,url,publicationTypes,publicationDate,openAccessPdf,citationCount,abstract",
    "year": "2023-"
}

# Directly define the API key (Reminder: Securely handle API keys in production environments)
api_key = os.getenv("API_KEY") # Replace with the actual API key

# Define headers with API key
headers = {"x-api-key": api_key}

# Send the API request
response = requests.get(url, params=query_params, headers=headers).json()

with open("test_get_results.json", "w") as f:
    json.dump(response, f, indent=2)