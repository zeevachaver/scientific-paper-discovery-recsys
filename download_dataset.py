import requests
import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

base_url = "https://api.semanticscholar.org/datasets/v1/release/"

# This endpoint requires authentication via api key
api_key = os.getenv("API_KEY") 
headers = {"x-api-key": api_key}

# Set the release id
release_id = "2026-05-05"

# Define dataset name you want to download
dataset_name = "abstracts"

# Send the GET request and store the response in a variable
response = requests.get(base_url + release_id + '/dataset/' + dataset_name, headers=headers)



with open("datasets.json", "w") as f:
    json.dump(response.json(), f, indent=2)
