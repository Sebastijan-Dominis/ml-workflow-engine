import os

import dotenv
import requests

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def call_pipeline(pipeline_name: str, payload: dict) -> dict:
    url = f"{API_URL}/pipelines/{pipeline_name}"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}
