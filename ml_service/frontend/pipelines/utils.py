"""Utility functions for calling backend pipelines from the frontend."""
import os

import dotenv
import requests

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def call_pipeline(pipeline_endpoint: str, payload: dict) -> dict:
    """Call a backend pipeline endpoint with the given payload.

    Args:
        pipeline_endpoint (str): The specific pipeline endpoint to call (e.g., "run-pipeline").
        payload (dict): The data to send in the request body.

    Returns:
        dict: The JSON response from the backend or an error message.
    """
    url = f"{API_URL}/{pipeline_endpoint}"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}
