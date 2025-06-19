import httpx
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = "incidents"
SOLUTION_DOC_URL = os.getenv("SOLUTION_DOC_URL")

def create_solutions_collection():
    url = f"{QDRANT_URL}/collections/solutions_guide"
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY,
    }
    payload = {
        "vectors": {
            "size": 1536,  # same as Cohere's embed-v4.0 output
            "distance": "Cosine"
        }
    }
    response = httpx.put(url, headers=headers, json=payload)
    print("✅ Solutions collection created:", response.status_code)
    response.raise_for_status()
create_solutions_collection()