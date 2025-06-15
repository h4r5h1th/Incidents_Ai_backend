import os
import uuid
import json
from dotenv import load_dotenv
import cohere
import httpx

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "incidents"

# Step 1: Initialize Cohere client
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# Step 2: Create Qdrant collection for 1536-dim vectors
httpx.put(
    f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
    headers={
        "api-key": QDRANT_API_KEY,
        "Content-Type": "application/json"
    },
    json={
        "vectors": {
            "size": 1536,
            "distance": "Cosine"
        }
    },
    timeout=10.0
)

# Step 3: Load data
with open("backend/incidents.json") as f:
    data = json.load(f)

# Step 4: Generate embeddings and upload
for incident in data:
    description = incident["incident_description"]

    embed_response = co.embed(
        texts=[description],
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    vector = embed_response.embeddings.float[0]

    point = {
        "id": str(uuid.uuid4()),
        "vector": vector,
        "payload": incident
    }

    res = httpx.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
        headers={
            "api-key": QDRANT_API_KEY,
            "Content-Type": "application/json"
        },
        json={"points": [point]},
        timeout=30.0
    )

    res.raise_for_status()

print("✅ Re-uploaded all incidents with 1536-dim embeddings.")
