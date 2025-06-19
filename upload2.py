import os
import uuid
import json
from dotenv import load_dotenv
import cohere
import httpx

# Load environment variables
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "incidents"

# Initialize Cohere client
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# Step 1: Delete existing collection (if any)
delete_response = httpx.delete(
    f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
    headers={
        "api-key": QDRANT_API_KEY,
        "Content-Type": "application/json"
    },
    timeout=10.0
)
print(f"🗑️ Deleted existing collection '{COLLECTION_NAME}' (if it existed).")

# Step 2: Create new collection with 1536-dim vectors
create_response = httpx.put(
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
print(f"✅ Created new collection '{COLLECTION_NAME}' with 1536-dim vectors.")

# Step 3: Load incidents from JSON file
with open("backend/incidents.json") as f:
    data = json.load(f)

# Step 4: Embed and upload incidents
for i, incident in enumerate(data, 1):
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
    print(f"📦 Uploaded incident {i}/{len(data)}")

print("🚀 Re-upload completed successfully.")
