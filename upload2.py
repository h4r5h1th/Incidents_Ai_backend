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
COLLECTION_NAME = "office_incidents"

# Initialize Cohere client
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# Step 1: Delete existing collection (if any)
httpx.delete(
    f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
    headers={
        "api-key": QDRANT_API_KEY,
        "Content-Type": "application/json"
    },
    timeout=10.0
)
print(f"🗑️ Deleted existing collection '{COLLECTION_NAME}' (if it existed).")

# Step 2: Create new collection with 1536-dim vectors
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
print(f"✅ Created new collection '{COLLECTION_NAME}' with 1536-dim vectors.")

# Step 3: Load incidents from JSON file
with open("backend/incidents.json", encoding='utf-8') as f:
    data = json.load(f)

# Step 4: Embed and upload each incident
for i, incident in enumerate(data, 1):
    description = incident.get("description", "")
    if not description:
        print(f"⚠️ Skipping incident {i}: missing description.")
        continue

    # Generate vector embedding for the description field
    embed_response = co.embed(
        texts=[description],
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    vector = embed_response.embeddings.float[0]

    # Construct vector point with full payload
    point = {
        "id": str(uuid.uuid4()),
        "vector": vector,
        "payload": incident  # full metadata for filtering/search
    }

    # Upload to Qdrant
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
