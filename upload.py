import os
import uuid
import json
from dotenv import load_dotenv
import cohere
import httpx

# Load environment variables
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")  # e.g., https://your-qdrant-cloud-instance.com
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "incidents"

# Initialize Cohere
co = cohere.ClientV2(api_key=COHERE_API_KEY)

# Load data
with open("incidents.json") as f:
    data = json.load(f)

# Prepare and upload points
for incident in data:
    description = incident["incident_description"]

    # Get embedding from Cohere v2
    response = co.embed(
        texts=[description],
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"]
    )

    vector = response.embeddings.float[0]

    # Format point
    point = {
        "id": str(uuid.uuid4()),
        "vector": vector,
        "payload": incident
    }

    # Upload to Qdrant cloud
    res = httpx.put(
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
        headers={
            "api-key": QDRANT_API_KEY,
            "Content-Type": "application/json"
        },
        json={"points": [point]},
        timeout=30.0
    )
    
    res.raise_for_status()  # Throw error if upload fails

print("✅ All incidents uploaded to Qdrant Cloud.")
