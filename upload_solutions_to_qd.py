import os
import httpx
from uuid import uuid4
from docx import Document
from dotenv import load_dotenv
from typing import List

# Load env vars
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_COLLECTION = "solutions_guide"  # change if needed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = os.path.join(SCRIPT_DIR, "solutions_docs")

def get_embedding(text: str) -> List[float]:
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "embed-v4.0",
        "texts": [text],
        "input_type": "search_query",
        "embedding_types": ["float"]
    }
    response = httpx.post("https://api.cohere.ai/v2/embed", headers=headers, json=payload, timeout=15.0)
    response.raise_for_status()
    return response.json()["embeddings"]["float"][0]

def create_qdrant_collection():
    collection_url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY,
    }

    # Check if the collection already exists
    check_response = httpx.get(collection_url, headers=headers)
    if check_response.status_code == 200:
        # Collection exists, delete it
        delete_response = httpx.delete(collection_url, headers=headers)
        delete_response.raise_for_status()
        print(f"🗑️ Deleted existing collection: {QDRANT_COLLECTION}")

    # Create a fresh collection
    payload = {
        "vectors": {
            "size": 1536,
            "distance": "Cosine"
        }
    }
    create_response = httpx.put(collection_url, headers=headers, json=payload)
    create_response.raise_for_status()
    print(f"✅ Created new collection: {QDRANT_COLLECTION}")



def load_docx_chunks(path: str, chunk_size: int = 500) -> List[str]:
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def upload_chunks(chunks: List[str], file_name: str):
    points = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        points.append({
            "id": str(uuid4()),
            "vector": embedding,
            "payload": {
                "text": chunk,
                "source_file": file_name
            }
        })

    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points"
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY,
    }
    response = httpx.put(url, headers=headers, json={"points": points})
    print(f"📤 Uploaded {len(points)} chunks from {file_name} - status {response.status_code}")
    response.raise_for_status()

def upload_folder(folder_path: str):
    create_qdrant_collection()
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".docx"):
            path = os.path.join(folder_path, filename)
            print(f"📄 Processing: {filename}")
            chunks = load_docx_chunks(path)
            upload_chunks(chunks, filename)

if __name__ == "__main__":
    upload_folder(FOLDER_PATH)
