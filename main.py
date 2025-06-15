from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
import httpx
import os
import traceback
from docx import Document
from io import BytesIO

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = "incidents"
SOLUTION_DOC_URL = os.getenv("SOLUTION_DOC_URL")


class PromptRequest(BaseModel):
    prompt: str


# 🔹 Load solution doc from Google Drive or URL
def get_docx_text_from_url(url: str) -> str:
    response = httpx.get(url, follow_redirects=True)  # ✅ FIXED: follow redirects
    response.raise_for_status()
    doc = Document(BytesIO(response.content))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# 🔹 Get embedding from Cohere
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
    res_json = response.json()

    embeddings_list = res_json.get("embeddings", {}).get("float", [])

    if not embeddings_list:
        raise ValueError(f"❌ Embeddings missing or empty: {res_json}")

    embedding = embeddings_list[0]

    if len(embedding) != 1536:
        raise ValueError(f"Expected embedding size 1536, got {len(embedding)}")

    return embedding


# 🔹 Search Qdrant vector DB
def get_incidents_from_qdrant(prompt: str, top_k: int = 30) -> List[dict]:
    embedded_prompt = get_embedding(prompt)

    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY,
    }
    payload = {
        "vector": embedded_prompt,
        "limit": top_k,
        "with_payload": True
    }

    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search"
    response = httpx.post(url, headers=headers, json=payload, timeout=15.0)

    print("🔍 Qdrant search status:", response.status_code)

    response.raise_for_status()
    results = response.json().get("result", [])

    if not results:
        raise ValueError("Qdrant search returned no results.")

    incidents = []
    for hit in results:
        payload = hit.get("payload", {})
        incident_id = payload.get("incident_number")  # FIXED field name

        if not incident_id:
            raise ValueError(f"Incident missing 'incident_number': {payload}")

        incidents.append({
            "incident": incident_id,
            "description": payload.get("incident_description", ""),
            "closure_notes": payload.get("closure_notes", ""),
            "assigned_to": payload.get("assigned_to", "Unknown"),
            "resolved_at": payload.get("resolved_at", "Unknown"),
            "priority": payload.get("priority", "Unknown"),
            "urgency": payload.get("urgency", "Unknown"),
            "state": payload.get("state", "Unknown")
        })
    return incidents


# 🔹 Call Groq LLM with full data
def call_groq_llm(user_prompt: str, incidents: List[dict]) -> dict:
    system_prompt = """
You are an AI incident support assistant.

You are given:
1. A user prompt,
2. A list of past incidents,
3. A mandatory solution guide.

You MUST respond in raw HTML format only.

**Rules**:
- For ANY incident-related question (even if not explicitly about resolution), you MUST include:
  - <h3>Steps to Resolution</h3> derived from the solution guide.
- If the user prompt is unrelated to incidents, respond with:
  <p>Sorry, I can only discuss incident-related issues.</p>

Respond using valid HTML only. Do not use Markdown.

Include the following sections:
- <h3>Summary</h3><p>...</p>
- <h3>Steps to Resolution</h3><ol>...</ol>
- <h3>People Involved</h3><ul>...</ul>
- <h3>Incident Summary Table</h3><table>...</table>
"""

    # Load solution guide
    try:
        print("⚙️ Loading solution doc and embeddings...")
        solution_guide = get_docx_text_from_url(SOLUTION_DOC_URL)
    except Exception as e:
        print(f"❌ Failed to load solution guide: {e}")
        solution_guide = "Solution guide not available."

    # Format incidents
    formatted_incidents = "\n\n".join([
        f"Incident {i+1}:\n"
        f"Incident ID: {d['incident']}\n"
        f"Description: {d['description']}\n"
        f"Closure Notes: {d['closure_notes']}\n"
        f"Resolved By: {d['assigned_to']}\n"
        f"Resolved At: {d['resolved_at']}\n"
        f"Priority: {d['priority']}, Urgency: {d['urgency']}, State: {d['state']}"
        for i, d in enumerate(incidents)
    ])

    full_prompt = f"""
User Prompt: "{user_prompt}"

Mandatory Solution Guide:
{solution_guide}

Relevant Incidents:
{formatted_incidents}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30.0)

    print("🧠 Groq response status:", response.status_code)

    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return {"html": content}


# 🔹 FastAPI endpoint
@app.post("/chat")
def chat(request: PromptRequest):
    try:
        print("📨 Prompt received:", request.prompt)
        incidents = get_incidents_from_qdrant(request.prompt)
        return call_groq_llm(request.prompt, incidents)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {str(e)}"}
