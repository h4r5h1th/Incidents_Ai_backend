from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
import httpx
import os

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

class PromptRequest(BaseModel):
    prompt: str

# Function: get embedding from Cohere
def get_embedding(text: str) -> List[float]:
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "embed-english-light-v3.0",  # ✅ This returns exactly 384 dims
        "texts": [text],
        "input_type": "search_query"
    }

    response = httpx.post("https://api.cohere.ai/v1/embed", headers=headers, json=payload, timeout=15.0)
    response.raise_for_status()
    embedding = response.json()["embeddings"][0]

    if len(embedding) != 384:
        raise ValueError(f"Expected embedding size 384, got {len(embedding)}")

    return embedding


# Function: search Qdrant
def get_incidents_from_qdrant(prompt: str, top_k: int = 5) -> List[dict]:
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
    response.raise_for_status()
    results = response.json()["result"]

    incidents = []
    for hit in results:
        payload = hit["payload"]
        incidents.append({
            "incident": payload.get("incident_id", f"INC{hit['id']}"),
            "description": payload.get("incident_description", ""),
            "closure_notes": payload.get("closure_notes", ""),
            "assigned_to": payload.get("assigned_to", "Unknown"),
            "resolved_at": payload.get("resolved_at", "Unknown"),
            "priority": payload.get("priority", "Unknown"),
            "urgency": payload.get("urgency", "Unknown"),
            "state": payload.get("state", "Unknown")
        })
    return incidents

# Function: call Groq LLM
def call_groq_llm(user_prompt: str, incidents: List[dict]) -> dict:
    system_prompt = """
    You are an AI incident support assistant.

    Given past incidents, return your response as raw HTML with these sections:
    - <h3>Summary</h3><p>...</p>
    - <h3>Steps to Resolution</h3><ol>...</ol>
    - <h3>People Involved</h3><ul>...</ul>
    - <h3>Incident Summary Table</h3>
    <table>
        Include columns: Incident ID, Description, Closure Notes, Assigned To, Resolved At, Priority, Urgency, State.
    </table>

    Important:
    - DO NOT use markdown or backticks.
    - Only respond with valid HTML.
    - If the prompt is not incident-related, return: <p>Sorry, I can only discuss incident-related issues.</p>
    """

    formatted_incidents = "\n\n".join([
        f"Incident {i+1}:\n"
        f"Description: {d['description']}\n"
        f"Closure Notes: {d['closure_notes']}\n"
        f"Resolved By: {d['assigned_to']}\n"
        f"Resolved At: {d['resolved_at']}\n"
        f"Priority: {d['priority']}, Urgency: {d['urgency']}, State: {d['state']}"
        for i, d in enumerate(incidents)
    ])

    full_prompt = f"""
User Prompt: "{user_prompt}"

Past Relevant Incidents:
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
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    return {"html": content}

# FastAPI endpoint
@app.post("/chat")
def chat(request: PromptRequest):
    try:
        incidents = get_incidents_from_qdrant(request.prompt)
        return call_groq_llm(request.prompt, incidents)
    except Exception as e:
        return {"error": str(e)}
