from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import json

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "incidents"

# Clients
qdrant = QdrantClient(url=QDRANT_URL, api_key=os.getenv("QDRANT_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Request schema
class PromptRequest(BaseModel):
    prompt: str

# Fetch incidents from Qdrant
def get_incidents_from_qdrant(prompt: str, top_k: int = 5) -> list[dict]:
    embedded_prompt = embedder.encode(prompt).tolist()
    results = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=embedded_prompt,
        limit=top_k
    )
    incidents = []
    for hit in results:
        payload = hit.payload
        incidents.append({
            "incident": payload.get("incident_id", f"INC{hit.id}"),
            "description": payload.get("incident_description", ""),
            "closure_notes": payload.get("closure_notes", ""),
            "assigned_to": payload.get("assigned_to", "Unknown"),
            "resolved_at": payload.get("resolved_at", "Unknown"),
            "priority": payload.get("priority", "Unknown"),
            "urgency": payload.get("urgency", "Unknown"),
            "state": payload.get("state", "Unknown")
        })
    return incidents

# Call LLM and return HTML
def call_groq_llm(user_prompt: str, incidents: list[dict]) -> dict:
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
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    return {"html": content}

# Endpoint
@app.post("/chat")
def chat(request: PromptRequest):
    incidents = get_incidents_from_qdrant(request.prompt)
    return call_groq_llm(request.prompt, incidents)
