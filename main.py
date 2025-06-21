from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
import httpx
import os
import traceback
from docx import Document
from io import BytesIO
from collections import Counter
from datetime import datetime
import json
import uuid  # ✅ Add for unique canvas IDs

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

# Env vars
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION = "office_incidents"


class PromptRequest(BaseModel):
    prompt: str

def query_solutions_from_qdrant(prompt: str, top_k: int = 1) -> str:
    embedding = get_embedding(prompt)
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY,
    }
    payload = {
        "vector": embedding,
        "limit": top_k,
        "with_payload": True
    }
    url = f"{QDRANT_URL}/collections/solutions_guide/points/search"
    response = httpx.post(url, headers=headers, json=payload)
    response.raise_for_status()
    results = response.json()["result"]
    return "\n\n".join(hit["payload"]["text"] for hit in results if "text" in hit["payload"])

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


def get_incidents_from_qdrant(prompt: str, top_k: int = 10) -> List[dict]:
    embedded_prompt = get_embedding(prompt)
    print(f"🔢 Embedding generated successfully, length: {len(embedded_prompt)}")
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
    print(f"🔍 Raw Qdrant results count: {len(results)}")
    if results:
        print(f"📊 First result score: {results[0].get('score', 'N/A')}")
        print(f"📄 First result payload keys: {list(results[0].get('payload', {}).keys())}")
    if not results:
        print("❌ No results returned from Qdrant search")
        return []

    incidents = []
    for hit in results:
        payload = hit.get("payload", {})
        incident_id = payload.get("number")  # Changed from "incident_number" to "number"
        if not incident_id:
            continue
        incidents.append({
            "incident": incident_id,
            "job_name": payload.get("job_name", ""),
            "description": payload.get("description", ""),  # Changed from "incident_description"
            "impact": payload.get("impact", ""),
            "closure_notes": payload.get("closure_notes", ""),
            "assigned_to": payload.get("assigned_to", ""),
            "assignment_group": payload.get("assignment_group", ""),
            "configuration_item": payload.get("configuration_item", ""),
            "ci_class": payload.get("ci_class", ""),
            "opened_by": payload.get("opened_by", ""),
            "resolved_by": payload.get("resolved_by", ""),
            "closed_by": payload.get("closed_by", ""),
            "opened_time": payload.get("opened_time", ""),
            "resolved": payload.get("resolved", ""),
            "closed": payload.get("closed", ""),
            "priority": payload.get("priority", ""),
            "urgency": payload.get("urgency", ""),
            "state": payload.get("state", ""),
            "similarity_score": hit.get("score", 0.0)
        })
    return incidents


def filter_incidents_for_analytics(incidents: List[dict], user_prompt: str) -> tuple[List[dict], List[dict]]:
    """
    Smart filtering for analytics only - separate relevant from non-relevant incidents
    This doesn't affect the LLM input, only the analytics charts
    """
    if not incidents:
        return [], []
    
    # Get average similarity score to determine threshold dynamically
    scores = [inc.get("similarity_score", 0.0) for inc in incidents]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    # Dynamic threshold: use 70% of average score, but minimum 0.6
    dynamic_threshold = max(avg_score * 0.7, 0.6)
    
    print(f"📊 Analytics filtering - Avg score: {avg_score:.3f}, Threshold: {dynamic_threshold:.3f}")
    
    # Extract keywords from user prompt for additional filtering
    prompt_keywords = set(word.lower() for word in user_prompt.split() if len(word) > 2)
    
    relevant_incidents = []
    non_relevant_incidents = []
    
    for incident in incidents:
        score = incident.get("similarity_score", 0.0)
        
        # Primary filter: similarity score
        if score >= dynamic_threshold:
            relevant_incidents.append(incident)
        else:
            # Secondary filter: keyword matching for edge cases
            incident_text = f"{incident.get('description', '')} {incident.get('closure_notes', '')}".lower()
            
            # Check if significant keywords from prompt appear in incident
            matching_keywords = sum(1 for keyword in prompt_keywords if keyword in incident_text)
            keyword_match_ratio = matching_keywords / max(len(prompt_keywords), 1)
            
            if keyword_match_ratio >= 0.3:  # At least 30% keyword match
                relevant_incidents.append(incident)
            else:
                non_relevant_incidents.append(incident)
    
    print(f"🎯 Analytics split: {len(relevant_incidents)} relevant, {40 - len(relevant_incidents)} non-relevant")
    return relevant_incidents, non_relevant_incidents


def compute_analytics(incidents: List[dict], user_prompt: str) -> str:
    """
    Premium themed analytics with mobile responsiveness and consistent styling
    Updated to use new JSON field names and include additional analytics
    """
    if not incidents:
        return """
        <div class="analytics-empty-state">
            <div class="empty-icon">📊</div>
            <p>No incident analytics available.</p>
        </div>
        <style>
            .analytics-empty-state {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 60px 20px;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 16px;
                border: 1px solid rgba(126, 87, 194, 0.2);
                margin: 20px 0;
                text-align: center;
            }
            .empty-icon {
                font-size: 48px;
                margin-bottom: 16px;
                opacity: 0.6;
            }
            .analytics-empty-state p {
                color: #b0b0b0;
                font-size: 16px;
                margin: 0;
            }
        </style>
        """

    # Filter incidents for analytics (this doesn't affect LLM processing)
    relevant_incidents, non_relevant_incidents = filter_incidents_for_analytics(incidents, user_prompt)
    
    if not relevant_incidents:
        return """
        <div class="analytics-warning">
            <div class="warning-icon">⚠️</div>
            <h4>Analytics Note</h4>
            <p>No highly relevant incidents found for detailed analytics. Consider refining your search query.</p>
        </div>
        <style>
            .analytics-warning {
                margin: 20px 0;
                padding: 20px;
                background: linear-gradient(135deg, #2d1b69 0%, #1a1a2e 100%);
                border-radius: 12px;
                border-left: 4px solid #7e57c2;
                border: 1px solid rgba(126, 87, 194, 0.3);
            }
            .warning-icon {
                font-size: 24px;
                margin-bottom: 8px;
            }
            .analytics-warning h4 {
                margin: 0 0 10px 0;
                color: #ce93d8;
                font-size: 18px;
                font-weight: 600;
            }
            .analytics-warning p {
                margin: 0;
                color: #b0b0b0;
                line-height: 1.5;
            }
        </style>
        """

    closed_states = {"closed", "resolved", "done", "completed"}
    open_states = {"open", "in_progress", "assigned", "pending", "new", "active"}

    # Analytics based on RELEVANT incidents only
    resolved_by_counter = Counter()
    assignment_group_counter = Counter()
    ci_class_counter = Counter()
    state_counter = Counter()

    for inc in relevant_incidents:
        state = (inc.get("state") or "").strip().lower()
        resolved_by = (inc.get("resolved_by") or "").strip()  # Updated field name
        assignment_group = (inc.get("assignment_group") or "").strip()
        ci_class = (inc.get("ci_class") or "").strip()

        if state in closed_states:
            state_counter["Closed"] += 1
            # Count resolver only if incident is actually resolved
            if resolved_by:
                resolved_by_counter[resolved_by] += 1
        elif state in open_states:
            state_counter["Open"] += 1
        else:
            state_counter["Other"] += 1
        
        # Count assignment groups and CI classes for all relevant incidents
        if assignment_group:
            assignment_group_counter[assignment_group] += 1
        if ci_class:
            ci_class_counter[ci_class] += 1

    related_count = len(relevant_incidents)
    non_related_count = 40 - len(relevant_incidents)

    closed = state_counter.get("Closed", 0)
    open_ = state_counter.get("Open", 0)
    other = state_counter.get("Other", 0)

    # Limit charts to top 10 to avoid clutter
    top_resolvers = dict(resolved_by_counter.most_common(10))
    top_groups = dict(assignment_group_counter.most_common(8))
    top_ci_classes = dict(ci_class_counter.most_common(6))

    uid = uuid.uuid4().hex[:8]
    relevance_chart_id = f"relevanceChart_{uid}"
    resolution_id = f"resolutionChart_{uid}"
    resolver_id = f"resolverChart_{uid}"
    group_id = f"groupChart_{uid}"
    ci_class_id = f"ciClassChart_{uid}"

    # Properly escape JSON data for JavaScript
    resolver_labels_json = json.dumps(list(top_resolvers.keys()), ensure_ascii=False)
    resolver_data_json = json.dumps(list(top_resolvers.values()), ensure_ascii=False)
    group_labels_json = json.dumps(list(top_groups.keys()) if top_groups else [], ensure_ascii=False)
    group_data_json = json.dumps(list(top_groups.values()) if top_groups else [], ensure_ascii=False)
    ci_class_labels_json = json.dumps(list(top_ci_classes.keys()), ensure_ascii=False)
    ci_class_data_json = json.dumps(list(top_ci_classes.values()), ensure_ascii=False)

    analytics_html = f"""
<div class="analytics-container">
    <div class="analytics-header">
        <h2><span class="analytics-icon">📊</span>Incident Analytics</h2>
        <div class="analytics-subtitle">Smart insights from your incident data</div>
    </div>
    
    <div class="charts-grid">
        <div class="chart-card">
            <div class="chart-header">
                <h3>Relevance Distribution</h3>
                <span class="chart-badge">Query Match</span>
            </div>
            <div class="chart-container">
                <canvas id="{relevance_chart_id}"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <div class="chart-header">
                <h3>Resolution Status</h3>
                <span class="chart-badge">Active</span>
            </div>
            <div class="chart-container">
                <canvas id="{resolution_id}"></canvas>
            </div>
        </div>

        <div class="chart-card chart-wide">
            <div class="chart-header">
                <h3>Top Resolvers</h3>
                <span class="chart-badge">Performance</span>
            </div>
            <div class="chart-container">
                <canvas id="{resolver_id}"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <div class="chart-header">
                <h3>Assignment Groups</h3>
                <span class="chart-badge">Teams</span>
            </div>
            <div class="chart-container">
                <canvas id="{group_id}"></canvas>
            </div>
        </div>

        <div class="chart-card">
            <div class="chart-header">
                <h3>CI Classes</h3>
                <span class="chart-badge">Categories</span>
            </div>
            <div class="chart-container">
                <canvas id="{ci_class_id}"></canvas>
            </div>
        </div>
    </div>

    <div class="analytics-summary">
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-value">{related_count + non_related_count}</div>
                <div class="summary-label">Total Retrieved</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{related_count}</div>
                <div class="summary-label">Relevant</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{non_related_count}</div>
                <div class="summary-label">Non-Relevant</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{round((closed / related_count * 100) if related_count > 0 else 0, 1)}%</div>
                <div class="summary-label">Resolution Rate</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{len(top_groups)}</div>
                <div class="summary-label">Active Teams</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{len(top_ci_classes)}</div>
                <div class="summary-label">CI Classes</div>
            </div>
        </div>
    </div>
</div>

<style>
    .analytics-container {{
        margin: 30px 0;
        font-family: 'Roboto', sans-serif;
    }}

    .analytics-header {{
        text-align: center;
        margin-bottom: 32px;
    }}

    .analytics-header h2 {{
        color: #ce93d8;
        font-size: 28px;
        font-weight: 600;
        margin: 0 0 8px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
    }}

    .analytics-icon {{
        font-size: 32px;
        background: linear-gradient(135deg, #7e57c2, #ce93d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .analytics-subtitle {{
        color: #b0b0b0;
        font-size: 16px;
        font-weight: 400;
    }}

    .charts-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 24px;
        margin-bottom: 32px;
    }}

    .chart-wide {{
        grid-column: 1 / -1;
    }}

    .chart-card {{
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        border: 1px solid rgba(126, 87, 194, 0.2);
        padding: 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }}

    .chart-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #7e57c2, transparent);
        opacity: 0.6;
    }}

    .chart-card:hover {{
        transform: translateY(-4px);
        border-color: rgba(126, 87, 194, 0.4);
        box-shadow: 0 12px 48px rgba(126, 87, 194, 0.2);
    }}

    .chart-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }}

    .chart-header h3 {{
        color: #ffffff;
        font-size: 18px;
        font-weight: 600;
        margin: 0;
    }}

    .chart-badge {{
        background: linear-gradient(135deg, #7e57c2, #ce93d8);
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .chart-container {{
        position: relative;
        height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .chart-wide .chart-container {{
        height: 350px;
    }}

    .analytics-summary {{
        background: linear-gradient(135deg, #2d1b69 0%, #1a1a2e 100%);
        border-radius: 16px;
        border: 1px solid rgba(126, 87, 194, 0.3);
        padding: 24px;
        position: relative;
        overflow: hidden;
    }}

    .analytics-summary::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #7e57c2, #ce93d8, #7e57c2);
    }}

    .summary-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 24px;
    }}

    .summary-item {{
        text-align: center;
        padding: 16px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        border: 1px solid rgba(126, 87, 194, 0.1);
        transition: all 0.3s ease;
    }}

    .summary-item:hover {{
        background: rgba(126, 87, 194, 0.1);
        transform: translateY(-2px);
    }}

    .summary-value {{
        font-size: 32px;
        font-weight: 700;
        color: #ce93d8;
        margin-bottom: 8px;
        background: linear-gradient(135deg, #7e57c2, #ce93d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    .summary-label {{
        font-size: 14px;
        color: #b0b0b0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Mobile Responsiveness */
    @media (max-width: 768px) {{
        .analytics-container {{
            margin: 20px 0;
        }}

        .analytics-header h2 {{
            font-size: 24px;
        }}

        .analytics-subtitle {{
            font-size: 14px;
        }}

        .charts-grid {{
            grid-template-columns: 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }}

        .chart-card {{
            padding: 16px;
        }}

        .chart-header {{
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }}

        .chart-container {{
            height: 250px;
        }}

        .chart-wide .chart-container {{
            height: 280px;
        }}

        .summary-grid {{
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }}

        .summary-value {{
            font-size: 24px;
        }}

        .summary-label {{
            font-size: 12px;
        }}
    }}

    @media (max-width: 480px) {{
        .analytics-header h2 {{
            font-size: 20px;
            flex-direction: column;
            gap: 8px;
        }}

        .analytics-icon {{
            font-size: 28px;
        }}

        .chart-card {{
            padding: 12px;
        }}

        .chart-container {{
            height: 200px;
        }}

        .chart-wide .chart-container {{
            height: 220px;
        }}

        .summary-grid {{
            grid-template-columns: 1fr;
        }}

        .summary-item {{
            padding: 12px;
        }}
    }}
</style>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
(() => {{
    // Premium Chart Configuration
    const chartDefaults = {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            legend: {{
                position: 'bottom',
                labels: {{
                    padding: 20,
                    font: {{
                        size: 13,
                        weight: '500'
                    }},
                    color: '#ffffff',
                    usePointStyle: true,
                    pointStyle: 'circle'
                }}
            }},
            tooltip: {{
                backgroundColor: 'rgba(26, 26, 46, 0.95)',
                titleColor: '#ce93d8',
                bodyColor: '#ffffff',
                borderColor: '#7e57c2',
                borderWidth: 1,
                cornerRadius: 8,
                titleFont: {{
                    size: 14,
                    weight: '600'
                }},
                bodyFont: {{
                    size: 13
                }},
                padding: 12
            }}
        }},
        animation: {{
            duration: 1500,
            easing: 'easeInOutQuart'
        }}
    }};

    // Chart 1: Relevant vs Non-Relevant incidents (Doughnut with gradient)
    const relevanceCtx = document.getElementById('{relevance_chart_id}').getContext('2d');
    
    // Create gradients
    const relevantGradient = relevanceCtx.createLinearGradient(0, 0, 0, 400);
    relevantGradient.addColorStop(0, '#7e57c2');
    relevantGradient.addColorStop(1, '#ce93d8');
    
    const nonRelevantGradient = relevanceCtx.createLinearGradient(0, 0, 0, 400);
    nonRelevantGradient.addColorStop(0, '#424242');
    nonRelevantGradient.addColorStop(1, '#616161');

    new Chart(relevanceCtx, {{
        type: 'doughnut',
        data: {{
            labels: ['Relevant Incidents', 'Non-Relevant Incidents'],
            datasets: [{{
                data: [{related_count}, {non_related_count}],
                backgroundColor: [relevantGradient, nonRelevantGradient],
                borderWidth: 0,
                hoverBorderWidth: 3,
                hoverBorderColor: '#ffffff',
                cutout: '60%'
            }}]
        }},
        options: {{
            ...chartDefaults,
            plugins: {{
                ...chartDefaults.plugins,
                title: {{
                    display: false
                }}
            }}
        }}
    }});

    // Chart 2: Resolution status (Doughnut with premium colors)
    const resolutionCtx = document.getElementById('{resolution_id}').getContext('2d');
    
    const closedGradient = resolutionCtx.createLinearGradient(0, 0, 0, 400);
    closedGradient.addColorStop(0, '#4caf50');
    closedGradient.addColorStop(1, '#66bb6a');
    
    const openGradient = resolutionCtx.createLinearGradient(0, 0, 0, 400);
    openGradient.addColorStop(0, '#ff9800');
    openGradient.addColorStop(1, '#ffb74d');
    
    const otherGradient = resolutionCtx.createLinearGradient(0, 0, 0, 400);
    otherGradient.addColorStop(0, '#9e9e9e');
    otherGradient.addColorStop(1, '#bdbdbd');

    new Chart(resolutionCtx, {{
        type: 'doughnut',
        data: {{
            labels: ['Closed/Resolved', 'Open/In Progress', 'Other Status'],
            datasets: [{{
                data: [{closed}, {open_}, {other}],
                backgroundColor: [closedGradient, openGradient, otherGradient],
                borderWidth: 0,
                hoverBorderWidth: 3,
                hoverBorderColor: '#ffffff',
                cutout: '60%'
            }}]
        }},
        options: {{
            ...chartDefaults,
            plugins: {{
                ...chartDefaults.plugins,
                title: {{
                    display: false
                }}
            }}
        }}
    }});

    // Chart 3: Incidents resolved by person (horizontal bar with gradient)
    const resolverCtx = document.getElementById('{resolver_id}').getContext('2d');
    const resolverLabels = {resolver_labels_json};
    const resolverData = {resolver_data_json};
    
    const barGradient = resolverCtx.createLinearGradient(0, 0, 400, 0);
    barGradient.addColorStop(0, '#7e57c2');
    barGradient.addColorStop(1, '#ce93d8');
    
    new Chart(resolverCtx, {{
        type: 'bar',
        data: {{
            labels: resolverLabels,
            datasets: [{{
                label: 'Resolved Incidents',
                data: resolverData,
                backgroundColor: barGradient,
                borderColor: '#7e57c2',
                borderWidth: 0,
                borderRadius: 8,
                borderSkipped: false,
                hoverBackgroundColor: '#ce93d8',
                hoverBorderWidth: 2,
                hoverBorderColor: '#ffffff'
            }}]
        }},
        options: {{
            ...chartDefaults,
            indexAxis: 'y',
            plugins: {{
                ...chartDefaults.plugins,
                legend: {{
                    display: false
                }},
                title: {{
                    display: false
                }}
            }},
            scales: {{
                x: {{ 
                    beginAtZero: true,
                    grid: {{
                        color: 'rgba(255, 255, 255, 0.1)',
                        borderColor: 'rgba(255, 255, 255, 0.2)'
                    }},
                    ticks: {{
                        color: '#b0b0b0',
                        font: {{
                            size: 12
                        }}
                    }},
                    title: {{
                        display: true,
                        text: 'Number of Incidents',
                        color: '#ffffff',
                        font: {{
                            size: 14,
                            weight: '500'
                        }}
                    }}
                }},
                y: {{ 
                    grid: {{
                        display: false
                    }},
                    ticks: {{
                        color: '#b0b0b0',
                        font: {{
                            size: 12,
                            weight: '500'
                        }}
                    }},
                    title: {{
                        display: true,
                        text: 'Team Members',
                        color: '#ffffff',
                        font: {{
                            size: 14,
                            weight: '500'
                        }}
                    }}
                }}
            }}
        }}
    }});

    // Chart 4: Assignment Groups (Pie Chart)
    const groupCtx = document.getElementById('{group_id}').getContext('2d');
    const groupLabels = {group_labels_json};
    const groupData = {group_data_json};
    
    // Create diverse colors for groups
    const groupColors = [
        '#7e57c2', '#ce93d8', '#4caf50', '#ff9800', 
        '#2196f3', '#e91e63', '#9c27b0', '#607d8b'
    ];
    
    new Chart(groupCtx, {{
        type: 'pie',
        data: {{
            labels: groupLabels,
            datasets: [{{
                data: groupData,
                backgroundColor: groupColors.slice(0, groupLabels.length),
                borderWidth: 0,
                hoverBorderWidth: 3,
                hoverBorderColor: '#ffffff'
            }}]
        }},
        options: {{
            ...chartDefaults,
            plugins: {{
                ...chartDefaults.plugins,
                title: {{
                    display: false
                }}
            }}
        }}
    }});

    // Chart 5: CI Classes (Polar Area Chart)
    const ciClassCtx = document.getElementById('{ci_class_id}').getContext('2d');
    const ciClassLabels = {ci_class_labels_json};
    const ciClassData = {ci_class_data_json};
    
    const ciClassColors = [
        'rgba(126, 87, 194, 0.8)', 'rgba(206, 147, 216, 0.8)', 
        'rgba(76, 175, 80, 0.8)', 'rgba(255, 152, 0, 0.8)',
        'rgba(33, 150, 243, 0.8)', 'rgba(233, 30, 99, 0.8)'
    ];
    
    new Chart(ciClassCtx, {{
        type: 'polarArea',
        data: {{
            labels: ciClassLabels,
            datasets: [{{
                data: ciClassData,
                backgroundColor: ciClassColors.slice(0, ciClassLabels.length),
                borderWidth: 0,
                hoverBorderWidth: 3,
                hoverBorderColor: '#ffffff'
            }}]
        }},
        options: {{
            ...chartDefaults,
            plugins: {{
                ...chartDefaults.plugins,
                title: {{
                    display: false
                }}
            }},
            scales: {{
                r: {{
                    grid: {{
                        color: 'rgba(255, 255, 255, 0.1)'
                    }},
                    ticks: {{
                        color: '#b0b0b0',
                        font: {{
                            size: 10
                        }}
                    }},
                    pointLabels: {{
                        color: '#ffffff',
                        font: {{
                            size: 12,
                            weight: '500'
                        }}
                    }}
                }}
            }}
        }}
    }});
}})();
</script>
"""
    return analytics_html

def call_groq_llm(user_prompt: str, incidents: List[dict]) -> dict:
    # ✅ UPDATED SYSTEM PROMPT TO MATCH NEW JSON STRUCTURE
    system_prompt = """
You are an AI incident support assistant...

You are given:
1. A user prompt,
2. A list of past incidents,
3. A mandatory solution guide.

You MUST respond in raw HTML format only.

**Rules**:
- For ANY incident-related question (even if not explicitly about resolution), you MUST include:
  - <h3>Summary of Closure notes</h3> - derived from closure notes of similar incidents and also provided in points not as paragraph
  - <h3>Steps to Resolution</h3> - from the solution guide (only if the guide is available)
- If the user prompt is unrelated to incidents or pointing out mistake in your previous prompt, respond with:
<p>Sorry, I can only discuss incident-related issues.</p>

**Important Formatting Logic**:
- While producing output follow the flow don't randomize the flow
- If NO incidents are relevant or found, simply say:
<p>No incidents found for this query.</p>
and skip all related sections.
- DO NOT render any value that is missing, "None", or "Unknown".
- OMIT entire sections if no valid data is available.
- For "Summary of Closure notes" - extract and synthesize steps from closure_notes field of incidents
- For "Steps to Resolution" - use the provided solution guide content
- Ensure all HTML output is clean and avoid displaying empty placeholders or filler data.
- Summary Table should atleast have 3 or more fields like Incident ID, Description, State or closure notes

Respond using valid HTML only. Do not use Markdown.

If incident data exists and is relevant, include these sections (with actual content only):
- <h3>Summary</h3><p>...</p>
- <h3>Summary of Closure notes</h3><ol>...</ol>
- <h3>Steps to Resolution</h3><ol>...</ol>
- <h3>Incident Summary Table</h3><table>...</table>
"""

    try:
        print("⚙️ Loading solution doc and embeddings...")
        solution_guide = query_solutions_from_qdrant(user_prompt)
    except Exception as e:
        print(f"❌ Failed to load solution guide: {e}")
        solution_guide = ""

    # ✅ UPDATED FORMATTING TO MATCH NEW JSON STRUCTURE
    formatted_incidents = "\n\n".join([
        f"Incident {i+1}:\n"
        f"Incident ID: {d['incident']}\n"
        f"Job Name: {d.get('job_name', 'N/A')}\n"
        f"Description: {d['description']}\n"
        f"Impact: {d.get('impact', 'N/A')}\n"
        f"Closure Notes: {d['closure_notes']}\n"
        f"Assigned To: {d['assigned_to']}\n"
        f"Assignment Group: {d.get('assignment_group', 'N/A')}\n"
        f"Configuration Item: {d.get('configuration_item', 'N/A')}\n"
        f"CI Class: {d.get('ci_class', 'N/A')}\n"
        f"Opened By: {d.get('opened_by', 'N/A')}\n"
        f"Resolved By: {d.get('resolved_by', 'N/A')}\n"
        f"Closed By: {d.get('closed_by', 'N/A')}\n"
        f"Opened Time: {d.get('opened_time', 'N/A')}\n"
        f"Resolved Time: {d.get('resolved', 'N/A')}\n"
        f"Closed Time: {d.get('closed', 'N/A')}\n"
        f"Priority: {d['priority']}, Urgency: {d['urgency']}, State: {d['state']}"
        for i, d in enumerate(incidents)
    ]) if incidents else "No incidents found."
    print(f"🔍 Number of incidents passed to LLM: {len(incidents)}")
    print(f"📖 Solution guide length: {len(solution_guide)} characters")

    # Extract closure notes for resolution steps
    closure_notes_summary = "\n".join([
        f"Incident {d['incident']}: {d['closure_notes']}" 
        for d in incidents 
        if d.get('closure_notes') and d['closure_notes'].strip() and d['closure_notes'] != 'N/A'
    ])

    full_prompt = f"""
    User Prompt: \"{user_prompt}\"

    Solution Guide (for Original Steps):
    {solution_guide}

    Closure Notes from Similar Incidents (for practical resolution steps):
    {closure_notes_summary}

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

    # ✅ UPDATED analytics function with smart filtering
    analytics_html = compute_analytics(incidents, user_prompt)
    return {"html": content + analytics_html}

@app.post("/chat")
def chat(request: PromptRequest):
    try:
        print("📨 Prompt received:", request.prompt)
        incidents = get_incidents_from_qdrant(request.prompt)
        result = call_groq_llm(request.prompt, incidents)

        async def content_generator():
            yield result["html"]

        return StreamingResponse(content_generator(), media_type="text/html")

    except Exception as e:
        traceback.print_exc()
        return StreamingResponse(
            iter([f"<p>⚠️ {type(e).__name__}: {str(e)}</p>"]),
            media_type="text/html"
        )