from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import json, uuid, os
from dotenv import load_dotenv

load_dotenv()
client = QdrantClient(api_key=os.getenv("QDRANT_API_KEY"), url=os.getenv("QDRANT_URL"))
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("incidents.json") as f:
    data = json.load(f)

points = []
for incident in data:
    vector = model.encode(incident["incident_description"]).tolist()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=incident
    )
    points.append(point)

client.upsert(collection_name="incidents", points=points)
print("Uploaded!")
