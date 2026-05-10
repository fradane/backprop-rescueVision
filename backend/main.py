from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import json

app = FastAPI(title="OAK 4D SAR Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Data Model ───────────────────────────────────────────────────────────────

class Detection(BaseModel):
    person_id:  int
    distance_m: float
    confidence: float
    bbox:       list[float]
    stationary: bool
    lat:        Optional[float] = None
    lon:        Optional[float] = None
    is_fallen:  Optional[bool]  = False
    still_secs: Optional[float] = 0.0
    timestamp:  Optional[str]   = None

# ── In-Memory Store ──────────────────────────────────────────────────────────

MAX_EVENTS        = 200
detections_store: list[dict]      = []
active_persons:   dict[int, dict] = {}
ws_clients:       list[WebSocket] = []

# ── Helpers ──────────────────────────────────────────────────────────────────

async def broadcast(message: dict):
    dead = []
    text = json.dumps(message)
    for client in ws_clients:
        try:
            await client.send_text(text)
        except Exception:
            dead.append(client)
    for c in dead:
        ws_clients.remove(c)

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "online",
        "endpoints": [
            "POST   /detections",
            "GET    /detections/latest?n=50",
            "GET    /detections/summary",
            "DELETE /detections/reset",
            "WS     /ws",
        ],
    }


@app.post("/detections", status_code=201)
async def receive_detection(det: Detection):
    event = det.model_dump()
    if not event["timestamp"]:
        event["timestamp"] = datetime.utcnow().isoformat() + "Z"

    detections_store.append(event)
    if len(detections_store) > MAX_EVENTS:
        detections_store.pop(0)
    active_persons[det.person_id] = event

    await broadcast({"type": "detection", "data": event})
    return {"status": "ok", "stored": len(detections_store)}


@app.get("/detections/latest")
async def get_latest(n: int = 50):
    persons = sorted(active_persons.values(), key=lambda e: e["timestamp"], reverse=True)
    return {"count": len(persons), "events": persons[:n]}


@app.get("/detections/summary")
async def get_summary():
    persons    = list(active_persons.values())
    stationary = [p for p in persons if p["stationary"]]
    distances  = [p["distance_m"] for p in persons]
    return {
        "total_persons_tracked": len(persons),
        "stationary_count":      len(stationary),
        "stationary_ids":        [p["person_id"] for p in stationary],
        "stationary_locations":  [
            {"person_id": p["person_id"], "lat": p.get("lat"), "lon": p.get("lon")}
            for p in stationary if p.get("lat") is not None
        ],
        "closest_person_m":      min(distances) if distances else None,
        "farthest_person_m":     max(distances) if distances else None,
        "last_event_ts":         detections_store[-1]["timestamp"] if detections_store else None,
    }


@app.delete("/detections/reset")
async def reset():
    detections_store.clear()
    active_persons.clear()
    return {"status": "reset"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        snapshot = {
            "type": "snapshot",
            "data": {
                "recent_events":  sorted(active_persons.values(), key=lambda e: e["timestamp"], reverse=True),
                "active_persons": list(active_persons.values()),
            },
        }
        await ws.send_text(json.dumps(snapshot))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in ws_clients:
            ws_clients.remove(ws)
