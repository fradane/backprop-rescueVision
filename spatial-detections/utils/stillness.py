import math
import time
import numpy as np
from typing import Optional

MOVE_THRESHOLD_MM = 150   # mm - sotto = inizia a contare immobilità
BREAK_THRESHOLD_MM = 250  # mm - sopra = rompe l'immobilità (hysteresis)
STILL_SECONDS = 5         # secondi per alert immobilità
REID_THRESHOLD_MM = 500   # mm - distanza entro cui cercare match
LOST_TIMEOUT_S = 30       # secondi - dopo quanto eliminare una persona persa
CSIM_THRESHOLD = 0.8      # soglia similarità coseno osnet
EMA_ALPHA = 0.3           # peso del frame corrente nell'EMA (0=max smooth, 1=no smooth)

_track_history = {}


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _find_lost_match(x: float, y: float, z: float, embedding: Optional[np.ndarray]) -> Optional[str]:
    best_id = None
    best_score = -1.0

    for tid, data in _track_history.items():
        if not data.get("lost"):
            continue
        dist = math.sqrt((x - data["x"]) ** 2 + (y - data["y"]) ** 2 + (z - data["z"]) ** 2)
        if dist > REID_THRESHOLD_MM:
            continue

        if embedding is not None and data.get("embedding") is not None:
            score = _cos_sim(embedding, data["embedding"])
            if score > CSIM_THRESHOLD and score > best_score:
                best_score = score
                best_id = tid
        else:
            pos_score = 1.0 - dist / REID_THRESHOLD_MM
            if pos_score > best_score:
                best_score = pos_score
                best_id = tid

    return best_id


def resolve_track_id(tracker_id: str, x: float, y: float, z: float, embedding: Optional[np.ndarray] = None) -> str:
    if tracker_id in _track_history and not _track_history[tracker_id].get("lost"):
        if embedding is not None:
            _track_history[tracker_id]["embedding"] = embedding
        return tracker_id

    match = _find_lost_match(x, y, z, embedding)
    if match:
        _track_history[tracker_id] = _track_history.pop(match)
        _track_history[tracker_id]["lost"] = False

    return tracker_id


def check_stillness(track_id: str, x: float, y: float, z: float, embedding: Optional[np.ndarray] = None):
    now = time.time()

    if track_id not in _track_history:
        _track_history[track_id] = {
            "x": x, "y": y, "z": z,
            "sx": x, "sy": y, "sz": z,  # posizione smoothed (EMA)
            "last_move_time": now,
            "last_seen": now,
            "lost": False,
            "is_still": False,
            "embedding": embedding,
        }
        return False, 0.0

    prev = _track_history[track_id]
    prev["last_seen"] = now
    prev["lost"] = False
    if embedding is not None:
        prev["embedding"] = embedding

    # aggiorna EMA
    prev["sx"] = EMA_ALPHA * x + (1 - EMA_ALPHA) * prev["sx"]
    prev["sy"] = EMA_ALPHA * y + (1 - EMA_ALPHA) * prev["sy"]
    prev["sz"] = EMA_ALPHA * z + (1 - EMA_ALPHA) * prev["sz"]

    dist = math.sqrt(
        (prev["sx"] - prev["x"]) ** 2 +
        (prev["sy"] - prev["y"]) ** 2 +
        (prev["sz"] - prev["z"]) ** 2
    )

    # threshold dipende dallo stato corrente (hysteresis)
    threshold = BREAK_THRESHOLD_MM if prev["is_still"] else MOVE_THRESHOLD_MM

    if dist > threshold:
        prev["x"], prev["y"], prev["z"] = prev["sx"], prev["sy"], prev["sz"]
        prev["last_move_time"] = now
        prev["is_still"] = False
        return False, 0.0
    else:
        still_duration = now - prev["last_move_time"]
        prev["is_still"] = still_duration >= STILL_SECONDS
        return prev["is_still"], still_duration


def cleanup_tracks(active_ids: set):
    now = time.time()
    for tid in list(_track_history.keys()):
        if tid not in active_ids:
            _track_history[tid]["lost"] = True
            if now - _track_history[tid]["last_seen"] > LOST_TIMEOUT_S:
                del _track_history[tid]
