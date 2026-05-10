import depthai as dai
import numpy as np
from depthai_nodes import PRIMARY_COLOR, TRANSPARENT_PRIMARY_COLOR
from depthai_nodes.utils import AnnotationHelper
from typing import List
import cv2
import threading
import requests
from utils.stillness import check_stillness, cleanup_tracks, resolve_track_id

# Placeholder — in produzione da GPS/IMU reali
GPS_LAT = 44.4056
GPS_LON = 8.9463

BACKEND_URL = "http://localhost:8000/detections"

# track_id -> last state sent: {"stationary": bool, "is_fallen": bool}
_sent_states: dict[str, dict] = {}

# aspect ratio width/height sopra cui la persona è considerata caduta
FALL_ASPECT_RATIO = 1.3

COLOR_NORMAL   = PRIMARY_COLOR
COLOR_FALLEN   = (1.0, 0.5, 0.0, 1.0)   # arancione
COLOR_STILL    = (1.0, 0.0, 0.0, 1.0)   # rosso
COLOR_FALLEN_STILL = (0.8, 0.0, 0.8, 1.0)  # viola — caduto + immobile

_next_id = 0
_embedding_db = {}  # stable_id -> embedding
CSIM_THRESHOLD = 0.8


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _send_detection(payload: dict) -> None:
    try:
        requests.post(BACKEND_URL, json=payload, timeout=2)
    except Exception:
        pass


def _should_send(track_id: str, stationary: bool, is_fallen: bool) -> bool:
    prev = _sent_states.get(track_id)
    if prev is None:
        return True
    return prev["stationary"] != stationary or prev["is_fallen"] != is_fallen


def _assign_id(embedding: np.ndarray) -> str:
    global _next_id
    best_id = None
    best_score = CSIM_THRESHOLD

    for sid, emb in _embedding_db.items():
        score = _cos_sim(embedding, emb)
        if score > best_score:
            best_score = score
            best_id = sid

    if best_id is None:
        best_id = str(_next_id)
        _next_id += 1
        _embedding_db[best_id] = embedding
    else:
        _embedding_db[best_id] = 0.8 * _embedding_db[best_id] + 0.2 * embedding
    return best_id


class AnnotationNode(dai.node.HostNode):
    def __init__(self) -> None:
        super().__init__()
        self.input_gathered = self.createInput()
        self.out_annotations = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )
        self.out_depth = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.labels = []

    def build(
        self,
        input_gathered: dai.Node.Output,
        depth: dai.Node.Output,
        labels: List[str],
    ) -> "AnnotationNode":
        self.labels = labels
        self.link_args(input_gathered, depth)
        return self

    def process(self, gathered_msg: dai.Buffer, depth_message: dai.ImgFrame) -> None:
        detections_msg: dai.SpatialImgDetections = gathered_msg.reference_data
        embeddings_list = gathered_msg.items

        assert isinstance(detections_msg, dai.SpatialImgDetections)

        detections = [d for d in detections_msg.detections if d.label == 0]

        annotation_helper = AnnotationHelper()
        active_ids = set()

        for i, detection in enumerate(detections):
            xmin, ymin, xmax, ymax = detection.xmin, detection.ymin, detection.xmax, detection.ymax
            x = detection.spatialCoordinates.x
            y = detection.spatialCoordinates.y
            z = detection.spatialCoordinates.z

            embedding = None
            if i < len(embeddings_list):
                try:
                    embedding = embeddings_list[i].getTensor("output", dequantize=True)
                except Exception:
                    pass

            if embedding is not None:
                embedding = embedding.flatten()
                track_id = _assign_id(embedding)
                track_id = resolve_track_id(track_id, x, y, z, embedding)
            else:
                track_id = resolve_track_id(f"det_{i}", x, y, z)

            active_ids.add(track_id)
            is_still, still_secs = check_stillness(track_id, x, y, z, embedding)

            box_w = xmax - xmin
            box_h = ymax - ymin
            is_fallen = (box_h > 0) and (box_w / box_h) > FALL_ASPECT_RATIO

            if is_fallen and is_still:
                box_color = COLOR_FALLEN_STILL
            elif is_fallen:
                box_color = COLOR_FALLEN
            elif is_still:
                box_color = COLOR_STILL
            else:
                box_color = COLOR_NORMAL

            annotation_helper.draw_rectangle(
                top_left=(xmin, ymin),
                bottom_right=(xmax, ymax),
                outline_color=box_color,
                fill_color=TRANSPARENT_PRIMARY_COLOR,
                thickness=2.0,
            )

            parts = [f"person #{track_id} {int(detection.confidence * 100)}%"]
            if is_fallen:
                parts.append("CADUTO")
            if is_still:
                parts.append(f"IMMOBILE {still_secs:.0f}s")
            parts.append(f"dist: {z/1000:.1f}m")
            parts.append(f"lat: {GPS_LAT:.6f}  lon: {GPS_LON:.6f}")

            annotation_helper.draw_text(
                text="\n".join(parts),
                position=(xmin + 0.01, ymin + 0.2),
                size=12,
                color=box_color,
            )

            # send to backend only on state change (no flood)
            if _should_send(track_id, is_still, is_fallen):
                _sent_states[track_id] = {"stationary": is_still, "is_fallen": is_fallen}
                payload = {
                    "person_id":  int(track_id) if track_id.isdigit() else hash(track_id) % 10000,
                    "distance_m": round(z / 1000, 3),
                    "confidence": round(float(detection.confidence), 3),
                    "bbox":       [round(xmin, 4), round(ymin, 4), round(xmax, 4), round(ymax, 4)],
                    "stationary": is_still,
                    "lat":        GPS_LAT,
                    "lon":        GPS_LON,
                    "is_fallen":  is_fallen,
                    "still_secs": round(still_secs, 1),
                }
                threading.Thread(target=_send_detection, args=(payload,), daemon=True).start()

        cleanup_tracks(active_ids)

        annotations = annotation_helper.build(
            timestamp=detections_msg.getTimestamp(),
            sequence_num=detections_msg.getSequenceNum(),
        )

        depth_map = depth_message.getCvFrame()
        depth_map = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_map, alpha=0.3), cv2.COLORMAP_JET
        )

        depth_frame = dai.ImgFrame()
        depth_frame.setCvFrame(depth_map, dai.ImgFrame.Type.BGR888i)
        depth_frame.setTimestamp(depth_message.getTimestamp())
        depth_frame.setSequenceNum(depth_message.getSequenceNum())

        self.out_annotations.send(annotations)
        self.out_depth.send(depth_frame)
