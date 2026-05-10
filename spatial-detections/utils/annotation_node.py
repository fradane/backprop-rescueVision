import depthai as dai
import numpy as np
from depthai_nodes import PRIMARY_COLOR, TRANSPARENT_PRIMARY_COLOR
from depthai_nodes.utils import AnnotationHelper
from typing import List
import cv2
import threading
import requests
from utils.stillness import check_stillness, cleanup_tracks, resolve_track_id
from utils.sensor_data import compute_absolute_coordinates, get_sensor_data

BACKEND_URL = "http://localhost:8000/detections"
MIN_CONFIDENCE = 0.65

# track_id -> last state sent: {"stationary": bool, "is_fallen": bool}
_sent_states: dict[str, dict] = {}

# aspect ratio width/height sopra cui la persona è considerata caduta
FALL_ASPECT_RATIO = 1.3

COLOR_NORMAL       = PRIMARY_COLOR
COLOR_FALLEN       = (1.0, 0.5, 0.0, 1.0)
COLOR_STILL        = (1.0, 0.0, 0.0, 1.0)
COLOR_FALLEN_STILL = (0.8, 0.0, 0.8, 1.0)

# BGR equivalenti per OpenCV (video recording)
CV_NORMAL       = (255, 255,   0)   # cyan
CV_FALLEN       = (  0, 128, 255)   # arancione
CV_STILL        = (  0,   0, 255)   # rosso
CV_FALLEN_STILL = (204,   0, 204)   # viola

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
        self.out_annotated_frame = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgFrame, True)
            ]
        )
        self.labels = []

    def build(
        self,
        input_gathered: dai.Node.Output,
        depth: dai.Node.Output,
        passthrough: dai.Node.Output,
        labels: List[str],
    ) -> "AnnotationNode":
        self.labels = labels
        self.link_args(input_gathered, depth, passthrough)
        return self

    def process(self, gathered_msg: dai.Buffer, depth_message: dai.ImgFrame, rgb_msg: dai.Buffer) -> None:
        rgb_frame: dai.ImgFrame = rgb_msg
        detections_msg: dai.SpatialImgDetections = gathered_msg.reference_data
        embeddings_list = gathered_msg.items

        assert isinstance(detections_msg, dai.SpatialImgDetections)

        detections = [d for d in detections_msg.detections if d.label == 0]

        annotation_helper = AnnotationHelper()
        active_ids = set()

        canvas = rgb_frame.getCvFrame().copy()
        h, w = canvas.shape[:2]

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

            sensor = get_sensor_data()
            abs_lat, abs_lon = compute_absolute_coordinates(x, y, z, sensor)

            box_w = xmax - xmin
            box_h = ymax - ymin
            is_fallen = (box_h > 0) and (box_w / box_h) > FALL_ASPECT_RATIO

            if is_fallen and is_still:
                box_color = COLOR_FALLEN_STILL
                cv_color  = CV_FALLEN_STILL
            elif is_fallen:
                box_color = COLOR_FALLEN
                cv_color  = CV_FALLEN
            elif is_still:
                box_color = COLOR_STILL
                cv_color  = CV_STILL
            else:
                box_color = COLOR_NORMAL
                cv_color  = CV_NORMAL

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
            parts.append(f"lat: {abs_lat:.6f}  lon: {abs_lon:.6f}")

            annotation_helper.draw_text(
                text="\n".join(parts),
                position=(xmin + 0.01, ymin + 0.2),
                size=12,
                color=box_color,
            )

            # send to backend only on state change and above confidence threshold
            if detection.confidence >= MIN_CONFIDENCE and _should_send(track_id, is_still, is_fallen):
                _sent_states[track_id] = {"stationary": is_still, "is_fallen": is_fallen}
                payload = {
                    "person_id":  int(track_id) if track_id.isdigit() else hash(track_id) % 10000,
                    "distance_m": round(z / 1000, 3),
                    "confidence": round(float(detection.confidence), 3),
                    "bbox":       [round(xmin, 4), round(ymin, 4), round(xmax, 4), round(ymax, 4)],
                    "stationary": is_still,
                    "lat":        round(abs_lat, 7),
                    "lon":        round(abs_lon, 7),
                    "is_fallen":  is_fallen,
                    "still_secs": round(still_secs, 1),
                }
                threading.Thread(target=_send_detection, args=(payload,), daemon=True).start()

            # disegna box e label sul canvas per il video
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), cv_color, 2)
            for j, line in enumerate(parts):
                cv2.putText(canvas, line, (x1 + 4, y1 + 16 + j * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, cv_color, 1, cv2.LINE_AA)

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

        annotated = dai.ImgFrame()
        annotated.setCvFrame(canvas, dai.ImgFrame.Type.BGR888i)
        annotated.setTimestamp(detections_msg.getTimestamp())
        annotated.setSequenceNum(detections_msg.getSequenceNum())

        self.out_annotations.send(annotations)
        self.out_depth.send(depth_frame)
        self.out_annotated_frame.send(annotated)
