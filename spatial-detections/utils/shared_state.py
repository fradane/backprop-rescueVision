import threading

_lock = threading.Lock()
_fall_boxes: list[tuple[float, float, float, float, str]] = []  # (xmin, ymin, xmax, ymax, label)


def update_fall_detections(detections: list) -> None:
    with _lock:
        _fall_boxes.clear()
        _fall_boxes.extend(detections)


def get_fall_detections() -> list[tuple[float, float, float, float, str]]:
    with _lock:
        return list(_fall_boxes)
