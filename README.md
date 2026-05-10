# Drone SAR — Search & Rescue Vision

An AI-powered system that detects and localizes people in danger using a drone-mounted **Luxonis OAK 4 D** camera. All inference runs on-device: no cloud, no latency.

**Hardware:** OAK 4 D (stereo depth + RGB + 48 TOPS on-device AI, 8 GB RAM), mounted on a drone. The host laptop only displays results — all processing happens on the camera.

---

## Idea

During disasters like floods, earthquakes, or wildfires, locating survivors quickly is critical. A drone equipped with a depth-sensing AI camera can autonomously scan large areas, detect people, estimate their real-world distance, identify if they are stationary (possibly injured or unconscious), and stream alerts to a rescue dashboard — without requiring a human operator to watch every frame.

---

## Implementation

The full pipeline runs on the OAK 4 D:

```
ColorCamera (RGB)          → person detection + re-ID
MonoCameras (left + right) → stereo depth map
StereoDepth node           → metric distance per detection
SpatialDetectionNetwork    → bounding box + XYZ coordinates
ObjectTracker              → stable person IDs across frames
AnnotationNode             → stillness detection (fallen / stationary)
VideoRecorder              → saves annotated footage to disk
FastAPI backend            → REST + WebSocket events to the dashboard
```

Each tracked person gets: a bounding box, a distance in meters, a stable ID, a `stationary` flag (if motionless for N seconds), and optionally a fall-detection label.

### Models

Two YOLOv8n models were trained on Google Colab and exported to RVC4 format:

| Model | Task | Dataset |
|---|---|---|
| `best.rvc4.tar.xz` | Person detection | Roboflow Universe |
| `best_fall_detection_model.rvc4.tar.xz` | Fall / lying-down detection | Roboflow fall-detection dataset |

Training augmentations (Albumentations) simulate aerial and disaster conditions:
- `RandomBrightnessContrast` — backlighting, shadows
- `RandomFog` — smoke, haze, water spray
- `GaussianBlur` — camera vibration from drone motors
- `CoarseDropout` — partial occlusion (debris, foliage)
- `GaussNoise` — low-light sensor noise

Re-identification uses **OSNet** (from the Luxonis model zoo) to assign consistent IDs across drone passes.

---

## Running the Camera Code

### Requirements

- Python >= 3.10
- Luxonis OAK 4 D connected (USB or PoE)
- [`oakctl`](https://docs.luxonis.com/software-v3/oak-apps/oakctl) installed for standalone mode

### Install dependencies

```bash
cd spatial-detections
pip install -r requirements.txt
```

### Peripheral mode (host + camera)

```bash
python3 main.py
```

Open `http://localhost:8082` in your browser to see the live view with detections and depth overlay.

Optional flags:
```
--device <IP or device ID>   Connect to a specific OAK device
--fps_limit <N>              Cap frame rate (default: 30 on RVC4)
```

### Standalone mode (runs entirely on the OAK 4 D)

```bash
oakctl connect <DEVICE_IP>
oakctl app run .
```

The app bundles and deploys itself to the camera — no host needed during flight.

### Backend (dashboard API)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

API available at `http://localhost:8000`. WebSocket stream at `ws://localhost:8000/ws`.
