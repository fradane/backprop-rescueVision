# Search & Rescue Vision

An AI-powered person detection system that runs entirely on a **Luxonis OAK 4 D camera**. The camera autonomously identifies and localizes people in dangerous, low-visibility environments — smoke-filled rooms, dark buildings, debris-cluttered spaces — without streaming any frames to an external computer.

## The Problem

A firefighter or rescue operator entering a dangerous building has limited visibility and can easily miss a survivor. A camera mounted on their equipment that detects people and reports their exact 3D position — even through smoke, in the dark, or partially hidden behind obstacles — can directly save lives.

## How It Works

The full pipeline runs on-device:

```
ColorCamera node       → captures RGB frame
MonoCamera nodes (×2)  → left + right grayscale for stereo depth
StereoDepth node       → computes the depth map
NeuralNetwork node     → runs the model → bounding boxes + confidence scores
ObjectTracker node     → assigns stable IDs across frames
Sync / Script node     → aligns depth to each detection → real-world distance
```

Each detected person gets a bounding box, a distance in meters, a stable track ID, and a flag if they have been stationary for more than N seconds (possibly incapacitated).

## The Model

- **Base:** YOLOv8 nano (Ultralytics), fine-tuned from pretrained weights
- **Dataset:** person detection images from Roboflow Universe
- **Augmentations** (via Albumentations) simulate real rescue conditions:
  - `RandomBrightnessContrast` — dark rooms
  - `RandomFog` — smoke and haze
  - `GaussianBlur` — camera shake
  - `CoarseDropout` — debris partially blocking people
  - `ToGray` — near-IR lighting simulation
- **Training:** 30–50 epochs on Google Colab
- **Deployment format:** converted from `.pt` → `.onnx` → RVC4-compatible model, packaged as an OAK App and deployed via `oakctl`

## Hardware

| Component | Role |
|---|---|
| OAK 4 D Camera | Stereo depth + color camera + on-device AI (48 TOPS INT8, 8GB RAM) |
| PoE+ Switch | Powers the camera through the Ethernet cable |
| ETH cable | Camera → switch → laptop |

The laptop only displays results. All inference, depth computation, and tracking happen on the camera itself.

## Stack

- [DepthAI v3 SDK](https://docs.luxonis.com/software-v3/depthai/) — on-device pipeline
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — model training and export
- [Albumentations](https://albumentations.ai/) — training augmentations
- [OAK Apps + oakctl](https://docs.luxonis.com/software-v3/oak-apps/) — on-device deployment
- [Google Colab](https://colab.research.google.com/) — GPU training environment

## Key Links

- [OAK4 Getting Started](https://docs.luxonis.com/hardware/platform/deploy/oak4-deployment-guide/oak4-getting-started/)
- [DepthAI Examples](https://docs.luxonis.com/software-v3/depthai/examples/)
- [Roboflow Universe](https://universe.roboflow.com/)
