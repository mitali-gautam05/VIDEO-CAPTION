# Silent Video Understanding AI

> An end-to-end multimodal AI system that analyzes silent videos using Computer Vision, Motion Analysis, Object Detection, and Natural Language Generation — and automatically generates contextual captions.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Silent Video Understanding AI bridges the gap between raw video data and human-readable understanding — without relying on audio. It combines three state-of-the-art AI models in a unified pipeline:

- **YOLOv8** for real-time object detection across frames
- **MediaPipe** for human pose estimation and motion trajectory analysis
- **BLIP Transformer** for intelligent, context-aware caption generation

The result is an automatically captioned video with an AI-generated scene description, deployable as a web app on Hugging Face Spaces.

---

## Demo

> Live demo: [[Hugging Face Spaces](https://huggingface.co/spaces/Mitali1234/Video_Captioning)]

**Example input:** Silent video of a person running in a park

**AI-generated output:**
```
The video shows a person running outdoors in a park environment.
The subject appears to be moving forward continuously with active
leg motion detected across multiple frames.
```

---

## Features

- Frame-by-frame object detection using YOLOv8
- Human pose and motion direction analysis via MediaPipe
- Transformer-based frame captioning with BLIP
- Context fusion across all extracted signals into a single video description
- Captioned video rendering with overlaid text output
- Interactive Gradio web interface — no setup needed for end users
- One-click deployment on Hugging Face Spaces

---

## Architecture

```
Input Video
     │
     ▼
Frame Extraction          (OpenCV)
     │
     ├──────────────────────────────────┐
     ▼                                  ▼
Object Detection               Pose & Motion Analysis
   (YOLOv8)                       (MediaPipe)
     │                                  │
     └──────────────┬───────────────────┘
                    ▼
          Frame Caption Generation
                 (BLIP)
                    │
                    ▼
          Context Fusion & Understanding
                    │
                    ▼
          Final AI Video Caption
                    │
                    ▼
         Captioned Video Rendering
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Deep Learning | PyTorch, TensorFlow |
| Object Detection | YOLOv8 (Ultralytics) |
| Pose Estimation | MediaPipe Pose Landmarker |
| Image Captioning | BLIP (Salesforce, via HuggingFace Transformers) |
| Video Processing | OpenCV |
| Web Interface | Gradio |
| Deployment | Hugging Face Spaces |

---

## Project Structure

```
AI-VIDEO-CAPTION/
│
├── app.py                    # Gradio app entry point
├── pipeline.py               # End-to-end orchestration pipeline
├── captioning.py             # BLIP captioning module
├── frame_extractor.py        # OpenCV frame extraction
├── motion_analysis.py        # MediaPipe motion & pose analysis
├── object_detection.py       # YOLOv8 detection module
├── video_renderer.py         # Caption overlay and video rendering
├── pose_landmarker_lite.task # MediaPipe pose model file
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/mitali-gautam05/AI-VIDEO-CAPTION.git
cd AI-VIDEO-CAPTION
```

### 2. Create and activate a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

Open your browser at `http://127.0.0.1:7860`

---

## Deployment on Hugging Face Spaces

This project is ready for Hugging Face Spaces deployment out of the box.

**runtime.txt**
```
python-3.11
```

**requirements.txt**
```
numpy==1.26.4
opencv-python-headless
mediapipe==0.10.35
tensorflow==2.20.0
torch
torchvision
ultralytics
transformers
pillow
accelerate
sentencepiece
```

Push your code to a Hugging Face Space with SDK set to **Gradio** and it will auto-deploy.

---

## How It Works

**Step 1 — Frame Extraction**
OpenCV splits the uploaded video into individual frames at a defined sampling rate.

**Step 2 — Object Detection**
YOLOv8 runs inference on each frame, identifying objects, people, and key subjects with bounding boxes and confidence scores.

**Step 3 — Motion Analysis**
MediaPipe's Pose Landmarker maps body keypoints across frames and computes motion direction, speed, and activity patterns.

**Step 4 — Caption Generation**
BLIP generates a natural language caption for each sampled frame based on its visual content.

**Step 5 — Context Fusion**
Object labels, motion vectors, and frame captions are aggregated into a unified video-level description using rule-based and model-assisted summarization.

**Step 6 — Video Rendering**
The final caption is overlaid onto the output video using OpenCV, and the result is returned to the user via the Gradio interface.

---

## Roadmap

- [ ] Real-time video captioning
- [ ] Scene segmentation using SAM
- [ ] Action recognition with Video Transformers
- [ ] Multi-language caption generation
- [ ] Audio understanding integration
- [ ] GPU optimization for faster inference

---
