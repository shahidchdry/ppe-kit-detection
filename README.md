# 🛡️ SafeGuard AI: PPE Kit Detection & Live Violation Alert System

An end-to-end Computer Vision platform for detecting **Personal Protective Equipment (PPE)**, calculating compliance rates, and triggering live violation alerts on video feeds.

Designed with a **Google Colab GPU training pipeline** and an interactive, cyber-industrial **Web Application** with video playback, one-click download, and live violation alerts.

---

## 🌟 Key Features

- **Object Detection & Compliance Logic**:
  - Detects **Hardhats / Helmets**, **High-Visibility Safety Vests**, **Face Masks**, and **Persons**.
  - Recognizes negative violation classes (`NO-Hardhat`, `NO-Safety Vest`, `NO-Mask`).
  - Spatial heuristics matching equipment to workers (head & torso bounding box association).
- **Interactive Web Application**:
  - **Video Upload & Analysis**: Drag & drop surveillance or site video (MP4, WebM, AVI, MOV).
  - **Dual Video Player**: Switch between AI-annotated output and original footage.
  - **One-Click Video Download**: Download the detected video as `.mp4`.
  - **Ephemeral Processing**: Videos are processed temporarily in memory/temp storage and automatically cleaned up without disk buildup.
  - **Live Violation Feed & Alerts**: Real-time alert cards with hazard snapshots and worker IDs.
  - **Click-to-Seek**: Clicking any alert card jumps the video player immediately to that exact second!
  - **Audible Hazard Chime**: Synthesized audio warning tone (Web Audio API) when non-compliant workers appear.
  - **Compliance Audit Report**: Export CSV / JSON compliance logs.
  - **Live Webcam Safety Scanner**: Scan yourself or webcam feed in real-time.
- **Model Training on Google Colab (Free GPU)**:
  - Complete, turnkey Jupyter Notebook (`model/PPE_Kit_Detection_Training_Colab.ipynb`).
  - Automated dataset download (Roboflow Construction PPE dataset).
  - YOLOv8 training with data augmentations for safety gear.
  - Validation metrics (mAP@50, mAP@50-95, Confusion Matrix).
  - One-click browser download of trained `best_ppe_yolov8.pt`.

---

## 📂 Project Structure

```
ppe-kit-detection/
├── model/
│   ├── PPE_Kit_Detection_Training_Colab.ipynb  # Google Colab notebook for training
│   ├── train.py                                # Modular CLI training script
│   ├── data.yaml                               # YOLO dataset configuration
│   └── README.md                               # Colab training guide
├── inference/
│   └── ppe_detector.py                         # Core PPE compliance & tracking engine
├── website/
│   ├── index.html                              # Modern glassmorphism dashboard
│   ├── style.css                               # Cyber-safety visual design system
│   ├── app.js                                  # Interactive video player, Web Audio alerts
│   ├── server.py                               # Multi-threaded server with range streaming
│   └── static/                                 # Static assets & web resources
├── requirements.txt                            # Python dependencies
├── run_website.py                              # Python launcher (auto-opens browser)
├── run_website.bat                             # Windows 1-click batch launcher
└── README.md                                   # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Run the Website Locally
Run either the Python launcher or double-click `run_website.bat`:

```bash
python run_website.py
```
Or:
```bash
python website/server.py 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser:
- Upload any construction or surveillance video file.
- Review real-time compliance tracking, snapshots, and alerts.
- Download the detected video by clicking **"Download Detected Video (.mp4)"**.

---

### 2. Train Model in Google Colab (Free T4 GPU)
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload [model/PPE_Kit_Detection_Training_Colab.ipynb](file:///c:/Users/topsy/OneDrive/Documents/ppe-kit-detection/model/PPE_Kit_Detection_Training_Colab.ipynb).
3. Select **Runtime &rarr; Change runtime type &rarr; T4 GPU**.
4. Click **Runtime &rarr; Run all**.
5. Once training completes, the notebook will automatically trigger a browser download for `best_ppe_yolov8.pt`.
6. Drop `best_ppe_yolov8.pt` into the website's **"Upload Custom best.pt"** button to run inference with your trained model!

---

## 🛠️ Target PPE Classes

| Class ID | Class Name | Category | Role |
|---|---|---|---|
| 0 | `Hardhat` | PPE Item | 🟢 Positive Compliance |
| 1 | `Mask` | PPE Item | 🟢 Positive Compliance |
| 2 | `NO-Hardhat` | Violation | 🚨 Direct Safety Violation |
| 3 | `NO-Mask` | Violation | 🚨 Direct Safety Violation |
| 4 | `NO-Safety Vest`| Violation | 🚨 Direct Safety Violation |
| 5 | `Person` | Subject | Target for PPE check |
| 6 | `Safety Cone` | Site Object| Hazard boundary |
| 7 | `Safety Vest` | PPE Item | 🟢 Positive Compliance |
| 8 | `Machinery` | Equipment | Heavy equipment hazard |
| 9 | `Vehicle` | Vehicle | Site vehicle |