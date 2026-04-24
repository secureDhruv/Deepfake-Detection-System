```
```
██████╗ ███████╗███████╗██████╗ ███████╗ █████╗ ██╗  ██╗███████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗██║ ██╔╝██╔════╝
██║  ██║█████╗  █████╗  ██████╔╝█████╗  ███████║█████╔╝ █████╗  
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══╝  ██╔══██║██╔═██╗ ██╔══╝  
██████╔╝███████╗███████╗██║     ██║     ██║  ██║██║  ██╗███████╗
╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝

██████╗ ███████╗████████╗███████╗ ██████╗████████╗ ██████╗ ██████╗ 
██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝
██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██╗
╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
```

<div align="center">

> **Next-Generation AI Forensic Platform — Multi-model ensemble detection for the modern web.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?style=for-the-badge&logo=sqlite)](https://www.sqlite.org/)

</div>

---

## 🧠 Overview

**Deepfake Detector** is a high-performance, forensic-grade AI application designed to identify AI-generated (fake) face images with surgical precision. By leveraging a **Multi-Model Ensemble Architecture**, it cross-references predictions across three distinct neural networks—**MobileNetV2**, **ResNet50V2**, and **Xception**—to provide a highly reliable consensus verdict.

---

## ✨ Features

### 🛡️ Multi-Model Ensemble
- **Triple-Model Validation:** Uses MobileNetV2, ResNet50V2, and Xception backbones.
- **Weighted Consensus:** Returns an aggregated confidence score based on the strengths of each model.
- **Explainable Results:** Breakdown of how each individual model voted on the forensic analysis.

### 📊 Advanced Analytics & UI
- **Live Forensic Dashboard:** Real-time visualization of detection history with high-resolution previews.
- **Statistical Intelligence:** Aggregated data on scan distributions, fake-vs-real ratios, and model performance metrics.
- **Glassmorphism Design:** A modern, high-fidelity UI that respects system dark/light modes.
- **Detailed Forensic View:** Drills down into specific detections with individual model results.

### ⚡ Technical Excellence
- **Hybrid Inference:** High-speed processing (~150ms per ensemble pass).
- **SQLite Persistence:** Atomic storage of forensic records and metadata.
- **Sanitized Upload Pipeline:** Enterprise-grade file handling and path-traversal protection.

---

## 🏗️ System Architecture

```text
                        ┌─────────────────────────────────────────────────┐
                        │              MODERN WEB INTERFACE               │
                        │   (Scan · History · Analytics · Settings)       │
                        └────────┬───────────────────────▲────────────────┘
                                 │ POST /image           │ JSON / HTML
                                 ▼                       │
                        ┌────────────────────────────────┴────────────────┐
                        │               FLASK ENGINE (app.py)             │
                        │  ┌────────────┐  ┌─────────────┐  ┌──────────┐  │
                        │  │ Validator  │  │ Predictor   │  │ DB Layer │  │
                        │  └────────────┘  └──────┬──────┘  └─────┬────┘  │
                        └─────────────────────────┼───────────────┼───────┘
                                                  │               │
            ┌─────────────────────────────────────▼───────┐       │
            │           MULTI-MODEL ENSEMBLE              │       │
            │  ┌────────────┐┌────────────┐┌────────────┐ │       ▼
            │  │ MobileNetV2││ ResNet50V2 ││  Xception  │ │   ┌─────────┐
            │  └─────┬──────┘└─────┬──────┘└─────┬──────┘ │   │ SQLite  │
            │        │             │             │        │   │ (DB)    │
            │        └─────────────┼─────────────┘        │   └─────────┘
            │                      ▼                      │
            │            ENSMEMBLE CONSENSUS              │
            └─────────────────────────────────────────────┘
```

---

## 🔬 Deep Learning Pipeline

### 1. Training Phase (`train_ensemble.py`)
The system employs **Transfer Learning** from ImageNet-pretrained weights. Each model is fine-tuned on a custom dataset of 100,000+ real and fake face images.
- **MobileNetV2:** Lightweight, focuses on high-level spatial features.
- **ResNet50V2:** Uses skip-connections to retain fine-grained detail.
- **Xception:** Utilizes depthwise separable convolutions for superior accuracy.

### 2. Inference Phase (`predictor.py`)
1. **Face Detection:** OpenCV Haar Cascades isolate the facial region to reduce noise.
2. **Standardization:** Images are normalized to 224x224x3 and preprocessed per-model requirements.
3. **Parallel Inference:** All three models run a forward pass simultaneously.
4. **Aggregation:** The results are averaged to mitigate single-model biases.

---

## 📁 Project Structure

```bash
Deepfake Detector/
├── 📄 app.py                  # Main Flask application & routes
├── 📄 database.py             # SQLite database operations
├── 📄 train_ensemble.py       # Orchestrator for training the 3-model suite
├── 📄 train_model.py          # Legacy single-model trainer
├── 📄 requirements.txt        # Pinned project dependencies
├── 📄 history.db              # Persistence layer
│
└── 📂 dataset/
    ├── 📂 model/              # Trained weights (*.h5)
    │   ├── 🧠 deepfake_model.h5 (MobileNetV2)
    │   ├── 🧠 resnet_model.h5
    │   └── 🧠 xception_model.h5
    │
    ├── 📂 templates/          # Jinja2 modern UI components
    │   ├── 🌐 index.html      # Unified Dashboard
    │   ├── 🌐 analytics.html  # Stats & Visualization
    │   └── 🌐 analysis.html   # Detailed Record View
    │
    └── 📂 utils/
        ├── 🐍 image_processing.py  # Computer Vision utilities
        └── 🐍 predictor.py         # Ensemble inference engine
```

---

## 🚀 Installation & Deployment

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/secureDhruv/Deepfake-Detection-System.git
cd Deepfake-Detection-System

# Setup Virtual Environment
python -m venv venv
# Win: venv\Scripts\activate | Unix: source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Model Initialization
If you are starting fresh, you must train the ensemble to generate the model files:
```bash
python train_ensemble.py
```
*Note: This will train MobileNetV2, ResNet50V2, and Xception in sequence.*

### 3. Launch the Platform
```bash
python app.py
```
└─────────────────────────────────────┘
```

### Dashboard — Detection History

```
┌────────────────────────────────────────────────────────────────────┐
│  DeepGuard AI | History                          [ New Analysis ]  │
├────────────────────────────────────────────────────────────────────┤
│  Recent Detections                         Total Records: 12       │
├──────┬──────────┬──────────────┬────────┬────────────┬────────────┤
│  ID  │ Preview  │  Filename    │ Result │ Confidence │ Timestamp  │
├──────┼──────────┼──────────────┼────────┼────────────┼────────────┤
│  #12 │  [img]   │ photo01.jpg  │  REAL  │ ████░  91% │ 2026-04-16 │
│  #11 │  [img]   │ face02.png   │  FAKE  │ ███░░  78% │ 2026-04-15 │
│  #10 │  [img]   │ test.webp    │  FAKE  │ █████  96% │ 2026-04-15 │
│  ... │          │              │        │            │            │
└──────┴──────────┴──────────────┴────────┴────────────┴────────────┘
```

---

## 🔒 Security Features

```
  Upload Request
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  1. File field present?        ✔ / ✖   │
  │  2. Filename not empty?        ✔ / ✖   │
  │  3. Extension in allowlist?    ✔ / ✖   │  allowlist: png, jpg, jpeg, webp
  │     (server-side, not just JS)          │
  │  4. secure_filename() applied  ✔       │  prevents path traversal
  │  5. MAX_CONTENT_LENGTH = 10 MB ✔       │  blocks oversized payloads
  └─────────────────────────────────────────┘
       │ all checks pass
       ▼
  File saved to uploads/
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `change-me-before-deploying` | Flask session secret. Set via env var in production. |
| `MAX_CONTENT_LENGTH` | `10 MB` | Maximum upload file size |
| `ALLOWED_EXTENSIONS` | `png, jpg, jpeg, webp` | Accepted image formats |
| `MODEL_PATH` | `dataset/model/deepfake_model.h5` | Path to the trained Keras model |
| `DB_PATH` | `history.db` | SQLite database path |

---

## 🗄️ Database Schema

```sql
CREATE TABLE detections (
    id         INTEGER  PRIMARY KEY AUTOINCREMENT,
    filename   TEXT,                              -- sanitized upload filename
    result     TEXT,                              -- "Real Image" | "Fake Image"
    confidence REAL,                              -- 0.0 – 100.0
    date       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🧪 How the Model Works

```
  INPUT IMAGE (any size)
         │
         ▼
  ┌──────────────────┐
  │  Face Detection  │   OpenCV Haar Cascade — crops face region
  │  (or full image  │   Falls back to full image if no face found
  │   as fallback)   │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Resize 224×224  │   Target size for MobileNetV2
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ preprocess_input │   Scales pixels from [0,255] → [-1, 1]
  │  (mobilenet_v2)  │   Must match training preprocessing exactly
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │   MobileNetV2    │   1280 feature maps from frozen ImageNet backbone
  │  feature vector  │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │  Dense(128,relu) │   Learns deepfake-specific patterns
  │  + Dropout(0.3)  │
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Dense(1, sigmoid)│   Output: P(real)  ∈ [0.0, 1.0]
  └────────┬─────────┘
           │
    ┌──────┴──────┐
    │             │
  > 0.5         ≤ 0.5
    │             │
  "Real"        "Fake"
  conf=p×100   conf=(1-p)×100
```

---

## 🐛 Known Limitations

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚠  Haar cascade face detector may miss non-frontal faces.      │
│     → Falls back to full image (slightly lower accuracy).       │
│                                                                 │
│  ⚠  Model accuracy depends heavily on training dataset size.    │
│     → Use at least 1000+ images per class for reliable results. │
│                                                                 │
│  ⚠  Not designed for video analysis (image-only pipeline).      │
│                                                                 │
│  ⚠  May struggle with heavily compressed or very low-res images.│
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔭 Roadmap

- [ ] **Video support** — frame-by-frame deepfake detection
- [ ] **MTCNN face detector** — replace Haar cascade for better accuracy
- [ ] **REST API endpoint** — `POST /api/detect` returning JSON
- [ ] **Batch upload** — detect multiple images at once
- [ ] **Grad-CAM heatmap** — visualise which regions the model focuses on
- [ ] **User authentication** — private detection history per user
- [ ] **Docker support** — one-command deployment

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Howard et al., Google Brain
- [OpenCV](https://opencv.org/) — Open Source Computer Vision Library
- [TensorFlow / Keras](https://www.tensorflow.org/) — ML framework by Google
- [Flask](https://flask.palletsprojects.com/) — Lightweight Python web framework
- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Deepfake dataset inspiration

---

<div align="center">

```
Built with ❤️  using Python, TensorFlow & Flask
```

</div>