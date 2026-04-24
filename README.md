
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
██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
```

<div align="center">

> **AI-powered deepfake detection in your browser — upload, analyse, know the truth.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=flat-square&logo=opencv)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?style=flat-square&logo=sqlite)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

</div>

---

## 🧠 What Is This?

**Deepfake Detector** is a full-stack AI application that uses a fine-tuned **MobileNetV2** convolutional neural network to determine whether a face image is **real** or **AI-generated (fake)**. Upload an image through the web interface and get an instant verdict — complete with a confidence score and a persistent history dashboard.

---

## ✨ Features

```
┌─────────────────────────────────────────────────────────────┐
│                     FEATURE OVERVIEW                        │
├───────────────────────┬─────────────────────────────────────┤
│  🔍 Smart Detection   │  MobileNetV2 transfer learning CNN  │
│  🖼️  Image Preview    │  Drag-and-drop with live preview     │
│  📊 History Dashboard │  SQLite-backed detection log        │
│  🎯 Confidence Score  │  0–100% certainty per prediction    │
│  🌙 Dark Mode         │  Respects system color scheme       │
│  🔒 Secure Uploads    │  Extension validation + sanitization│
│  📱 Responsive UI     │  Works on desktop & mobile          │
│  ⚡ Fast Inference    │  ~100ms prediction on CPU           │
└───────────────────────┴─────────────────────────────────────┘
```

---

## 🏗️ System Architecture

```
                        ┌──────────────────────────────────────┐
                        │           USER'S BROWSER             │
                        │                                      │
                        │  ┌────────────────────────────────┐  │
                        │  │        index.html              │  │
                        │  │  ┌──────────┐  ┌───────────┐  │  │
                        │  │  │ Upload   │  │  Result   │  │  │
                        │  │  │  Form    │  │   Box     │  │  │
                        │  │  └────┬─────┘  └─────▲─────┘  │  │
                        │  │       │   script.js   │        │  │
                        │  │       │  (drag,preview│valid.) │  │
                        │  └───────┼───────────────┼────────┘  │
                        └──────────┼───────────────┼───────────┘
                                   │ POST /         │ HTML
                                   ▼               │
                        ┌──────────────────────────────────────┐
                        │           FLASK SERVER               │
                        │                                      │
                        │  app.py                              │
                        │  ┌────────────────────────────────┐  │
                        │  │  @route("/")                   │  │
                        │  │  1. Validate file (ext, size)  │  │
                        │  │  2. secure_filename()          │  │
                        │  │  3. Save to uploads/           │  │
                        │  │  4. Call predict_image()  ─────┼──┼──┐
                        │  │  5. save_detection()           │  │  │
                        │  │  6. render_template()          │  │  │
                        │  └────────────────────────────────┘  │  │
                        │                                      │  │
                        │  ┌────────────────────────────────┐  │  │
                        │  │  @route("/dashboard")          │  │  │
                        │  │  get_all_detections() ─────────┼──┼──┼──┐
                        │  └────────────────────────────────┘  │  │  │
                        └──────────────────────────────────────┘  │  │
                                                                   │  │
                   ┌───────────────────────────────────────────────┘  │
                   │                                                   │
                   ▼                                                   ▼
        ┌──────────────────────┐                         ┌────────────────────┐
        │   ML PIPELINE        │                         │     DATABASE       │
        │                      │                         │                    │
        │  image_processing.py │                         │  history.db        │
        │  ┌──────────────┐    │                         │  ┌──────────────┐  │
        │  │ cv2.imread() │    │                         │  │  detections  │  │
        │  │ BGR → RGB    │    │                         │  │  ─────────── │  │
        │  │ Face Detect  │    │                         │  │  id          │  │
        │  │ (Haar casc.) │    │                         │  │  filename    │  │
        │  └──────┬───────┘    │                         │  │  result      │  │
        │         │            │                         │  │  confidence  │  │
        │  predictor.py        │                         │  │  date        │  │
        │  ┌──────▼───────┐    │                         │  └──────────────┘  │
        │  │preprocess_   │    │                         └────────────────────┘
        │  │  input()     │    │
        │  │ model.predict│    │
        │  │ label+conf   │    │
        │  └──────────────┘    │
        │                      │
        │  dataset/model/      │
        │  deepfake_model.h5   │
        └──────────────────────┘
```

---

## 🤖 ML Pipeline

```
  TRAINING PHASE                              INFERENCE PHASE
  ─────────────                              ───────────────

  dataset/train/                             User uploads image
  ├── real/  (real faces)                            │
  └── fake/  (AI-generated faces)                    ▼
        │                                   ┌─────────────────┐
        ▼                                   │  cv2.imread()   │
  ImageDataGenerator                        │  BGR → RGB      │
  ┌─────────────────────┐                   └────────┬────────┘
  │ preprocess_input()  │                            │
  │ rotation_range=15°  │                            ▼
  │ horizontal_flip     │                   ┌─────────────────┐
  │ zoom_range=0.1      │                   │  Haar Cascade   │
  └────────┬────────────┘                   │  Face Detection │
           │                                └────────┬────────┘
           ▼                                         │
  ┌─────────────────────┐                            ▼
  │    MobileNetV2      │                   ┌─────────────────┐
  │  (ImageNet weights) │                   │ Resize 224×224  │
  │  trainable=False    │                   │ preprocess_     │
  └────────┬────────────┘                   │   input()       │
           │                                └────────┬────────┘
           ▼                                         │
  GlobalAveragePooling2D                             ▼
           │                                ┌─────────────────┐
           ▼                                │  MobileNetV2    │──── frozen
  Dense(128, relu)                          │  feature extrac │     weights
           │                                └────────┬────────┘
           ▼                                         │
      Dropout(0.3)                                   ▼
           │                                ┌─────────────────┐
           ▼                                │  Dense(1)       │
  Dense(1, sigmoid)                         │  sigmoid output │
           │                                └────────┬────────┘
           ▼                                         │
  Binary Cross-Entropy                               ▼
  Adam Optimizer                             p_real = output[0]
           │                                         │
           ▼                               ┌─────────┴──────────┐
  ModelCheckpoint                          │                    │
  EarlyStopping                      p > 0.5              p ≤ 0.5
           │                          "Real Image"        "Fake Image"
           ▼                          conf = p*100      conf=(1-p)*100
  deepfake_model.h5
```

---

## 📁 Project Structure

```
Deepfake Detector/
│
├── 📄 app.py                    ← Flask entry point & routes
├── 📄 database.py               ← SQLite helpers (init, save, fetch)
├── 📄 train_model.py            ← Model training script
├── 📄 requirements.txt          ← Pinned dependencies
├── 📄 history.db                ← Auto-created SQLite database
├── 📄 README.md                 ← You are here!
│
└── 📂 dataset/
    │
    ├── 📂 model/
    │   └── 🧠 deepfake_model.h5     ← Trained Keras model (~11 MB)
    │
    ├── 📂 train/                    ← Training images
    │   ├── 📂 fake/                 ← AI-generated face images
    │   └── 📂 real/                 ← Authentic face images
    │
    ├── 📂 validation/               ← Validation images
    │   ├── 📂 fake/
    │   └── 📂 real/
    │
    ├── 📂 templates/
    │   ├── 🌐 index.html            ← Upload page
    │   └── 🌐 dashboard.html        ← Detection history
    │
    ├── 📂 static/
    │   ├── 📂 css/
    │   │   └── 🎨 style.css         ← Styles + dark mode
    │   ├── 📂 js/
    │   │   └── ⚡ script.js          ← Drag-drop, preview, validation
    │   └── 📂 uploads/              ← Saved user-uploaded images
    │
    └── 📂 utils/
        ├── 🐍 image_processing.py   ← Face extraction (Haar cascade)
        └── 🐍 predictor.py          ← Model load & inference
```

---

## 🛠️ Tech Stack

```
┌──────────────────────────────────────────────────────────────┐
│                        TECH STACK                            │
├────────────────┬─────────────────────────────────────────────┤
│  Layer         │  Technology                                  │
├────────────────┼─────────────────────────────────────────────┤
│  Deep Learning │  TensorFlow 2.12+ / Keras                   │
│  Base Model    │  MobileNetV2 (ImageNet pretrained)          │
│  Face Detect   │  OpenCV Haar Cascade                        │
│  Web Framework │  Flask 2.3+                                 │
│  Database      │  SQLite 3 (via Python sqlite3)              │
│  Image I/O     │  OpenCV, Pillow, NumPy                      │
│  Frontend      │  Vanilla HTML5 + CSS3 + JavaScript ES6      │
│  Dashboard CSS │  Tailwind CSS (CDN)                         │
└────────────────┴─────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1 · Prerequisites

```
Python 3.10 or higher  ──►  https://www.python.org/downloads/
Git                    ──►  https://git-scm.com/
```

### 2 · Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 · Install Dependencies

```bash
pip install -r requirements.txt
```

### 4 · Download the Model & Dataset from Google Drive

Because the model weights and training datasets are extremely large, they are not hosted on GitHub. You should download the **assets zip file** directly from Google Drive.

1. **Download the Zip**
   * [Click here to download `deepfake_assets.zip`](YOUR_GOOGLE_DRIVE_PUBLIC_LINK_HERE)

2. **Extract the Assets**  
   Extract the contents of the zip file into the `dataset/` directory. Once extracted, your structure MUST look like this:

```text
Deepfake Detector/
└── dataset/
    ├── model/
    │   └── deepfake_model.h5     ← The 11 MB Keras model
    ├── train/
    │   ├── fake/                 ← AI-generated background images
    │   └── real/                 ← Authentic faces
    └── validation/
        ├── fake/
        └── real/
```

### 5 · Train the Model

```bash
python train_model.py
```

```
Expected output:
  Class indices: {'fake': 0, 'real': 1}
  Epoch 1/20 ────────────── loss: 0.6821 - accuracy: 0.5873 - val_accuracy: 0.6210
  Epoch 2/20 ────────────── loss: 0.5914 - accuracy: 0.6892 - val_accuracy: 0.7140
  ...
  Epoch 8/20: val_accuracy did not improve. Early stopping.
  Model saved to: dataset/model/deepfake_model.h5
```

> **Note:** Training uses `ModelCheckpoint` — only the best epoch is saved.
> `EarlyStopping` will halt training automatically when accuracy plateaus.

### 6 · Launch the Web App

```bash
python app.py
```

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

Open your browser and navigate to **`http://127.0.0.1:5000`** 🎉

---

## 🌐 Web Interface

### Home Page — Upload & Detect

```
┌─────────────────────────────────────┐
│          🔍 AI Deepfake Detector    │
│   Upload an image to check if it's  │
│       AI-generated or real.         │
│                                     │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │
│       🖼️  Click or drag here        │
│  │   PNG, JPG or WEBP · Max 10 MB │ │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│                                     │
│  ✓ my_photo.jpg                     │
│  [image preview thumbnail]          │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  ⟳  Analysing…             │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ ✅  Real Image  · 91.4%    │   │  ← green
│  └─────────────────────────────┘   │
│                                     │
│  📊 View Detection History          │
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