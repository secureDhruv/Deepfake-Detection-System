import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_prep
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_prep
from tensorflow.keras.applications.xception import preprocess_input as xception_prep
from .image_processing import extract_face

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "model"))

# Define our ensemble architecture mappings
ENSEMBLE_CONFIG = {
    "MobileNetV2": {
        "path": os.path.join(MODEL_DIR, "deepfake_model.h5"),
        "preprocess": mobilenet_prep
    },
    "ResNet50V2": {
        "path": os.path.join(MODEL_DIR, "resnet_model.h5"),
        "preprocess": resnet_prep
    },
    "Xception": {
        "path": os.path.join(MODEL_DIR, "xception_model.h5"),
        "preprocess": xception_prep
    }
}

loaded_models = {}
for name, config in ENSEMBLE_CONFIG.items():
    if os.path.exists(config["path"]):
        try:
            print(f"[Predictor] Loading {name}...")
            loaded_models[name] = {
                "model": load_model(config["path"]),
                "preprocess": config["preprocess"]
            }
        except Exception as e:
            print(f"[Predictor] Failed to load {name}: {e}")

def get_loaded_model_names() -> list:
    """Return list of model names that loaded successfully."""
    return list(loaded_models.keys())


def predict_image(image_path: str) -> tuple:
    """
    Run deepfake detection on the image using an ensemble of available models.

    Returns:
        (label, confidence_pct, details_dict)
          label          – "Real Image" or "Fake Image" (Soft Voting Aggregated)
          confidence_pct – float 0-100 indicating ensemble certainty
          details_dict   - dictionary specifying per-model confidence
    """
    face = extract_face(image_path)

    if face.ndim == 2:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    elif face.shape[2] == 4:
        face = face[:, :, :3]

    img_base = cv2.resize(face, (224, 224)).astype("float32")
    
    details = {}
    total_p_real = 0.0
    valid_models = 0

    for name, m_config in loaded_models.items():
        # Preprocess specific to the model
        img = m_config["preprocess"](img_base.copy())
        img = np.expand_dims(img, axis=0)
        
        # Predict
        raw = m_config["model"].predict(img, verbose=0)
        p_real = float(raw[0][0])
        
        total_p_real += p_real
        valid_models += 1
        
        if p_real > 0.5:
            details[name] = {"label": "Real Image", "confidence": round(p_real * 100, 1)}
        else:
            details[name] = {"label": "Fake Image", "confidence": round((1.0 - p_real) * 100, 1)}

    if valid_models == 0:
        raise ValueError("No models available for prediction. Please train them first.")

    # Ensemble Soft Voting
    avg_p_real = total_p_real / valid_models
    
    if avg_p_real > 0.5:
        label = "Real Image"
        confidence = round(avg_p_real * 100, 1)
    else:
        label = "Fake Image"
        confidence = round((1.0 - avg_p_real) * 100, 1)

    return label, confidence, details