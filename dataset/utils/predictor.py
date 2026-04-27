import os
import threading

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

from .image_processing import extract_face

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "model"))

ENSEMBLE_CONFIG = {
    "MobileNetV2": {
        "path": os.path.join(MODEL_DIR, "deepfake_model.h5"),
        "preprocess": "mobilenet",
        "target_size": (224, 224),
    },
    "ResNet50V2": {
        "path": os.path.join(MODEL_DIR, "resnet_model.h5"),
        "preprocess": "resnet",
        "target_size": (224, 224),
    },
    "Xception": {
        "path": os.path.join(MODEL_DIR, "xception_model.h5"),
        "preprocess": "xception",
        "target_size": (224, 224),
    },
}

_models_lock = threading.Lock()
_loaded_models: dict[str, dict] = {}
_load_errors: dict[str, str] = {}
_preprocessors = None


def get_loaded_model_names() -> list[str]:
    """
    Return model names available to the app.

    Model files are loaded lazily on the first prediction so normal page loads do
    not pay the TensorFlow startup cost.
    """
    if _loaded_models:
        return list(_loaded_models.keys())
    return [name for name, cfg in ENSEMBLE_CONFIG.items() if os.path.exists(cfg["path"])]


def _get_preprocessors() -> dict[str, object]:
    global _preprocessors
    if _preprocessors is None:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_prep
        from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_prep
        from tensorflow.keras.applications.xception import preprocess_input as xception_prep

        _preprocessors = {
            "mobilenet": mobilenet_prep,
            "resnet": resnet_prep,
            "xception": xception_prep,
        }
    return _preprocessors


def _load_available_models() -> dict[str, dict]:
    if _loaded_models:
        return _loaded_models

    with _models_lock:
        if _loaded_models:
            return _loaded_models

        try:
            from tensorflow.keras.models import load_model
        except ImportError as exc:
            raise ValueError("TensorFlow is not installed. Install requirements.txt before prediction.") from exc

        preprocessors = _get_preprocessors()
        for name, config in ENSEMBLE_CONFIG.items():
            model_path = config["path"]
            if not os.path.exists(model_path):
                _load_errors[name] = f"Missing model file: {model_path}"
                continue

            try:
                print(f"[Predictor] Loading {name} from {model_path}...")
                _loaded_models[name] = {
                    "model": load_model(model_path, compile=False),
                    "preprocess": preprocessors[config["preprocess"]],
                    "target_size": config["target_size"],
                }
            except Exception as exc:
                _load_errors[name] = str(exc)
                print(f"[Predictor] Failed to load {name}: {exc}")

        return _loaded_models


def _model_target_size(model, default_size: tuple[int, int]) -> tuple[int, int]:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if input_shape and len(input_shape) >= 4 and input_shape[1] and input_shape[2]:
        return int(input_shape[2]), int(input_shape[1])
    return default_size


def _extract_real_probability(raw_prediction) -> float:
    arr = np.asarray(raw_prediction, dtype="float32")
    if arr.size == 0:
        raise ValueError("Model returned an empty prediction.")

    arr = np.squeeze(arr)
    if arr.ndim == 0:
        p_real = float(arr)
    elif arr.shape[-1] == 1:
        p_real = float(arr.reshape(-1)[0])
    elif arr.shape[-1] == 2:
        # For a two-unit softmax classifier, flow_from_directory maps
        # alphabetically, so class 1 is "real" when classes are fake/real.
        p_real = float(arr.reshape(-1, 2)[0, 1])
    else:
        raise ValueError(f"Unsupported prediction shape: {np.asarray(raw_prediction).shape}")

    if not np.isfinite(p_real):
        raise ValueError("Model returned a non-finite probability.")
    return float(np.clip(p_real, 0.0, 1.0))


def predict_image(image_path: str) -> tuple[str, float, dict]:
    """
    Run deepfake detection on an image using the available ensemble models.

    Returns:
        (label, confidence_pct, details_dict)
    """
    loaded_models = _load_available_models()
    if not loaded_models:
        error_summary = "; ".join(f"{name}: {msg}" for name, msg in _load_errors.items())
        raise ValueError(f"No models available for prediction. {error_summary}".strip())

    face = extract_face(image_path)
    if face.ndim == 2:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    elif face.ndim == 3 and face.shape[2] == 4:
        face = face[:, :, :3]
    elif face.ndim != 3 or face.shape[2] != 3:
        raise ValueError(f"Unsupported image array shape: {face.shape}")

    details = {}
    total_p_real = 0.0

    for name, model_config in loaded_models.items():
        model = model_config["model"]
        target_size = _model_target_size(model, model_config["target_size"])
        img = cv2.resize(face, target_size).astype("float32")
        img = model_config["preprocess"](img)
        img = np.expand_dims(img, axis=0)

        raw = model.predict(img, verbose=0)
        p_real = _extract_real_probability(raw)
        total_p_real += p_real

        if p_real > 0.5:
            details[name] = {"label": "Real Image", "confidence": round(p_real * 100, 1)}
        else:
            details[name] = {"label": "Fake Image", "confidence": round((1.0 - p_real) * 100, 1)}

    avg_p_real = total_p_real / len(loaded_models)
    if avg_p_real > 0.5:
        label = "Real Image"
        confidence = round(avg_p_real * 100, 1)
    else:
        label = "Fake Image"
        confidence = round((1.0 - avg_p_real) * 100, 1)

    return label, confidence, details
