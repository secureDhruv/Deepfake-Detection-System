import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(min(4, os.cpu_count() or 1)))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import cv2
import numpy as np

from .image_processing import extract_face

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (BASE_DIR / ".." / "model").resolve()
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

# The current MobileNet model is weakly calibrated: validation sampling showed
# useful separation near 0.026-0.030, while the old 0.002 threshold over-called
# real images. Full calibration can override this via model_metadata.json.
DEFAULT_DECISION_THRESHOLD = float(os.environ.get("DEEPFAKE_THRESHOLD", "0.03"))
MAX_CONFIDENCE = float(os.environ.get("DEEPFAKE_MAX_CONFIDENCE", "85.0"))


ENSEMBLE_CONFIG: dict[str, dict[str, Any]] = {
    "MobileNetV2": {
        "path": MODEL_DIR / "deepfake_model.h5",
        "preprocess": "mobilenet",
        "target_size": (224, 224),
        "weight": 1.0,
        "threshold": DEFAULT_DECISION_THRESHOLD,
        "enabled": True,
    },
    # Xception is large and was documented as near-random in the current code.
    # Keep it opt-in for CPU inference until evaluate_models.py proves it helps.
    "Xception": {
        "path": MODEL_DIR / "xception_model.h5",
        "preprocess": "xception",
        "target_size": (224, 224),
        "weight": 0.0,
        "threshold": DEFAULT_DECISION_THRESHOLD,
        "enabled": os.environ.get("DEEPFAKE_ENABLE_XCEPTION") == "1",
    },
}

_models_lock = threading.Lock()
_loaded_models: dict[str, dict[str, Any]] = {}
_load_errors: dict[str, str] = {}
_preprocessors: dict[str, Any] | None = None
_metadata_cache: dict[str, Any] | None = None


def get_loaded_model_names() -> list[str]:
    """
    Return model names available to the app.

    Models are loaded lazily on the first prediction so normal page loads avoid
    the TensorFlow startup cost.
    """
    if _loaded_models:
        return list(_loaded_models.keys())

    metadata = _load_model_metadata()
    names: list[str] = []
    for name, config in ENSEMBLE_CONFIG.items():
        resolved = _resolve_model_config(name, config, metadata)
        if resolved["enabled"] and Path(resolved["path"]).exists():
            names.append(name)
    return names


def _load_model_metadata() -> dict[str, Any]:
    global _metadata_cache
    if _metadata_cache is not None:
        return _metadata_cache

    if not MODEL_METADATA_PATH.exists():
        _metadata_cache = {}
        return _metadata_cache

    try:
        with MODEL_METADATA_PATH.open("r", encoding="utf-8") as handle:
            _metadata_cache = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read model metadata %s: %s", MODEL_METADATA_PATH, exc)
        _metadata_cache = {}

    return _metadata_cache


def _resolve_model_config(
    name: str,
    base_config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(base_config)
    model_meta = metadata.get("models", {}).get(name, {})

    for key in ("threshold", "weight", "enabled"):
        if key in model_meta:
            resolved[key] = model_meta[key]

    if name == "Xception" and os.environ.get("DEEPFAKE_ENABLE_XCEPTION") == "1":
        resolved["enabled"] = True
        resolved["weight"] = max(float(resolved.get("weight", 0.0)), 0.1)

    resolved["threshold"] = float(resolved.get("threshold", DEFAULT_DECISION_THRESHOLD))
    resolved["weight"] = float(resolved.get("weight", 1.0))
    resolved["enabled"] = bool(resolved.get("enabled", True))
    return resolved


def _get_preprocessors() -> dict[str, Any]:
    global _preprocessors
    if _preprocessors is None:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_prep
        from tensorflow.keras.applications.xception import preprocess_input as xception_prep

        _preprocessors = {
            "mobilenet": mobilenet_prep,
            "xception": xception_prep,
        }
    return _preprocessors


def _configure_tensorflow_runtime():
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        # TensorFlow may already be initialized; CUDA_VISIBLE_DEVICES still keeps
        # native Windows inference on CPU for this project.
        pass

    try:
        intra_threads = int(os.environ.get("TF_NUM_INTRAOP_THREADS", "4"))
        inter_threads = int(os.environ.get("TF_NUM_INTEROP_THREADS", "1"))
        tf.config.threading.set_intra_op_parallelism_threads(max(1, intra_threads))
        tf.config.threading.set_inter_op_parallelism_threads(max(1, inter_threads))
    except Exception:
        pass

    return tf


def _load_available_models() -> dict[str, dict[str, Any]]:
    if _loaded_models:
        return _loaded_models

    with _models_lock:
        if _loaded_models:
            return _loaded_models

        try:
            _configure_tensorflow_runtime()
            from tensorflow.keras.models import load_model
        except ImportError as exc:
            raise ValueError("TensorFlow is not installed. Install requirements.txt before prediction.") from exc

        metadata = _load_model_metadata()
        preprocessors = _get_preprocessors()

        for name, config in ENSEMBLE_CONFIG.items():
            resolved = _resolve_model_config(name, config, metadata)
            model_path = Path(resolved["path"])

            if not resolved["enabled"] or resolved["weight"] <= 0:
                logger.info("Skipping disabled model: %s", name)
                continue
            if not model_path.exists():
                _load_errors[name] = f"Missing model file: {model_path}"
                continue

            try:
                start = time.perf_counter()
                logger.info("Loading model %s from %s", name, model_path)
                model = load_model(model_path, compile=False)
                target_size = _model_target_size(model, resolved["target_size"])
                _loaded_models[name] = {
                    "model": model,
                    "preprocess": preprocessors[resolved["preprocess"]],
                    "target_size": target_size,
                    "weight": resolved["weight"],
                    "threshold": resolved["threshold"],
                    "path": str(model_path),
                }
                logger.info(
                    "Loaded %s in %.2fs with input_size=%s threshold=%.6f weight=%.3f",
                    name,
                    time.perf_counter() - start,
                    target_size,
                    resolved["threshold"],
                    resolved["weight"],
                )
            except Exception as exc:
                _load_errors[name] = str(exc)
                logger.exception("Failed to load model %s", name)

        return _loaded_models


def _model_target_size(model: Any, default_size: tuple[int, int]) -> tuple[int, int]:
    """Return (width, height) for cv2.resize from model input_shape."""
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if input_shape and len(input_shape) >= 4 and input_shape[1] and input_shape[2]:
        height = int(input_shape[1])
        width = int(input_shape[2])
        return (width, height)
    return default_size


def _extract_real_probability(raw_prediction: Any) -> float:
    arr = np.asarray(raw_prediction, dtype="float32")
    if arr.size == 0:
        raise ValueError("Model returned an empty prediction.")

    arr = np.squeeze(arr)
    if arr.ndim == 0:
        p_real = float(arr)
    elif arr.shape[-1] == 1:
        p_real = float(arr.reshape(-1)[0])
    elif arr.shape[-1] == 2:
        # flow_from_directory maps alphabetically, so class 1 is "real" for
        # the expected fake/real directory layout.
        p_real = float(arr.reshape(-1, 2)[0, 1])
    else:
        raise ValueError(f"Unsupported prediction shape: {np.asarray(raw_prediction).shape}")

    if not np.isfinite(p_real):
        raise ValueError("Model returned a non-finite probability.")
    return float(np.clip(p_real, 0.0, 1.0))


def _calibrate_confidence(p_real: float, threshold: float, is_real: bool) -> float:
    """
    Map distance from the decision threshold to a conservative confidence score.

    The current model is poorly calibrated, so raw sigmoid values are not shown
    directly. At the threshold, confidence is 50%. Farther away approaches the
    configured cap instead of pretending the model is certain.
    """
    threshold = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    max_confidence = float(
        _load_model_metadata().get("ensemble", {}).get("max_confidence", MAX_CONFIDENCE)
    )
    if is_real:
        ratio = (p_real - threshold) / (1.0 - threshold)
    else:
        ratio = (threshold - p_real) / threshold

    ratio = float(np.clip(ratio, 0.0, 1.0))
    score = 50.0 + ratio * (max_confidence - 50.0)
    return round(min(score, max_confidence), 1)


def _prepare_face_for_model(face_rgb: np.ndarray, target_size: tuple[int, int], preprocess) -> np.ndarray:
    if face_rgb.ndim == 2:
        face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_GRAY2RGB)
    elif face_rgb.ndim == 3 and face_rgb.shape[2] == 4:
        face_rgb = face_rgb[:, :, :3]
    elif face_rgb.ndim != 3 or face_rgb.shape[2] != 3:
        raise ValueError(f"Unsupported image array shape: {face_rgb.shape}")

    img = cv2.resize(face_rgb, target_size, interpolation=cv2.INTER_AREA).astype("float32")
    img = preprocess(img)
    return np.expand_dims(img, axis=0)


def _ensemble_threshold(models: dict[str, dict[str, Any]]) -> float:
    ensemble_meta = _load_model_metadata().get("ensemble", {})
    if "threshold" in ensemble_meta:
        return float(ensemble_meta["threshold"])

    total_weight = sum(float(cfg.get("weight", 1.0)) for cfg in models.values())
    if total_weight <= 0:
        return DEFAULT_DECISION_THRESHOLD
    return sum(
        float(cfg.get("threshold", DEFAULT_DECISION_THRESHOLD)) * float(cfg.get("weight", 1.0))
        for cfg in models.values()
    ) / total_weight


def predict_image(image_path: str) -> tuple[str, float, dict[str, dict[str, Any]]]:
    """
    Run deepfake detection on one image.

    Returns:
        (label, confidence_pct, details_dict)
    """
    loaded_models = _load_available_models()
    if not loaded_models:
        error_summary = "; ".join(f"{name}: {msg}" for name, msg in _load_errors.items())
        raise ValueError(f"No models available for prediction. {error_summary}".strip())

    start = time.perf_counter()
    face_rgb, face_metadata = extract_face(image_path, return_metadata=True)

    details: dict[str, dict[str, Any]] = {}
    weighted_p_real = 0.0
    total_weight = 0.0

    for name, model_config in loaded_models.items():
        model = model_config["model"]
        weight = float(model_config.get("weight", 1.0))
        threshold = float(model_config.get("threshold", DEFAULT_DECISION_THRESHOLD))
        target_size = model_config["target_size"]
        batch = _prepare_face_for_model(face_rgb, target_size, model_config["preprocess"])

        raw = model(batch, training=False).numpy()
        p_real = _extract_real_probability(raw)

        weighted_p_real += p_real * weight
        total_weight += weight

        is_real_model = p_real > threshold
        confidence = _calibrate_confidence(p_real, threshold, is_real_model)
        details[name] = {
            "label": "Real Image" if is_real_model else "Fake Image",
            "confidence": confidence,
            "p_real": round(p_real, 6),
            "threshold": round(threshold, 6),
            "margin": round(p_real - threshold, 6),
            "weight": round(weight, 4),
            "face_detected": face_metadata.face_detected,
            "crop_box": list(face_metadata.crop_box) if face_metadata.crop_box else None,
            "detector": face_metadata.detector,
        }

        logger.info(
            "%s p_real=%.6f threshold=%.6f label=%s confidence=%.1f weight=%.3f",
            name,
            p_real,
            threshold,
            details[name]["label"],
            confidence,
            weight,
        )

    if total_weight <= 0:
        raise ValueError("All loaded model weights are zero; cannot produce a prediction.")

    avg_p_real = weighted_p_real / total_weight
    threshold = _ensemble_threshold(loaded_models)
    is_real_ensemble = avg_p_real > threshold
    label = "Real Image" if is_real_ensemble else "Fake Image"
    confidence = _calibrate_confidence(avg_p_real, threshold, is_real_ensemble)

    logger.info(
        "Prediction complete label=%s confidence=%.1f p_real=%.6f threshold=%.6f elapsed=%.2fs",
        label,
        confidence,
        avg_p_real,
        threshold,
        time.perf_counter() - start,
    )

    return label, confidence, details
