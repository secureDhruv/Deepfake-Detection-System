import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(min(4, os.cpu_count() or 1)))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import cv2
import numpy as np

from dataset.utils.image_processing import extract_face, load_image_rgb

BASE_DIR = Path(__file__).resolve().parent
VAL_DIR = BASE_DIR / "dataset" / "validation"
MODEL_DIR = BASE_DIR / "dataset" / "model"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_CLASS_NAMES = ["fake", "real"]

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "MobileNetV2": {
        "path": MODEL_DIR / "deepfake_model.h5",
        "preprocess": "mobilenet",
        "default_input_size": (224, 224),
    },
    "Xception": {
        "path": MODEL_DIR / "xception_model.h5",
        "preprocess": "xception",
        "default_input_size": (224, 224),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and calibrate deepfake detector models.")
    parser.add_argument("--validation-dir", type=Path, default=VAL_DIR)
    parser.add_argument("--metadata-out", type=Path, default=METADATA_PATH)
    parser.add_argument("--models", nargs="+", default=["MobileNetV2"], choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-per-class", type=int, default=0, help="0 means use every validation image.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-face-crop", action="store_true")
    parser.add_argument("--write-metadata", action="store_true")
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.56)
    return parser.parse_args()


def configure_tensorflow():
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            max(1, int(os.environ.get("TF_NUM_INTRAOP_THREADS", "4")))
        )
        tf.config.threading.set_inter_op_parallelism_threads(
            max(1, int(os.environ.get("TF_NUM_INTEROP_THREADS", "1")))
        )
    except Exception:
        pass

    return tf


def get_preprocess(name: str):
    if name == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        return preprocess_input
    if name == "xception":
        from tensorflow.keras.applications.xception import preprocess_input

        return preprocess_input
    raise ValueError(f"Unknown preprocessing function: {name}")


def list_validation_images(validation_dir: Path, sample_per_class: int, seed: int) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    samples: list[tuple[Path, int]] = []

    for label, class_name in enumerate(EXPECTED_CLASS_NAMES):
        class_dir = validation_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing validation class folder: {class_dir}")

        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No validation images found in: {class_dir}")

        if sample_per_class > 0:
            files = rng.sample(files, min(sample_per_class, len(files)))

        samples.extend((path, label) for path in files)

    rng.shuffle(samples)
    return samples


def model_target_size(model: Any, default_size: tuple[int, int]) -> tuple[int, int]:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape and len(input_shape) >= 4 and input_shape[1] and input_shape[2]:
        return int(input_shape[2]), int(input_shape[1])
    return default_size


def real_probability(raw_prediction: Any) -> float:
    arr = np.asarray(raw_prediction, dtype="float32")
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        value = float(arr)
    elif arr.shape[-1] == 1:
        value = float(arr.reshape(-1)[0])
    elif arr.shape[-1] == 2:
        value = float(arr.reshape(-1, 2)[0, 1])
    else:
        raise ValueError(f"Unsupported prediction shape: {np.asarray(raw_prediction).shape}")
    return float(np.clip(value, 0.0, 1.0))


def prepare_image(path: Path, target_size: tuple[int, int], preprocess, use_face_crop: bool) -> np.ndarray:
    if use_face_crop:
        img_rgb = extract_face(str(path))
    else:
        img_rgb = load_image_rgb(str(path))

    img = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA).astype("float32")
    return preprocess(img)


def predict_scores(
    model: Any,
    samples: list[tuple[Path, int]],
    target_size: tuple[int, int],
    preprocess,
    batch_size: int,
    use_face_crop: bool,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_score: list[float] = []
    batch: list[np.ndarray] = []
    batch_labels: list[int] = []

    for index, (path, label) in enumerate(samples, start=1):
        batch.append(prepare_image(path, target_size, preprocess, use_face_crop))
        batch_labels.append(label)

        if len(batch) == batch_size or index == len(samples):
            raw = model(np.asarray(batch, dtype="float32"), training=False).numpy()
            y_score.extend(real_probability(row) for row in raw)
            y_true.extend(batch_labels)
            batch.clear()
            batch_labels.clear()
            print(f"  processed {index}/{len(samples)}", flush=True)

    return np.asarray(y_true, dtype=np.int32), np.asarray(y_score, dtype=np.float32)


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score > threshold).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "threshold": round(float(threshold), 8),
        "accuracy": round(float(accuracy), 4),
        "precision_real": round(float(precision), 4),
        "recall_real": round(float(recall), 4),
        "specificity_fake": round(float(specificity), 4),
        "f1_real": round(float(f1), 4),
        "balanced_accuracy": round(float((recall + specificity) / 2), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positive = y_true == 1
    negative = y_true == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.arange(1, len(y_score) + 1, dtype=np.float64)

    unique_scores, inverse, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    rank_sums = np.bincount(inverse, weights=ranks)
    average_ranks = rank_sums / counts

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = average_ranks[inverse]
    auc = (original_ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return round(float(auc), 4)


def summarize_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, list[float]]:
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {
        "fake_p_real_percentiles": np.percentile(y_score[y_true == 0], percentiles).round(6).tolist(),
        "real_p_real_percentiles": np.percentile(y_score[y_true == 1], percentiles).round(6).tolist(),
    }


def find_best_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, dict[str, Any]]:
    thresholds = np.unique(np.concatenate(([0.0, 0.002, 0.5, 1.0], y_score)))
    all_metrics = [metrics_at_threshold(y_true, y_score, float(threshold)) for threshold in thresholds]
    best_balanced = max(all_metrics, key=lambda item: item["balanced_accuracy"])
    best_f1 = max(all_metrics, key=lambda item: item["f1_real"])
    return {
        "threshold_0_5": metrics_at_threshold(y_true, y_score, 0.5),
        "threshold_0_002": metrics_at_threshold(y_true, y_score, 0.002),
        "best_balanced": best_balanced,
        "best_f1": best_f1,
    }


def evaluate_one_model(model_name: str, samples: list[tuple[Path, int]], args: argparse.Namespace) -> dict[str, Any]:
    from tensorflow.keras.models import load_model

    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file for {model_name}: {model_path}")

    print(f"\n[MODEL] {model_name}")
    print(f"  path: {model_path}")
    start = time.perf_counter()
    model = load_model(model_path, compile=False)
    target_size = model_target_size(model, config["default_input_size"])
    preprocess = get_preprocess(config["preprocess"])
    print(f"  input_size: {target_size}")

    y_true, y_score = predict_scores(
        model=model,
        samples=samples,
        target_size=target_size,
        preprocess=preprocess,
        batch_size=max(1, args.batch_size),
        use_face_crop=args.use_face_crop,
    )

    thresholds = find_best_thresholds(y_true, y_score)
    result = {
        "model": model_name,
        "path": str(model_path),
        "sample_count": int(len(y_true)),
        "sample_per_class": args.sample_per_class,
        "use_face_crop": bool(args.use_face_crop),
        "input_size": list(target_size),
        "auc": roc_auc_score(y_true, y_score),
        "thresholds": thresholds,
        "score_summary": summarize_scores(y_true, y_score),
        "seconds": round(time.perf_counter() - start, 2),
    }

    print(json.dumps(result, indent=2))
    return result


def write_metadata(results: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    models: dict[str, Any] = {}
    for model_name, result in results.items():
        best = result["thresholds"]["best_balanced"]
        enabled = best["balanced_accuracy"] >= args.min_balanced_accuracy
        weight = max(0.0, best["balanced_accuracy"] - 0.5) if enabled else 0.0
        models[model_name] = {
            "enabled": enabled,
            "weight": round(weight, 4),
            "threshold": best["threshold"],
            "metrics": {
                "auc": result["auc"],
                "sample_count": result["sample_count"],
                "use_face_crop": result["use_face_crop"],
                "best_balanced": best,
                "best_f1": result["thresholds"]["best_f1"],
            },
        }

    metadata = {
        "version": 1,
        "calibration": {
            "source": "evaluate_models.py",
            "validation_dir": str(args.validation_dir),
            "sample_per_class": args.sample_per_class,
            "use_face_crop": bool(args.use_face_crop),
            "min_balanced_accuracy": args.min_balanced_accuracy,
        },
        "models": models,
    }

    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata_out.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"\n[OK] Wrote metadata: {args.metadata_out}")


def main() -> int:
    args = parse_args()
    configure_tensorflow()
    samples = list_validation_images(args.validation_dir, args.sample_per_class, args.seed)
    print(f"[INFO] Validation samples: {len(samples)}")
    print(f"[INFO] Face crop: {args.use_face_crop}")

    results = {}
    for model_name in args.models:
        results[model_name] = evaluate_one_model(model_name, samples, args)

    if args.write_metadata:
        write_metadata(results, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
