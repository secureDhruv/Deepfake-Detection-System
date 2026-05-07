import argparse
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(min(4, os.cpu_count() or 1)))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2, Xception
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_prep
from tensorflow.keras.applications.xception import preprocess_input as xception_prep
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
VAL_DIR = BASE_DIR / "dataset" / "validation"
MODEL_DIR = BASE_DIR / "dataset" / "model"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_CLASS_INDICES = {"fake": 0, "real": 1}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "MobileNetV2": {
        "class": MobileNetV2,
        "preprocess": mobilenet_prep,
        "input_size": (224, 224),
        "save_path": MODEL_DIR / "deepfake_model.h5",
        "finetune_layers": 30,
        "batch_size": 32,
    },
    "Xception": {
        "class": Xception,
        "preprocess": xception_prep,
        "input_size": (224, 224),
        "save_path": MODEL_DIR / "xception_model.h5",
        "finetune_layers": 24,
        "batch_size": 16,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CPU-friendly deepfake detector models.")
    parser.add_argument(
        "--model",
        choices=["MobileNetV2", "Xception", "all"],
        default="MobileNetV2",
        help="MobileNetV2 is the recommended CPU model. Xception is opt-in.",
    )
    parser.add_argument("--epochs-head", type=int, default=15)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=0, help="0 uses each model default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    return parser.parse_args()


def configure_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
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


def list_images(folder: Path) -> list[Path]:
    return [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def validate_dataset() -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_dir in (TRAIN_DIR, VAL_DIR):
        for label in ("fake", "real"):
            folder = split_dir / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing dataset folder: {folder}")
            images = list_images(folder)
            if not images:
                raise ValueError(f"No supported images found in: {folder}")
            counts[f"{split_dir.name}_{label}"] = len(images)

    for label in ("fake", "real"):
        train_names = {path.name for path in list_images(TRAIN_DIR / label)}
        val_names = {path.name for path in list_images(VAL_DIR / label)}
        overlap = train_names & val_names
        if overlap:
            raise ValueError(
                f"Dataset leakage detected for '{label}': {len(overlap)} filenames "
                "exist in both train and validation. Regenerate with prepare_dataset.py."
            )

    return counts


def assert_class_mapping(class_indices: dict[str, int]) -> None:
    if class_indices != EXPECTED_CLASS_INDICES:
        raise ValueError(
            f"Unexpected class mapping {class_indices}. Expected {EXPECTED_CLASS_INDICES}; "
            "prediction code assumes sigmoid output is P(real)."
        )


def compute_class_weights() -> dict[int, float]:
    n_fake = len(list_images(TRAIN_DIR / "fake"))
    n_real = len(list_images(TRAIN_DIR / "real"))
    n_total = n_fake + n_real

    w_fake = n_total / (2 * n_fake) if n_fake else 1.0
    w_real = n_total / (2 * n_real) if n_real else 1.0
    w_fake = min(w_fake, 3.0)
    w_real = min(w_real, 3.0)

    print(f"[Balance] fake={n_fake:,} real={n_real:,} ratio={n_fake / max(n_real, 1):.2f}")
    print(f"[Balance] class_weight fake(0)={w_fake:.3f} real(1)={w_real:.3f}")
    return {0: w_fake, 1: w_real}


def build_model(config: dict[str, Any], label_smoothing: float) -> tuple[models.Sequential, models.Model]:
    img_h, img_w = config["input_size"]
    base_model = config["class"](
        input_shape=(img_h, img_w, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
            ),
            layers.Dropout(0.35),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    compile_model(model, learning_rate=1e-3, label_smoothing=label_smoothing)
    return model, base_model


def compile_model(model: models.Model, learning_rate: float, label_smoothing: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def make_generators(config: dict[str, Any], batch_size: int):
    img_h, img_w = config["input_size"]
    train_gen = ImageDataGenerator(
        preprocessing_function=config["preprocess"],
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        zoom_range=0.12,
        brightness_range=(0.85, 1.15),
        fill_mode="nearest",
    )
    val_gen = ImageDataGenerator(preprocessing_function=config["preprocess"])

    train_data = train_gen.flow_from_directory(
        str(TRAIN_DIR),
        target_size=(img_h, img_w),
        batch_size=batch_size,
        class_mode="binary",
        seed=42,
        shuffle=True,
    )
    val_data = val_gen.flow_from_directory(
        str(VAL_DIR),
        target_size=(img_h, img_w),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    return train_data, val_data


def callbacks_for(model_name: str, phase: str, save_path: Path) -> list:
    history_path = MODEL_DIR / f"{model_name.lower()}_{phase}_history.csv"
    return [
        ModelCheckpoint(
            str(save_path),
            save_best_only=True,
            monitor="val_auc",
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_auc",
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(str(history_path), append=False),
    ]


def unfreeze_for_finetuning(base_model: models.Model, trainable_layers: int) -> None:
    base_model.trainable = True
    cutoff = max(0, len(base_model.layers) - trainable_layers)
    for index, layer in enumerate(base_model.layers):
        layer.trainable = index >= cutoff and not isinstance(layer, layers.BatchNormalization)


def best_value(history, key: str) -> float:
    values = history.history.get(key, [])
    return float(max(values)) if values else 0.0


def train_model_pipeline(model_name: str, args: argparse.Namespace) -> None:
    from tensorflow.keras.models import load_model

    config = MODEL_CONFIGS[model_name]
    batch_size = args.batch_size or config["batch_size"]
    print("\n" + "=" * 72)
    print(f"Training {model_name} | input={config['input_size']} | batch_size={batch_size}")
    print("=" * 72)

    train_data, val_data = make_generators(config, batch_size=batch_size)
    print("Class indices:", train_data.class_indices)
    assert_class_mapping(train_data.class_indices)
    class_weight = compute_class_weights()

    model, base_model = build_model(config, label_smoothing=args.label_smoothing)
    model.summary()

    save_path = config["save_path"]
    phase1_path = MODEL_DIR / f"{model_name.lower()}_phase1_best.h5"
    phase2_path = MODEL_DIR / f"{model_name.lower()}_phase2_best.h5"
    print("\n[Phase 1] Training classification head")
    history_head = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs_head,
        callbacks=callbacks_for(model_name, "phase1", phase1_path),
        class_weight=class_weight,
    )
    best_phase1_auc = best_value(history_head, "val_auc")

    if args.epochs_finetune > 0:
        print(f"\n[Phase 2] Fine-tuning top {config['finetune_layers']} non-BN base layers")
        unfreeze_for_finetuning(base_model, config["finetune_layers"])
        compile_model(model, learning_rate=1e-5, label_smoothing=args.label_smoothing)
        history_finetune = model.fit(
            train_data,
            validation_data=val_data,
            epochs=args.epochs_finetune,
            callbacks=callbacks_for(model_name, "phase2", phase2_path),
            class_weight=class_weight,
        )
    else:
        history_finetune = None

    best_phase2_auc = best_value(history_finetune, "val_auc") if history_finetune else 0.0
    best_source = phase2_path if history_finetune and best_phase2_auc >= best_phase1_auc else phase1_path
    best_model = load_model(str(best_source), compile=False)
    best_model.save(str(save_path))

    best_auc = max(best_phase1_auc, best_phase2_auc)
    best_acc = max(
        best_value(history_head, "val_accuracy"),
        best_value(history_finetune, "val_accuracy") if history_finetune else 0.0,
    )
    print(f"[{model_name}] best val_auc={best_auc:.4f} best val_accuracy={best_acc:.4f}")
    print(f"[{model_name}] saved model: {save_path}")


def selected_models(name: str) -> list[str]:
    if name == "all":
        return ["MobileNetV2", "Xception"]
    return [name]


def main() -> int:
    args = parse_args()
    configure_runtime(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Validating dataset structure")
    counts = validate_dataset()
    print("[INFO] Dataset counts:", counts)

    for model_name in selected_models(args.model):
        train_model_pipeline(model_name, args)

    print("\n[OK] Training complete.")
    print("[NEXT] Run: python evaluate_models.py --use-face-crop --write-metadata")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
