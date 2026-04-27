"""
generate_dummy_models.py
------------------------
Generates randomly-initialized (UNTRAINED) dummy .h5 model files for the
complete DeepGuard AI ensemble:

    Model          │ Save File            │ Preprocessor
    ───────────────┼──────────────────────┼─────────────────────────
    MobileNetV2    │ deepfake_model.h5    │ mobilenet_v2.preprocess_input
    ResNet50V2     │ resnet_model.h5      │ resnet_v2.preprocess_input
    Xception       │ xception_model.h5    │ xception.preprocess_input

Architecture (identical to train_ensemble.py):
    BaseModel (frozen, weights=None) → GlobalAveragePooling2D →
    Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)

WARNING:
    These models have RANDOM weights and will produce meaningless predictions.
    They are intended for:
      • Verifying the Flask app loads without crashing
      • Checking the ensemble pipeline end-to-end
      • UI/template development without needing a GPU
    Replace with properly trained weights (run train_ensemble.py) before
    any real-world use.

Usage:
    # Generate only missing models (default)
    python generate_dummy_models.py

    # Force-regenerate all models (overwrites existing)
    python generate_dummy_models.py --force
"""

import os
import sys
import time

# ── Imports ────────────────────────────────────────────────────────────────────
from tensorflow.keras.applications import MobileNetV2, ResNet50V2, Xception
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_prep
from tensorflow.keras.applications.resnet_v2    import preprocess_input as resnet_prep
from tensorflow.keras.applications.xception     import preprocess_input as xception_prep
from tensorflow.keras import layers, models

# ── Shared constants (mirrors predictor.py and train_ensemble.py) ──────────────
INPUT_SHAPE   = (224, 224, 3)   # All three models use the same input resolution
DENSE_UNITS   = 128
DROPOUT_RATE  = 0.3

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "dataset", "model")

# ── Ensemble registry ──────────────────────────────────────────────────────────
# Kept in the same order and with the same keys as predictor.py's ENSEMBLE_CONFIG
# so dummy models map 1-to-1 with what the app expects to load.
ENSEMBLE_REGISTRY = [
    {
        "name":       "MobileNetV2",
        "class":      MobileNetV2,
        "preprocess": mobilenet_prep,           # maps [0,255] → [-1, 1]
        "save_path":  os.path.join(MODEL_DIR, "deepfake_model.h5"),
    },
    {
        "name":       "ResNet50V2",
        "class":      ResNet50V2,
        "preprocess": resnet_prep,              # maps [0,255] → [-1, 1]
        "save_path":  os.path.join(MODEL_DIR, "resnet_model.h5"),
    },
    {
        "name":       "Xception",
        "class":      Xception,
        "preprocess": xception_prep,            # maps [0,255] → [-1, 1]
        "save_path":  os.path.join(MODEL_DIR, "xception_model.h5"),
    },
]


# ── Core builder ──────────────────────────────────────────────────────────────
def build_dummy_model(base_model_class: type) -> models.Sequential:
    """
    Build a Sequential model with a frozen base and a binary classification head.

    The architecture is kept identical to train_ensemble.py so that the saved
    .h5 files are structurally compatible with trained checkpoints — you can
    drop real weights in without changing anything else.

    Args:
        base_model_class: One of MobileNetV2, ResNet50V2, or Xception.

    Returns:
        A compiled (but untrained) Keras Sequential model.
    """
    # weights=None → random initialization; skips the ~80-100 MB ImageNet
    # download that `weights="imagenet"` would trigger.
    base_model = base_model_class(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights=None,
    )
    base_model.trainable = False   # mirrors train_ensemble.py behaviour

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(DENSE_UNITS, activation="relu"),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(1, activation="sigmoid"),
        ],
        name=f"{base_model_class.__name__}_dummy",
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def generate_model(config: dict, force: bool = False) -> bool:
    """
    Generate and save a single dummy model from the registry entry.

    Args:
        config: One entry from ENSEMBLE_REGISTRY.
        force:  If True, overwrite an existing file.

    Returns:
        True if the model was (re-)generated, False if it was skipped.
    """
    name      = config["name"]
    save_path = config["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path) and not force:
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  [SKIP]  {name:12s} -> already exists  ({size_mb:.1f} MB)  {save_path}")
        return False

    print(f"  [BUILD] {name:12s} -> building model architecture ...")
    t0    = time.time()
    model = build_dummy_model(config["class"])
    model.save(save_path, include_optimizer=False)
    elapsed = time.time() - t0

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  [DONE]  {name:12s} -> saved  ({size_mb:.1f} MB, {elapsed:.1f}s)  {save_path}")
    return True


def verify_models() -> None:
    """
    Post-generation sanity check: confirm every expected file exists and
    print a summary table matching predictor.py's ENSEMBLE_CONFIG.
    """
    print()
    print("  Verification")
    print("  " + "-" * 56)
    all_ok = True
    for cfg in ENSEMBLE_REGISTRY:
        exists  = os.path.exists(cfg["save_path"])
        status  = "[OK]     " if exists else "[MISSING]"
        size    = f"{os.path.getsize(cfg['save_path']) / (1024*1024):.1f} MB" if exists else "--"
        fname   = os.path.basename(cfg["save_path"])
        print(f"  {status}  {cfg['name']:12s}  {fname:22s}  {size}")
        if not exists:
            all_ok = False
    print("  " + "-" * 56)
    if all_ok:
        print("  All 3 ensemble models are present. The app should load cleanly.")
    else:
        print("  Some models are missing — re-run this script to generate them.")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    force_regen = "--force" in sys.argv

    print()
    print("=" * 60)
    print("  DeepGuard AI -- Dummy Model Generator")
    print("=" * 60)
    print("  [!] WARNING: Random weights -- for development use ONLY.")
    if force_regen:
        print("  --force flag detected: existing files will be overwritten.")
    print("=" * 60)
    print()

    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"  Model directory: {MODEL_DIR}")
    print()

    generated = 0
    skipped   = 0

    for cfg in ENSEMBLE_REGISTRY:
        result = generate_model(cfg, force=force_regen)
        if result:
            generated += 1
        else:
            skipped += 1

    verify_models()

    print(f"  Summary: {generated} generated, {skipped} skipped.")
    if generated > 0:
        print("  Reminder: replace these with trained weights (run train_ensemble.py)")
        print("  before running real deepfake detections.")
    print()
