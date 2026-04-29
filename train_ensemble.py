import os
from tensorflow.keras.applications import MobileNetV2, Xception
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_prep
from tensorflow.keras.applications.xception import preprocess_input as xception_prep
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR    = os.path.join(BASE_DIR, "dataset", "validation")
MODEL_DIR  = os.path.join(BASE_DIR, "dataset", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

EXPECTED_CLASS_INDICES = {"fake": 0, "real": 1}

# ── Ensemble Configuration ──────────────────────────────────────────────────────
MODELS_TO_TRAIN = [
    {
        "name": "MobileNetV2",
        "class": MobileNetV2,
        "preprocess": mobilenet_prep,
        "save_path": os.path.join(MODEL_DIR, "deepfake_model.h5")
    },
    {
        "name": "Xception",
        "class": Xception,
        "preprocess": xception_prep,
        "save_path": os.path.join(MODEL_DIR, "xception_model.h5")
    }
]

def validate_dataset():
    required_folders = [
        os.path.join(TRAIN_DIR, "fake"),
        os.path.join(TRAIN_DIR, "real"),
        os.path.join(VAL_DIR, "fake"),
        os.path.join(VAL_DIR, "real"),
    ]
    for folder in required_folders:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing dataset folder: {folder}")

    for label in ["fake", "real"]:
        train_names = set(os.listdir(os.path.join(TRAIN_DIR, label)))
        val_names = set(os.listdir(os.path.join(VAL_DIR, label)))
        overlap = train_names & val_names
        if overlap:
            raise ValueError(
                f"Dataset leakage detected for '{label}': {len(overlap)} filenames "
                "exist in both train and validation. Regenerate with prepare_dataset.py."
            )


def assert_class_mapping(class_indices):
    if class_indices != EXPECTED_CLASS_INDICES:
        raise ValueError(
            f"Unexpected class mapping {class_indices}. Expected {EXPECTED_CLASS_INDICES}; "
            "prediction code assumes sigmoid output is P(real)."
        )


def train_model_pipeline(config):
    print(f"\n========================================")
    print(f"  Training {config['name']}...")
    print(f"========================================\n")
    
    # 1. Generators
    train_gen = ImageDataGenerator(
        preprocessing_function=config['preprocess'],
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    val_gen = ImageDataGenerator(preprocessing_function=config['preprocess'])

    train_data = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        seed=42,
    )
    val_data = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=False,
    )
    print("Class indices:", train_data.class_indices)
    assert_class_mapping(train_data.class_indices)

    # 2. Build Model
    base_model = config['class'](input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 3. Callbacks
    callbacks = [
        ModelCheckpoint(config['save_path'], save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
    ]

    # 4. Train
    print("[INFO] Starting training for 3 epochs (Reduced for CPU time limits)")
    model.fit(train_data, validation_data=val_data, epochs=3, callbacks=callbacks)
    print(f"Finished training {config['name']}. Saved to {config['save_path']}")

if __name__ == "__main__":
    validate_dataset()
    for m_config in MODELS_TO_TRAIN:
        train_model_pipeline(m_config)
    print("\n[SUCCESS] Ensemble training complete!")
