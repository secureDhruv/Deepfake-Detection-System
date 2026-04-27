import os

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
MODEL_DIR = os.path.join(BASE_DIR, "dataset", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_model.h5")

EXPECTED_CLASS_INDICES = {"fake": 0, "real": 1}


def validate_dataset() -> None:
    for folder in [
        os.path.join(TRAIN_DIR, "fake"),
        os.path.join(TRAIN_DIR, "real"),
        os.path.join(VAL_DIR, "fake"),
        os.path.join(VAL_DIR, "real"),
    ]:
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


def assert_class_mapping(class_indices: dict[str, int]) -> None:
    if class_indices != EXPECTED_CLASS_INDICES:
        raise ValueError(
            f"Unexpected class mapping {class_indices}. Expected {EXPECTED_CLASS_INDICES}; "
            "prediction code assumes sigmoid output is P(real)."
        )


def build_model() -> models.Sequential:
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    validate_dataset()
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

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

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_accuracy",
        ),
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=callbacks,
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
