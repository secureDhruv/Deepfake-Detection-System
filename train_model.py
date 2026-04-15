import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR    = os.path.join(BASE_DIR, "dataset", "validation")

# Save model to the same location predictor.py reads from
MODEL_DIR  = os.path.join(BASE_DIR, "dataset", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_model.h5")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data generators ────────────────────────────────────────────────────────────
# Use MobileNetV2's preprocess_input (maps [0,255] → [-1,1]) — NOT rescale=1/255
# Training augmentation added to reduce overfitting
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
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
)

# Confirm class mapping — should be: {'fake': 0, 'real': 1}
print("Class indices:", train_data.class_indices)

# ── Model ───────────────────────────────────────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False   # freeze pretrained weights

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),          # regularize to reduce overfitting
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Callbacks ───────────────────────────────────────────────────────────────────
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

# ── Training ─────────────────────────────────────────────────────────────────────
model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,       # EarlyStopping will stop earlier if val_accuracy plateaus
    callbacks=callbacks,
)

print(f"Model saved to: {MODEL_PATH}")