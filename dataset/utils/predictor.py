import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from .image_processing import extract_face

# Resolve model path robustly against working-directory changes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "deepfake_model.h5"))

model = load_model(MODEL_PATH)


def predict_image(image_path: str) -> tuple:
    """
    Run deepfake detection on the image at *image_path*.

    Returns:
        (label, confidence_pct)
          label          – "Real Image" or "Fake Image"
          confidence_pct – float 0-100 indicating model certainty in that label

    Raises:
        ValueError: propagated from extract_face if the file cannot be read.
    """
    face = extract_face(image_path)   # returns an RGB numpy array

    # Ensure the array has exactly 3 colour channels
    if face.ndim == 2:
        # Grayscale image — duplicate to 3 channels
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    elif face.shape[2] == 4:
        # RGBA — strip the alpha channel
        face = face[:, :, :3]

    img = cv2.resize(face, (224, 224)).astype("float32")

    # preprocess_input maps pixels from [0,255] to [-1,1] as MobileNetV2 expects.
    # Both training (train_model.py) and inference must use the same function.
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)   # shape: (1, 224, 224, 3)

    raw = model.predict(img, verbose=0)
    p_real = float(raw[0][0])
    # Sigmoid output = P(real), because flow_from_directory sorts alphabetically:
    #   fake → class 0,  real → class 1  →  output = P(class 1) = P(real)

    if p_real > 0.5:
        label = "Real Image"
        confidence = round(p_real * 100, 1)
    else:
        label = "Fake Image"
        confidence = round((1.0 - p_real) * 100, 1)

    return label, confidence