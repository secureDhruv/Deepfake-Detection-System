import cv2

# Load the Haar cascade once at module level — avoids repeated disk I/O per call
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_face(image_path: str):
    """
    Read an image, convert it to RGB, attempt to detect and crop a face.

    Returns:
        numpy.ndarray (RGB) — the face crop if one was detected,
        otherwise the full image.

    Raises:
        ValueError: if the image file cannot be read (bad path, corrupt file, etc.)
    """
    img_bgr = cv2.imread(image_path)

    # Guard against unreadable / missing files
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert BGR → RGB to match the RGB ordering used during training
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Return the first detected face region in RGB
        return img_rgb[y : y + h, x : x + w]

    # No face found — fall back to full image with a warning
    print(f"[WARNING] No face detected in '{image_path}'. Using full image.")
    return img_rgb