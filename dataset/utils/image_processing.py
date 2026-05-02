import cv2

MAX_DETECTION_DIMENSION = 900
FACE_PADDING_RATIO = 0.18

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if _face_cascade.empty():
    raise RuntimeError("OpenCV Haar cascade could not be loaded.")


def extract_face(image_path: str):
    """
    Read an image, convert it to RGB, and crop the largest detected face.

    If no face is found, the full RGB image is returned.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    height, width = img_bgr.shape[:2]
    largest_dimension = max(width, height)
    scale = 1.0
    detection_bgr = img_bgr

    if largest_dimension > MAX_DETECTION_DIMENSION:
        scale = MAX_DETECTION_DIMENSION / largest_dimension
        detection_bgr = cv2.resize(
            img_bgr,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(detection_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces) == 0:
        print(f"[WARNING] No face detected in '{image_path}'. Using full image.")
        return img_rgb

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

    padding = int(max(w, h) * FACE_PADDING_RATIO)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)

    face = img_rgb[y0:y1, x0:x1]
    if face.size == 0:
        print(f"[WARNING] Empty face crop in '{image_path}'. Using full image.")
        return img_rgb

    return face
