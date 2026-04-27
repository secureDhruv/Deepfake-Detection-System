import cv2

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
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
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
    face = img_rgb[y : y + h, x : x + w]
    if face.size == 0:
        print(f"[WARNING] Empty face crop in '{image_path}'. Using full image.")
        return img_rgb

    return face
