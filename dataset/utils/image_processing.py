import logging
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MAX_DETECTION_DIMENSION = 900
FACE_PADDING_RATIO = 0.18
MIN_FACE_SIZE_RATIO = 0.08


@dataclass(frozen=True)
class FaceCropMetadata:
    face_detected: bool
    source_width: int
    source_height: int
    crop_box: tuple[int, int, int, int] | None
    detector: str
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.crop_box is not None:
            data["crop_box"] = list(self.crop_box)
        return data


def _load_cascade(filename: str) -> cv2.CascadeClassifier | None:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + filename)
    if cascade.empty():
        logger.warning("OpenCV Haar cascade could not be loaded: %s", filename)
        return None
    return cascade


_CASCADES: list[tuple[str, cv2.CascadeClassifier | None, bool]] = [
    ("frontal_default", _load_cascade("haarcascade_frontalface_default.xml"), False),
    ("frontal_alt2", _load_cascade("haarcascade_frontalface_alt2.xml"), False),
    ("profile_left", _load_cascade("haarcascade_profileface.xml"), False),
    ("profile_right", _load_cascade("haarcascade_profileface.xml"), True),
]

if not any(cascade is not None for _, cascade, _ in _CASCADES):
    raise RuntimeError("No OpenCV Haar cascades could be loaded.")


def load_image_rgb(image_path: str) -> np.ndarray:
    """Load an image as RGB while honoring EXIF orientation."""
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return np.array(image, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Could not read image: {image_path}") from exc


def _resize_for_detection(img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = img_rgb.shape[:2]
    largest_dimension = max(width, height)
    scale = 1.0

    if largest_dimension > MAX_DETECTION_DIMENSION:
        scale = MAX_DETECTION_DIMENSION / largest_dimension
        resized = cv2.resize(
            img_rgb,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = img_rgb

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray, scale


def _detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int, str]]:
    img_h, img_w = gray.shape[:2]
    min_side = max(32, int(min(img_h, img_w) * MIN_FACE_SIZE_RATIO))
    detections: list[tuple[int, int, int, int, str]] = []

    for detector_name, cascade, use_flip in _CASCADES:
        if cascade is None:
            continue

        detect_gray = cv2.flip(gray, 1) if use_flip else gray
        faces = cascade.detectMultiScale(
            detect_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_side, min_side),
        )

        for x, y, w, h in faces:
            if use_flip:
                x = img_w - x - w
            detections.append((int(x), int(y), int(w), int(h), detector_name))

    return detections


def _full_image_result(
    img_rgb: np.ndarray,
    reason: str,
    return_metadata: bool,
):
    height, width = img_rgb.shape[:2]
    metadata = FaceCropMetadata(
        face_detected=False,
        source_width=width,
        source_height=height,
        crop_box=None,
        detector="full_image",
        fallback_reason=reason,
    )
    logger.info("Using full image fallback: %s", reason)
    if return_metadata:
        return img_rgb, metadata
    return img_rgb


def extract_face(image_path: str, return_metadata: bool = False):
    """
    Read an image, convert it to RGB, and crop the largest detected face.

    The function falls back to the full RGB image when no reliable face crop is
    available. This mirrors the previous public behavior while giving callers
    optional metadata for traceability.
    """
    img_rgb = load_image_rgb(image_path)
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return _full_image_result(img_rgb, "unsupported_rgb_shape", return_metadata)

    height, width = img_rgb.shape[:2]
    gray, scale = _resize_for_detection(img_rgb)
    faces = _detect_faces(gray)

    if not faces:
        return _full_image_result(img_rgb, "no_face_detected", return_metadata)

    x, y, w, h, detector = max(faces, key=lambda box: box[2] * box[3])
    if scale != 1.0:
        x = int(round(x / scale))
        y = int(round(y / scale))
        w = int(round(w / scale))
        h = int(round(h / scale))

    padding = int(max(w, h) * FACE_PADDING_RATIO)
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)

    face = img_rgb[y0:y1, x0:x1]
    if face.size == 0:
        return _full_image_result(img_rgb, "empty_face_crop", return_metadata)

    metadata = FaceCropMetadata(
        face_detected=True,
        source_width=width,
        source_height=height,
        crop_box=(x0, y0, x1, y1),
        detector=detector,
    )
    if return_metadata:
        return face, metadata
    return face
