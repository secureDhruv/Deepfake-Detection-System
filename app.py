import os
import json
import math
import secrets
import uuid
from flask import Flask, abort, flash, redirect, render_template, request, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from dataset.utils.predictor import predict_image, get_loaded_model_names
from database import (
    init_db,
    save_detection,
    get_all_detections,
    get_detection_count,
    get_detection_stats,
    get_detection_by_id,
    delete_detection,
    clear_detections,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "dataset", "templates"),
    static_folder=os.path.join(BASE_DIR, "dataset", "static"),
)

IS_PRODUCTION = os.environ.get("APP_ENV") == "production" or os.environ.get("FLASK_ENV") == "production"
SECRET_KEY = os.environ.get("SECRET_KEY")

if IS_PRODUCTION and not SECRET_KEY:
    raise RuntimeError("SECRET_KEY must be set when APP_ENV or FLASK_ENV is production.")

# Secret key required for sessions, flash() messages, and CSRF tokens.
app.secret_key = SECRET_KEY or "dev-only-change-me"

UPLOAD_FOLDER = os.path.join(BASE_DIR, "dataset", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = not IS_PRODUCTION

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
CSRF_SESSION_KEY = "_csrf_token"
CSRF_HEADER_NAME = "X-CSRFToken"
MAX_IMAGE_PIXELS = 16_000_000
DASHBOARD_PAGE_SIZE = 50

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_upload_filename(original_filename: str) -> str:
    """Return a sanitized, collision-resistant filename for the upload folder."""
    filename = secure_filename(original_filename)
    if not filename:
        raise ValueError("Invalid filename.")

    stem, ext = os.path.splitext(filename)
    stem = (stem or "upload")[:80]
    ext = ext.lower()

    while True:
        candidate = f"{stem}_{uuid.uuid4().hex[:12]}{ext}"
        if not os.path.exists(os.path.join(UPLOAD_FOLDER, candidate)):
            return candidate


def validate_saved_image(filepath: str) -> None:
    """Verify that the uploaded file is a real image before running inference."""
    from PIL import Image, UnidentifiedImageError
    from PIL.Image import DecompressionBombError

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

    try:
        with Image.open(filepath) as img:
            if img.format not in ALLOWED_IMAGE_FORMATS:
                raise ValueError("Unsupported image content. Please upload a PNG, JPG, or WEBP image.")

            width, height = img.size
            if width <= 0 or height <= 0:
                raise ValueError("Uploaded image has invalid dimensions.")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Image resolution is too large. Please upload an image under 16 megapixels.")

            img.verify()
    except ValueError:
        raise
    except DecompressionBombError as exc:
        raise ValueError("Image resolution is too large. Please upload an image under 16 megapixels.") from exc
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc


def delete_upload_file(filename: str | None) -> bool:
    """Delete one stored upload by sanitized filename."""
    if not filename:
        return False

    safe_name = secure_filename(filename)
    if safe_name != filename:
        return False

    upload_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, safe_name))
    upload_root = os.path.abspath(UPLOAD_FOLDER)
    if os.path.commonpath([upload_root, upload_path]) != upload_root:
        return False
    if not os.path.isfile(upload_path):
        return False

    os.remove(upload_path)
    return True


def positive_int_arg(name: str, default: int, maximum: int | None = None) -> int:
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default

    value = max(1, value)
    if maximum is not None:
        value = min(value, maximum)
    return value


def csrf_token() -> str:
    """Return the current session CSRF token, creating one when needed."""
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


@app.before_request
def protect_post_requests():
    """Require a session CSRF token for every mutating request."""
    if request.method != "POST":
        return

    expected_token = session.get(CSRF_SESSION_KEY)
    submitted_token = request.form.get("csrf_token") or request.headers.get(CSRF_HEADER_NAME)

    if not expected_token or not submitted_token:
        abort(400, description="Missing CSRF token.")
    if not secrets.compare_digest(expected_token, submitted_token):
        abort(400, description="Invalid CSRF token.")


# Initialize database
init_db()

# ── Inject engine model list into every template ───────────────────────────
@app.context_processor
def inject_engine_models():
    """
    Makes `engine_models` (list of loaded model names) available in every
    Jinja2 template so the sidebar panel can render without extra per-route code.
    """
    return {
        "engine_models": get_loaded_model_names(),
        "csrf_token": csrf_token,
    }


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    flash("File too large. Please upload an image under 10 MB.")
    return redirect(url_for("index"))


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    details = None
    new_record_id = None

    if request.method == "POST":
        # Validate that the request contains an image field
        if "image" not in request.files:
            flash("No file part in the request.")
            return redirect(url_for("index"))

        file = request.files["image"]

        if file.filename == "":
            flash("No file selected.")
            return redirect(url_for("index"))

        # Server-side extension validation
        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload a PNG, JPG, or WEBP image.")
            return redirect(url_for("index"))

        # Sanitize filename and avoid overwriting previous uploads.
        try:
            filename = unique_upload_filename(file.filename)
        except ValueError as e:
            flash(str(e))
            return redirect(url_for("index"))

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            validate_saved_image(filepath)
            # predict_image now returns (label, confidence_pct, details)
            result, confidence, details = predict_image(filepath)
        except Exception as e:
            try:
                os.remove(filepath)
            except OSError:
                pass
            flash(f"Prediction error: {e}")
            return redirect(url_for("index"))

        # Persist detection result including confidence score and details summary
        details_json = json.dumps(details, sort_keys=True)
        new_record_id = save_detection(filename=filename, result=result, confidence=confidence, details=details_json)

    record_count = get_detection_count()
    
    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        details=details,
        new_record_id=new_record_id,
        record_count=record_count,
    )


@app.route("/dashboard")
def dashboard():
    page = positive_int_arg("page", 1)
    search_query = request.args.get("q", "").strip()
    dashboard_stats = get_detection_stats(search_query)
    total_records = dashboard_stats["total"]
    total_pages = max(1, math.ceil(total_records / DASHBOARD_PAGE_SIZE))
    page = min(page, total_pages)
    offset = (page - 1) * DASHBOARD_PAGE_SIZE
    data = get_all_detections(limit=DASHBOARD_PAGE_SIZE, offset=offset, search=search_query)

    return render_template(
        "dashboard.html",
        data=data,
        page=page,
        per_page=DASHBOARD_PAGE_SIZE,
        total_pages=total_pages,
        total_records=total_records,
        fake_count=dashboard_stats["fake_count"],
        search_query=search_query,
    )


@app.route("/analysis/<int:record_id>")
def analysis(record_id: int):
    """Heatmap / detail view for a single detection record."""
    record = get_detection_by_id(record_id)
    if record is None:
        flash("Detection record not found.")
        return redirect(url_for("dashboard"))

    details = {}
    if len(record) > 5 and record[5]:
        try:
            details = json.loads(record[5])
        except json.JSONDecodeError:
            details = {}

    confidence_value = float(record[3] or 0)
    is_fake = "fake" in str(record[2]).lower()
    fake_score = confidence_value if is_fake else max(0.0, 100.0 - confidence_value)
    real_score = confidence_value if not is_fake else max(0.0, 100.0 - confidence_value)

    return render_template(
        "analysis.html",
        record=record,
        details=details,
        is_fake=is_fake,
        fake_score=round(fake_score, 1),
        real_score=round(real_score, 1),
    )


@app.route("/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id: int):
    """Delete a detection record from the database."""
    record = get_detection_by_id(record_id)
    deleted = delete_detection(record_id)
    if deleted:
        upload_deleted = False
        if record is not None:
            try:
                upload_deleted = delete_upload_file(record[1])
            except OSError:
                upload_deleted = False

        if upload_deleted:
            flash(f"Record #{record_id} and its upload were deleted successfully.")
        else:
            flash(f"Record #{record_id} deleted successfully.")
    else:
        flash(f"Record #{record_id} was not found.")
    return redirect(url_for("dashboard"))


@app.route("/analytics")
def analytics():
    """Aggregated statistics for deepfake detection."""
    stats = get_detection_stats()
    data = get_all_detections(limit=8)

    return render_template("analytics.html", stats=stats, data=data)

@app.route("/settings")
def settings():
    """Redirect to the merged home page, Settings tab."""
    return redirect(url_for('index') + '?tab=settings')


@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Delete all detection records from DB."""
    clear_detections()
    return "", 204


@app.route("/clear-uploads", methods=["POST"])
def clear_uploads():
    """Delete all files in the uploads folder."""
    import glob
    for f in glob.glob(os.path.join(UPLOAD_FOLDER, "*")):
        if os.path.basename(f) == ".gitkeep" or not os.path.isfile(f):
            continue
        try:
            os.remove(f)
        except OSError:
            pass
    return "", 204

@app.route("/test-ui")
def test_ui():
    if os.environ.get("ENABLE_TEST_UI") != "1":
        abort(404)

    details = {
        "MobileNetV2": {"label": "Fake Image", "confidence": 88.5},
        "Xception": {"label": "Real Image", "confidence": 60.0}
    }
    return render_template("index.html", result="Fake Image", confidence=82.5, details=details, record_count=42, active_tab="scan")

if __name__ == "__main__":
    debug_enabled = os.environ.get("FLASK_DEBUG") == "1"
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=debug_enabled)
