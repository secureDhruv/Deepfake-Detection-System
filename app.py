import os
import json
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
from dataset.utils.predictor import predict_image, get_loaded_model_names
from database import (
    init_db,
    save_detection,
    get_all_detections,
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

# Secret key required for flash() messages
app.secret_key = os.environ.get("SECRET_KEY", "change-me-before-deploying")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "dataset", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True  # always pick up fresh HTML/CSS

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

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

    try:
        with Image.open(filepath) as img:
            img.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc


# Initialize database
init_db()

# ── Inject engine model list into every template ───────────────────────────
@app.context_processor
def inject_engine_models():
    """
    Makes `engine_models` (list of loaded model names) available in every
    Jinja2 template so the sidebar panel can render without extra per-route code.
    """
    return {"engine_models": get_loaded_model_names()}


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

    record_count = len(get_all_detections())
    
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
    data = get_all_detections()
    return render_template("dashboard.html", data=data)


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
    deleted = delete_detection(record_id)
    if deleted:
        flash(f"Record #{record_id} deleted successfully.")
    else:
        flash(f"Record #{record_id} was not found.")
    return redirect(url_for("dashboard"))


@app.route("/analytics")
def analytics():
    """Aggregated statistics for deepfake detection."""
    data = get_all_detections()
    
    total_scans = len(data)
    fake_count = sum(1 for row in data if "fake" in str(row[2]).lower())
    authentic_count = total_scans - fake_count
    
    avg_conf = 0
    dist = {'0-25': 0, '25-50': 0, '50-75': 0, '75-100': 0}
    high_conf_count = 0
    
    if total_scans > 0:
        confs = [row[3] for row in data if row[3] is not None]
        if confs:
            avg_conf = round(sum(confs) / len(confs), 1)
            for c in confs:
                if c <= 25:
                    dist['0-25'] += 1
                elif c <= 50:
                    dist['25-50'] += 1
                elif c <= 75:
                    dist['50-75'] += 1
                else:
                    dist['75-100'] += 1
                if c > 75:
                    high_conf_count += 1
                    
    fake_pct = (fake_count / total_scans * 100) if total_scans > 0 else 0
    real_pct = (authentic_count / total_scans * 100) if total_scans > 0 else 0
    high_conf_pct = (high_conf_count / total_scans * 100) if total_scans > 0 else 0
    
    stats = {
        'total': total_scans,
        'fake_count': fake_count,
        'real_count': authentic_count,
        'avg_conf': avg_conf,
        'fake_pct': fake_pct,
        'real_pct': real_pct,
        'dist': dist,
        'high_conf_pct': high_conf_pct
    }

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
    details = {
        "MobileNetV2": {"label": "Fake Image", "confidence": 88.5},
        "ResNet50V2": {"label": "Fake Image", "confidence": 92.1},
        "Xception": {"label": "Real Image", "confidence": 60.0}
    }
    return render_template("index.html", result="Fake Image", confidence=82.5, details=details, record_count=42, active_tab="scan")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)  # auto-restarts on any .py change
