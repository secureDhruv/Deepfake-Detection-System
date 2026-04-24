import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dataset.utils.predictor import predict_image, get_loaded_model_names
from database import init_db, save_detection, get_all_detections, get_detection_by_id

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

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None

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

        # Sanitize filename to prevent path-traversal attacks
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            # predict_image now returns (label, confidence_pct, details)
            result, confidence, details = predict_image(filepath)
        except Exception as e:
            flash(f"Prediction error: {e}")
            return redirect(url_for("index"))

        # Persist detection result including confidence score and details summary
        details_json = json.dumps(details)
        new_record_id = save_detection(filename=filename, result=result, confidence=confidence, details=details_json)

    record_count = len(get_all_detections())
    
    # We provide 'details' if available from a POST, otherwise None
    return render_template("index.html", result=result, confidence=confidence, details=locals().get('details'), new_record_id=locals().get('new_record_id'), record_count=record_count)


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
    return render_template("analysis.html", record=record)


@app.route("/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id: int):
    """Delete a detection record from the database."""
    import sqlite3 as _sqlite3
    from database import DB_PATH
    conn = _sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM detections WHERE id = ?", (record_id,))
        conn.commit()
    finally:
        conn.close()
    flash(f"Record #{record_id} deleted successfully.")
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
    import sqlite3 as _sqlite3
    from database import DB_PATH
    conn = _sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM detections")
        conn.commit()
    finally:
        conn.close()
    return "", 204


@app.route("/clear-uploads", methods=["POST"])
def clear_uploads():
    """Delete all files in the uploads folder."""
    import glob
    for f in glob.glob(os.path.join(UPLOAD_FOLDER, "*")):
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
    app.run(debug=True)
