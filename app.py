import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dataset.utils.predictor import predict_image
from database import init_db, save_detection, get_all_detections

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
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB server-side hard limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def allowed_file(filename: str) -> bool:
    """Return True only if *filename* has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialise DB on startup — creates the table if it doesn't exist
init_db()


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
            # predict_image now returns (label, confidence_pct)
            result, confidence = predict_image(filepath)
        except Exception as e:
            flash(f"Prediction error: {e}")
            return redirect(url_for("index"))

        # Persist detection result including confidence score
        save_detection(filename=filename, result=result, confidence=confidence)

    return render_template("index.html", result=result, confidence=confidence)


@app.route("/dashboard")
def dashboard():
    data = get_all_detections()
    return render_template("dashboard.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)