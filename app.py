from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from integrator import run_integrated_detection
import os
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            # Generate a unique filename to prevent collisions and path issues
            ext = os.path.splitext(file.filename)[1]
            if not ext:
                ext = ".jpg"
            unique_filename = str(uuid.uuid4()) + ext
            
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(img_path)
            
            output_filename = "output_" + unique_filename
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            
            result_path, detected_signs = run_integrated_detection(img_path, output_path=output_path)
            
            if result_path is None:
                if os.path.exists(img_path):
                    os.remove(img_path)
                return render_template("index.html", result=None, signs=None, error="Invalid image file uploaded.")
            
            # Use url_for to properly link static files
            result_url = url_for('static', filename=f'uploads/{output_filename}')
            
            return render_template("index.html", result=result_url, signs=detected_signs)

    return render_template("index.html", result=None, signs=None)

if __name__ == "__main__":
    app.run(debug=True)
