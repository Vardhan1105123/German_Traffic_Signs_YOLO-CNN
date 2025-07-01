from flask import Flask, render_template, request
from integrator import run_integrated_detection
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)
            result_path, detected_signs = run_integrated_detection(img_path)
            return render_template("index.html", result=result_path, signs=detected_signs)

    return render_template("index.html", result=None, signs=None)

if __name__ == "__main__":
    app.run(debug=True)
