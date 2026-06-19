# German Traffic Sign Detection and Classification

This project uses a combined **YOLOv8-based object detection** and **CNN-based classification** pipeline to identify and label German traffic signs in images. A web application built with Flask allows users to upload images and view the detected and classified traffic signs.

---

## Project Structure

```text
German_Traffic_Signs_YOLO-CNN/
├── app.py                     # Flask backend serving the web application
├── integrator.py              # Core logic combining YOLO and CNN predictions
├── label_map.json             # Mapping from CNN class indices to sign names
├── requirements.txt           # Python dependencies
├── models/                    # Directory containing pre-trained models
│   ├── best.pt                # YOLOv8 object detection weights
│   └── the_cnn_model.h5       # CNN classification model
├── notebooks/                 # Jupyter Notebooks for training
│   ├── CNN_Training.ipynb     # CNN classification model training
│   └── YOLO_Training.ipynb    # YOLOv8 model training
├── static/                    # Static files (CSS, Uploads)
│   ├── css/                   # Stylesheets
│   │   └── style.css
│   └── uploads/               # Directory for uploaded and processed images
├── templates/                 # HTML templates
│   └── index.html             # Web UI
└── test_images/               # Sample images for testing
```

---

## Technologies Used

- **Python**
- **YOLOv8 (Ultralytics)**: Object detection
- **TensorFlow/Keras**: Image classification
- **Flask**: Web backend
- **OpenCV**, **NumPy**: Image processing

---

## Getting Started (Setup Manual)

Follow these instructions to get the application up and running on your local machine.

### 1. Clone the repository

```bash
git clone https://github.com/Vardhan1105123/German_Traffic_Signs_YOLO-CNN.git
cd German_Traffic_Signs_YOLO-CNN
```

### 2. Create a Virtual Environment (Recommended)

It is highly recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Start the Flask web server:

```bash
python app.py
```

### 5. Access the Web UI

Open your web browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

You can now upload images (from the `test_images/` directory or your own) to detect and classify German traffic signs!
