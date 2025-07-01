# German Traffic Sign Detection and Classification 🚦

This project uses a combined **YOLOv8-based object detection** and **CNN-based classification** pipeline to identify and label German traffic signs in large images.

---

## 🧠 Project Structure

- `YOLO_Training.ipynb`: YOLOv8 model training and detection
- `CNN_Training.ipynb`: CNN classification model (TensorFlow/Keras)
- `integrator.py`: Combines YOLO + CNN predictions
- `app.py`: Flask backend serving detection/classification results
- `templates/`, `static/`: HTML + CSS frontend for the web UI

---

## 🏗️ Technologies Used

- **Python**
- **YOLOv8 (Ultralytics)**
- **TensorFlow/Keras or PyTorch**
- **Flask**
- **OpenCV**, **NumPy**, **Matplotlib**

---

## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/Vardhan1105123/German_Traffic_Signs_YOLO-CNN.git
   cd German_Traffic_Signs_YOLO-CNN
