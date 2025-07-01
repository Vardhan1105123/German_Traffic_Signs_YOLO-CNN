import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import json
import math

# Load models
yolo_model = YOLO("C:/Users/vardh/Desktop/Traffic_Signs/runs/detect/the_yolov8n_model/weights/best.pt")
cnn_model = load_model("C:/Users/vardh/Desktop/Traffic_Signs/models/the_cnn_model.h5")

# Load CNN label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)

def run_integrated_detection(image_path, tile_size=160 , output_path="static/output.jpg"):
    i_image = cv2.imread(image_path)
    image = cv2.resize(i_image, (256, 256))
    #image = cv2.imread(image_path)
    h, w, _ = image.shape
    full_result = image.copy()
    detected_labels = []

    # Tiling
    n_tiles_x = math.ceil(w / tile_size)
    n_tiles_y = math.ceil(h / tile_size)

    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            x_start = j * tile_size
            y_start = i * tile_size
            x_end = min((j + 1) * tile_size, w)
            y_end = min((i + 1) * tile_size, h)

            tile = image[y_start:y_end, x_start:x_end]

            # ✅ Run YOLO on the tile
            yolo_results = yolo_model(tile, conf = 0.4)[0]

            for box in yolo_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                global_x1 = x_start + x1
                global_y1 = y_start + y1
                global_x2 = x_start + x2
                global_y2 = y_start + y2

                detected_roi = image[global_y1:global_y2, global_x1:global_x2]
                label = "Unknown"  # Default

                try:
                    if detected_roi.shape[0] < 10 or detected_roi.shape[1] < 10:
                        raise ValueError("ROI too small to process")

                    roi_resized = cv2.resize(detected_roi, (30, 30))
                    roi_array = img_to_array(roi_resized)
                    roi_array = preprocess_input(roi_array)
                    roi_array = np.expand_dims(roi_array, axis=0)

                    prediction = cnn_model.predict(roi_array)
                    #class_idx = int(np.argmax(prediction))
                    confidence = np.max(prediction)
                    if confidence > 0.5:
                        label = label_map[str(np.argmax(prediction))]
                    else :
                        label = "Unknown"
                    #if str(class_idx) in label_map:
                    #    label = label_map[str(class_idx)]
                    #else:
                    #    print(f"Class index {class_idx} not in label_map")

                except Exception as e:
                    print(f"Error classifying with CNN: {e}")

                detected_labels.append(label)

                # Draw label on the image
                cv2.rectangle(full_result, (global_x1, global_y1), (global_x2, global_y2), (0, 255, 0), 2)
                cv2.putText(full_result, label, (global_x1, global_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                

    h, w, _ = i_image.shape
    f_full_result = cv2.resize(full_result, (h, w))
    cv2.imwrite(output_path, f_full_result)
    return output_path, detected_labels
