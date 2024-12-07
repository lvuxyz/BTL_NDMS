from collections import Counter, defaultdict
import cv2
import numpy as np
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import time

# Load pre-trained YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Định nghĩa mapping màu cho bounding box
color_mapping = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'orange': (0, 165, 255),
    'purple': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'brown': (42, 42, 165),
    'grey': (128, 128, 128),
    'unknown': (128, 128, 128),
    'uncertain': (192, 192, 192),
    'error': (0, 0, 0)
}

def get_box_color(color_prediction):
    return color_mapping.get(color_prediction.lower(), (128, 128, 128))


def detect_objects_and_classify_color(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    max_history = 5
    confidence_threshold = 0.6
    object_color_histories = {}

    # Detect objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    color_predictions = []

    # Process each detected object
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            object_id = f"{label}_{x}_{y}_{w}_{h}"

            if object_id not in object_color_histories:
                object_color_histories[object_id] = []

            # Extract and process ROI
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            object_roi = frame[y1:y2, x1:x2]

            try:
                # Preprocess ROI
                gray_roi = cv2.cvtColor(object_roi, cv2.COLOR_BGR2GRAY)
                equalized_roi = cv2.equalizeHist(gray_roi)
                processed_roi = cv2.cvtColor(equalized_roi, cv2.COLOR_GRAY2BGR)

                # Predict color
                cv2.imwrite('test_object.jpg', processed_roi)
                color_histogram_feature_extraction.color_histogram_of_test_image(processed_roi)
                current_color = knn_classifier.main('training.data', 'test.data')

                if current_color:
                    # Update color history
                    object_color_histories[object_id].append(current_color)
                    if len(object_color_histories[object_id]) > max_history:
                        object_color_histories[object_id].pop(0)

                    # Get most common color
                    color_counts = Counter(object_color_histories[object_id])
                    if color_counts:
                        most_common_color = color_counts.most_common(1)[0]
                        if most_common_color[1] / len(object_color_histories[object_id]) >= confidence_threshold:
                            color_prediction = most_common_color[0]
                        else:
                            color_prediction = "uncertain"
                    else:
                        color_prediction = "unknown"
                else:
                    color_prediction = "unknown"

                # Draw results with color-based bounding box
                color_predictions.append(color_prediction)
                box_color = get_box_color(color_prediction)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

                # Add text with black background
                text = f'{label}: {color_prediction}'
                confidence_text = f'Conf: {confidences[i]:.2f}'

                # Draw text background
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x, y - 40), (x + text_width, y - 10), (0, 0, 0), -1)

                # Draw text
                cv2.putText(frame, text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            except Exception as e:
                print(f"Error processing ROI: {e}")
                color_predictions.append("error")
                continue

    return frame, color_predictions


def main():
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            processed_frame, color_predictions = detect_objects_and_classify_color(frame)
            cv2.imshow('Object and Color Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
