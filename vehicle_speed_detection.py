import cv2
import numpy as np

# Paths to YOLO files
weights_path = r"./yolo/yolov4.weights"
config_path = r"./yolo/yolov4.cfg"
names_path = r"./yolo/coco.names"

# Load the YOLO model
net = cv2.dnn.readNet(weights_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load COCO class labels
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load video
video_path = r"D:\Vehicle speed detection\12691893_3840_2160_30fps.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Video display size
resize_width, resize_height = 1280, 720

# Video FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# Known real-world distance for speed calculation
known_distance = 150  # Adjust this based on your scenario

# Variables for tracking vehicles
previous_frame_boxes = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (resize_width, resize_height))
    height, width = frame.shape[:2]

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, and centers
    boxes, confidences, class_ids, centers = [], [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] in ["car", "truck", "bus", "motorbike"]:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append bounding box coordinates and center
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centers.append((center_x, center_y))

    # Perform Non-Maximum Suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    current_frame_boxes = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            center_x, center_y = centers[i]

            matched = False
            for prev_box in previous_frame_boxes:
                prev_x, prev_y, prev_w, prev_h, prev_center_x, prev_center_y = prev_box
                distance_pixels = np.linalg.norm(np.array([center_x, center_y]) - np.array([prev_center_x, prev_center_y]))

                if distance_pixels < 50:  # Threshold for matching
                    matched = True
                    time_seconds = 1 / fps
                    speed_mps = (distance_pixels / width) * known_distance / time_seconds
                    speed_kmph = speed_mps * 3.6

                    # Determine direction
                    dx = center_x - prev_center_x
                    dy = center_y - prev_center_y
                    direction = ""
                    if abs(dx) > abs(dy):
                        direction = "Right" if dx > 0 else "Left"
                    else:
                        direction = "Down" if dy > 0 else "Up"

                    # Display speed and direction
                    cv2.putText(frame, f"Speed: {speed_kmph:.2f} km/h, {direction}", (x, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Add the current bounding box with center for tracking
            current_frame_boxes.append([x, y, w, h, center_x, center_y])
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update the previous frame boxes
    previous_frame_boxes = current_frame_boxes

    # Display the frame
    cv2.imshow("Vehicle Speed and Direction Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
