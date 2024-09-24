import cv2
import torch
import numpy as np
from collections import deque
import os

# Path to your uploaded local video file
video_path = "C:\\Users\\abhis\\Gen al learning\\Reak time vehicle detection and Counting\\Cars Moving On Road Stock Footage - Free Download.mp4"
  # Update with the correct local path of your video

# Check if the video exists in the specified path
if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found.")
else:
    # Load YOLOv5 Model (Pre-trained on COCO dataset)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create a deque to store object IDs to track
    vehicle_counter = deque(maxlen=100)  # Limited to the last 100 vehicle detections

    # Initialize a counter for vehicles
    vehicle_count = 0

    def filter_vehicles(results):
        vehicles = []
        for *box, conf, cls in results:
            if int(cls) in [2, 3, 5, 7]:  # Car, motorcycle, bus, truck classes in COCO
                vehicles.append((*box, conf, cls))
        return vehicles

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get YOLO predictions for the frame
        results = model(frame)

        # Filter to retain only vehicles
        vehicle_detections = filter_vehicles(results.xyxy[0].cpu().numpy())  # Get predictions as NumPy array

        for *box, conf, cls in vehicle_detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Track unique vehicles using the centroid of the bounding box
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            vehicle_counter.append((centroid_x, centroid_y))

        for i in range(1, len(vehicle_counter)):
            if vehicle_counter[i - 1] is None or vehicle_counter[i] is None:
                continue
            # Check if a vehicle has crossed a defined line (simulating passing through a road)
            if vehicle_counter[i - 1][1] < 300 and vehicle_counter[i][1] >= 300:  # Arbitrary line at y=300
                vehicle_count += 1

        # Display the current vehicle count on the frame
        cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the video feed with bounding boxes and labels
        cv2.imshow('YOLOv5 Vehicle Detection', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
