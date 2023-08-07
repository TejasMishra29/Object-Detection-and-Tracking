#!/usr/bin/env python
# coding: utf-8

# # Object Detection and Tracking using Yolov5 and Sort Algorithms
Object Detection
# In[1]:


import torch
from PIL import Image
from pathlib import Path
import sys

# Add YOLOv5 repository to the path
sys.path.append(str(Path("path/to/yolov5").resolve()))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, plot_one_box
from utils.datasets import LoadImages

# Load YOLOv5 model
weights = 'path/to/weights.pt'  # Path to YOLOv5 weights file
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = attempt_load(weights, map_location=device)
model.to(device).eval()

# Define class names (customize according to your dataset)
class_names = ['class1', 'class2', 'class3']  # List of class names

# Load image
image_path = 'path/to/your/image.jpg'
img = Image.open(image_path).convert('RGB')

# Perform object detection
img_tensor = torch.from_numpy(img).unsqueeze(0).to(device) / 255.0
results = model(img_tensor)[0]
results = non_max_suppression(results, conf_thres=0.3, iou_thres=0.5)[0]

# Display results on the image
for *xyxy, conf, cls in results:
    xyxy = scale_coords(img_tensor.shape[2:], xyxy, img.size).round()
    plot_one_box(xyxy, img, label=f'{class_names[int(cls)]} {conf:.2f}')

img.show()  # Display the annotated image

Object Tacking
# In[ ]:


import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from sort import Sort

# Initialize SORT tracker
tracker = Sort()

# Initialize Kalman filter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 1, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 1],
                 [0, 0, 0, 1]])  # State transition matrix
kf.H = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0]])  # Measurement matrix
kf.P *= 10.0  # Initial covariance matrix
kf.R *= 1.0   # Measurement noise
kf.Q *= 0.01  # Process noise

# Open video capture
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using an object detection model (not provided in this example)
    # You'll need to replace this with your object detection code
    
    # Get detection results (boxes and scores)
    detections = [(x1, y1, x2, y2, score) for (x1, y1, x2, y2), score in detection_results]
    
    # Update SORT tracker
    trackers = tracker.update(np.array(detections))

    # Draw bounding boxes and IDs on the frame
    for d in trackers:
        x1, y1, x2, y2, track_id = map(int, d)
        
        # Predict using Kalman filter
        predicted = kf.predict()
        kf.update(np.array([[x1 + (x2 - x1) / 2], [y1 + (y2 - y1) / 2]]))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Track ID: {int(track_id)}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

