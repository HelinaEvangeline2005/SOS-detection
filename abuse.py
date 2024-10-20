import cv2
import numpy as np
import torch
from yolov5 import YOLOv5
import mediapipe as mp

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize YOLOv5 model
model = YOLOv5("yolov5s.pt")  # Update this path if needed

def detect_abuse(frame):
    # Perform object detection with YOLOv5
    results = model.predict(frame)  # Use the correct method for inference

    # Extract bounding boxes and labels
    boxes = results.xyxy[0].cpu().numpy()  # Convert results to numpy array
    class_ids = results.names

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform hand detection with MediaPipe
    hand_results = hands.process(frame_rgb)
    
    # Check for detected hands
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Convert frame to grayscale for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and approximate bounding boxes
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Minimum area to filter out small contours
            cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)

    # Draw bounding boxes on detected objects
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        label = f"{class_ids[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Abuse Detection with Pose Estimation', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Open a video file or capture from webcam
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detect_abuse(frame)
    
    cap.release()

if _name_ == "_main_":
    main()