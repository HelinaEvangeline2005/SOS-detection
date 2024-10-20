import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import torch
import yolov5  # Ensure the correct package is installed

# Load VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define hand detection and tracking model using MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load YOLOv5 model
# Adjust the path if using a different method to load the model
yolo_model = yolov5.load('yolov5x.pt')  # Ensure 'yolov5x.pt' is in the correct directory

# Define abuse detection model (simple placeholder)
abuse_model = Sequential([
    Dense(64, activation='relu', input_shape=(14,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
abuse_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Define function to detect abuse
def detect_abuse(image):
    # Convert image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Detect people using YOLOv5
    results = yolo_model.predict(image)  # Perform detection
    people = results.xyxy[0].numpy()  # Get detection results as numpy array
    
    # Estimate pose for each detected person using VGG16
    poses = []
    for person in people:
        x_min, y_min, x_max, y_max, conf, cls = person
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        
        if x_min >= 0 and y_min >= 0 and x_max > x_min and y_max > y_min:
            roi = image[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                try:
                    roi = cv2.resize(roi, (224, 224))
                    roi = roi / 255.0
                    pose = vgg16.predict(np.expand_dims(roi, axis=0))
                    poses.append(pose)
                except cv2.error as e:
                    print(f"Error resizing ROI for person {person}: {e}")
            else:
                print("Empty ROI detected for person:", person)
        else:
            print("Invalid ROI coordinates for person:", person)
    
    # Detect hands and track movements
    hand_tracks = []
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            x_min = min([lm.x for lm in landmarks.landmark])
            y_min = min([lm.y for lm in landmarks.landmark])
            x_max = max([lm.x for lm in landmarks.landmark])
            y_max = max([lm.y for lm in landmarks.landmark])
            x_min, x_max = int(x_min * image.shape[1]), int(x_max * image.shape[1])
            y_min, y_max = int(y_min * image.shape[0]), int(y_max * image.shape[0])
            
            if x_min >= 0 and y_min >= 0 and x_max > x_min and y_max > y_min:
                roi = image[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    try:
                        roi = cv2.resize(roi, (224, 224))
                        roi = roi / 255.0
                        hand_track = np.mean(roi)  # Placeholder for actual hand tracking
                        hand_tracks.append(hand_track)
                    except cv2.error as e:
                        print(f"Error resizing ROI for hand landmarks: {e}")
                else:
                    print("Empty ROI detected for hand landmarks.")
            else:
                print("Invalid ROI coordinates for hand landmarks.")
    
    # Detect abuse based on poses and hand tracks
    abuse = []
    for i, pose in enumerate(poses):
        for j, hand_track in enumerate(hand_tracks):
            if hand_track > np.mean(pose):  # Placeholder condition
                abuse.append(1)
            else:
                abuse.append(0)
    
    return abuse

# Load video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect abuse
    abuse = detect_abuse(frame)
    
    # Display output
    for i, abuse_label in enumerate(abuse):
        if abuse_label == 1:
            cv2.rectangle(frame, (10, 10), (100, 100), (0, 0, 255), 2)
            cv2.putText(frame, 'Abuse Detected', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow('Abuse Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()