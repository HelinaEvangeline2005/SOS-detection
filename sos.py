import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=21, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
# Function to detect if thumb is touching index, middle, or ring finger tips
def is_thumb_touching_fingers(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    
    # Calculate the Euclidean distance between thumb and finger tips
    thumb_index_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    thumb_middle_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([middle_tip.x, middle_tip.y]))
    thumb_ring_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([ring_tip.x, ring_tip.y]))
    
    # If the thumb is touching any of the fingers, return True
    return thumb_index_distance < 0.05 or thumb_middle_distance < 0.05 or thumb_ring_distance < 0.05

# Function to check if all other fingers are bent
def are_all_fingers_bent(landmarks):
    fingers = [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]
    
    # Check if the tip of each finger is below its corresponding proximal interphalangeal joint (PIP)
    for tip, pip in fingers:
        if landmarks[tip].y < landmarks[pip].y:  # If the tip is higher than the PIP, the finger is not bent
            return False
    return True

# Initialize video capture
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        height, width, _ = frame.shape

        # Convert the frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hand landmarks
        hand_results = hands.process(rgb_frame)

        # Flag to indicate if danger is detected
        danger_detected = False

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = hand_landmarks.landmark

                # Check both conditions:
                # 1. Thumb (Point 1) touches Index (Point 8), Middle (Point 12), or Ring (Point 16) Finger
                # 2. All other fingers are bent
                if is_thumb_touching_fingers(landmarks) and are_all_fingers_bent(landmarks):
                    danger_detected = True

        # Display "DANGER" if detected in any hand
        if danger_detected:
            cv2.putText(frame, "DANGER", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)

        # Display the frame with landmarks and text
        cv2.imshow("Hand SOS Gesture Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()