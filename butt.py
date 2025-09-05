import cv2
import mediapipe as mp
import numpy as np
from math import sqrt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Button properties
button_x, button_y = 200, 200
button_w, button_h = 100, 50
button_state = False  # False = OFF, True = ON

def calculate_distance(p1, p2):
    """Calculate distance between two points"""
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for natural view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)
    
    # Draw button
    button_color = (0, 255, 0) if button_state else (0, 0, 255)
    cv2.rectangle(frame, (button_x, button_y), 
                 (button_x + button_w, button_y + button_h), 
                 button_color, -1)
    status_text = "ON" if button_state else "OFF"
    cv2.putText(frame, status_text, 
                (button_x + 20, button_y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get coordinates of index finger tip (landmark 8) and thumb tip (landmark 4)
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            
            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            
            # Calculate distance between index finger and thumb
            distance = calculate_distance((index_x, index_y), (thumb_x, thumb_y))
            
            # Check if finger is over button
            finger_over_button = (button_x <= index_x <= button_x + button_w and 
                                button_y <= index_y <= button_y + button_h)
            
            # Detect click (when index and thumb are close together over button)
            if finger_over_button and distance < 30:  # 30 pixels threshold for click
                # Toggle button state (with simple debouncing)
                cv2.waitKey(200)  # Delay to prevent multiple toggles
                button_state = not button_state
                
            # Draw circle at index finger tip
            cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), -1)
    
    # Display the frame
    cv2.imshow('Virtual Button with Hand Tracking', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()