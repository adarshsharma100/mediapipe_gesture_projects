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

# Button properties (dictionary for each button)
buttons = {
    'LED': {'x': 100, 'y': 150, 'w': 100, 'h': 50, 'state': False},
    'Buzzer': {'x': 250, 'y': 150, 'w': 100, 'h': 50, 'state': False},
    'Relay': {'x': 400, 'y': 150, 'w': 100, 'h': 50, 'state': False}
}

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
    
    # Draw all buttons
    for button_name, props in buttons.items():
        button_color = (0, 255, 0) if props['state'] else (0, 0, 255)
        cv2.rectangle(frame, (props['x'], props['y']), 
                     (props['x'] + props['w'], props['y'] + props['h']), 
                     button_color, -1)
        status_text = "ON" if props['state'] else "OFF"
        text_x = props['x'] + 10 if button_name != 'Relay' else props['x'] + 5  # Adjust text position
        cv2.putText(frame, button_name, 
                   (text_x, props['y'] + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status_text, 
                   (text_x, props['y'] + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
            
            # Check each button
            for button_name, props in buttons.items():
                finger_over_button = (props['x'] <= index_x <= props['x'] + props['w'] and 
                                    props['y'] <= index_y <= props['y'] + props['h'])
                
                # Detect click (when index and thumb are close together over button)
                if finger_over_button and distance < 30:  # 30 pixels threshold for click
                    # Toggle button state (with simple debouncing)
                    cv2.waitKey(200)  # Delay to prevent multiple toggles
                    buttons[button_name]['state'] = not buttons[button_name]['state']
            
            # Draw circle at index finger tip
            cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), -1)
    
    # Display the frame
    cv2.imshow('Virtual Control Panel with Hand Tracking', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()