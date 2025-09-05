import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Finger tip landmarks indices
finger_tips = [4, 8, 12, 16, 20]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip and convert frame to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = hands.process(rgb_frame)
    
    # Count fingers
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Count raised fingers
            fingers = []
            for tip in finger_tips:
                # Check if finger tip is above finger base (MCP joint)
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
            
            # Special case for thumb
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                fingers[0] = 1
            else:
                fingers[0] = 0
                
            # Count total raised fingers
            count = sum(fingers)
            
            # Display count
            cv2.putText(frame, f'Fingers: {count}', (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Hand Finger Counting', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()