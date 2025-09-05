import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get the landmark of the wrist (landmark index 0)
            wrist_x = hand_landmarks.landmark[0].x * w
            wrist_y = hand_landmarks.landmark[0].y * h
            
            # Determine hand position relative to the frame
            position = ""
            if wrist_x < w / 3:
                position += "Left "
            elif wrist_x > 2 * w / 3:
                position += "Right "
            
            if wrist_y < h / 3:
                position += "Up"
            elif wrist_y > 2 * h / 3:
                position += "Down"
            
            if position:
                print(position.strip())
                cv2.putText(frame, position.strip(), (int(wrist_x), int(wrist_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()