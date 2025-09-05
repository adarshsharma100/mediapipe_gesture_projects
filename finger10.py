import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define fingertip landmarks (Thumb, Index, Middle, Ring, Pinky)
FINGERTIP_IDS = [4, 8, 12, 16, 20]

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Convert back to BGR for OpenCV rendering
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw all hand landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Highlight fingertips
            for idx, landmark in enumerate(hand_landmarks.landmark):
                if idx in FINGERTIP_IDS:
                    # Get the pixel coordinates of the fingertip
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    # Draw a circle at the fingertip
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                    # Label the fingertip
                    finger_names = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}
                    cv2.putText(frame, finger_names[idx], (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Five Fingers Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()