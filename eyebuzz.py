import cv2
import mediapipe as mp
import numpy as np
import lgpio
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# GPIO Pins
LED_PIN = 12
BUZZER_PIN = 22

# Initialize GPIO
h = lgpio.gpiochip_open(0)

# Eye aspect ratio (EAR) threshold
EAR_THRESHOLD = 0.25

def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR) for a given eye."""
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (v1 + v2) / (2.0 * h)
    return ear

def safe_claim_output(pin, initial_value):
    """Safely claim GPIO pin as output"""
    try:
        lgpio.gpio_claim_output(h, pin)
        lgpio.gpio_write(h, pin, initial_value)
    except lgpio.error as e:
        print(f"Error claiming pin {pin}: {e}")
        lgpio.gpiochip_close(h)
        exit(1)

# Initialize actuator pins
for pin in [LED_PIN, BUZZER_PIN]:
    safe_claim_output(pin, 0)

# Initialize the webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from webcam.")
            break

        # Flip and convert image
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process face detection
        results = face_mesh.process(img_rgb)
        
        # Clear terminal
        print("\033c", end="")

        if results.multi_face_landmarks:
            # Use the first detected face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

            # Get image dimensions
            h_img, w, _ = img.shape
            
            # Left eye landmarks
            left_eye = [
                np.array([face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h_img]),
                np.array([face_landmarks.landmark[160].x * w, face_landmarks.landmark[160].y * h_img]),
                np.array([face_landmarks.landmark[158].x * w, face_landmarks.landmark[158].y * h_img]),
                np.array([face_landmarks.landmark[133].x * w, face_landmarks.landmark[133].y * h_img]),
                np.array([face_landmarks.landmark[153].x * w, face_landmarks.landmark[153].y * h_img]),
                np.array([face_landmarks.landmark[144].x * w, face_landmarks.landmark[144].y * h_img])
            ]
            
            # Right eye landmarks
            right_eye = [
                np.array([face_landmarks.landmark[362].x * w, face_landmarks.landmark[362].y * h_img]),
                np.array([face_landmarks.landmark[385].x * w, face_landmarks.landmark[385].y * h_img]),
                np.array([face_landmarks.landmark[387].x * w, face_landmarks.landmark[387].y * h_img]),
                np.array([face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h_img]),
                np.array([face_landmarks.landmark[373].x * w, face_landmarks.landmark[373].y * h_img]),
                np.array([face_landmarks.landmark[380].x * w, face_landmarks.landmark[380].y * h_img])
            ]

            # Calculate EAR
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Determine eye status and control actuators
            if avg_ear < EAR_THRESHOLD:
                eye_status = "Closed"
                lgpio.gpio_write(h, LED_PIN, 0)    # LED OFF when eyes closed
                lgpio.gpio_write(h, BUZZER_PIN, 1) # Buzzer ON when eyes closed
                print("Eyes Closed - Buzzer ON")
            else:
                eye_status = "Open"
                lgpio.gpio_write(h, LED_PIN, 1)    # LED ON when eyes open
                lgpio.gpio_write(h, BUZZER_PIN, 0) # Buzzer OFF when eyes open
                print("Eyes Open - LED ON")

            # Display status
            nose_tip = (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h_img))
            text_pos = (nose_tip[0] - 50, nose_tip[1] - 50)
            cv2.putText(img, f"Eyes: {eye_status}", text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(f"EAR: {avg_ear:.2f}")

        # Show image
        cv2.imshow("Face Detection with Eye Status", img)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    # Cleanup
    lgpio.gpio_write(h, LED_PIN, 0)    # Turn off LED
    lgpio.gpio_write(h, BUZZER_PIN, 0) # Turn off Buzzer
    lgpio.gpiochip_close(h)
    cap.release()
    cv2.destroyAllWindows()