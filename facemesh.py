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

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Eye aspect ratio (EAR) threshold
EAR_THRESHOLD = 0.25

# GPIO Initialization
LED_PIN = 12
h = None  # Initialize GPIO handle

try:
    h = lgpio.gpiochip_open(0)  # Open GPIO chip
    lgpio.gpio_claim_output(h, LED_PIN)  # Set LED pin as output
except lgpio.error as e:
    print(f"Error initializing GPIO: {e}")
    exit(1)

def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR) for a given eye."""
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (v1 + v2) / (2.0 * h)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    print("\033c", end="")

    if results.multi_face_landmarks:
        for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

            h, w, _ = img.shape
            left_eye = [
                np.array([face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h])
                for i in [33, 160, 158, 133, 153, 144]
            ]
            right_eye = [
                np.array([face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h])
                for i in [362, 385, 387, 263, 373, 380]
            ]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            eye_open = avg_ear > EAR_THRESHOLD
            text_status = "Open" if eye_open else "Closed"
            nose_tip = (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h))
            text_pos = (nose_tip[0] - 50, nose_tip[1] - 50)
            cv2.putText(img, f"Face {face_idx + 1}: {text_status}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Face {face_idx + 1}: Eyes {text_status} (EAR: {avg_ear:.2f})")

            if h is not None:
                try:
                    lgpio.gpio_write(h, LED_PIN, 1 if eye_open else 0)
                except lgpio.error as e:
                    print(f"Error writing to GPIO: {e}")
    
    cv2.imshow("Face Detection with Eye Status", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if h is not None:
    lgpio.gpiochip_close(h)
